import email
import logging
import re
import string
from email.header import decode_header
from imaplib import IMAP4
from itertools import batched
import time

import backoff
import bs4
import imapclient
from imapclient.exceptions import LoginError, IMAPClientError

from settings import Settings

_RE_SYMBOL_SEQ = re.compile(r"(?<=\s)\W+(?=\s)")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_COMBINE_CR = re.compile(r"\n+")
_RE_NO_ARROWS = re.compile(r"^([>])+", re.MULTILINE)
_BATCH_SIZE = 40

HEADER_KEY = b"BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]"
BODY_KEY = b"BODY[]"


def html2text(html: str) -> str:
    """Convert html to plain-text using beautifulsoup"""
    soup = bs4.BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = "".join(filter(lambda x: x in string.printable, text))
    text = re.sub(r"&[a-z]{3,4};", " ", text)
    return text


def get_header(raw_header, key):
    header_str = ""
    email_headers = email.message_from_bytes(raw_header)
    header = email_headers[key]
    if header is None:
        return header_str

    for _header, c_set in decode_header(header):
        if isinstance(_header, str):
            header_str = f"{header_str} {_header}"
        else:
            try:
                header_str = f"{header_str} {_header.decode(c_set or "utf-8")}"
            except LookupError:
                # Retry with utf-8, if we didn't find the codec.
                header_str = f"{header_str} {_header.decode("utf-8")}"

    return header_str.strip()


def mesg_to_text(mesg: email.message.Message) -> str:
    """Convert an email message to plain-text"""
    text = ""
    for part in mesg.walk():
        charset = part.get_content_charset() or "utf-8"
        if part.get_content_type() == "text/plain":
            text += part.get_payload(decode=True).decode(charset, errors="ignore")
        elif part.get_content_type() == "text/html":
            text += html2text(
                part.get_payload(decode=True).decode(charset, errors="ignore")
            )

    text = _RE_SYMBOL_SEQ.sub("", text)
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_COMBINE_CR.sub(" ", text)
    text = _RE_NO_ARROWS.sub("", text)
    return text


class ImapHandler:
    def __init__(self, settings: Settings) -> None:
        self.__settings = settings
        self.__imap_conn = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_connection(self):
        return self.__imap_conn

    def connect_imap(self) -> bool:
        if not self.__settings.imap_host or not self.__settings.username:
            return False

        try:
            self.__imap_conn = imapclient.IMAPClient(
                self.__settings.imap_host, ssl=True
            )
            self.__imap_conn.login(self.__settings.username, self.__settings.password)
        except LoginError as e:
            self.logger.error("Could not login to %s", self.__settings.imap_host)
            return False
        except Exception as e:
            self.logger.error(
                "Unknow error logging in to imap server %s", self.__settings.imap_host
            )
            self.logger.error(e, exc_info=True)
            return False

        self.logger.debug(self.__imap_conn.capabilities())

        return True

    def __del__(self):
        self.close()

    def __reconnect(self):
        if self.__imap_conn is not None:
            self.logger.warning("Error communicating with IMAP server. Reconnecting.")
            self.connect_imap()

    def close(self):
        if self.__imap_conn is not None:
            self.logger.debug("Cleaning up imap connection")
            try:
                self.__imap_conn.logout()
            except IMAP4.error as se:
                self.logger.debug("Error logging out from imap.")
            except Exception as e:
                self.logger.debug("Other error %s", e)
            finally:
                self.__imap_conn = None

    def list_folders(self) -> list[str]:
        try:
            return self.__list_folders()
        except IMAPClientError as e:
            self.logger.warning(
                "Got exception on listing folders, will reconnect %s", e
            )
            self.__reconnect()
            return self.__list_folders()

    def __list_folders(self) -> list[str]:
        return [t[2] for t in self.__imap_conn.list_folders()]

    def fetch(self, uids: list) -> dict:
        """Will fetch a set of email based on the list of uids. Any list extending
            a certain size will be batched.

        Args:
            uids (list): a list of uids for emails to be feteched

        Returns:
            dict: All fetched emails
        """
        try:
            return self.__fetch(uids)
        except IMAPClientError as e:
            self.logger.warning("Got exception on fetching, will reconnect %s", e)
            self.__reconnect()
            return self.__fetch(uids)

    def __fetch(self, uids) -> dict:
        all_mails = {}
        batched_uids = list(batched(uids, _BATCH_SIZE))
        index = 0
        for uid_batch in batched_uids:
            index += 1
            self.logger.info("\t Batch %i/%i", index, len(batched_uids))
            time.sleep(1)
            all_mails.update(self.__fetch_batch(uid_batch))

        return all_mails

    @backoff.on_exception(
        backoff.expo,
        (IOError, IMAP4.error, IMAPClientError),
        max_tries=4,
        on_backoff=lambda self, details: self.logger.warning(
            "Backing off %0.1f seconds after %i tries",
            details["wait"],
            details["tries"],
        ),
    )
    def __fetch_batch(self, uid_batch: list):
        return self.__imap_conn.fetch(uid_batch, [HEADER_KEY, BODY_KEY])

    def search(self, folder: str, search_args=None) -> list[int]:
        """Searches for messages in imap folder

        Args:
            folder (str): the folder
            search_args (Any, optional): Search criteria. Defaults to ["ALL"].

        Returns:
            list: list of uids
        """
        try:
            return self.__search(folder, search_args)
        except IMAPClientError as e:
            self.logger.warning("Got exception on searching, will reconnect %s", e)
            self.__reconnect()
            return self.__search(folder, search_args)

    def __search(self, folder: str, search_args=None) -> list[int]:
        if search_args is None:
            search_args = ["ALL"]
        self.__imap_conn.select_folder(folder)
        results = self.__imap_conn.search(search_args)
        return results

    @backoff.on_exception(
        backoff.expo,
        (IOError, IMAP4.error, IMAPClientError),
        max_tries=2,
    )
    def move(
        self,
        folder: str,
        uids: list,
        dest_folder: str,
        flag_messages=True,
        flag_unseen=True,
    ) -> int:
        """Move a message from one folder to another

        Args:
            folder (str): source folder
            uids (list): message uids
            dest_folder (str): destination folder
            flag_messages (bool, optional): Whether to flag messages moved. Defaults to True.
        Returns:
            int: number of messages moved
        """
        if not isinstance(uids, list):
            self.logger.error(
                "Expected the uids to be a list \
                              moving from folder %s to folder %s",
                folder,
                dest_folder,
            )
            raise ValueError("Expected uids to be a list")
        self.__imap_conn.select_folder(folder)
        if flag_messages:
            self.__imap_conn.add_flags(uids, [imapclient.FLAGGED])
        if flag_unseen:
            # For some reason it seems like adding a flag also put a message as "SEEN"
            # thus we need to make it "UNSEEN" again
            self.__imap_conn.remove_flags(uids, [imapclient.SEEN])

        # Move in imap is a combination of operations. Copy, delete and expunge.
        self.__imap_conn.copy(uids, dest_folder)
        self.__imap_conn.add_flags(uids, [imapclient.DELETED], silent=True)
        self.__imap_conn.uid_expunge(uids)
        self.logger.info(
            "REALLY moved from %s to %s: %i", folder, dest_folder, len(uids)
        )
        return len(uids)

    def parse_mesg(self, mesg: dict) -> dict:
        """Parse a raw message into a string

        Args:
            mesg (dict): the message

        Returns:
            dict: the message as a string
        """
        raw_header = mesg[HEADER_KEY]
        raw_body = mesg[BODY_KEY]
        payload = email.message_from_bytes(raw_body)
        body_text = mesg_to_text(payload)

        to_addr = get_header(raw_header, "TO")
        to_addr += get_header(raw_header, "CC")
        from_addr = get_header(raw_header, "FROM")
        subject = get_header(raw_header, "SUBJECT").removeprefix("**SPAM**").strip()

        mesg_dict = {
            "from": from_addr,
            "tocc": to_addr,
            "body": f"Subject: {subject}. {body_text}",
        }

        return mesg_dict
