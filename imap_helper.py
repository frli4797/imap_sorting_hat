import email
import imaplib
import logging
import re
import string

import bs4
import imapclient

re_header_item = re.compile(r"(\w+): (.*)")
re_address = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
re_newline = re.compile(r"[\r\n]+")
re_symbol_sequence = re.compile(r"(?<=\s)\W+(?=\s)")
re_whitespace = re.compile(r"\s+")

hkey = b"BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]"
bkey = b"BODY[]"


def html2text(html: str) -> str:
    """Convert html to plain-text using beautifulsoup"""
    soup = bs4.BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = "".join(filter(lambda x: x in string.printable, text))
    text = re.sub(r"&[a-z]{3,4};", " ", text)
    return text


def mesg_to_text(mesg: email.message.Message) -> str:
    """Convert an email message to plain-text"""
    text = ""
    for part in mesg.walk():
        if part.get_content_type() == "text/plain":
            text += part.get_payload(decode=True).decode("utf-8", errors="ignore")
        elif part.get_content_type() == "text/html":
            text += html2text(
                part.get_payload(decode=True).decode("utf-8", errors="ignore")
            )

    text = re_symbol_sequence.sub("", text)
    text = re_whitespace.sub(" ", text)
    return text


class ImapHelper:
    def __init__(self, settings) -> None:
        self.__settings = settings
        self.__imap_conn = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect_imap(self) -> bool:
        if not self.__settings["host"] or not self.__settings["username"]:
            return False

        try:
            self.__imap_conn = imapclient.IMAPClient(self.__settings["host"], ssl=True)
            self.__imap_conn.login(
                self.__settings["username"], self.__settings["password"]
            )
        except imaplib.IMAP4.error as e:
            self.logger.error(e)
            return False
        self.logger.debug(self.__imap_conn.capabilities())

        return True

    def list_folders(self) -> list[str]:
        return [t[2] for t in self.__imap_conn.list_folders()]

    def fetch(self, uids) -> dict:
        return self.__imap_conn.fetch(uids, [hkey, bkey])

    def search(self, folder: str, search_args=None) -> list[int]:
        """Searches for messages in imap folder

        Args:
            folder (str): the folder
            search_args (Any, optional): Search criteria. Defaults to ["ALL"].

        Returns:
            list: list of uids
        """
        if search_args is None:
            search_args = ["ALL"]
        self.__imap_conn.select_folder(folder)
        results = self.__imap_conn.search(search_args)
        return results

    def move(
        self, folder: str, uids: list, dest_folder: str, flag_messages=True
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
            # imap_conn.remove_flags([uid], [imapclient.SEEN])
        self.__imap_conn.copy(uids, dest_folder)
        self.__imap_conn.add_flags(uids, imapclient.DELETED, silent=True)
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
        header = mesg[hkey].decode("utf-8")
        raw_body = mesg[bkey]
        payload = email.message_from_bytes(raw_body)
        body_text = mesg_to_text(payload)
        header_lines = re_newline.split(header)

        header_dict = {}
        for item in header_lines:
            m = re_header_item.match(item)
            if m:
                header_dict[m.group(1)] = m.group(2)

        # remove spam prefix because we want spam training data to be as similar as possible
        # to non-spam training data
        header_dict["Subject"] = (
            header_dict.get("Subject", "").removeprefix("**SPAM**").strip()
        )

        from_addr = []
        to_addr = []

        try:
            from_addr = [m.group(1) for m in re_address.finditer(header_dict["FROM"])]
            to_addr = [
                m.group(1)
                for m in re_address.finditer(
                    header_dict["TO"] + header_dict.get("CC", "")
                )
            ]
        except KeyError:
            self.logger.warning("Did not find a TO or CC in the headers.")

        mesg_dict = {
            "from": from_addr,
            "tocc": to_addr,
            "body": f'Subject: {header_dict["Subject"]}. {body_text}',
        }

        return mesg_dict
