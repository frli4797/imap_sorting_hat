import logging
import shelve
from hashlib import sha256
from itertools import batched
from os.path import join
from time import perf_counter
from typing import Dict, List

import backoff
import numpy as np
from openai import OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse

base_logger = logging.getLogger("emailservice")

embed_max_chars = 16384


class EmailFeatureService:
    def __init__(
        self, settings, imap_handler, openai_client, dry_run=False, interactive=False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.setLevel(logging.DEBUG)
        self.__settings = settings

        self.__interactive = interactive
        self.__dry_run = dry_run
        self.__imap_conn = imap_handler
        self.__client: OpenAI = openai_client

    @property
    def msgs_file(self) -> str:
        return join(self.__settings.data_directory, "msgs")

    @property
    def embd_file(self) -> str:
        return join(self.__settings.data_directory, "embd")

    @property
    def interactive(self) -> bool:
        return self.__interactive

    def get_msgs(self, folder: str, uids: List[int]) -> Dict[int, str]:
        """Fetch new messages through cache {uid: 'msg'}

        Args:
            folder (str): source folder
            uids (List[int]): uids

        Returns:
            Dict[int, str]: message body by uid
        """
        d = {}
        new_uids = []
        imap_conn = self.__imap_conn
        self.logger.info("Getting %i messages from %s", len(uids), folder)
        with shelve.open(self.msgs_file, writeback=False) as fm:
            # Check if the message/folder combination is in the cache.
            for uid in uids:
                if f"{folder}:{uid}" not in fm:
                    new_uids.append(uid)
                else:
                    msg_hash = fm[f"{folder}:{uid}"]
                    mesg = fm[f"{msg_hash}.mesg"]
                    d[uid] = mesg
                    continue
            # If not in the cache, fetch the message and put in cache.
            if len(new_uids) > 0:
                self.logger.debug("Found %s messages not in cache", len(new_uids))
                msgs = imap_conn.fetch(new_uids)
                for uid in new_uids:
                    mesg = imap_conn.parse_mesg(msgs[uid])
                    msg_hash = self.__mesg_hash(mesg)
                    fm[f"{folder}:{uid}"] = msg_hash
                    fm[f"{msg_hash}.mesg"] = mesg
                    d[uid] = mesg
                self.logger.debug("Found %i messages to cache", len(new_uids))
        self.logger.info("Total messages found/added %i in %s.", len(d), folder)
        return d

    def get_embeddings(self, folder: str, uids: List[int]) -> Dict[int, np.ndarray]:
        """Get embeddings using OpenAI API through cache {uid: embedding}

        Args:
            folder (str): the source folder
            uids (List[int]): uids to get embeddings for

        Returns:
            Dict[int, np.ndarray]: embeddings by uid
        """
        dhash = {}
        dembd = {}
        self.logger.info("Getting %i embeddings from %s", len(uids), folder)

        with shelve.open(self.msgs_file, writeback=False) as fm:
            new_uids = []  # uids that need a new hash
            # Check which folder/messages that are not in the cache.
            for uid in uids:
                if f"{folder}:{uid}" in fm:
                    dhash[uid] = fm[f"{folder}:{uid}"]
                else:
                    new_uids.append(uid)
                    continue

            self.logger.debug(
                "Found %i out of %i needing hash", len(new_uids), len(uids)
            )
            dmesg = self.get_msgs(folder, new_uids)

            self.logger.debug("Adding hashes for %i messages", len(dmesg))
            # Saving hashes for unseen messages.
            for uid, mesg in dmesg.items():
                msg_hash = self.__mesg_hash(mesg)
                dhash[uid] = msg_hash
                fm[f"{folder}:{uid}"] = msg_hash
            self.logger.debug("Added hashes for messages.")
            fm.close()

            new_uids = []  # uids that need a new embedding
            self.logger.debug("Finding embedding for %s messages", len(uids))
        with shelve.open(self.embd_file, writeback=False) as fe:
            for uid, msg_hash in dhash.items():  # dhash is {uid: hash}
                if f"{msg_hash}.embd" in fe:
                    embd = fe[f"{msg_hash}.embd"]
                    dembd[uid] = embd
                    continue
                else:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                self.logger.debug("Found %i embeddings not in cache", len(new_uids))
                msgs = self.get_msgs(folder, new_uids)

                t_batch_0 = perf_counter()
                embeddings = self.__get_embeddings(
                    list(msg["body"][:embed_max_chars] for msg in msgs.values())
                )
                t_batch_1 = perf_counter()
                if len(embeddings) == len(msgs.keys()):
                    self.logger.debug(
                        "Took %.2f to BATCH %i embedings.",
                        t_batch_1 - t_batch_0,
                        len(embeddings),
                    )
                else:
                    self.logger.error(
                        "Embeddings list is not same length as messages list"
                    )
                    raise IndexError(
                        "Embeddings list is not same length as messages list"
                    )
                self.logger.debug("Storing embeddings for %i messages", len(dmesg))

                t_0 = perf_counter()
                for idx, uid in enumerate(msgs):
                    # dembd[uid] = self.__get_embedding(msg["body"][:embed_max_chars])
                    fe[f"{dhash[uid]}.embd"] = embeddings[idx]
                    dembd[uid] = embeddings[idx]
                t_1 = perf_counter()
                self.logger.debug("Took %.2f to download embedings.", t_1 - t_0)

        self.logger.info("Total embeddings found/added %i in %s.", len(dembd), folder)
        return dembd

    def move_messages(self, folder: str, messages: dict[str, list]) -> int:
        """Move the messages market for moving, by target folder.

        Args:
            folder (str): source folder
            messages (dict[str,list]): dict of destination folder names and lists of messages

        Returns:
            int: number of messages moved
        """
        imap_conn = self.__imap_conn
        moved = 0
        for dest_folder in messages:
            messages_list = messages[dest_folder]
            uids: list = [mess["uid"] for mess in messages_list]
            if not self.__dry_run:
                if len(uids) > 0:
                    imap_conn.move(
                        folder,
                        uids,
                        dest_folder,
                        flag_messages=self.interactive,
                        flag_unseen=not self.interactive,
                    )
                moved += len(uids)
            else:
                self.logger.info(
                    "Dry run. WOULD have moved UID %s from %s to %s",
                    uids,
                    folder,
                    dest_folder,
                )
        return moved

    @backoff.on_exception(
        backoff.expo,
        RateLimitError,
        on_backoff=lambda details: base_logger.warning(
            "Backing off %0.1f seconds after %i tries",
            details["wait"],
            details["tries"],
        ),
    )
    def __get_embeddings(self, texts: list[str]) -> list:
        """Get the embedding from OpenAI

        Args:
            text (str): the message text

        Returns:
            CreateEmbeddingResponse: the embeddings
        """
        result = []
        batched_list = list(batched(texts, 20))
        index = 0
        for batch in batched_list:
            index += 1
            self.logger.debug("\tBatch %i/%i", index, len(batched_list))
            e: CreateEmbeddingResponse = self.__client.embeddings.create(
                input=batch, model=self.__settings.openai_model
            )
            embeddings = [emb_obj.embedding for emb_obj in e.data]
            result = result + embeddings

        return result

    def __mesg_hash(self, mesg: str) -> str:
        return sha256(mesg["body"].encode("utf-8")).hexdigest()[:12]
