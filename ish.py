# ish.py
"""
# imap_sorting_hat = "ish"
Magically sort email into smart folders

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly
  learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

Status: Early development
"""

import logging
import shelve
import sys
from hashlib import sha256
from os.path import join
from time import perf_counter
from typing import Dict, List

import backoff
import joblib
import numpy as np
from openai import APIError, OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from imap_helper import ImapHelper
from settings import Settings

logging.basicConfig(level=logging.INFO)

base_logger = logging.getLogger("ish")

embed_max_chars = 16384


class ISH:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.__settings = Settings()
        self.__client: OpenAI = None
        self.__imap_conn: ImapHelper = ImapHelper(self.__settings)
        self.classifier: RandomForestClassifier = None
        self.moved = 0
        self.skipped = 0

    @property
    def msgs_file(self) -> str:
        return join(self.__settings["data_directory"], "msgs")

    @property
    def embd_file(self) -> str:
        return join(self.__settings["data_directory"], "embd")

    @property
    def model_file(self) -> str:
        return join(self.__settings["data_directory"], "model.pkl")

    def __mesg_hash(self, mesg: str) -> str:
        return sha256(mesg["body"].encode("utf-8")).hexdigest()[:12]

    def __connect_openai(self) -> bool:
        if not self.__settings["openai_api_key"]:
            return False

        # check if api key is valid
        try:
            self.__client = OpenAI(api_key=self.__settings["openai_api_key"])
        except APIError as e:
            self.logger.error(e)
            return False
        return True

    def configure_and_connect(self):
        """Configure ish and connect to imap and openai"""
        settings = self.__settings

        self.__settings.update_data_settings()

        while not self.__imap_conn.connect_imap():
            settings.update_login_settings()

        while not self.__connect_openai():
            settings.update_openai_settings()

        folders = self.__imap_conn.list_folders()
        settings.update_folder_settings(folders)

        print("Configuration complete")

    def connect_noninteractive(self) -> bool:
        """Connect to imap and openai without user interaction"""
        if not self.__imap_conn.connect_imap():
            self.logger.error(
                "Failed to connect to imap server. Configure, or check your settings in %s",
                self.__settings.settings_file,
            )
            return False

        if not self.__connect_openai():
            self.logger.error(
                "Failed to connect to openai. Configure or check your settings in %s",
                self.__settings.settings_file,
            )
            return False

        return True

    @staticmethod
    def __backoff_debug(details):
        base_logger.info(
            "Backing off {wait:0.1f} seconds after {tries} tries ".format(**details)
        )

    @backoff.on_exception(backoff.expo, RateLimitError, on_backoff=__backoff_debug)
    def __get_embedding(self, text: str) -> list:
        """Get the embedding from OpenAI

        Args:
            text (str): the message text

        Returns:
            CreateEmbeddingResponse: the embeddings
        """
        e: CreateEmbeddingResponse = self.__client.embeddings.create(
            input=[text], model=self.__settings["openai_model"]
        )
        return e.data[0].embedding

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
                # messages = list(msg['body'][:embed_max_chars] for msg in msgs.values() )
                # e = self.client.embeddings.create(input = messages,
                #   model=self.settings['openai_model'])
                # if len(e.data) == len( msgs.keys()):
                #     self.logger.info("EQUAL")
                # else:
                #     self.logger.info("NOT equal")
                self.logger.debug("Adding embeddings for %i messages", len(dmesg))
                # TODO: Batch the embeddings lookup.
                for uid, msg in msgs.items():
                    dembd[uid] = self.__get_embedding(msg["body"][:embed_max_chars])
                    fe[f"{dhash[uid]}.embd"] = dembd[uid]

        self.logger.info("Total embeddings found/added %i in %s.", len(dembd), folder)
        return dembd

    def learn_folders(self, folders: List[str]) -> RandomForestClassifier:
        """Learn the (source) folders using cached and fetched embeddings
            Also always saves the model pickle to disk

        Args:
            folders (List[str]): the source folders

        Returns:
            RandomForestClassifier: the classifier being used
        """
        imap_conn = self.__imap_conn
        embed_array = []
        folder_array = []

        t0 = perf_counter()

        for folder in folders:
            self.logger.info("Learning folder %s", folder)
            # Retrieve the UIDs of all messages in the folder
            uids = imap_conn.search(folder, ["ALL"])
            embd = self.get_embeddings(folder, uids[:80])
            embed_array.extend(embd.values())
            folder_array.extend([folder] * len(embd))

        t1 = perf_counter()
        self.logger.info(
            "Fetched %i embeddings in %.2f seconds", len(embed_array), t1 - t0
        )

        # Train a classifier
        self.logger.info("Training classifier...")

        all_embeddings: List = []
        for embd in embed_array:
            embedding = embd
            all_embeddings.append(embedding)

        X = np.array(all_embeddings)
        y = np.array([folder for folder in folder_array])

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        self.classifier = clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        self.logger.info("Accuracy: %.2f", accuracy)
        self.logger.info("Classifier: %s", self.classifier)

        t2 = perf_counter()

        self.logger.info("Trained classifier in %.2f seconds", t2 - t1)

        joblib.dump(self.classifier, self.model_file)
        self.logger.info("Saved classifier.")
        return clf

    def classify_messages(self, source_folders: List[str], interactive=True) -> None:
        """Classify and move messages for all source folders

        Args:
            source_folders (List[str]): list of source folders
            interactive (bool, optional):
                Interactive or non-interactive mode. Defaults to interactive.
        """
        imap_conn: ImapHelper = self.__imap_conn

        self.skipped = 0
        self.moved = 0

        if not interactive:
            classifier = joblib.load(self.model_file)
        else:
            classifier = self.classifier

        for folder in source_folders:
            uids = []
            self.logger.info("Classifying messages for folder %s", folder)
            # Retrieve the UIDs of all messages in the folder
            if not interactive:
                uids = imap_conn.search(folder, [b"UNSEEN"])
            else:
                uids = imap_conn.search(folder, ["ALL"])

            embd = self.get_embeddings(folder, uids[:160])
            mesgs = self.get_msgs(folder, uids[:160])

            to_move: dict[str, list] = {}
            for uid, embd in embd.items():
                dest_folder = classifier.predict([embd])[0]
                proba = classifier.predict_proba([embd])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)
                (top_probability, predcited_class) = ranks[0]
                mess_to_move = {
                    "uid": uid,
                    "probability": top_probability,
                    "from": mesgs[uid]["from"][0],
                    "body": mesgs[uid]["body"][0:100],
                }
                if top_probability > 0.25:
                    print(
                        f'\n{uid:3} From {mess_to_move["from"]}: {mess_to_move["body"]}'
                    )

                    for p, c in ranks[:3]:
                        print(f"{p:.2f}: {c}")

                    if interactive and not self.__select_move(dest_folder):
                        self.logger.debug(
                            f"""Skipping due to probability {top_probability:.2f}%
                                {uid:3} From {mess_to_move["from"]}: {mess_to_move["body"]}"""
                        )
                        self.skipped += 1
                        continue

                    if dest_folder not in to_move:
                        to_move[dest_folder] = [mess_to_move]
                    else:
                        to_move[dest_folder].append(mess_to_move)

                else:
                    self.logger.debug(
                        f"""Skipping due to probability {top_probability:.2f}%
                            {uid:3} From {mess_to_move["from"]}: {mess_to_move["body"]}"""
                    )
                    self.skipped += 1
            self.logger.info("Finished predicting %s", folder)
            self.moved += self.move_messages(folder, to_move)
        self.logger.info("Finished moved %i and skipped %i", self.moved, self.skipped)

    def __select_move(self, dest_folder: str) -> bool:
        """Interactively ask user if to move.

        Args:
            dest_folder (str): destination folder

        Returns:
            bool: True if to move message
        """
        opt = ""
        while opt not in ["y", "n", "q"]:
            opt = input(f"Move message to {dest_folder}? [y]yes, [n]no, [q]quit:")
            if opt == "y":
                return True
            elif opt == "q":
                self.logger.info("Quitting.")
                sys.exit(0)
            elif opt == "n":
                return False

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

            imap_conn.move(folder, uids, dest_folder)
            moved += len(uids)

        return moved

    def run(self, interactive: bool) -> int:
        try:
            if interactive:
                self.configure_and_connect()
            else:
                if not self.connect_noninteractive():
                    return 1

            settings = self.__settings

            for f in settings["source_folders"]:
                self.logger.info("Source folder: %s", f)

            for f in settings["destination_folders"]:
                self.logger.info("Destination folder: %s", f)

            if interactive:
                self.learn_folders(settings["destination_folders"])

            self.classify_messages(settings["source_folders"], interactive=interactive)
        except Exception as e:
            base_logger.error("Something went wrong. Unknown error.")
            base_logger.debug(e, stack_info=True)
            raise e
        return 0


def main():
    ish = ISH()
    r = ish.run(interactive=False)
    sys.exit(r)


if __name__ == "__main__":
    main()
