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
base_logger.setLevel(level=logging.DEBUG)

class ISH:
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings = Settings()
        self.client: OpenAI = None
        self.imap_conn:ImapHelper = ImapHelper()
        self.hkey = b"BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]"
        self.bkey = b"BODY[]"
        self.max_chars = 16384
        self.classifier: RandomForestClassifier = None

    @property
    def msgs_file(self) -> str:
        return join(self.settings["data_directory"], "msgs")

    @property
    def embd_file(self) -> str:
        return join(self.settings["data_directory"], "embd")

    @property
    def model_file(self) -> str:
        return join(self.settings["data_directory"], "model.pkl")

    def mesg_hash(self, mesg: str) -> str:
        return sha256(mesg["body"].encode("utf-8")).hexdigest()[:12]

    def connect_openai(self) -> bool:
        if not self.settings["openai_api_key"]:
            return False

        # check if api key is valid
        try:
            self.client = OpenAI(api_key=self.settings["openai_api_key"])
        except APIError as e:
            self.logger.error(e)
            return False
        return True

    def configure_and_connect(self):
        """Configure ish and connect to imap and openai"""
        settings = self.settings

        self.settings.update_data_settings()

        while not self.imap_conn.connect_imap():
            settings.update_login_settings()

        while not self.imap_conn.connect_openai():
            settings.update_openai_settings()

        folders = self.imap_conn.list_folders()
        settings.update_folder_settings(folders)

        print("Configuration complete")


    def connect_noninteractive(self) -> bool:
        """Connect to imap and openai without user interaction"""
        if not self.imap_conn.connect_imap():
            self.logger.error(
                "Failed to connect to imap server. Configure, or check your settings in %s",
                    self.settings.settings_file
            )
            return False

        if not self.connect_openai():
            self.logger.error(
                "Failed to connect to openai. Configure or check your settings in %s",
                    self.settings.settings_file)
            return False

        return True

    @staticmethod
    def backoff_debug(details):
        base_logger.info("Backing off {wait:0.1f} seconds after {tries} tries ")

    @backoff.on_exception(backoff.expo, RateLimitError, on_backoff=backoff_debug)
    def get_embedding(self, text):
        e: CreateEmbeddingResponse = self.client.embeddings.create(
            input=[text], model=self.settings["openai_model"]
        )
        return e

    def get_msgs(self, folder: str, uids: List[int]) -> Dict[int, str]:
        """Fetch new messages through cache {uid: 'msg'}"""
        d = {}
        new_uids = []
        imap_conn = self.imap_conn
        self.logger.info("Getting %i messages from %s",len(uids) , folder)
        with shelve.open(self.msgs_file, writeback=False) as fm:
            for uid in uids:
                try:
                    msg_hash = fm[f"{folder}:{uid}"]
                    mesg = fm[f"{msg_hash}.mesg"]
                    d[uid] = mesg
                    continue
                except KeyError:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                self.logger.info("Found %s messages not in cache", {len(new_uids)})

                msgs = imap_conn.fetch(new_uids, [self.hkey, self.bkey])
                for uid in new_uids:
                    mesg = self.parse_mesg(msgs[uid])
                    msg_hash = self.mesg_hash(mesg)
                    fm[f"{folder}:{uid}"] = msg_hash
                    fm[f"{msg_hash}.mesg"] = mesg
                    d[uid] = mesg
                self.logger.info("Found %i messages to cache", len(new_uids))
        self.logger.info("Total messages found/added %i in %s.", len(d), folder)
        return d

    def get_embeddings(self, folder: str, uids: List[int]) -> Dict[int, np.ndarray]:
        """Get embeddings using OpenAI API through cache {uid: embedding}"""
        dhash = {}
        dembd = {}
        self.logger.info("Getting %i embeddings from %s",len(uids) , folder)
        # with embd and msgs db open at the same time
        with shelve.open(self.msgs_file, writeback=False) as fm:
            new_uids = []  # uids that need a new hash
            for uid in uids:
                try:
                    dhash[uid] = fm[f"{folder}:{uid}"]
                except KeyError:
                    new_uids.append(uid)
                    continue

            self.logger.info("Found %i out of %i needing hash", len(new_uids), len(uids))
            dmesg = self.get_msgs(folder, new_uids)

            self.logger.debug("Adding hashes for %i messages", len(dmesg))
            for uid, mesg in dmesg.items():
                msg_hash = self.mesg_hash(mesg)
                dhash[uid] = msg_hash
                fm[f"{folder}:{uid}"] = msg_hash
            self.logger.debug("Added hashes for messages.")
            fm.close()

            new_uids = []  # uids that need a new embedding
            self.logger.debug("Finding embedding for %s messages", len(uids))
        with shelve.open(self.embd_file, writeback=False) as fe:
            for uid, msg_hash in dhash.items():  # dhash is {uid: hash}
                try:
                    embd = fe[f"{msg_hash}.embd"]
                    dembd[uid] = embd
                    continue
                except KeyError:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                self.logger.info("Found %i embeddings not in cache", len(new_uids))
                msgs = self.get_msgs(folder, new_uids)
                # messages = list(msg['body'][:self.max_chars] for msg in msgs.values() )
                # e = self.client.embeddings.create(input = messages,
                #   model=self.settings['openai_model'])
                # if len(e.data) == len( msgs.keys()):
                #     self.logger.info("EQUAL")
                # else:
                #     self.logger.info("NOT equal")
                self.logger.debug("Adding embeddings for %i messages", len(dmesg))
                for uid, msg in msgs.items():
                    dembd[uid] = self.get_embedding(msg["body"][: self.max_chars])
                    fe[f"{dhash[uid]}.embd"] = dembd[uid]

        self.logger.info("Total embeddings found/added %i in %s.", len(dembd), folder)
        return dembd

    def learn_folders(self, folders: List[str]) -> RandomForestClassifier:
        imap_conn = self.imap_conn
        embed_array = []
        folder_array = []

        t0 = perf_counter()

        for folder in folders:
            self.logger.info("Learning folder %s", folder)
            # Retrieve the UIDs of all messages in the folder
            uids = imap_conn.search(["ALL"])
            embd = self.get_embeddings(folder, uids[:80])
            embed_array.extend(embd.values())
            folder_array.extend([folder] * len(embd))

        t1 = perf_counter()
        self.logger.info(
            "Fetched %i embeddings in %.2f seconds", len(embed_array), t1-t0
        )

        # Train a classifier
        self.logger.info("Training classifier...")

        all_embeddings: List = []
        for embd in embed_array:
            embedding = embd.data[0].embedding
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

        self.logger.info("Trained classifier in %.2f seconds", t2-t1)

        joblib.dump(self.classifier, self.model_file)
        self.logger.info("Saved classifier.")

    def classify_messages(
        self, source_folders: List[str], inference_mode=False
    ) -> None:
        imap_conn:ImapHelper = self.imap_conn

        if inference_mode:
            classifier = joblib.load(self.model_file)
        else:
            classifier = self.classifier

        skipped = 0
        moved = 0

        for folder in source_folders:
            uid = []
            self.logger.info("Classifying messages for folder {folder}")
            # Retrieve the UIDs of all messages in the folder
            if inference_mode:
                uids = imap_conn.search(folder,[b"UNSEEN"])
            else:
                uids = imap_conn.search(folder, ["ALL"])

            embd = self.get_embeddings(folder, uids[:160])
            mesgs = self.get_msgs(folder, uids[:160])

            for uid, embd in embd.items():
                dest_folder = classifier.predict([embd.data[0].embedding])[0]
                proba = classifier.predict_proba([embd.data[0].embedding])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)

                (top_probability,) = ranks[0]
                if top_probability > 0.25:
                    print(
                        f'\n{uid:3} From {mesgs[uid]["from"][0]}: {mesgs[uid]["body"][0:100]}'
                    )
                    for p, c in ranks:
                        print(f"{p:.2f}: {c}")
                    self.move_message(
                        skipped, moved, folder, uid, dest_folder, inference_mode
                    )
                else:
                    skipped += 1

        self.logger.info("Finished moved %i and skipped %i", moved, skipped)

    def move_message(self, skipped, moved, folder, uid, dest_folder, interactive):
        opt = None
        imap_conn = self.imap_conn
        if not interactive:
            imap_conn.move(folder, uid, dest_folder)
            moved += 1
        else:
            while opt not in ["y", "n", "q"]:
                opt = input(f"Move message to {dest_folder}? [y]yes, [n]no, [q]quit:")
                if opt == "y":
                    imap_conn.move(folder, uid, dest_folder)
                    moved += 1
                elif opt == "q":
                    self.logger.info(
                        "Quitting. Finished moved %i and skipped %i", moved, skipped
                    )
                    sys.exit(0)
                elif opt == "n":
                    skipped += 1


    def run(self, interactive: bool) -> int:
        try:
            if interactive:
                self.configure_and_connect()
            else:
                if not self.connect_noninteractive():
                    return 1

            settings = self.settings

            for f in settings["source_folders"]:
                self.logger.info("Source folder: %s", f)

            for f in settings["destination_folders"]:
                self.logger.info("Destination folder: %s", f)

            if interactive:
                self.learn_folders(settings["destination_folders"])

            self.classify_messages(
                settings["source_folders"], inference_mode=(not interactive)
            )
        except Exception as e:
            base_logger.error("Something went wrong. Unknown error.")
            base_logger.error(e)
        return 0


def main():
    ish = ISH()
    r = ish.run(interactive=False)
    sys.exit(r)


if __name__ == "__main__":
    main()
