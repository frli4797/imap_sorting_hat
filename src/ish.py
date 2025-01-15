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
import os
import signal
import sys
from datetime import datetime, timedelta
from enum import Enum
from os.path import join
from threading import Event
from time import perf_counter, time
from typing import Dict, List

import joblib
import numpy as np
from openai import APIError, OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from emailservice import EmailFeatureService
from imap import ImapHandler
from settings import Settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
base_logger = logging.getLogger("ish")

embed_max_chars = 8192
max_source_messages = 160
max_learn_messages = 1600
POLL_TIME_SEC = 20


def env_to_bool(key: str):
    return os.environ.get(key) is not None


Action = Enum("Action", ["YES", "NO", "QUIT"])


class ISH:
    debug = False
    _exit_event = Event()

    _interactive = False
    _train = False
    _daemon = False

    def __init__(
        self,
        interactive: bool = False,
        train: bool = False,
        daemon: bool = False,
        dry_run=False,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if ISH.debug:
            self.logger.setLevel(logging.DEBUG)
        self.__settings = Settings(ISH.debug)
        self.__client: OpenAI = None
        self.__imap_conn: ImapHandler = ImapHandler(self.__settings)

        self._interactive = interactive
        self._train = train
        self._daemon = daemon
        self._dry_run = dry_run

        self.classifier: RandomForestClassifier = None
        self.moved = 0
        self.skipped = 0

        # signal.signal(signal.SIGHUP, self.__do_reload)
        signal.signal(signal.SIGINT, self.__do_exit)
        signal.signal(signal.SIGTERM, self.__do_exit)

    @property
    def model_file(self) -> str:
        return join(self.__settings.data_directory, "model.pkl")

    @property
    def train(self) -> bool:
        return self._train

    @property
    def daemon(self) -> bool:
        return self._daemon

    @property
    def interactive(self) -> bool:
        return self._interactive

    def __connect_openai(self) -> bool:
        if not self.__settings.openai_api_key:
            return False

        # check if api key is valid
        try:
            self.__client = OpenAI(api_key=self.__settings.openai_api_key)
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

        self.logger.info("Connect and configuration complete")

    def connect(self) -> bool:
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

        self.__emailservice = EmailFeatureService(
            settings=self.__settings,
            imap_handler=self.__imap_conn,
            openai_client=self.__client,
            dry_run=self._dry_run,
            interactive=self.interactive,
        )

        return True

    def learn_folders(self, folders: List[str]) -> RandomForestClassifier:
        """Learn the (target) folders using cached and fetched embeddings
            Also always saves the model pickle to disk

        Args:
            folders (List[str]): the target folders

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
            embd = self.__emailservice.get_embeddings_new(
                folder, uids[:max_learn_messages]
            )
            embed_array.extend(embd.values())
            folder_array.extend([folder] * len(embd))
            if self._exit_event.is_set():
                return None

        t1 = perf_counter()
        self.logger.debug(
            "Fetched %i embeddings in %.2f seconds", len(embed_array), t1 - t0
        )

        # Train a classifier
        self.logger.info("Training classifier...")

        X = np.array(embed_array)
        y = np.array(folder_array)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        self.classifier = clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        self.logger.info("Trained with %i embeddings", len(embed_array))
        self.logger.info("Accuracy: %.2f", accuracy)
        self.logger.info("Classifier: %s", self.classifier)

        y_pred = clf.predict(X_test)
        class_report = classification_report(y_test, y_pred, labels=folders)
        self.logger.debug("Classification Report:")
        self.logger.debug("\n%s", class_report)

        t2 = perf_counter()

        self.logger.debug("Trained classifier in %.2f seconds", t2 - t1)

        joblib.dump(self.classifier, self.model_file)
        self.logger.debug("Saved classifier.")
        return clf

    def classify_messages(self, source_folders: List[str]) -> None:
        """Classify and move messages for all source folders

        Args:
            source_folders (List[str]): list of source folders
        """
        imap_conn: ImapHandler = self.__imap_conn
        self.skipped = 0
        self.moved = 0
        classifier = self.classifier
        if self.classifier is None:
            classifier = joblib.load(self.model_file)

        for folder in source_folders:
            uids = []
            self.logger.info("Classifying messages for folder %s", folder)
            # Retrieve the UIDs of all messages in the folder
            if not self.interactive:
                uids = imap_conn.search(folder, [b"UNSEEN"])
            else:
                uids = imap_conn.search(folder, ["ALL"])

            mesgs = self.__emailservice.get_emails_w_embeddings(
                folder, uids[:max_source_messages]
            )

            to_move: dict[str, list] = {}
            for uid, mesg in mesgs.items():
                emb = np.asarray(mesg.emdeddings)
                dest_folder = classifier.predict([emb])[0]
                proba = classifier.predict_proba([emb])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)
                (top_probability, predcited_class) = ranks[0]
                mess_to_move = {
                    "uid": uid,
                    "probability": top_probability,
                    "from": mesg.from_,
                    "body": mesg.body[0:100],
                }
                if top_probability > 0.25:
                    self._log_move(uid, "Going to move", ranks, mess_to_move)

                    if self.interactive:
                        answer = self.__select_move(dest_folder)
                        if answer == Action.NO:
                            self.skipped += 1
                            continue
                        elif answer == Action.QUIT:
                            break

                    if dest_folder not in to_move:
                        to_move[dest_folder] = [mess_to_move]
                    else:
                        to_move[dest_folder].append(mess_to_move)

                else:
                    self._log_move(
                        uid, "Skipping due to probability", ranks, mess_to_move
                    )
                    self.skipped += 1

            self.moved += self.move_messages(folder, to_move)
        if (self.moved + self.skipped) > 0:
            self.logger.info(
                "Finished moved %i and skipped %i", self.moved, self.skipped
            )

    def _log_move(self, uid, text, ranks, mess_to_move):
        self.logger.debug(
            "%s\n%3i From %s: %s",
            text,
            uid,
            mess_to_move["from"],
            mess_to_move["body"],
        )

        for p, c in ranks[:3]:
            self.logger.debug("%.2f: %s", p, c)

    def __select_move(self, dest_folder: str) -> Action:
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
                return Action.YES
            if opt == "q":
                self.logger.info("Quitting.")
                return Action.QUIT
        return Action.NO

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
            if not self._dry_run:
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
                self.logger.debug(
                    "Dry run. WOULD have moved UID %s from %s to %s",
                    uids,
                    folder,
                    dest_folder,
                )
        return moved

    def run(self) -> int:

        settings = self.__settings
        next_training = time()

        for f in settings.source_folders:
            self.logger.debug("Source folder: %s", f)

        for f in settings.destination_folders:
            self.logger.debug("Destination folder: %s", f)

        if not self.connect():
            return 1

        if not os.path.isfile(self.model_file):
            self.logger.info("No classifier at %s. Going to learning folders.")
            self._train = True

        while not self._exit_event.is_set():
            if self.train and time() >= next_training:
                next_training: datetime = time() + timedelta(hours=24).total_seconds()
                self.learn_folders(settings.destination_folders)

            self.classify_messages(settings.source_folders)
            if not self.daemon:
                break
            self._exit_event.wait(POLL_TIME_SEC)

        return 0

    def close(self):
        if self.__imap_conn is None:
            self.__imap_conn.close()
            self.__imap_conn = None

        if self.__client is None:
            self.__client.close()
            self.__client = None

    def __del__(self):
        self.close()

    def __do_exit(self, signum, frame):
        self.logger.debug("Got %s ", signal.strsignal(signum))
        self.logger.info(
            "Shutting down.",
        )
        self.close()
        self._exit_event.set()


def main(args: Dict[str, str]):
    ISH.debug = bool(args.pop("verbose"))
    dry_run = bool(args.pop("dry_run"))  # noqa: F841
    daemonize = bool(args.pop("daemon"))
    interactive = bool(args.pop("interactive"))
    train = bool(args.pop("learn_folders"))
    config_path = args.pop("config_path")
    if config_path is not None and not config_path == "":
        os.environ["ISH_CONFIG_PATH"] = config_path

    ish = ISH(interactive=interactive, train=train, daemon=daemonize, dry_run=dry_run)
    r = ish.run()
    sys.exit(r)


if __name__ == "__main__":

    import argparse

    userhomedir = Settings.get_user_directory()
    parser = argparse.ArgumentParser(
        description="""Magically sort email into smart folders.
                            **ish** works by downloading plain text versions of all the \
                            emails in the source email folders and move those unread to \
                            the destination folders, by using a multi class classifier."""
    )
    # Environment variables always takes precedence.
    parser.add_argument(
        "--learn-folders",
        "-l",
        help="Learn based on the contents of the destination folders",
        action="store_true",
        default=env_to_bool("ISH_LEARN_FOLDERS"),
    )

    parser.add_argument(
        "--interactive",
        "-i",
        help="Prompt user before moving anything",
        action="store_true",
        default=env_to_bool("ISH_INTERACTIVE"),
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        help="Don't actually move emails",
        action="store_true",
        default=env_to_bool("ISH_DRY_RUN"),
    )

    parser.add_argument(
        "--daemon",
        "-D",
        help="Run in daemon/polling mode",
        action="store_true",
        default=env_to_bool("ISH_DAEMON"),
    )

    parser.add_argument(
        "--config-path",
        "-C",
        type=str,
        help=f"Path for config file and data. Will default to {userhomedir}/.ish",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        help="Verbose/debug mode",
        action="store_true",
        default=env_to_bool("ISH_DEBUG"),
    )
    main(vars(parser.parse_args()))
