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
from typing import Dict, List, Optional

import backoff
import joblib
import numpy as np
from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
try:
    from itertools import batched  # Python 3.12+
except ImportError:
    def batched(iterable, n):
        it = iter(iterable)
        while True:
            batch = []
            for _ in range(n):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break
            yield batch

from . import metrics
from .imap import ImapHandler
from .message import Message  # added
from .settings import Settings
from .db import SQLiteCache
from .migrate_shelve_to_sql import migrate as migrate_legacy_cache


metrics.configure_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
base_logger = logging.getLogger("ish")
metrics.start_metrics_server_if_configured()

embed_max_chars = 8192
max_source_messages = 160
max_learn_messages = 1600
POLL_TIME_SEC = 30
TRAINING_INTERVAL_SEC = timedelta(hours=24).total_seconds()
LEGACY_SHELVE_EXTENSIONS = ("", ".db", ".dat", ".dir", ".bak")


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
        self.__settings.update_data_settings()
        self.__client: Optional[OpenAI] = None
        self.__imap_conn: ImapHandler = ImapHandler(self.__settings)

        self._interactive = interactive
        self._train = train
        self._daemon = daemon
        self._dry_run = dry_run

        self.classifier: Optional[RandomForestClassifier] = None
        # sqlite cache placed in data directory
        self._db_file = join(self.__settings.data_directory, "cache.sqlite")
        self._maybe_migrate_legacy_cache()
        self._cache = SQLiteCache(self._db_file)
        metrics.record_db_size(self._db_file)

        self.moved = 0
        self.skipped = 0

        # signal.signal(signal.SIGHUP, self.__do_reload)
        signal.signal(signal.SIGINT, self.__do_exit)
        signal.signal(signal.SIGTERM, self.__do_exit)

    @property
    def msgs_file(self) -> str:
        # kept for backward compat with code referencing msgs_file; now points to DB
        return self._db_file

    @property
    def embd_file(self) -> str:
        # kept for backward compat with code referencing embd_file; now points to DB
        return self._db_file

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
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def interactive(self) -> bool:
        return self._interactive

    def _increment_skipped(self, amount: int = 1) -> None:
        self.skipped += amount
        metrics.increment_skipped(amount)

    def _maybe_migrate_legacy_cache(self) -> None:
        """Ensure legacy shelve data is imported when cache.sqlite is missing."""
        if os.path.isfile(self._db_file):
            return

        if not self._legacy_cache_exists():
            return

        self.logger.info("cache.sqlite not found; attempting legacy cache migration")
        try:
            migrate_legacy_cache(sqlite_path=self._db_file)
        except Exception:
            self.logger.exception("Automatic legacy cache migration failed; starting with a fresh cache")

    def _legacy_cache_exists(self) -> bool:
        data_dir = self.__settings.data_directory
        for base in ("msgs", "embd"):
            if self._any_legacy_file_exists(join(data_dir, base)):
                return True
        return False

    @staticmethod
    def _any_legacy_file_exists(base_path: str) -> bool:
        for ext in LEGACY_SHELVE_EXTENSIONS:
            candidate = f"{base_path}{ext}" if ext else base_path
            if os.path.exists(candidate):
                return True
        return False

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

        return True

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, TimeoutError, APIError, BadRequestError),
        max_tries=5,
        on_backoff=lambda details: base_logger.warning(
            "Backing off %0.1f seconds after %i tries",
            details["wait"],
            details["tries"],
        ),
    )
    def __get_embeddings(self, texts: list[str]) -> list:
        """Get embeddings from OpenAI API for a list of texts
        Args:
            texts (list[str]): list of texts to get embeddings for
        Returns:
            list: list of embeddings
        """

        result = []
        batched_list = list(batched(texts, 20))
        index = 0
        for batch in batched_list:
            index += 1
            self.logger.debug("\tBatch %i/%i", index, len(batched_list))
            try:
                e = self.__client.embeddings.create(input=batch, model=self.__settings.openai_model)
            except (APITimeoutError, BadRequestError) as bre:
                self.logger.error("Can't send request to openai %s", bre)
                raise bre
            embeddings = [emb_obj.embedding for emb_obj in e.data]
            result.extend(embeddings)
        return result

    def get_msgs(self, folder: str, uids: List[int]) -> Dict[int, Message]:
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
        if len(uids) > 0:
            self.logger.info("Getting %i messages from %s", len(uids), folder)

        # First try to fetch cached messages using the SQL cache
        for uid in uids:
            msg_hash = self._cache.get_hash(folder, uid)
            if msg_hash is None:
                new_uids.append(uid)
            else:
                mesg = self._cache.get_message_by_hash(msg_hash)
                if mesg:
                    # attach uid (cache message content doesn't store uid)
                    d[uid] = Message(uid=uid, from_addr=mesg.from_addr, to_addr=mesg.to_addr, subject=mesg.subject, body=mesg.body)
                else:
                    # Not found; treat as new
                    new_uids.append(uid)

        # If not in the cache, fetch the message and put in cache.
        if len(new_uids) > 0:
            self.logger.debug("Found %s messages not in cache", len(new_uids))
            msgs = imap_conn.fetch(new_uids)
            for uid in new_uids:
                mesg = imap_conn.parse_mesg(msgs[uid])  # returns Message
                # store in sqlite cache (message content and folder->uid mapping)
                self._cache.store_message(folder, uid, mesg)
                d[uid] = mesg
            self.logger.debug("Found %i messages to cache", len(new_uids))
        self.logger.debug("Total messages found/added %i in %s.", len(d), folder)
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
        if len(uids) > 0:
            self.logger.info("Getting %i embeddings from %s", len(uids), folder)

        self.logger.debug("Opening message cache '%s'", self.msgs_file)

        # Output the contents of self.__settings.data_directory to the log for debugging
        self.logger.debug("Data directory contents: %s", os.listdir(self.__settings.data_directory))

    
        # First try to use the SQL cache to get msg hash for each uid
        new_uids = []
        for uid in uids:
            msg_hash = self._cache.get_hash(folder, uid)
            if msg_hash:
                dhash[uid] = msg_hash
            else:
                new_uids.append(uid)

        self.logger.debug(
            "Found %i out of %i needing hash", len(new_uids), len(uids)
        )
        dmesg = self.get_msgs(folder, new_uids)

        self.logger.debug("Adding hashes for %i messages", len(dmesg))
        # Saving unseen messages / mappings into sqlite via store_message
        for uid, mesg in dmesg.items():
            msg_hash = self._cache.store_message(folder, uid, mesg)
            dhash[uid] = msg_hash
        self.logger.debug("Added hashes for messages.")

        new_uids = []  # uids that need a new embedding
        self.logger.debug("Finding embedding for %s messages", len(uids))
        # check existing embeddings in sqlite
        for uid, msg_hash in dhash.items():  # dhash is {uid: hash}
            embd = self._cache.get_embedding(msg_hash)
            if embd is not None:
                dembd[uid] = embd
                continue
            else:
                new_uids.append(uid)

        if len(new_uids) > 0:
            self.logger.debug("Found %i embeddings not in cache", len(new_uids))
            msgs = self.get_msgs(folder, new_uids)

            t_batch_0 = perf_counter()
            # change to use msg.body
            embeddings = self.__get_embeddings(
                list(msg.body[:embed_max_chars] for msg in msgs.values())
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
                emb = embeddings[idx]
                # store in SQL embeddings table
                self._cache.store_embedding(dhash[uid], emb)
                dembd[uid] = emb
            t_1 = perf_counter()
            self.logger.debug("Took %.2f to download embedings.", t_1 - t_0)

        self.logger.debug("Total embeddings found/added %i in %s.", len(dembd), folder)
        return dembd

    def learn_folders(self, folders: List[str]) -> Optional[RandomForestClassifier]:
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
            embd = self.get_embeddings(folder, uids[:max_learn_messages])
            cached_embd = self._cache.get_folder_embeddings(folder)
            for uid, emb in cached_embd.items():
                embd.setdefault(uid, emb)
            if len(embd) == 0:
                self.logger.warning("No embeddings available for folder %s; skipping.", folder)
                continue
            embed_array.extend(embd.values())
            folder_array.extend([folder] * len(embd))
            if self._exit_event.is_set():
                return None

        if len(embed_array) < 2:
            self.logger.warning(
                "Need at least two messages to train; gathered %i. Skipping training.",
                len(embed_array),
            )
            return None

        if len(set(folder_array)) < 2:
            self.logger.warning(
                "Need samples from at least two folders to train; gathered folders: %s. Skipping training.",
                sorted(set(folder_array)),
            )
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
        report_dict = classification_report(y_test, y_pred, labels=folders, output_dict=True)
        class_report = classification_report(y_test, y_pred, labels=folders)
        self.logger.info("Classification Report:")
        self.logger.info("\n%s", class_report)
        metrics.record_classification_metrics(report_dict)

        t2 = perf_counter()
        duration = t2 - t1
        self.logger.debug("Trained classifier in %.2f seconds", duration)
        metrics.record_training_stats(len(embed_array), accuracy, duration)

        joblib.dump(self.classifier, self.model_file)
        self.logger.info("Saved classifier.")
        return clf

    def _train_on_destination_folders(self) -> bool:
        """Ensure we can train on the configured destination folders."""
        folders = self.__settings.destination_folders
        if not folders:
            self.logger.error(
                "No destination folders configured. Cannot train classifier."
            )
            return False

        classifier = self.learn_folders(folders)
        if classifier is None:
            self.logger.error(
                "Learning aborted before a classifier was produced. Aborting training cycle."
            )
            return False

        return True

    def classify_messages(self, source_folders: List[str]) -> None:
        """Classify and move messages for all source folders

        Args:
            source_folders (List[str]): list of source folders
        """
        imap_conn: ImapHandler = self.__imap_conn
        self.skipped = 0
        self.moved = 0
        classifier: RandomForestClassifier = self.classifier
        if self.classifier is None:
            if not os.path.isfile(self.model_file):
                self.logger.error("Classifier file %s not found; cannot classify.", self.model_file)
                return
            try:
                classifier = joblib.load(self.model_file)
            except Exception as exc:
                self.logger.error("Failed to load classifier from %s: %s", self.model_file, exc)
                return

        stop_processing = False
        for folder in source_folders:
            if self._exit_event.is_set():
                break
            uids = []
            self.logger.info("Classifying messages for folder %s", folder)
            # Retrieve the UIDs of all messages in the folder
            if not self.interactive:
                uids = imap_conn.search(folder, [b"UNSEEN"])
            else:
                uids = imap_conn.search(folder, ["ALL"])

            embd = self.get_embeddings(folder, uids[:max_source_messages])
            metrics.record_folder_embedding_count(folder, len(embd))
            mesgs = self.get_msgs(folder, uids[:max_source_messages])

            to_move: dict[str, list] = {}
            for uid, embd in embd.items():
                dest_folder = classifier.predict([embd])[0]
                proba = classifier.predict_proba([embd])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)
                (top_probability, predcited_class) = ranks[0]
                mess_to_move = {
                    "uid": uid,
                    "probability": top_probability,
                    "from": mesgs[uid].from_addr,
                    "body": mesgs[uid].preview(100),
                }
                if top_probability > 0.25:
                    self._log_move(uid, "Going to move", ranks, mess_to_move)

                    if self.interactive:
                        answer = self.__select_move(dest_folder)
                        if answer == Action.NO:
                            self._increment_skipped()
                            continue
                        elif answer == Action.QUIT:
                            self.logger.info("User requested quit; stopping classification.")
                            self._exit_event.set()
                            stop_processing = True
                            break

                    if dest_folder not in to_move:
                        to_move[dest_folder] = [mess_to_move]
                    else:
                        to_move[dest_folder].append(mess_to_move)

                else:
                    self._log_move(
                        uid, "Skipping due to probability", ranks, mess_to_move
                    )
                    self._increment_skipped()

            if stop_processing or self._exit_event.is_set():
                break

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
                moved_count = len(uids)
                moved += moved_count
                metrics.increment_moved(moved_count)
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
        # schedule training immediately when requested, otherwise wait one interval
        next_training = (
            time() if self.train else time() + TRAINING_INTERVAL_SEC
        )

        for f in settings.source_folders:
            self.logger.debug("Source folder: %s", f)

        for f in settings.destination_folders:
            self.logger.debug("Destination folder: %s", f)

        if not self.connect():
            return 1
        
        if not os.path.isfile(self.model_file):
            self.logger.info(
                "No classifier at %s. Going to learning folders.",
                self.model_file,
            )
            if not self._train_on_destination_folders():
                return 1
            next_training = time() + TRAINING_INTERVAL_SEC

        while not self._exit_event.is_set():
            if self.train and time() >= next_training:
                next_training = time() + TRAINING_INTERVAL_SEC
                if not self._train_on_destination_folders():
                    self.logger.error("Training requested but failed.")
            self.classify_messages(settings.source_folders)
            metrics.record_db_size(self._db_file)
            
            if not self.daemon:
                break
            self._exit_event.wait(POLL_TIME_SEC)

        return 0

    def close(self):
        # Fix inverted checks: only close if objects exist
        if getattr(self, "_cache", None) is not None:
            try:
                self._cache.close()
            except Exception:
                pass
            finally:
                self._cache = None
                metrics.record_db_size(self._db_file)

        if self.__imap_conn is not None:
            try:
                self.__imap_conn.close()
            except Exception:
                pass

        if self.__client is not None:
            try:
                self.__client.close()
            except Exception:
                pass

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
