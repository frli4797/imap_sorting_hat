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
from datetime import timedelta
from os.path import join
from threading import Event
from time import time
from typing import Dict, List, Optional
from pytimeparse.timeparse import timeparse

import backoff
import numpy as np
from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from sklearn.ensemble import RandomForestClassifier

from itertools import batched  # Python 3.12+

from . import metrics
from .classification_service import ClassificationService
from .embedding_store import EmbeddingStore
from .imap import ImapHandler
from .message import Message  # added
from .settings import Settings
from .db import SQLiteCache
from .migrate_shelve_to_sql import migrate as migrate_legacy_cache
from .message_repository import MessageRepository
from .training_manager import TrainingManager


metrics.configure_logging()
logging.getLogger("httpx").setLevel(logging.WARNING)
base_logger = logging.getLogger("ish")
metrics.start_metrics_server_if_configured()

EMBED_MAX_CHARS = 8192
MAX_SOURCE_MESSAGES = 160
MAX_LEARN_MESSAGES = 1600
POLL_TIME_SEC = 30
TRAINING_INTERVAL_SEC = timedelta(hours=24).total_seconds()
LEGACY_SHELVE_EXTENSIONS = ("", ".db", ".dat", ".dir", ".bak")


def env_to_bool(key: str):
    return os.environ.get(key) is not None


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
        self.logger = logging.getLogger("ish").getChild(self.__class__.__name__)

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

        # sqlite cache placed in data directory
        self._db_file = join(self.__settings.data_directory, "cache.sqlite")
        self._maybe_migrate_legacy_cache()
        self._cache = SQLiteCache(self._db_file)
        metrics.record_db_size(self._db_file)

        self._message_repository = MessageRepository(
            cache=self._cache,
            imap_conn_provider=lambda: self.__imap_conn,
        )
        self._embedding_store = EmbeddingStore(
            cache=self._cache,
            message_repository=self._message_repository,
            embedder=lambda texts: self.__get_embeddings(texts),
            max_chars=EMBED_MAX_CHARS,
            data_directory=self.__settings.data_directory
        )
        self._training_manager = TrainingManager(
            imap_conn_provider=lambda: self.__imap_conn,
            get_embeddings=lambda folder, uids: self.get_embeddings(folder, uids),
            get_cache_embeddings=self._cache.get_folder_embeddings,
            model_file=self.model_file,
            max_learn_messages=MAX_LEARN_MESSAGES,
            exit_event=self._exit_event
        )
        self._classification_service = ClassificationService(
            imap_conn_provider=lambda: self.__imap_conn,
            get_embeddings=lambda folder, uids: self.get_embeddings(folder, uids),
            get_messages=lambda folder, uids: self.get_msgs(folder, uids),
            model_file=self.model_file,
            max_source_messages=MAX_SOURCE_MESSAGES,
            interactive=self._interactive,
            dry_run=self._dry_run,
            exit_event=self._exit_event
        )

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

    @property
    def classifier(self) -> Optional[RandomForestClassifier]:
        return self._classification_service.classifier

    @classifier.setter
    def classifier(self, classifier: Optional[RandomForestClassifier]) -> None:
        self._classification_service.classifier = classifier

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
        """Fetch new messages through cache {uid: 'msg'}."""
        return self._message_repository.get_messages(folder, uids)

    def get_embeddings(self, folder: str, uids: List[int]) -> Dict[int, np.ndarray]:
        """Get embeddings using the EmbeddingStore through cache {uid: embedding}."""
        return self._embedding_store.get_embeddings(folder, uids)

    def learn_folders(self, folders: List[str]) -> Optional[RandomForestClassifier]:
        """Delegate learning to the TrainingManager."""
        classifier = self._training_manager.learn_folders(folders)
        if classifier is not None:
            self.classifier = classifier
        return classifier

    def train_on_destination_folders(self) -> bool:
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
        """Delegate classification workflow to the ClassificationService."""
        self.moved, self.skipped = self._classification_service.classify_messages(
            source_folders
        )

    def move_messages(self, folder: str, messages: dict[str, list]) -> int:
        """Proxy to the classification service for backwards compatibility."""
        return self._classification_service.move_messages(folder, messages)

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
            if not self.train_on_destination_folders():
                return 1
            next_training = time() + TRAINING_INTERVAL_SEC

        while not self._exit_event.is_set():
            if self.train and time() >= next_training:
                next_training = time() + TRAINING_INTERVAL_SEC
                if not self.train_on_destination_folders():
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

    if ISH.debug:
        ish_logger = logging.getLogger("ish")
        ish_logger.setLevel(logging.DEBUG)
        if not ish_logger.handlers:
            debug_handler = logging.StreamHandler()
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(logging.Formatter(metrics.LOG_FORMAT))
            ish_logger.addHandler(debug_handler)
        else:
            for handler in ish_logger.handlers:
                handler.setLevel(logging.DEBUG)
        ish_logger.propagate = False

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
