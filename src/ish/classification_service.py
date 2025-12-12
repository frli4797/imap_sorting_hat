import logging
import os
from enum import Enum
from threading import Event
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import metrics
from .imap import ImapHandler
from .message import Message


Action = Enum("Action", ["YES", "NO", "QUIT"])


class ClassificationService:
    """Handle classification and message moving workflow."""

    def __init__(
        self,
        *,
        imap_conn_provider: Callable[[], ImapHandler],
        get_embeddings: Callable[[str, List[int]], Dict[int, np.ndarray]],
        get_messages: Callable[[str, List[int]], Dict[int, Message]],
        model_file: str,
        max_source_messages: int,
        interactive: bool,
        dry_run: bool,
        exit_event: Event,
        logger: Optional[logging.Logger] = None,
        probability_threshold: float = 0.25,
    ) -> None:
        self._imap_conn_provider = imap_conn_provider
        self._get_embeddings = get_embeddings
        self._get_messages = get_messages
        self._model_file = model_file
        self._max_source_messages = max_source_messages
        self._interactive = interactive
        self._dry_run = dry_run
        self._exit_event = exit_event
        self._probability_threshold = probability_threshold
        self._logger = logger or logging.getLogger("ish").getChild(self.__class__.__name__)

        self._classifier: Optional[RandomForestClassifier] = None
        self.moved = 0
        self.skipped = 0

    @property
    def classifier(self) -> Optional[RandomForestClassifier]:
        return self._classifier

    @classifier.setter
    def classifier(self, classifier: Optional[RandomForestClassifier]) -> None:
        self._classifier = classifier

    def classify_messages(self, source_folders: List[str]) -> tuple[int, int]:
        """Classify and optionally move messages for provided folders."""
        classifier = self._ensure_classifier()
        if classifier is None:
            return (0, 0)

        self._reset_counters()

        for source_folder in source_folders:
            if self._exit_event.is_set():
                break

            self._logger.debug("Classifying messages for folder %s", source_folder)
            imap_conn = self._imap_conn_provider()
            uids = (
                imap_conn.search(source_folder, [b"UNSEEN"])
                if not self._interactive
                else imap_conn.search(source_folder, ["ALL"])
            )

            embeddings = self._get_embeddings(source_folder, uids[: self._max_source_messages])
            metrics.record_folder_embedding_count(source_folder, len(embeddings))
            messages = self._get_messages(source_folder, uids[: self._max_source_messages])

            to_move: Dict[str, List[dict]] = {}
            for uid, embedding in embeddings.items():
                message = messages.get(uid)
                if message is None:
                    continue

                dest_folder = classifier.predict([embedding])[0]
                proba = classifier.predict_proba([embedding])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)
                top_probability, _ = ranks[0]
                message_entry = {
                    "uid": uid,
                    "probability": top_probability,
                    "from": message.from_addr,
                    "body": message.preview(100),
                }

                if top_probability <= self._probability_threshold:
                    self._log_move(uid, "Skipping due to probability", ranks, message_entry)
                    self._increment_skipped()
                    continue

                self._log_move(uid, "Going to move", ranks, message_entry)
                if self._interactive:
                    action = self._select_move(dest_folder)
                    if action == Action.NO:
                        self._increment_skipped()
                        continue
                    if action == Action.QUIT:
                        self._logger.info("User requested quit; stopping classification.")
                        self._exit_event.set()
                        stop_processing = True
                        break

                to_move.setdefault(dest_folder, []).append(message_entry)

            if self._exit_event.is_set():
                break

            self.moved += self.move_messages(source_folder, to_move)

        if (self.moved + self.skipped) > 0:
            self._logger.info("Finished moved %i and skipped %i", self.moved, self.skipped)

        return (self.moved, self.skipped)

    def move_messages(self, folder: str, messages: Dict[str, List[dict]]) -> int:
        """Move grouped messages to their destination folders."""
        imap_conn = self._imap_conn_provider()
        moved = 0
        for dest_folder, message_list in messages.items():
            uids = [message["uid"] for message in message_list]
            if self._dry_run:
                if uids:
                    self._logger.debug(
                        "Dry run. WOULD have moved UID %s from %s to %s",
                        uids,
                        folder,
                        dest_folder,
                    )
                continue

            if uids:
                imap_conn.move(
                    folder,
                    uids,
                    dest_folder,
                    flag_messages=self._interactive,
                    flag_unseen=not self._interactive,
                )
            moved_count = len(uids)
            moved += moved_count
            metrics.increment_moved(moved_count)
        return moved

    def _ensure_classifier(self) -> Optional[RandomForestClassifier]:
        if self._classifier is not None:
            return self._classifier

        if not os.path.isfile(self._model_file):
            self._logger.error("Classifier file %s not found; cannot classify.", self._model_file)
            return None

        try:
            self._classifier = joblib.load(self._model_file)
        except Exception as exc:
            self._logger.error("Failed to load classifier from %s: %s", self._model_file, exc)
            return None
        return self._classifier

    def _increment_skipped(self, amount: int = 1) -> None:
        self.skipped += amount
        metrics.increment_skipped(amount)

    def _log_move(self, uid: int, text: str, ranks, message_entry) -> None:
        self._logger.debug(
            "%s\n%3i From %s: %s",
            text,
            uid,
            message_entry["from"],
            message_entry["body"],
        )
        for probability, predicted_class in ranks[:3]:
            self._logger.debug("%.2f: %s", probability, predicted_class)

    def _reset_counters(self) -> None:
        self.moved = 0
        self.skipped = 0

    def _select_move(self, dest_folder: str) -> Action:
        opt = ""
        while opt not in ["y", "n", "q"]:
            opt = input(f"Move message to {dest_folder}? [y]yes, [n]no, [q]quit:")
            if opt == "y":
                return Action.YES
            if opt == "q":
                return Action.QUIT
        return Action.NO
