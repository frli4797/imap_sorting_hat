import logging
from threading import Event
from time import perf_counter
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from . import metrics
from .imap import ImapHandler


class TrainingManager:
    """Handles training the classifier based on destination folders."""

    def __init__(
        self,
        *,
        imap_conn_provider: Callable[[], ImapHandler],
        get_embeddings: Callable[[str, List[int]], Dict[int, np.ndarray]],
        get_cache_embeddings: Callable[[str], Dict[int, np.ndarray]],
        model_file: str,
        max_learn_messages: int,
        exit_event: Event,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._imap_conn_provider = imap_conn_provider
        self._get_embeddings = get_embeddings
        self._get_cache_embeddings = get_cache_embeddings
        self._model_file = model_file
        self._max_learn_messages = max_learn_messages
        self._exit_event = exit_event
        self._logger = logger or logging.getLogger("ish").getChild(self.__class__.__name__)

    def learn_folders(self, folders: List[str]) -> Optional[RandomForestClassifier]:
        """Train a classifier from the provided folders."""
        imap_conn = self._imap_conn_provider()
        fetch_start = perf_counter()

        training_data = self._collect_training_data(imap_conn, folders)
        if training_data is None:
            return None
        embed_array, folder_array = training_data

        if not self._has_sufficient_training_data(embed_array, folder_array):
            return None

        fetch_end = perf_counter()
        self._log_embedding_stats(len(embed_array), fetch_end - fetch_start)

        X = np.array(embed_array)
        y = np.array(folder_array)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = self._train_classifier(X_train, y_train)
        accuracy = self._evaluate_classifier(clf, X_test, y_test, folders)
        self._log_training_summary(len(embed_array), accuracy, clf)

        eval_end = perf_counter()
        duration = eval_end - fetch_end
        metrics.record_training_stats(len(embed_array), accuracy, duration)

        joblib.dump(clf, self._model_file)
        self._logger.info("Saved classifier.")
        return clf

    def _collect_training_data(
        self, imap_conn: ImapHandler, folders: List[str]
    ) -> Optional[tuple[List[np.ndarray], List[str]]]:
        """ Collect embeddings and labels from the specified folders."""
        embed_array: List[np.ndarray] = []
        folder_array: List[str] = []

        for folder in folders:
            self._logger.info("Learning folder %s", folder)
            uids = imap_conn.search(folder, ["ALL"])
            embeddings = self._get_embeddings(folder, uids[: self._max_learn_messages])
            cached_embeddings = self._get_cache_embeddings(folder)
            for uid, emb in cached_embeddings.items():
                embeddings.setdefault(uid, emb)
            if len(embeddings) == 0:
                self._logger.warning("No embeddings available for folder %s; skipping.", folder)
                continue
            embed_array.extend(embeddings.values())
            folder_array.extend([folder] * len(embeddings))
            if self._exit_event.is_set():
                return None

        return embed_array, folder_array

    def _has_sufficient_training_data(
        self, embed_array: List[np.ndarray], folder_array: List[str]
    ) -> bool:
        """Check if there is enough data to train the classifier."""
        if len(embed_array) < 2:
            self._logger.warning(
                "Need at least two messages to train; gathered %i. Skipping training.",
                len(embed_array),
            )
            return False

        if len(set(folder_array)) < 2:
            self._logger.warning(
                "Need samples from at least two folders to train; gathered folders: %s. Skipping training.",
                sorted(set(folder_array)),
            )
            return False

        return True

    def _log_embedding_stats(self, total_embeddings: int, duration: float) -> None:
        self._logger.debug(
            "Fetched %i embeddings in %.2f seconds", total_embeddings, duration
        )

    def _train_classifier(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestClassifier:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        return clf

    def _evaluate_classifier(
        self,
        clf: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        folders: List[str],
    ) -> float:
        """Evaluate the classifier and record metrics."""

        accuracy = clf.score(X_test, y_test)

        y_pred = clf.predict(X_test)
        report_dict = classification_report(
            y_test, y_pred, labels=folders, output_dict=True
        )
        class_report = classification_report(y_test, y_pred, labels=folders)
        self._logger.info("Classification Report:")
        self._logger.info("\n%s", class_report)
        metrics.record_classification_metrics(report_dict)

        return accuracy

    def _log_training_summary(
        self, total_embeddings: int, accuracy: float, clf: RandomForestClassifier
    ) -> None:
        self._logger.info("Trained with %i embeddings", total_embeddings)
        self._logger.info("Accuracy: %.2f", accuracy)
        self._logger.info("Classifier: %s", clf)
