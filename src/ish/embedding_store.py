import logging
import os
from time import perf_counter
from typing import Callable, Dict, List, Optional

import numpy as np

from ish import metrics

from .db import SQLiteCache
from .message_repository import MessageRepository


class EmbeddingStore:
    """Manage cached embeddings and fetch new ones via embedder callback."""

    def __init__(
        self,
        *,
        cache: SQLiteCache,
        message_repository: MessageRepository,
        embedder: Callable[[List[str]], List[np.ndarray]],
        max_chars: int,
        data_directory: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cache = cache
        self._message_repository = message_repository
        self._embedder = embedder
        self._max_chars = max_chars
        self._data_directory = data_directory
        self._logger = logger or logging.getLogger("ish").getChild(self.__class__.__name__)

    def get_embeddings(self, folder: str, uids: List[int]) -> Dict[int, np.ndarray]:
        """Get embeddings for the given folder and UIDs, using cache when possible. 
            - New embeddings are fetched via the embedder callback and stored in the cache.
            - Only embeddings for the provided UIDs are returned.
            - Any new embeddings are also stored in the cache for future use.
        """

        embeddings: Dict[int, np.ndarray] = {}
        if not uids:
            return embeddings

        self._logger.info("Getting %i embeddings from %s", len(uids), folder)
        self._logger.debug("Opening message cache '%s'", self._cache.path)
        self._logger.debug("Data directory contents: %s", self._list_data_directory())

        hash_map = self._message_repository.get_hashes(folder, uids)
        missing: List[int] = []

        for uid, msg_hash in hash_map.items():
            embd = self._cache.get_embedding(msg_hash)
            if embd is not None:
                embeddings[uid] = embd
            else:
                missing.append(uid)

        if missing:
            self._logger.debug("Found %i embeddings not in cache", len(missing))
            messages = self._message_repository.get_messages(folder, missing)
            ordered_uids: List[int] = []
            texts: List[str] = []
            for uid in missing:
                message = messages.get(uid)
                if message is None:
                    continue
                ordered_uids.append(uid)
                texts.append(message.body[: self._max_chars])

            if texts:
                embeddings_list = self._batch_embeddings(texts)
                for uid, emb in zip(ordered_uids, embeddings_list):
                    message = messages.get(uid)
                    if message is None:
                        continue
                    msg_hash = hash_map.get(uid)
                    if msg_hash is None:
                        msg_hash = self._cache.store_message(folder, uid, message)
                        hash_map[uid] = msg_hash
                    self._cache.store_embedding(msg_hash, emb)
                    embeddings[uid] = emb

        self._logger.debug("Total embeddings found/added %i in %s.", len(embeddings), folder)
        metrics.record_folder_embedding_count(folder, len(embeddings))
        return embeddings

    def _batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        t0 = perf_counter()
        embeddings = self._embedder(texts)
        t1 = perf_counter()
        if len(embeddings) != len(texts):
            self._logger.error("Embeddings list is not same length as messages list")
            raise IndexError("Embeddings list is not same length as messages list")
        self._logger.debug("Took %.2f to BATCH %i embedings.", t1 - t0, len(embeddings))
        return embeddings

    def _list_data_directory(self):
        try:
            return os.listdir(self._data_directory)
        except Exception:
            return []
