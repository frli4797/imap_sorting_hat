import logging
from typing import Callable, Dict, List, Optional

from .db import SQLiteCache
from .imap import ImapHandler
from .message import Message


class MessageRepository:
    """Fetch and cache messages from IMAP."""

    def __init__(
        self,
        cache: SQLiteCache,
        imap_conn_provider: Callable[[], ImapHandler],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cache = cache
        self._imap_conn_provider = imap_conn_provider
        self._logger = logger or logging.getLogger("ish").getChild(self.__class__.__name__)

    def get_messages(self, folder: str, uids: List[int]) -> Dict[int, Message]:
        messages: Dict[int, Message] = {}
        new_uids: List[int] = []
        imap_conn = self._imap_conn_provider()

        if uids:
            self._logger.info("Getting %i messages from %s", len(uids), folder)

        for uid in uids:
            msg_hash = self._cache.get_hash(folder, uid)
            if msg_hash is None:
                new_uids.append(uid)
                continue

            cached = self._cache.get_message_by_hash(msg_hash)
            if cached:
                messages[uid] = Message(
                    uid=uid,
                    from_addr=cached.from_addr,
                    to_addr=cached.to_addr,
                    subject=cached.subject,
                    body=cached.body,
                )
            else:
                new_uids.append(uid)

        if new_uids:
            self._logger.debug("Found %s messages not in cache", len(new_uids))
            fetched = imap_conn.fetch(new_uids)
            for uid in new_uids:
                msg = imap_conn.parse_mesg(fetched[uid])
                self._cache.store_message(folder, uid, msg)
                messages[uid] = msg
            self._logger.debug("Found %i messages to cache", len(new_uids))

        self._logger.debug("Total messages found/added %i in %s.", len(messages), folder)
        return messages

    def get_hashes(self, folder: str, uids: List[int]) -> Dict[int, str]:
        """Return hashes for UIDs, fetching and caching missing ones."""
        hash_map: Dict[int, str] = {}
        missing: List[int] = []

        for uid in uids:
            msg_hash = self._cache.get_hash(folder, uid)
            if msg_hash:
                hash_map[uid] = msg_hash
            else:
                missing.append(uid)

        if missing:
            self._logger.debug("Ensuring hashes for %i messages in %s", len(missing), folder)
            self.get_messages(folder, missing)
            for uid in missing:
                msg_hash = self._cache.get_hash(folder, uid)
                if msg_hash is None:
                    continue
                hash_map[uid] = msg_hash

        return hash_map
