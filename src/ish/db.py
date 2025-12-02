import os
import pickle
import sqlite3
from typing import Dict, List, Optional

import numpy as np

from .message import Message


class SQLiteCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        # allow multi-thread use in the app
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages_content (
                msg_hash TEXT PRIMARY KEY,
                from_addr TEXT,
                to_addr TEXT,
                subject TEXT,
                body TEXT
            );
            CREATE TABLE IF NOT EXISTS folder_messages (
                folder TEXT,
                uid INTEGER,
                msg_hash TEXT,
                PRIMARY KEY (folder, uid),
                FOREIGN KEY (msg_hash) REFERENCES messages_content(msg_hash) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                msg_hash TEXT PRIMARY KEY,
                data BLOB,
                FOREIGN KEY (msg_hash) REFERENCES messages_content(msg_hash) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # Messages / mappings

    def get_hash(self, folder: str, uid: int) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT msg_hash FROM folder_messages WHERE folder = ? AND uid = ? LIMIT 1",
            (folder, int(uid)),
        )
        r = cur.fetchone()
        return r[0] if r else None

    def get_message_by_hash(self, msg_hash: str) -> Optional[Message]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT from_addr, to_addr, subject, body FROM messages_content WHERE msg_hash = ?",
            (msg_hash,),
        )
        r = cur.fetchone()
        if not r:
            return None
        from_addr, to_addr, subject, body = r
        return Message(uid=None, from_addr=from_addr, to_addr=to_addr, subject=subject, body=body)

    def get_messages(self, folder: str, uids: List[int]) -> Dict[int, Message]:
        if not uids:
            return {}
        placeholders = ",".join("?" for _ in uids)
        params = [folder, *map(int, uids)]
        # Query: folder_messages join messages_content
        query = f"""
        SELECT fm.uid, mc.from_addr, mc.to_addr, mc.subject, mc.body, mc.msg_hash
        FROM folder_messages fm
        JOIN messages_content mc on fm.msg_hash = mc.msg_hash
        WHERE fm.folder = ? AND fm.uid IN ({placeholders})
        """
        cur = self.conn.cursor()
        cur.execute(query, params)
        out = {}
        for row in cur.fetchall():
            uid, from_addr, to_addr, subject, body, _ = row
            out[int(uid)] = Message(uid=int(uid), from_addr=from_addr, to_addr=to_addr, subject=subject, body=body)
        return out

    def store_message(self, folder: str, uid: int, msg: Message) -> str:
        # store content in messages_content if not exists, then upsert mapping folder_messages
        msg_hash = msg.hash()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO messages_content (msg_hash, from_addr, to_addr, subject, body) VALUES (?, ?, ?, ?, ?)",
            (msg_hash, msg.from_addr, msg.to_addr, msg.subject, msg.body),
        )
        cur.execute(
            "INSERT OR REPLACE INTO folder_messages (folder, uid, msg_hash) VALUES (?, ?, ?)",
            (folder, int(uid), msg_hash),
        )
        self.conn.commit()
        return msg_hash

    # Embeddings

    def get_embedding(self, msg_hash: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT data FROM embeddings WHERE msg_hash = ? LIMIT 1", (msg_hash,))
        r = cur.fetchone()
        if not r:
            return None
        data = r[0]
        return pickle.loads(data)

    def store_embedding(self, msg_hash: str, emb: np.ndarray):
        cur = self.conn.cursor()
        pickled = pickle.dumps(emb, protocol=pickle.HIGHEST_PROTOCOL)
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (msg_hash, data) VALUES (?, ?)",
            (msg_hash, sqlite3.Binary(pickled)),
        )
        self.conn.commit()