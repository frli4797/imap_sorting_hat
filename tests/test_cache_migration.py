import pickle
import sqlite3

import numpy as np

from ish.db import LEGACY_EMBEDDING_PROFILE, SQLiteCache
from ish.embedding_store import EMBEDDING_INPUT_PROFILE
from ish.message import Message
from ish.migrate_shelve_to_sql import _rehash_and_reconcile


def test_sqlite_cache_migrates_legacy_embedding_schema_with_profile_and_timestamps(tmp_path):
    db_path = tmp_path / "cache.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE messages_content (
            msg_hash TEXT PRIMARY KEY,
            from_addr TEXT,
            to_addr TEXT,
            subject TEXT,
            body TEXT
        );
        CREATE TABLE folder_messages (
            folder TEXT,
            uid INTEGER,
            msg_hash TEXT,
            PRIMARY KEY (folder, uid),
            FOREIGN KEY (msg_hash) REFERENCES messages_content(msg_hash) ON DELETE CASCADE
        );
        CREATE TABLE embeddings (
            msg_hash TEXT PRIMARY KEY,
            data BLOB,
            FOREIGN KEY (msg_hash) REFERENCES messages_content(msg_hash) ON DELETE CASCADE
        );
        """
    )
    message = Message(uid=1, from_addr="a", to_addr="b", subject="s", body="body")
    msg_hash = message.hash()
    conn.execute(
        "INSERT INTO messages_content (msg_hash, from_addr, to_addr, subject, body) VALUES (?, ?, ?, ?, ?)",
        (msg_hash, message.from_addr, message.to_addr, message.subject, message.body),
    )
    conn.execute(
        "INSERT INTO embeddings (msg_hash, data) VALUES (?, ?)",
        (msg_hash, sqlite3.Binary(pickle.dumps(np.array([1.0])))),
    )
    conn.commit()
    conn.close()

    cache = SQLiteCache(str(db_path))
    cur = cache.conn.cursor()
    cur.execute("PRAGMA table_info(embeddings)")
    columns = {row[1] for row in cur.fetchall()}
    cur.execute("SELECT profile, created_at, updated_at FROM embeddings WHERE msg_hash = ?", (msg_hash,))
    profile, created_at, updated_at = cur.fetchone()

    assert {"msg_hash", "profile", "data", "created_at", "updated_at"}.issubset(columns)
    assert profile == LEGACY_EMBEDDING_PROFILE
    assert created_at
    assert updated_at
    assert cache.get_embedding(msg_hash, EMBEDDING_INPUT_PROFILE) is None
    assert cache.get_embedding(msg_hash, LEGACY_EMBEDDING_PROFILE).tolist() == [1.0]


def test_rehash_preserves_embedding_profile_and_timestamps(tmp_path):
    cache = SQLiteCache(str(tmp_path / "cache.sqlite"))
    conn = cache.conn
    message = Message(uid=1, from_addr="a", to_addr="b", subject="s", body="body")
    old_hash = "legacy-hash"
    computed_hash = message.hash()
    created_at = "2024-01-01T00:00:00+00:00"
    updated_at = "2024-02-01T00:00:00+00:00"

    conn.execute(
        "INSERT INTO messages_content (msg_hash, from_addr, to_addr, subject, body) VALUES (?, ?, ?, ?, ?)",
        (old_hash, message.from_addr, message.to_addr, message.subject, message.body),
    )
    conn.execute(
        "INSERT INTO folder_messages (folder, uid, msg_hash) VALUES (?, ?, ?)",
        ("INBOX", 1, old_hash),
    )
    conn.execute(
        """
        INSERT INTO embeddings (msg_hash, profile, data, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            old_hash,
            LEGACY_EMBEDDING_PROFILE,
            sqlite3.Binary(pickle.dumps(np.array([1.0]))),
            created_at,
            updated_at,
        ),
    )
    conn.commit()

    assert _rehash_and_reconcile(conn) == {"updated": 1, "merged": 0, "skipped": 0}

    cur = conn.cursor()
    cur.execute("SELECT msg_hash FROM folder_messages WHERE folder = ? AND uid = ?", ("INBOX", 1))
    assert cur.fetchone()[0] == computed_hash
    cur.execute(
        "SELECT profile, data, created_at, updated_at FROM embeddings WHERE msg_hash = ?",
        (computed_hash,),
    )
    profile, data, stored_created_at, stored_updated_at = cur.fetchone()

    assert profile == LEGACY_EMBEDDING_PROFILE
    assert pickle.loads(data).tolist() == [1.0]
    assert stored_created_at == created_at
    assert stored_updated_at == updated_at
    cur.execute("SELECT 1 FROM messages_content WHERE msg_hash = ?", (old_hash,))
    assert cur.fetchone() is None
