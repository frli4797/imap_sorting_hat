"""
Migrate old shelve caches to the new normalized SQLite cache.

Expected old shelve format (same as used by previous ish versions):
- msgs shelve:
    "folder:uid" -> "<msg_hash>"
    "<msg_hash>.mesg" -> Message(...)
- embd shelve:
    "<msg_hash>.embd" -> numpy.ndarray

The script will iterate the message mappings and embeddings and store them in SQLiteCache
while preserving the original shelve hashes. After inserting all legacy keys it will
recalculate message hashes and reconcile/merge rows inside the sqlite DB.
"""
from __future__ import annotations

import argparse
import logging
import os
import shelve
import sqlite3
import pickle
from typing import Optional

import numpy as np

from .settings import Settings
from .db import SQLiteCache
from .message import Message

logger = logging.getLogger("ish.migrate")
logging.basicConfig(level=logging.DEBUG)


def _find_shelve_base(data_dir: str, base_name: str) -> str:
    candidates = [
        os.path.join(data_dir, base_name),
        os.path.join(data_dir, base_name + ".db"),
        os.path.join(data_dir, base_name + ".dat"),
        os.path.join(data_dir, base_name + ".dir"),
        os.path.join(data_dir, base_name + ".bak"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            for ext in (".db", ".dat", ".dir", ".bak"):
                if candidate.endswith(ext):
                    return candidate[: -len(ext)]
            return candidate
    return os.path.join(data_dir, base_name)


def _insert_message_preserve_hash(conn: sqlite3.Connection, folder: str, uid: int, msg_hash: str, mesg: Message):
    cur = conn.cursor()
    # insert content if missing (preserve legacy msg_hash)
    
    cur.execute(
        "INSERT OR IGNORE INTO messages_content (msg_hash, from_addr, to_addr, subject, body) VALUES (?, ?, ?, ?, ?)",
        (msg_hash, mesg.from_addr, mesg.to_addr, mesg.subject, mesg.body),
    )
    # ensure latest folder assignment wins for this hash
    cur.execute("DELETE FROM folder_messages WHERE msg_hash = ?", (msg_hash,))
    # insert mapping (folder, uid) -> msg_hash
    cur.execute(
        "INSERT OR REPLACE INTO folder_messages (folder, uid, msg_hash) VALUES (?, ?, ?)",
        (folder, int(uid), msg_hash),
    )


def _insert_embedding_preserve_hash(conn: sqlite3.Connection, msg_hash: str, emb: np.ndarray):
    cur = conn.cursor()
    pickled = pickle.dumps(emb, protocol=pickle.HIGHEST_PROTOCOL)
    cur.execute(
        "INSERT OR REPLACE INTO embeddings (msg_hash, data) VALUES (?, ?)",
        (msg_hash, sqlite3.Binary(pickled)),
    )


def _rehash_and_reconcile(conn: sqlite3.Connection) -> dict:
    """
    Recompute message hashes for all messages_content rows and reconcile differences.
    - If computed_hash already exists: move folder_messages -> computed_hash, move embedding if needed, delete old row.
    - Else: create new messages_content row with computed_hash, move folder_messages and embedding, delete old row.
    Returns summary dict.
    """
    cur = conn.cursor()
    cur.execute("SELECT msg_hash, from_addr, to_addr, subject, body FROM messages_content")
    rows = cur.fetchall()

    updated = merged = skipped = 0

    for old_hash, from_addr, to_addr, subject, body in rows:
        try:
            msg = Message(uid=None, from_addr=from_addr or "", to_addr=to_addr, subject=subject or "", body=body or "")
            computed = msg.hash()
        except Exception:
            skipped += 1
            continue

        if computed == old_hash:
            continue

        try:
            conn.execute("BEGIN")
            # does computed already exist?
            cur.execute("SELECT 1 FROM messages_content WHERE msg_hash = ? LIMIT 1", (computed,))
            computed_exists = cur.fetchone() is not None

            if computed_exists:
                # move folder_messages
                cur.execute("UPDATE folder_messages SET msg_hash = ? WHERE msg_hash = ?", (computed, old_hash))
                # move embedding only if computed doesn't have embedding
                cur.execute("SELECT data FROM embeddings WHERE msg_hash = ? LIMIT 1", (old_hash,))
                emb_old = cur.fetchone()
                cur.execute("SELECT 1 FROM embeddings WHERE msg_hash = ? LIMIT 1", (computed,))
                emb_new_exists = cur.fetchone() is not None
                if emb_old and not emb_new_exists:
                    cur.execute("INSERT OR REPLACE INTO embeddings (msg_hash, data) VALUES (?, ?)", (computed, emb_old[0]))
                    cur.execute("DELETE FROM embeddings WHERE msg_hash = ?", (old_hash,))
                # delete old content
                cur.execute("DELETE FROM messages_content WHERE msg_hash = ?", (old_hash,))
                merged += 1
            else:
                # create new content row
                cur.execute(
                    "INSERT INTO messages_content (msg_hash, from_addr, to_addr, subject, body) VALUES (?, ?, ?, ?, ?)",
                    (computed, from_addr, to_addr, subject, body),
                )
                # move folder_messages and embedding
                cur.execute("UPDATE folder_messages SET msg_hash = ? WHERE msg_hash = ?", (computed, old_hash))
                cur.execute("SELECT data FROM embeddings WHERE msg_hash = ? LIMIT 1", (old_hash,))
                emb_old = cur.fetchone()
                if emb_old:
                    cur.execute("INSERT OR REPLACE INTO embeddings (msg_hash, data) VALUES (?, ?)", (computed, emb_old[0]))
                    cur.execute("DELETE FROM embeddings WHERE msg_hash = ?", (old_hash,))
                cur.execute("DELETE FROM messages_content WHERE msg_hash = ?", (old_hash,))
                updated += 1

            conn.commit()
        except Exception:
            conn.rollback()
            skipped += 1

    return {"updated": updated, "merged": merged, "skipped": skipped}


def migrate(
    shelve_msgs_path: Optional[str] = None,
    shelve_embd_path: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    dry_run: bool = False,
):
    settings = Settings(debug=False)
    settings.update_data_settings()

    if not shelve_msgs_path:
        shelve_msgs_path = _find_shelve_base(settings.data_directory, "msgs")
    if not shelve_embd_path:
        shelve_embd_path = _find_shelve_base(settings.data_directory, "embd")
    if not sqlite_path:
        sqlite_path = os.path.join(settings.data_directory, "cache.sqlite")

    logger.info("Migrating shelve -> sqlite")
    logger.info("msgs shelve base: %s", shelve_msgs_path)
    logger.info("embd shelve base: %s", shelve_embd_path)
    logger.info("sqlite db: %s", sqlite_path)
    if dry_run:
        logger.info("Running in dry-run mode; nothing will be written to sqlite.")

    # Ensure DB/schema exists by creating a SQLiteCache instance
    cache = SQLiteCache(sqlite_path) if not dry_run else None
    conn = cache.conn if cache is not None else sqlite3.connect(sqlite_path)

    migrated_msg_count = 0
    migrated_emb_count = 0
    skipped_missing_message = 0

    try:
        with shelve.open(shelve_msgs_path, writeback=False) as s_msgs:
            for key in list(s_msgs.keys()):
                if key.endswith(".mesg"):
                    continue
                if ":" not in key:
                    logger.debug("Skipping unknown key in msgs shelve: %s", key)
                    continue

                folder, uid_s = key.split(":", 1)
                try:
                    uid = int(uid_s)
                except ValueError:
                    logger.debug("Invalid uid for key %s, skipping", key)
                    continue

                msg_hash = s_msgs.get(key)
                if not msg_hash:
                    logger.debug("No hash stored for %s -> %s, skipping", folder, uid)
                    continue

                mesg_key = f"{msg_hash}.mesg"
                if mesg_key not in s_msgs:
                    logger.warning("Mapping %s -> %s found but content %s is missing; skipping", key, msg_hash, mesg_key)
                    skipped_missing_message += 1
                    continue

                mesg = s_msgs[mesg_key]
                if not isinstance(mesg, Message):
                    try:
                        mesg = Message(
                            uid=uid,
                            from_addr=mesg.get("from_addr", "") or mesg.get("from", ""),
                            to_addr=mesg.get("to_addr", "") or  mesg.get("tocc", ""),
                            subject=mesg.get("subject", "") or mesg.get("subject", mesg.get("subject", "")),
                            body=mesg.get("body", "") or mesg.get("'body'", mesg.get("body", "")),
                        )
                    except Exception:
                        logger.exception("Failed to coerce message object for key %s", mesg_key)
                        skipped_missing_message += 1
                        continue

                if dry_run:
                    migrated_msg_count += 1
                else:
                    _insert_message_preserve_hash(conn, folder, uid, msg_hash, mesg)
                    conn.commit()
                    migrated_msg_count += 1

    except Exception as e:
        logger.exception("Failed while reading messages shelve: %s", e)

    try:
        with shelve.open(shelve_embd_path, writeback=False) as s_embd:
            for key in list(s_embd.keys()):
                if not key.endswith(".embd"):
                    logger.debug("Skipping non-embedding key in embd shelve: %s", key)
                    continue
                msg_hash = key[:-5]
                emb = s_embd.get(key)
                if emb is None:
                    logger.debug("No embedding found for %s, skipping", key)
                    continue

                if not isinstance(emb, (np.ndarray, list, tuple)):
                    logger.warning("Embedding for %s is unexpected type %s; attempting to use as-is", key, type(emb))

                if dry_run:
                    migrated_emb_count += 1
                else:
                    if not isinstance(emb, np.ndarray):
                        emb = np.asarray(emb)
                    _insert_embedding_preserve_hash(conn, msg_hash, emb)
                    conn.commit()

    except Exception as e:
        logger.exception("Failed while reading embeddings shelve: %s", e)

    # Post-migration: recalculate hashes inside migration (db.py left unchanged)
    if not dry_run:
        logger.info("Running post-migration rehash pass (inside migrator)...")
        summary = _rehash_and_reconcile(conn)
        logger.info("Rehash pass summary: %s", summary)

    if cache:
        try:
            cache.close()
        except Exception:
            pass
    else:
        try:
            conn.close()
        except Exception:
            pass

    logger.info("Migration finished: messages=%d (skipped_missing=%d), embeddings=%d",
                migrated_msg_count, skipped_missing_message, migrated_emb_count)
    return migrated_msg_count, skipped_missing_message, migrated_emb_count


def main():
    parser = argparse.ArgumentParser(description="Migrate shelve caches to SQLiteCache")
    parser.add_argument("--msgs", "-m", help="Old messages shelve base path (no extension)")
    parser.add_argument("--embd", "-e", help="Old embeddings shelve base path (no extension)")
    parser.add_argument("--sqlite", "-s", help="Destination sqlite path to write", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to sqlite")
    args = parser.parse_args()
    migrate(args.msgs, args.embd, args.sqlite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
