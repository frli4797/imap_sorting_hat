from unittest import mock

import numpy as np

from ish.db import LEGACY_EMBEDDING_PROFILE, SQLiteCache
from ish.embedding_store import (
    EMBEDDING_INPUT_PROFILE,
    EmbeddingStore,
    embedding_profile_for_model,
)
from ish.message import Message


def test_message_embedding_text_includes_structured_headers():
    message = Message(
        uid=1,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Travel",
        body="Boarding pass attached",
    )

    assert message.embedding_text() == (
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Travel\n"
        "Body: Boarding pass attached"
    )


def test_embedding_store_ignores_legacy_profile_and_regenerates_active_embedding(tmp_path):
    cache = SQLiteCache(str(tmp_path / "cache.sqlite"))
    message = Message(
        uid=1,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Travel",
        body="Boarding pass attached",
    )
    msg_hash = cache.store_message("INBOX", 1, message)
    cache.store_embedding(msg_hash, np.array([1.0]), LEGACY_EMBEDDING_PROFILE)

    repository = mock.MagicMock()
    repository.get_hashes.return_value = {1: msg_hash}
    repository.get_messages.return_value = {1: message}
    embedder = mock.MagicMock(return_value=[np.array([2.0])])
    store = EmbeddingStore(
        cache=cache,
        message_repository=repository,
        embedder=embedder,
        max_chars=8192,
        data_directory=str(tmp_path),
    )

    embeddings = store.get_embeddings("INBOX", [1])

    assert embeddings[1].tolist() == [2.0]
    assert cache.get_embedding(msg_hash, LEGACY_EMBEDDING_PROFILE).tolist() == [1.0]
    assert cache.get_embedding(msg_hash, EMBEDDING_INPUT_PROFILE).tolist() == [2.0]
    embedder.assert_called_once_with([message.embedding_text()])


def test_embedding_store_reuses_active_profile_embedding(tmp_path):
    cache = SQLiteCache(str(tmp_path / "cache.sqlite"))
    message = Message(
        uid=1,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Travel",
        body="Boarding pass attached",
    )
    msg_hash = cache.store_message("INBOX", 1, message)
    cache.store_embedding(msg_hash, np.array([3.0]), EMBEDDING_INPUT_PROFILE)

    repository = mock.MagicMock()
    repository.get_hashes.return_value = {1: msg_hash}
    embedder = mock.MagicMock()
    store = EmbeddingStore(
        cache=cache,
        message_repository=repository,
        embedder=embedder,
        max_chars=8192,
        data_directory=str(tmp_path),
    )

    embeddings = store.get_embeddings("INBOX", [1])

    assert embeddings[1].tolist() == [3.0]
    embedder.assert_not_called()
    repository.get_messages.assert_not_called()


def test_embedding_profile_includes_openai_model_and_model_change_misses_cache(tmp_path):
    cache = SQLiteCache(str(tmp_path / "cache.sqlite"))
    message = Message(
        uid=1,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        subject="Travel",
        body="Boarding pass attached",
    )
    msg_hash = cache.store_message("INBOX", 1, message)
    old_profile = embedding_profile_for_model("text-embedding-old")
    new_profile = embedding_profile_for_model("text-embedding-new")
    cache.store_embedding(msg_hash, np.array([1.0]), old_profile)

    repository = mock.MagicMock()
    repository.get_hashes.return_value = {1: msg_hash}
    repository.get_messages.return_value = {1: message}
    embedder = mock.MagicMock(return_value=[np.array([2.0])])
    store = EmbeddingStore(
        cache=cache,
        message_repository=repository,
        embedder=embedder,
        max_chars=8192,
        data_directory=str(tmp_path),
        embedding_profile=new_profile,
    )

    embeddings = store.get_embeddings("INBOX", [1])

    assert embeddings[1].tolist() == [2.0]
    assert cache.get_embedding(msg_hash, old_profile).tolist() == [1.0]
    assert cache.get_embedding(msg_hash, new_profile).tolist() == [2.0]
    embedder.assert_called_once_with([message.embedding_text()])
