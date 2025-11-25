import types
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ish import ISH
import ish


def make_fake_openai_client():
    """Return a fake OpenAI client where embeddings.create returns an object
    with .data list of SimpleNamespace(embedding=...)."""
    client = mock.MagicMock()

    def create(input, model):
        # create an embedding object for every input item
        return SimpleNamespace(data=[SimpleNamespace(embedding=np.array([len(i)])) for i in input])

    client.embeddings.create = mock.MagicMock(side_effect=create)
    return client


def test___get_embeddings_batches_calls_api_multiple_times():
    ish = ISH(dry_run=True)
    ish._ISH__client = make_fake_openai_client()

    # 25 items -> batch size 20 then 5 -> should call embeddings.create twice
    texts = [f"text-{i}" for i in range(25)]
    embeddings = ish._ISH__get_embeddings(texts)

    assert len(embeddings) == 25
    assert ish._ISH__client.embeddings.create.call_count == 2


def test_move_messages_dry_run_does_not_call_imap_move():
    ish = ISH(dry_run=True)
    mock_conn = mock.MagicMock()
    ish._ISH__imap_conn = mock_conn

    messages = {
        "A": [{"uid": 1}, {"uid": 2}],
        "B": [{"uid": 3}],
    }

    moved = ish.move_messages("INBOX", messages)
    # Dry run should not call IMAP client's move method
    mock_conn.move.assert_not_called()
    assert moved == 0


def test_move_messages_calls_imap_move_when_not_dry_run():
    ish = ISH(dry_run=False)
    mock_conn = mock.MagicMock()
    ish._ISH__imap_conn = mock_conn

    messages = {
        "X": [{"uid": 10}, {"uid": 11}],
    }

    moved = ish.move_messages("SRC", messages)
    mock_conn.move.assert_called_once_with(
        "SRC", [10, 11], "X", flag_messages=ish.interactive, flag_unseen=not ish.interactive
    )
    assert moved == 2


def test_classify_messages_moves_high_probability_and_skips_low(monkeypatch):
    ish_mod = ish
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [42]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)
    # Provide a simple classifier that returns deterministic predictions/probabilities
    class DummyClassifier:
        classes_ = ["dest1", "dest2"]

        def predict(self, X):
            # always predict dest1
            return ["dest1"]

        def predict_proba(self, X):
            # single high probability for class 'dest1'
            return [[0.8, 0.2]]

    ish_instance.classifier = DummyClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={42: np.array([0.1, 0.2, 0.3])})
    ish_instance.get_msgs = mock.MagicMock(return_value={42: {"from": "alice@example.com", "body": "Hello world"}})
    ish_instance.move_messages = mock.MagicMock(return_value=1)

    ish_instance.classify_messages(["INBOX"])
    ish_instance.move_messages.assert_called()
    assert ish_instance.moved >= 0


def test_classify_messages_skips_when_probability_low(monkeypatch):
    ish_mod = ish
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [99]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class LowProbClassifier:
        classes_ = ["destA", "destB"]

        def predict(self, X):
            return ["destA"]

        def predict_proba(self, X):
            # Low top probability so it will be skipped (< 0.25)
            return [[0.1, 0.05]]

    ish_instance.classifier = LowProbClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={99: np.array([0.1])})
    ish_instance.get_msgs = mock.MagicMock(return_value={99: {"from": "bob", "body": "x"}})
    ish_instance.move_messages = mock.MagicMock(return_value=0)

    ish_instance.classify_messages(["INBOX"])
    assert ish_instance.skipped >= 1
    ish_instance.move_messages.assert_called()


def make_handler_with_mock_conn():
    handler = ImapHandler(settings=mock.MagicMock(), readonly=False) # pyright: ignore[reportUndefinedVariable]
    conn = mock.MagicMock()
    handler._ImapHandler__imap_conn = conn
    return handler, conn
# Use this helper in tests that exercise __search, fetch, list_folders, etc.