from threading import Event
from unittest import mock

import joblib
import numpy as np

import ish.app as ish_mod
from ish.app import ISH
from ish.classification_service import ClassificationService
from ish.embedding_store import embedding_profile_for_model
from ish.message import Message
from ish.model_store import make_model_bundle


def test_classify_messages_moves_high_probability_and_skips_low(monkeypatch):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [42]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class DummyClassifier:
        classes_ = ["dest1", "dest2"]

        def predict(self, X):
            return ["dest1"]

        def predict_proba(self, X):
            return [[0.8, 0.2]]

    test_msg = Message(
        uid=42,
        from_addr="alice@example.com",
        to_addr="bob@example.com",
        body="Hello world",
        subject="Test",
    )

    ish_instance.classifier = DummyClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={42: np.array([0.1, 0.2, 0.3])})
    ish_instance.get_msgs = mock.MagicMock(return_value={42: test_msg})
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=1)

    ish_instance.classify_messages(["INBOX"])
    ish_instance._classification_service.move_messages.assert_called()
    assert ish_instance.moved >= 0


def test_classify_messages_skips_when_probability_low(monkeypatch):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [99]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class LowProbClassifier:
        classes_ = ["destA", "destB"]

        def predict_proba(self, X):
            return [[0.52, 0.48]]

    ish_instance.classifier = LowProbClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={99: np.array([0.1])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={99: Message(uid=99, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=0)

    ish_instance.classify_messages(["INBOX"])
    assert ish_instance.skipped >= 1
    ish_instance._classification_service.move_messages.assert_called()


def test_classify_messages_skips_when_runner_up_is_too_close(monkeypatch):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [101]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class AmbiguousClassifier:
        classes_ = ["destA", "destB"]

        def predict(self, X):
            return ["destA"]

        def predict_proba(self, X):
            return [[0.56, 0.44]]

    ish_instance.classifier = AmbiguousClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={101: np.array([0.2])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={101: Message(uid=101, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=0)

    ish_instance.classify_messages(["INBOX"])

    assert ish_instance.skipped >= 1
    ish_instance._classification_service.move_messages.assert_called_once_with("INBOX", {})


def test_classify_messages_logs_specific_reason_when_runner_up_is_too_close(monkeypatch, caplog):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [104]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class AmbiguousClassifier:
        classes_ = ["destA", "destB"]

        def predict_proba(self, X):
            return [[0.56, 0.44]]

    ish_instance.classifier = AmbiguousClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={104: np.array([0.2])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={104: Message(uid=104, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=0)

    with caplog.at_level("DEBUG", logger="ish"):
        ish_instance.classify_messages(["INBOX"])

    assert "Skipping due to ambiguous top prediction" in caplog.text
    assert "runner_up=0.44" in caplog.text
    assert "min_gap=0.15" in caplog.text


def test_classify_messages_moves_when_gap_meets_threshold(monkeypatch):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [102]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class ClearEnoughClassifier:
        classes_ = ["destA", "destB"]

        def predict_proba(self, X):
            return [[0.58, 0.42]]

    ish_instance.classifier = ClearEnoughClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={102: np.array([0.2])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={102: Message(uid=102, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=1)

    ish_instance.classify_messages(["INBOX"])

    assert ish_instance.skipped == 0
    ish_instance._classification_service.move_messages.assert_called_once()


def test_classify_messages_skips_when_probability_equals_threshold(monkeypatch):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [103]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class ThresholdEdgeClassifier:
        classes_ = ["destA", "destB"]

        def predict_proba(self, X):
            return [[0.55, 0.45]]

    ish_instance.classifier = ThresholdEdgeClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={103: np.array([0.2])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={103: Message(uid=103, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=0)

    ish_instance.classify_messages(["INBOX"])

    assert ish_instance.skipped >= 1
    ish_instance._classification_service.move_messages.assert_called_once_with("INBOX", {})


def test_classify_messages_logs_specific_reason_when_probability_too_low(monkeypatch, caplog):
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [105]
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *a, **k: fake_imap)

    ish_instance = ISH(dry_run=True)

    class ThresholdEdgeClassifier:
        classes_ = ["destA", "destB"]

        def predict_proba(self, X):
            return [[0.55, 0.45]]

    ish_instance.classifier = ThresholdEdgeClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={105: np.array([0.2])})
    ish_instance.get_msgs = mock.MagicMock(
        return_value={105: Message(uid=105, from_addr="", to_addr="", body="", subject="")}
    )
    ish_instance._classification_service.move_messages = mock.MagicMock(return_value=0)

    with caplog.at_level("DEBUG", logger="ish"):
        ish_instance.classify_messages(["INBOX"])

    assert "Skipping due to low confidence" in caplog.text
    assert "top=0.55" in caplog.text
    assert "min=0.55" in caplog.text


def test_classification_sorts_source_uids_newest_first():
    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [1, 3, 2]

    class DummyClassifier:
        classes_ = ["A", "B"]

        def predict_proba(self, X):
            return [[0.9, 0.1]]

    service = ClassificationService(
        imap_conn_provider=lambda: fake_imap,
        get_embeddings=mock.MagicMock(return_value={}),
        get_messages=mock.MagicMock(return_value={}),
        model_file="unused.pkl",
        max_source_messages=2,
        interactive=False,
        dry_run=True,
        exit_event=Event(),
    )
    service.classifier = DummyClassifier()

    service.classify_messages(["INBOX"])

    service._get_embeddings.assert_called_once_with("INBOX", [3, 2])


def test_classification_rejects_model_with_mismatched_embedding_profile(tmp_path):
    model_file = tmp_path / "model.pkl"
    joblib.dump(
        make_model_bundle(object(), embedding_profile_for_model("text-embedding-old")),
        model_file,
    )
    fake_imap = mock.MagicMock()
    service = ClassificationService(
        imap_conn_provider=lambda: fake_imap,
        get_embeddings=mock.MagicMock(return_value={}),
        get_messages=mock.MagicMock(return_value={}),
        model_file=str(model_file),
        max_source_messages=10,
        interactive=False,
        dry_run=True,
        exit_event=Event(),
        embedding_profile=embedding_profile_for_model("text-embedding-new"),
    )

    assert service.classify_messages(["INBOX"]) == (0, 0)
    fake_imap.search.assert_not_called()
