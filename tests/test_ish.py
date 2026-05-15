import os
import types
from threading import Event
from types import SimpleNamespace
from unittest import mock

import joblib
import numpy as np
import pytest

import ish.app as ish_mod
from ish.app import ISH
from ish.classification_service import ClassificationService
from ish.message import Message
from ish.training_manager import TrainingManager


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

    test_msg = Message(uid=42, from_addr="alice@example.com", to_addr="bob@example.com", body="Hello world",subject="Test")

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
            # Top probability stays below the configured minimum.
            return [[0.52, 0.48]]

    ish_instance.classifier = LowProbClassifier()
    ish_instance.get_embeddings = mock.MagicMock(return_value={99: np.array([0.1])})
    ish_instance.get_msgs = mock.MagicMock(return_value={99: Message(uid=99, from_addr="", to_addr="", body="", subject="")})
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
            # Top prediction clears the minimum probability threshold,
            # but the runner-up is too close to move safely.
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
            # Clears both thresholds with room for float precision.
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


def test_ish_reads_thresholds_from_settings(monkeypatch):
    fake_settings = SimpleNamespace(
        data_directory="/tmp/data",
        source_folders=["INBOX"],
        destination_folders=["Important"],
        ignore_folders=[],
        openai_api_key="key",
        openai_model="text-embedding-3-small",
        classification_probability_threshold=0.72,
        classification_runner_up_gap_threshold=0.21,
    )
    fake_settings.update_data_settings = lambda: None
    monkeypatch.setattr(ish_mod, "Settings", lambda debug=False: fake_settings)
    monkeypatch.setattr(ish_mod, "SQLiteCache", mock.MagicMock())
    monkeypatch.setattr(ish_mod, "ImapHandler", lambda *args, **kwargs: mock.MagicMock())

    ish_instance = ISH(dry_run=True)

    assert ish_instance._classification_service._probability_threshold == pytest.approx(0.72)
    assert ish_instance._classification_service._runner_up_gap_threshold == pytest.approx(0.21)


def make_handler_with_mock_conn():
    handler = ImapHandler(settings=mock.MagicMock(), readonly=False) # pyright: ignore[reportUndefinedVariable]
    conn = mock.MagicMock()
    handler._ImapHandler__imap_conn = conn
    return handler, conn
# Use this helper in tests that exercise __search, fetch, list_folders, etc.


def test_init_sets_flags_and_dry_run():
    ish = ISH(interactive=True, train=True, daemon=True, dry_run=True)
    assert ish.interactive is True
    assert ish.train is True
    assert ish.daemon is True
    # internal attribute for dry-run exists and was set
    assert getattr(ish, "_dry_run", None) is True


def test_init_runs_migration_when_legacy_cache_exists(monkeypatch, tmp_path):
    config_dir = tmp_path / "ish_conf"
    data_dir = config_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "msgs.db").write_text("legacy")
    monkeypatch.setenv("ISH_CONFIG_PATH", str(config_dir))

    migrate_mock = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "migrate_legacy_cache", migrate_mock)

    ISH()

    assert migrate_mock.call_count == 1


def test_init_skips_migration_without_legacy_cache(monkeypatch, tmp_path):
    config_dir = tmp_path / "ish_conf"
    (config_dir / "data").mkdir(parents=True)
    monkeypatch.setenv("ISH_CONFIG_PATH", str(config_dir))

    migrate_mock = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "migrate_legacy_cache", migrate_mock)

    ISH()

    migrate_mock.assert_not_called()


def test_run_calls_learn_when_no_model_file(tmp_path, monkeypatch):
    fake_settings = SimpleNamespace(
        data_directory="/tmp/data",
        source_folders=["INBOX"],
        destination_folders=["Important"],
        ignore_folders=[],
        openai_api_key="key",
        openai_model="text-embedding-ada-002",
        classification_probability_threshold=0.55,
        classification_runner_up_gap_threshold=0.15,
    )
    fake_settings.update_data_settings = lambda: None
    monkeypatch.setattr(ish_mod, "Settings", lambda debug=False: fake_settings)
    
    ish = ISH(dry_run=True, train=True)
    # Ensure training has folders to learn from
    #ish.__settings["destination_folders"] = ["Important"]

    monkeypatch.setattr(os.path, "isfile", lambda path: False)

    # make connect succeed
    monkeypatch.setattr(ish, "connect", lambda: True)
    # stub classify_messages so run doesn't try to use network
    monkeypatch.setattr(ish, "classify_messages", lambda _: None)

    # replace learn_folders with a mock to confirm it's called
    learn_mock = mock.MagicMock(return_value="trained")
    monkeypatch.setattr(ish, "learn_folders", learn_mock)

    rc = ish.run()
    assert rc == 0
    # since no model file existed, learn_folders should have been invoked once
    assert learn_mock.called is True


def test_learn_folders_uses_cached_embeddings(monkeypatch, tmp_path):
    config_dir = tmp_path / "ish_conf"
    monkeypatch.setenv("ISH_CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(joblib, "dump", lambda *a, **k: None)

    ish = ISH(dry_run=True)
    fake_imap = mock.MagicMock()
    fake_imap.search.side_effect = [[0, 1, 2], [10, 11, 12]]
    ish._ISH__imap_conn = fake_imap
    monkeypatch.setattr(ish, "get_embeddings", lambda folder, uids: {})

    folders = ["FolderA", "FolderB"]
    for folder_idx, folder in enumerate(folders):
        for sample_idx in range(3):
            uid = folder_idx * 10 + sample_idx
            msg = Message(uid=uid, from_addr="a", to_addr="b", subject="s", body=f"body-{folder_idx}-{sample_idx}")
            msg_hash = ish._cache.store_message(folder, uid, msg)
            ish._cache.store_embedding(msg_hash, np.array([float(uid), float(sample_idx)]))

    clf = ish.learn_folders(folders)
    assert clf is not None


def test_training_ignores_cached_embeddings_for_uids_not_in_folder_search():
    fake_imap = mock.MagicMock()
    fake_imap.search.side_effect = [[1, 2], [10, 11]]

    cached = {
        "FolderA": {
            1: np.array([1.0]),
            2: np.array([2.0]),
            999: np.array([999.0]),
        },
        "FolderB": {
            10: np.array([10.0]),
            11: np.array([11.0]),
            998: np.array([998.0]),
        },
    }

    manager = TrainingManager(
        imap_conn_provider=lambda: fake_imap,
        get_embeddings=lambda folder, uids: {},
        get_cache_embeddings=lambda folder: dict(cached[folder]),
        model_file="unused.pkl",
        max_learn_messages=100,
        exit_event=Event(),
    )

    embeddings, labels = manager._collect_training_data(fake_imap, ["FolderA", "FolderB"])

    assert len(embeddings) == 4
    assert labels == ["FolderA", "FolderA", "FolderB", "FolderB"]
    assert all(float(emb[0]) not in {998.0, 999.0} for emb in embeddings)


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
