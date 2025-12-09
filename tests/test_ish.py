import logging
import os
import types
from types import SimpleNamespace
from unittest import mock

import joblib
import numpy as np
import pytest

import ish.app as ish_mod
from ish.app import ISH
from ish.message import Message


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


def test_move_messages_dry_run_does_not_call_imap_move(monkeypatch):
    ish = ISH(dry_run=True)
    mock_conn = mock.MagicMock()
    ish._ISH__imap_conn = mock_conn
    counter_mock = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "MESSAGES_MOVED_COUNTER", counter_mock)

    messages = {
        "A": [{"uid": 1}, {"uid": 2}],
        "B": [{"uid": 3}],
    }

    moved = ish.move_messages("INBOX", messages)
    # Dry run should not call IMAP client's move method
    mock_conn.move.assert_not_called()
    assert moved == 0
    counter_mock.inc.assert_not_called()


def test_move_messages_calls_imap_move_when_not_dry_run(monkeypatch):
    ish = ISH(dry_run=False)
    mock_conn = mock.MagicMock()
    ish._ISH__imap_conn = mock_conn
    counter_mock = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "MESSAGES_MOVED_COUNTER", counter_mock)

    messages = {
        "X": [{"uid": 10}, {"uid": 11}],
    }

    moved = ish.move_messages("SRC", messages)
    mock_conn.move.assert_called_once_with(
        "SRC", [10, 11], "X", flag_messages=ish.interactive, flag_unseen=not ish.interactive
    )
    assert moved == 2
    counter_mock.inc.assert_called_once_with(2)


def test_configure_logging_basicconfig_called_without_handlers(monkeypatch):
    root = logging.getLogger()
    prev_handlers = root.handlers[:]
    prev_level = root.level
    for handler in prev_handlers:
        root.removeHandler(handler)

    called = {}

    def fake_basic_config(*, level, format):
        called["level"] = level
        called["format"] = format

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    try:
        ish_mod._configure_logging()
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in prev_handlers:
            root.addHandler(handler)
        root.setLevel(prev_level)

    assert called["level"] == ish_mod.DEFAULT_LOG_LEVEL
    assert called["format"] == ish_mod.LOG_FORMAT


def test_configure_logging_updates_existing_handlers(monkeypatch):
    root = logging.getLogger()
    prev_handlers = root.handlers[:]
    prev_level = root.level
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    try:
        ish_mod._configure_logging()
        assert handler.level == ish_mod.DEFAULT_LOG_LEVEL
        assert handler.formatter._style._fmt == ish_mod.LOG_FORMAT
        assert root.level == ish_mod.DEFAULT_LOG_LEVEL
    finally:
        root.removeHandler(handler)
        for original in prev_handlers:
            root.addHandler(original)
        root.setLevel(prev_level)


def test_env_log_level_overrides_root_and_handlers(monkeypatch):
    root = logging.getLogger()
    prev_handlers = root.handlers[:]
    prev_level = root.level
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    monkeypatch.setenv(ish_mod.LOG_LEVEL_ENV, "DEBUG")

    try:
        ish_mod._configure_logging()
        assert root.level == logging.DEBUG
        assert handler.level == logging.DEBUG
    finally:
        monkeypatch.delenv(ish_mod.LOG_LEVEL_ENV, raising=False)
        root.removeHandler(handler)
        for original in prev_handlers:
            root.addHandler(original)
        root.setLevel(prev_level)


def test_env_named_log_levels_override_specific_loggers(monkeypatch):
    target_a = logging.getLogger("ISH")
    target_b = logging.getLogger("ish.imap")
    prev_a = target_a.level
    prev_b = target_b.level
    monkeypatch.setenv(ish_mod.LOG_LEVELS_ENV, "ISH=DEBUG,ish.imap=WARNING,invalid")

    try:
        ish_mod._configure_logging()
        assert target_a.level == logging.DEBUG
        assert target_b.level == logging.WARNING
    finally:
        monkeypatch.delenv(ish_mod.LOG_LEVELS_ENV, raising=False)
        target_a.setLevel(prev_a)
        target_b.setLevel(prev_b)


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
    ish_instance.move_messages = mock.MagicMock(return_value=1)

    ish_instance.classify_messages(["INBOX"])
    ish_instance.move_messages.assert_called()
    assert ish_instance.moved >= 0


def test_classify_messages_skips_when_probability_low(monkeypatch):
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
    ish_instance.get_msgs = mock.MagicMock(return_value={99: Message(uid=99, from_addr="", to_addr="", body="", subject="")})
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


def test_metrics_server_starts_when_env_set(monkeypatch):
    prev_started = ish_mod._METRICS_SERVER_STARTED
    monkeypatch.setattr(ish_mod, "_METRICS_SERVER_STARTED", False)
    fake_start = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "start_http_server", fake_start)
    monkeypatch.setenv("ISH_METRICS_PORT", "9410")
    monkeypatch.setenv("ISH_METRICS_ADDR", "127.0.0.1")

    ish_mod._start_metrics_server_if_configured()

    fake_start.assert_called_once_with(9410, addr="127.0.0.1")
    assert ish_mod._METRICS_SERVER_STARTED is True

    monkeypatch.delenv("ISH_METRICS_PORT", raising=False)
    monkeypatch.delenv("ISH_METRICS_ADDR", raising=False)
    monkeypatch.setattr(ish_mod, "_METRICS_SERVER_STARTED", prev_started)


def test_metrics_server_invalid_port(monkeypatch):
    prev_started = ish_mod._METRICS_SERVER_STARTED
    monkeypatch.setattr(ish_mod, "_METRICS_SERVER_STARTED", False)
    fake_start = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "start_http_server", fake_start)
    monkeypatch.setenv("ISH_METRICS_PORT", "not-a-port")

    ish_mod._start_metrics_server_if_configured()

    fake_start.assert_not_called()
    assert ish_mod._METRICS_SERVER_STARTED is False

    monkeypatch.delenv("ISH_METRICS_PORT", raising=False)
    monkeypatch.setattr(ish_mod, "_METRICS_SERVER_STARTED", prev_started)


def test_increment_skipped_updates_counter(monkeypatch):
    ish = ISH(dry_run=True)
    counter_mock = mock.MagicMock()
    monkeypatch.setattr(ish_mod, "MESSAGES_SKIPPED_COUNTER", counter_mock)
    start_value = ish.skipped

    ish._increment_skipped(3)

    assert ish.skipped == start_value + 3
    counter_mock.inc.assert_called_once_with(3)


def test_learn_folders_sets_training_metrics(monkeypatch, tmp_path):
    config_dir = tmp_path / "ish_conf"
    monkeypatch.setenv("ISH_CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(joblib, "dump", lambda *a, **k: None)

    ish = ISH(dry_run=True)
    gauge_embeddings = mock.MagicMock()
    gauge_accuracy = mock.MagicMock()
    gauge_duration = mock.MagicMock()
    gauge_classification = mock.MagicMock()
    labels_mock = mock.MagicMock()
    gauge_classification.labels.return_value = labels_mock
    monkeypatch.setattr(ish_mod, "TRAINING_EMBEDDINGS_GAUGE", gauge_embeddings)
    monkeypatch.setattr(ish_mod, "TRAINING_ACCURACY_GAUGE", gauge_accuracy)
    monkeypatch.setattr(ish_mod, "TRAINING_DURATION_GAUGE", gauge_duration)
    monkeypatch.setattr(ish_mod, "TRAINING_CLASSIFICATION_GAUGE", gauge_classification)

    fake_imap = mock.MagicMock()
    fake_imap.search.return_value = [1, 2]
    ish._ISH__imap_conn = fake_imap

    embeddings_map = {
        "FolderA": {1: np.array([0.1, 0.2]), 2: np.array([0.2, 0.3])},
        "FolderB": {1: np.array([0.4, 0.5]), 2: np.array([0.6, 0.7])},
    }

    def fake_get_embeddings(folder, uids):
        return embeddings_map[folder]

    monkeypatch.setattr(ish, "get_embeddings", fake_get_embeddings)

    clf = ish.learn_folders(["FolderA", "FolderB"])
    assert clf is not None
    gauge_embeddings.set.assert_called_once_with(4)
    gauge_accuracy.set.assert_called_once()
    gauge_duration.set.assert_called_once()
    gauge_classification.labels.assert_called()
    labels_mock.set.assert_called()
