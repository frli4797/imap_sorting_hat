import logging
from unittest import mock

import pytest

from ish import metrics


@pytest.fixture(autouse=True)
def reset_metrics_state(monkeypatch):
    # ensure we don't carry over logging handlers or env between tests
    monkeypatch.setattr(metrics, "_METRICS_SERVER_STARTED", False)
    yield
    monkeypatch.undo()


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
        metrics.configure_logging()
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in prev_handlers:
            root.addHandler(handler)
        root.setLevel(prev_level)

    assert called["level"] == metrics.DEFAULT_LOG_LEVEL
    assert called["format"] == metrics.LOG_FORMAT


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
        metrics.configure_logging()
        assert handler.level == metrics.DEFAULT_LOG_LEVEL
        assert handler.formatter._style._fmt == metrics.LOG_FORMAT
        assert root.level == metrics.DEFAULT_LOG_LEVEL
    finally:
        root.removeHandler(handler)
        for original in prev_handlers:
            root.addHandler(original)
        root.setLevel(prev_level)


def test_metrics_server_starts_when_env_set(monkeypatch):
    fake_start = mock.MagicMock()
    monkeypatch.setattr(metrics, "start_http_server", fake_start)
    monkeypatch.setenv("ISH_METRICS_PORT", "9410")
    monkeypatch.setenv("ISH_METRICS_ADDR", "127.0.0.1")

    metrics.start_metrics_server_if_configured()

    fake_start.assert_called_once_with(9410, addr="127.0.0.1")
    assert metrics._METRICS_SERVER_STARTED is True

    monkeypatch.delenv("ISH_METRICS_PORT", raising=False)
    monkeypatch.delenv("ISH_METRICS_ADDR", raising=False)


def test_metrics_server_invalid_port(monkeypatch):
    fake_start = mock.MagicMock()
    monkeypatch.setattr(metrics, "start_http_server", fake_start)
    monkeypatch.setenv("ISH_METRICS_PORT", "not-a-port")

    metrics.start_metrics_server_if_configured()

    fake_start.assert_not_called()
    assert metrics._METRICS_SERVER_STARTED is False

    monkeypatch.delenv("ISH_METRICS_PORT", raising=False)


def test_record_training_stats_updates_gauges(monkeypatch):
    gauge_embeddings = mock.MagicMock()
    gauge_accuracy = mock.MagicMock()
    gauge_duration = mock.MagicMock()
    monkeypatch.setattr(metrics, "TRAINING_EMBEDDINGS_GAUGE", gauge_embeddings)
    monkeypatch.setattr(metrics, "TRAINING_ACCURACY_GAUGE", gauge_accuracy)
    monkeypatch.setattr(metrics, "TRAINING_DURATION_GAUGE", gauge_duration)

    metrics.record_training_stats(10, 0.9, 5.5)

    gauge_embeddings.set.assert_called_once_with(10)
    gauge_accuracy.set.assert_called_once_with(0.9)
    gauge_duration.set.assert_called_once_with(5.5)


def test_record_classification_metrics(monkeypatch):
    gauge = mock.MagicMock()
    labels_mock = mock.MagicMock()
    gauge.labels.return_value = labels_mock
    monkeypatch.setattr(metrics, "TRAINING_CLASSIFICATION_GAUGE", gauge)

    report = {
        "label1": {"precision": 0.5, "recall": 0.7},
        "accuracy": 0.8,
    }
    metrics.record_classification_metrics(report)

    gauge.labels.assert_any_call("label1", "precision")
    gauge.labels.assert_any_call("label1", "recall")
    labels_mock.set.assert_called()


def test_record_folder_embedding_count(monkeypatch):
    gauge = mock.MagicMock()
    labels_mock = mock.MagicMock()
    gauge.labels.return_value = labels_mock
    monkeypatch.setattr(metrics, "CLASSIFY_FOLDER_EMBEDDINGS_GAUGE", gauge)

    metrics.record_folder_embedding_count("Inbox", 5)

    gauge.labels.assert_called_once_with("Inbox")
    labels_mock.set.assert_called_once_with(5)


def test_record_db_size(monkeypatch, tmp_path):
    gauge = mock.MagicMock()
    monkeypatch.setattr(metrics, "DB_SIZE_GAUGE", gauge)

    db_file = tmp_path / "cache.sqlite"
    db_file.write_bytes(b"123456789")

    metrics.record_db_size(str(db_file))

    gauge.set.assert_called_once_with(9)
