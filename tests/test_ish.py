import os
from types import SimpleNamespace
from unittest import mock

import joblib
import numpy as np
import pytest

import ish.app as ish_mod
from ish.app import ISH
from ish.embedding_store import embedding_profile_for_model
from ish.model_store import make_model_bundle


def make_fake_openai_client():
    client = mock.MagicMock()

    def create(input, model):
        return SimpleNamespace(data=[SimpleNamespace(embedding=np.array([len(i)])) for i in input])

    client.embeddings.create = mock.MagicMock(side_effect=create)
    return client


def test___get_embeddings_batches_calls_api_multiple_times():
    ish = ISH(dry_run=True)
    ish._ISH__client = make_fake_openai_client()

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
        "SRC",
        [10, 11],
        "X",
        flag_messages=ish.interactive,
        flag_unseen=not ish.interactive,
    )
    assert moved == 2


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


def test_init_sets_flags_and_dry_run():
    ish = ISH(interactive=True, train=True, daemon=True, dry_run=True)

    assert ish.interactive is True
    assert ish.train is True
    assert ish.daemon is True
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
    monkeypatch.setattr(os.path, "isfile", lambda path: False)
    monkeypatch.setattr(ish, "connect", lambda: True)
    monkeypatch.setattr(ish, "classify_messages", lambda _: None)
    learn_mock = mock.MagicMock(return_value="trained")
    monkeypatch.setattr(ish, "learn_folders", learn_mock)

    rc = ish.run()

    assert rc == 0
    assert learn_mock.called is True


def test_run_retrains_when_existing_model_has_mismatched_embedding_profile(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fake_settings = SimpleNamespace(
        data_directory=str(data_dir),
        source_folders=["INBOX"],
        destination_folders=["Important"],
        ignore_folders=[],
        openai_api_key="key",
        openai_model="text-embedding-new",
        classification_probability_threshold=0.55,
        classification_runner_up_gap_threshold=0.15,
    )
    fake_settings.update_data_settings = lambda: None
    monkeypatch.setattr(ish_mod, "Settings", lambda debug=False: fake_settings)

    ish = ISH(dry_run=True)
    joblib.dump(
        make_model_bundle(object(), embedding_profile_for_model("text-embedding-old")),
        ish.model_file,
    )
    monkeypatch.setattr(ish, "connect", lambda: True)
    train_mock = mock.MagicMock(return_value=True)
    monkeypatch.setattr(ish, "train_on_destination_folders", train_mock)
    monkeypatch.setattr(ish, "classify_messages", lambda source_folders: None)

    assert ish.run() == 0
    train_mock.assert_called_once_with()
