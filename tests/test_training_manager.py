from threading import Event
from unittest import mock

import joblib
import numpy as np

from ish.app import ISH
from ish.embedding_store import EMBEDDING_INPUT_PROFILE, embedding_profile_for_model
from ish.message import Message
from ish.model_store import make_model_bundle
from ish.training_manager import TrainingManager


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
            ish._cache.store_embedding(
                msg_hash,
                np.array([float(uid), float(sample_idx)]),
                EMBEDDING_INPUT_PROFILE,
            )

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
        embedding_profile=EMBEDDING_INPUT_PROFILE,
    )

    embeddings, labels = manager._collect_training_data(fake_imap, ["FolderA", "FolderB"])

    assert len(embeddings) == 4
    assert labels == ["FolderA", "FolderA", "FolderB", "FolderB"]
    assert all(float(emb[0]) not in {998.0, 999.0} for emb in embeddings)


def test_training_saves_classifier_with_embedding_profile_metadata(monkeypatch):
    profile = embedding_profile_for_model("text-embedding-3-small")
    classifier = object()
    dump_mock = mock.MagicMock()
    monkeypatch.setattr(joblib, "dump", dump_mock)

    manager = TrainingManager(
        imap_conn_provider=lambda: mock.MagicMock(),
        get_embeddings=lambda folder, uids: {},
        get_cache_embeddings=lambda folder: {},
        model_file="model.pkl",
        max_learn_messages=100,
        exit_event=Event(),
        embedding_profile=profile,
    )
    monkeypatch.setattr(
        manager,
        "_collect_training_data",
        lambda imap_conn, folders: (
            [np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0])],
            ["A", "A", "B", "B"],
        ),
    )
    monkeypatch.setattr(
        manager,
        "_split_training_data",
        lambda X, y: (X[:2], X[2:], y[:2], y[2:]),
    )
    monkeypatch.setattr(manager, "_train_classifier", lambda X_train, y_train: classifier)
    monkeypatch.setattr(manager, "_evaluate_classifier", lambda clf, X_test, y_test, folders: 0.5)

    assert manager.learn_folders(["A", "B"]) is classifier
    payload, model_file = dump_mock.call_args.args

    assert model_file == "model.pkl"
    assert payload == make_model_bundle(classifier, profile)
