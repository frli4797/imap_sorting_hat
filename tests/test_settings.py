import os

from ish.settings import Settings


def make_settings(tmp_path, monkeypatch):
    cfg = tmp_path / ".ish"
    cfg.mkdir(exist_ok=True)
    monkeypatch.setenv("ISH_CONFIG_PATH", str(cfg))
    return Settings(debug=True)


def test_update_folder_settings_adds_missing_keys(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch)
    # Clear any potential keys
    if "source_folders" in s:
        del s["source_folders"]
    if "destination_folders" in s:
        del s["destination_folders"]
    if "ignore_folders" in s:
        del s["ignore_folders"]

    s.update_folder_settings([])
    assert "source_folders" in s and isinstance(s["source_folders"], list)
    assert "destination_folders" in s and isinstance(s["destination_folders"], list)
    assert "ignore_folders" in s and isinstance(s["ignore_folders"], list)


def test_update_data_settings_creates_data_dir(tmp_path, monkeypatch):
    cfg = tmp_path / ".ish"
    os.makedirs(cfg, exist_ok=True)

    s = make_settings(tmp_path, monkeypatch)
    s.config_directory = str(cfg)
    s["data_directory"] = str(cfg / "data-dir-test")
    # ensure cleanup if present
    if os.path.exists(s["data_directory"]):
        os.rmdir(s["data_directory"])

    s.update_data_settings()
    assert os.path.isdir(s["data_directory"])


def test_classification_threshold_defaults_are_present(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch)

    assert s.classification_probability_threshold == 0.55
    assert s.classification_runner_up_gap_threshold == 0.15


def test_classification_threshold_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("ISH_CLASSIFICATION_PROBABILITY_THRESHOLD", "0.8")
    monkeypatch.setenv("ISH_CLASSIFICATION_RUNNER_UP_GAP_THRESHOLD", "0.22")

    s = make_settings(tmp_path, monkeypatch)

    assert s.classification_probability_threshold == 0.8
    assert s.classification_runner_up_gap_threshold == 0.22
