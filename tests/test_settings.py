import os
from unittest import mock

import pytest

from settings import Settings


def test_update_folder_settings_adds_missing_keys(tmp_path, monkeypatch):
    s = Settings(debug=True)
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
    settings_file = cfg / "settings.yaml"

    s = Settings(debug=True)
    s.config_directory = str(cfg)
    s["data_directory"] = str(cfg / "data-dir-test")
    # ensure cleanup if present
    if os.path.exists(s["data_directory"]):
        os.rmdir(s["data_directory"])

    s.update_data_settings()
    assert os.path.isdir(s["data_directory"])