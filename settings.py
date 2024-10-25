import logging
import os

from os.path import exists, isdir

import yaml


class Settings(dict):
    """Settings for the application"""

    _config_directory: str

    @staticmethod
    def get_user_directory() -> str:
        directory: str
        if os.name == "nt":
            directory = os.path.expandvars("%USERPROFILE%")
        elif os.name == "posix":
            directory = os.path.expandvars("$HOME")
        elif os.name == "mac":
            directory = os.path.expandvars("$HOME")
        return directory

    @property
    def settings_file(self):
        ishfile = os.path.join(self.config_directory, "settings.yaml")
        return ishfile

    @property
    def config_directory(self) -> str:
        return self._config_directory

    @config_directory.setter
    def config_directory(self, value) -> str:
        self._config_directory = value

    @property
    def data_directory(self) -> str:
        return self["data_directory"]

    @property
    def source_folders(self) -> list:
        return self["source_folders"]

    @property
    def ignore_folders(self) -> list:
        return self["ignore_folders"]

    @property
    def destination_folders(self) -> list:
        return self["destination_folders"]

    @property
    def imap_host(self) -> str:
        return self["host"]

    @property
    def username(self) -> str:
        return self["username"]

    @property
    def password(self) -> str:
        return self["password"]

    @property
    def openai_api_key(self) -> str:
        return self["openai_api_key"]

    @property
    def openai_model(self) -> str:
        return self["openai_model"]

    def __set_directories(self):
        if self["data_directory"] == "":
            self["data_directory"] = os.path.join(self.config_directory, "data")

        if not isdir(self["data_directory"]):
            try:
                os.makedirs(self["data_directory"], exist_ok=True)
            except IOError:
                self.logger.error(
                    "Could not create data directory %s", self["data_directory"]
                )
                # TODO: Pass specific error

    def __read(self):
        with open(self.settings_file, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.update(data)

    def __write(self):
        with open(self.settings_file, "w", encoding="utf-8") as f:
            yaml.dump(dict(self), f)
            self.logger.info("Settings saved to %s", self.settings_file)

    def __init__(self, debug=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)

        self["source_folders"] = []
        self["destination_folders"] = []
        self["ignore_folders"] = []
        self["host"] = ""
        self["username"] = ""
        self["password"] = ""
        self["data_directory"] = ""
        self["openai_api_key"] = ""
        self["openai_model"] = "text-embedding-ada-002"

        self.config_directory = os.environ.get("ISH_CONFIG_PATH")
        if self.config_directory is None or self.config_directory == "":
            self.config_directory = os.path.join(self.get_user_directory(), ".ish")
        self.logger.debug("Setting config dir to %s", self.config_directory)

        if exists(self.settings_file):
            self.__read()
        self.__set_directories()
        self.logger.debug("Settings\n%s", self)
