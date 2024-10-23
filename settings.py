import logging
import os
from getpass import getpass
from os.path import exists, isdir, join
from typing import List

import yaml


class Settings(dict):
    """Settings for the application"""

    @staticmethod
    def get_user_directory():
        directory = ""
        if os.name == "nt":
            directory = os.path.expandvars("%USERPROFILE%")
        elif os.name == "posix":
            directory = os.path.expandvars("$HOME")
        elif os.name == "mac":
            directory = os.path.expandvars("$HOME")
        return directory

    @property
    def settings_file(self):
        ishd = os.environ.get("ISH_CONFIG_PATH")
        if ishd is None or ishd == "":
            ishd = os.path.join(self.get_user_directory(), ".ish")
            os.makedirs(ishd, exist_ok=True)
        ishfile = os.path.join(ishd, "settings.yaml")
        return ishfile

    def read(self):
        with open(self.settings_file, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.update(data)

    def write(self):
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

        if exists(self.settings_file):
            self.read()

    def update_data_settings(self):
        while not isdir(self["data_directory"]):
            if self["data_directory"] == "":
                self["data_directory"] = join(self.get_user_directory(), ".ish", "data")
            else:
                print(f'Invalid data directory: {self["data_directory"]}')

            self["data_directory"] = (
                input(f'Enter data directory[{self["data_directory"]}]: ').strip()
                or self["data_directory"]
            )
            try:
                os.makedirs(self["data_directory"], exist_ok=True)
            except IOError:
                pass

        self.write()

    def update_openai_settings(self):
        self["openai_api_key"] = input(
            "Enter OpenAI API key (see https://beta.openai.com)): "
        )
        self.write()

    def update_login_settings(self):
        self.update(
            {
                "host": input(f'Enter imap server [{self["host"]}]: ').strip()
                or self["host"],
                "username": input(f'Enter username [{self["username"]}]: ').strip()
                or self["username"],
                "password": getpass("Enter password: "),
            }
        )
        self.write()

    def update_folder_settings(self, folders: List[str]):
        source_folders = set(self["source_folders"])
        destination_folders = set(self["destination_folders"])
        ignore_folders = set(self["ignore_folders"])
        all_folders = source_folders | destination_folders | ignore_folders

        for folder in folders:
            if folder not in all_folders:
                opt = None
                while opt not in ["s", "d", "i"]:
                    opt = input(
                        f"""Folder {folder} is not configured. \
                        What do you want to do with it? [s]ource, [d]estination, [i]gnore: """
                    )

                if opt == "s":
                    source_folders.add(folder)
                elif opt == "d":
                    destination_folders.add(folder)
                elif opt == "i":
                    ignore_folders.add(folder)

        missing_folders = all_folders - set(folders)
        if missing_folders:
            for folder in missing_folders:
                self.logger.info(
                    "Folder %s is missing. It will be removed from the settings.",
                    folder,
                )

            source_folders -= missing_folders
            destination_folders -= missing_folders
            ignore_folders -= missing_folders

        self["source_folders"] = sorted(source_folders)
        self["destination_folders"] = sorted(destination_folders)
        self["ignore_folders"] = sorted(ignore_folders)

        self.write()
