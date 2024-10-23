import logging
from abc import abstractmethod
from typing import override

import backoff
import ollama
from openai import APIError, OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse

from settings import Settings

base_logger = logging.getLogger("ModelManager")


class ModelManager:

    @abstractmethod
    def get_embeddings(self, texts: list) -> list:
        pass

    def connect(self) -> bool:
        pass


class OllamaManager(ModelManager):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__client: ollama.Client = None
        self._settings = Settings()

    @override
    def connect(self) -> bool:
        if not self.__settings["api_host"]:
            return False

        # check if api key is valid
        try:
            self.__client = ollama.Client(host=self.__settings["api_host"])
        except APIError as e:
            self.logger.error(e)
            return False
        return True

    @override
    def get_embeddings(self, texts: list[str]) -> list:
        result = None
        try:
            result = self.__client.embed(
                model=self.__settings["api_model"], input=texts
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                self.__client.pull(self.__settings["api_model"])
                # Retry after pulling the model.
                result = self.__client.embed(
                    model=self.__settings["api_model"], input=texts
                )
            else:
                raise e

        return result["embeddings"]


class OpenAIManager(ModelManager):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__client: OpenAI = None
        self._settings = Settings()

    @override
    def connect(self) -> bool:
        if not self.__settings["api_key"]:
            return False

        # check if api key is valid
        try:
            self.__client = OpenAI(api_key=self.__settings["api_key"])
        except APIError as e:
            self.logger.error(e)
            return False
        return True

    @backoff.on_exception(
        backoff.expo,
        RateLimitError,
        on_backoff=lambda details: base_logger.warning(
            "Backing off %0.1f seconds after %i tries",
            details["wait"],
            details["tries"],
        ),
    )
    @override
    def get_embeddings(self, texts: list[str]) -> list:
        e: CreateEmbeddingResponse = self.__client.embeddings.create(
            input=texts, model=self.__settings["openai_model"]
        )
        return [emb_obj.embedding for emb_obj in e.data]
