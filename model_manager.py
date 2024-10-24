import logging
from abc import abstractmethod
from itertools import batched
from typing import override

import backoff
import ollama
from openai import APIError, OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse

from settings import Settings

base_logger = logging.getLogger("ModelManager")


class ModelManager:

    @abstractmethod
    def get_embeddings(self, texts: list, batch_size=20) -> list:
        pass

    def connect(self) -> bool:
        pass


class OllamaManager(ModelManager):
    def __init__(self, settings):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__client: ollama.Client = None
        self.__settings = settings

    @override
    def connect(self) -> bool:
        if not self.__settings["ai_api_host"]:
            return False

        # check if api key is valid
        try:
            self.__client = ollama.Client(host=self.__settings["ai_api_host"])
        except APIError as e:
            self.logger.error(e)
            return False
        return True

    @override
    def get_embeddings(self, texts: list[str], batch_size=20) -> list:
        result = []
        try:
            # result = self.__client.embed(model=self.__settings["ai_api_model"], input=texts)
            in_batches = list(batched(texts, batch_size))
            for batch in in_batches:
                self.logger.info("Batching %i emails", len(batch))
                e = self.__client.embed(
                    model=self.__settings["ai_api_model"], input=batch
                )
                result = result + e["embeddings"]
        except ollama.ResponseError as e:
            if e.status_code == 404:
                # self.__client.pull(self.__settings["ai_api_model"])
                # Retry after pulling the model.
                #result = self.__client.embed(
                #    model=self.__settings["ai_api_model"], input=texts
                #)
                raise e
            else:
                raise e
        except Exception as e:
            self.logger.error(e)

        return result


class OpenAIManager(ModelManager):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__client: OpenAI = None
        self._settings = Settings()

    @override
    def connect(self) -> bool:
        if not self.__settings["ai_api_key"]:
            return False

        # check if api key is valid
        try:
            self.__client = OpenAI(api_key=self.__settings["ai_api_key"])
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
    def get_embeddings(self, texts: list[str], batch_size=0) -> list:
        e: CreateEmbeddingResponse = self.__client.embeddings.create(
            input=texts, model=self.__settings["ai_model"]
        )
        return [emb_obj.embedding for emb_obj in e.data]
