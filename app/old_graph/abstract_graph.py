from abc import ABC, abstractmethod
from typing import Optional

from app.tools.models.model_tokens import models_tokens
from app.tools.models.openai_model import OpenAI


class AbstractGraph(ABC):
    def __init__(self, prompt: str, config: dict, source: Optional[str] = None):
        self.prompt = prompt
        self.config = config
        self.source = source
        self.llm_model = self._create_llm(config["llm"], chat=True)

        # Create the old_graph
        self.graph = self._create_graph()
        self.final_state = None
        self.execution_info = None

    def _create_llm(self, llm_config: dict, chat=False) -> object:
        llm_defaults = {
            "temperature": 0,
            "streaming": False
        }
        llm_params = {**llm_defaults, **llm_config}

        if "gpt-" in llm_params["model"]:
            try:
                self.model_token = models_tokens["openai"][llm_params["model"]]
            except KeyError as exc:
                raise KeyError("Model not supported") from exc
            return OpenAI(llm_params)

    def get_state(self, key=None) -> dict:
        """""
        Get the final state of the old_graph.

        Args:
            key (str, optional): The key of the final state to retrieve.

        Returns:
            dict: The final state of the old_graph.
        """

        if key is not None:
            return self.final_state[key]
        return self.final_state

    def get_execution_info(self):
        """
        Returns the execution information of the old_graph.

        Returns:
            dict: The execution information of the old_graph.
        """

        return self.execution_info

    @abstractmethod
    def _create_graph(self):
        """
        Abstract method to create a old_graph representation.
        """
        pass

    @abstractmethod
    def run(self) -> str:
        """
        Abstract method to execute the old_graph and return the result.
        """
        pass
