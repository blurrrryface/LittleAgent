from abc import ABC, abstractmethod
from typing import Literal

from app.tools.models.select_llms import get_llms


class StructureNode(ABC):
    def __init__(self, llm_type: Literal["cheap", "fast", "cleaver"], **kwargs):
        self.llm = self.get_llm(llm_type, kwargs)

    def get_llm(self, llm_type, kwargs):
        if llm_type == "cheap":
            return get_llms({
                "provider": "deepseek",
                "model": "deepseek-chat",
                **kwargs
            })
        elif llm_type == "fast":
            return get_llms({
                "provider": "openai",
                "model": "gpt-4o-ca",
                **kwargs
            })
        elif llm_type == "cleaver":
            return get_llms({
                "provider": "openai",
                "model": "gpt-4-turbo-ca",
                **kwargs
            })
        else:
            raise ValueError("Invalid llm_type")

    @abstractmethod
    def get_agent(self):
        pass
