from langchain_core.language_models import LLM

from app.tools.models.deepseek import DeepSeekChat
from app.tools.models.glm import GLMChat
from app.tools.models.openai_model import OpenAI
from app.tools.models.spark_model import SparkModel


def get_llms(llm_config: dict) -> OpenAI | SparkModel:
    """
    Returns a list of LLMs based on the provided configuration.

    Args:
        llm_config (dict): A dictionary containing the configuration for the LLMs.

    Returns:
        list: A list of LLMs.

    Raises:
        ValueError: If the specified LLM type is not supported.

    """
    llm_defaults = {
        "temperature": 0.2,
        "streaming": False
    }
    provider = llm_config["provider"]
    del llm_config["provider"]
    llm_params = {**llm_defaults, **llm_config}

    if provider == "openai":
        return OpenAI(llm_params)
    elif provider == "spark":
        return SparkModel(llm_params)
    elif provider == "deepseek":
        return DeepSeekChat(llm_params)
    elif provider == "glm":
        return GLMChat(llm_params)


def get_llm_by_name(llm_name: str,streaming:bool=False) :
    if llm_name == "gpt4o":
        return get_llms({
            "provider": "openai",
            "model": "gpt-4o-ca",
            "streaming": streaming
        })
    elif llm_name == "glm4v":
        return get_llms({
            "provider": "glm",
            "model": "glm-4v",
            "streaming": streaming
        })
