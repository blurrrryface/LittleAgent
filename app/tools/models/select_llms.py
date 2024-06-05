from langchain_core.language_models import LLM

from app.tools.models.deepseek import DeepSeekChat
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
        "temperature": 0.6,
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