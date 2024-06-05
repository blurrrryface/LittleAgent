"""
OpenAI Module
"""
from langchain_openai import ChatOpenAI
from typing import Dict, Any


class OpenAI(ChatOpenAI):
    """
    A wrapper for the ChatOpenAI class that provides default configuration
    and could be extended with additional methods if needed.

    Args:
        llm_config (Dict[str, Any]): Configuration parameters for the language model.
    """



    def __init__(self, llm_config: Dict[str, Any]):
        DEFAULT_CONFIG = {
            "openai_api_key": "sk-DJFI6WpvxYievGbVjwidiXSyoZzvfqqiCCDsyk4BsNqz6pww",
            "openai_api_base": "https://api.chatanywhere.com.cn",
        }
        # Merge default config with user-provided config, giving priority to user values
        config = {**DEFAULT_CONFIG, **llm_config}


        super().__init__(**config)
