from typing import Dict, Any

from langchain_openai import ChatOpenAI


class DeepSeekChat(ChatOpenAI):
    def __init__(self, llm_config: Dict[str, Any]):
        DEFAULT_CONFIG = {
            "openai_api_key": "sk-4216e0956a174a33bae15907a8a02440",
            "openai_api_base": "https://api.deepseek.com",
        }
        # Merge default config with user-provided config, giving priority to user values
        config = {**DEFAULT_CONFIG, **llm_config}


        super().__init__(**config)