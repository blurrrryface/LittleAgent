from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

class DeepSeekChat(ChatOpenAI):
    def __init__(self, llm_config: Dict[str, Any]):
        DEFAULT_CONFIG = {
            "openai_api_key": os.getenv('DEEPSEEK_API_KEY'),
            "openai_api_base": "https://api.deepseek.com",
        }
        # Merge default config with user-provided config, giving priority to user values
        config = {**DEFAULT_CONFIG, **llm_config}


        super().__init__(**config)