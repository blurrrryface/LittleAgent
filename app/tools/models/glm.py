from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()


class GLMChat(ChatOpenAI):
    def __init__(self, llm_config: Dict[str, Any]):
        DEFAULT_CONFIG = {
            "openai_api_key": os.getenv('GLM_API_KEY'),
            "openai_api_base": "https://open.bigmodel.cn/api/paas/v4/",
        }
        # Merge default config with user-provided config, giving priority to user values
        config = {**DEFAULT_CONFIG, **llm_config}

        super().__init__(**config)
