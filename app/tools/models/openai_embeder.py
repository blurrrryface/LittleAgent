from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

class OpenaiEmbeder(OpenAIEmbeddings):

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        if llm_config is None:
            llm_config = {
                "model": "text-embedding-3-small"
            }
        DEFAULT_CONFIG = {
            "openai_api_key": os.getenv('OPENAI_API_KEY'),
            "openai_api_base": "https://api.chatanywhere.com.cn/v1",
        }
        # Merge default config with user-provided config, giving priority to user values
        config = {**DEFAULT_CONFIG, **llm_config}

        super().__init__(**config)


if __name__ == '__main__':
    embed = OpenaiEmbeder()
    response = embed.embed_query("你是傻逼")
    print(len(response))
    print(response)
