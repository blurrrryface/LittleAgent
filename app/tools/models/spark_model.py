from typing import Dict, Any

from langchain_community.llms import SparkLLM


class SparkModel(SparkLLM):
    def __init__(self, llm_config: Dict[str, Any]):
        # client: Any = None  #: :meta private:
        # spark_app_id: Optional[str] = None
        # spark_api_key: Optional[str] = None
        # spark_api_secret: Optional[str] = None
        # spark_api_url: Optional[str] = None
        # spark_llm_domain: Optional[str] = None
        # spark_user_id: str = "lc_user"
        # streaming: bool = False
        # request_timeout: int = 30
        # temperature: float = 0.5
        # top_k: int = 4
        # model_kwargs: Dict[str, Any] = Field(default_factory=dict)
        model = llm_config.get("model", "Spark3.5 Max")
        model_url = {
            # 最强大的星火大模型版本，效果最优
            # 支持联网搜索、天气、日期等多个内置插件
            # 核心能力全面升级，各场景应用效果普遍提升
            # 支持System角色人设与FunctionCall函数调用
            "Spark3.5 Max": {"base_url": "wss://spark-api.xf-yun.com/v3.5/chat",
                             "spark_llm_domain": "generalv3.5"},
            # 专业级大语言模型，兼顾模型效果与性能
            # 数学、代码、医疗、教育等场景专项优化
            # 支持联网搜索、天气、日期等多个内置插件
            # 覆盖大部分知识问答、语言理解、文本创作等多个场景
            "Spark Pro": {"base_url": "wss://spark-api.xf-yun.com/v3.1/chat",
                          "spark_llm_domain": "generalv3.1"},

            # 轻量级大语言模型，低延迟，全免费
            # 支持在线联网搜索功能
            # 响应快速、便捷，全面免费开放
            # 适用于低算力推理与模型精调等定制化场景
            "Spark Lite": {"base_url": "wss://spark-api.xf-yun.com/v1.1/chat",
                           "spark_llm_domain": "general"},
        }
        DEFAULT_CONFIG = {
            "spark_app_id": "9f9d5017",
            "spark_api_key": "6087921c15527dfd84aa19b7a2482b84",
            "spark_api_secret": "MTA3ZDkyYmZkNjc0OTYwMWE1MDljMjZj",
            "spark_api_url": model_url[model]["base_url"],
            "spark_llm_domain": model_url[model]["spark_llm_domain"],


        }
        config = {**DEFAULT_CONFIG, **llm_config}

        super().__init__(**config)
