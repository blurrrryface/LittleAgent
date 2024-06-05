"""
Models token
"""

models_tokens = {
    "openai": {
        "gpt-3.5-turbo-ca":4096,
        "gpt-4o-ca":128000,
        "gpt-3.5-turbo-0125":16385,
        # -----------------------
        #
        # "gpt-3.5-turbo": 4096,
        # "gpt-3.5-turbo-1106": 16385,
        # "gpt-3.5-turbo-instruct": 4096,
        # "gpt-4-0125-preview": 128000,
        # "gpt-4-turbo-preview": 128000,
        # "gpt-4-turbo": 128000,
        # "gpt-4-turbo-2024-04-09": 128000,
        # "gpt-4-1106-preview": 128000,
        # "gpt-4-vision-preview": 128000,
        # "gpt-4": 8192,
        # "gpt-4-0613": 8192,
        # "gpt-4-32k": 32768,
        # "gpt-4-32k-0613": 32768,
        # "gpt-4o": 128000,
        # "gpt-3.5-turbo-16k-0613":16385,
    },
    "spark": {
        "Spark3.5 Max": 32768,
        "Spark Pro": 32768,
        "Spark Lite": 32768
    },
    "deepseek": {
        "deepseek-chat": 32768,

    }

}