from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from common.constants import QWEN_BASE_URL, QWEN_MODEL_OLLAMA, REASONING_MODEL


def get_antibody_runnable_config(thread_id=None) -> RunnableConfig:
    config = {
        "configurable": {"thread_id": thread_id},
        "model_config": {
            "default_model": {
                "provider": "Ollama",
                "model": REASONING_MODEL,
                "params": {"temperature": 0.2},
            },
            "embedding_model": {
                "provider": "Ollama",
                "model": "text-embedding-ada-002",
            },
            "summarize_model": {
                "provider": "Ollama",
                "model": QWEN_MODEL_OLLAMA,
            },
            "reasoning_model": {
                "provider": "Ollama",
                "model": QWEN_MODEL_OLLAMA,
                "params": {
                    # "base_url": QWEN_BASE_URL,
                    # "api_key": SecretStr("dummy"),
                    # "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    "temperature": 0.2
                },
            },
        },
    }
    return config
