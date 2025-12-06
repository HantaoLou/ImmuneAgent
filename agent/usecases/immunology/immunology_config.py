from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from common.constants import (
    QWEN_BASE_URL,
    QWEN_MODEL_OLLAMA,
    QWEN_MODEL_VLLM,
    REASONING_MODEL,
)
from config.api_keys import APIKeys


def get_immunology_runnable_config(thread_id=None) -> RunnableConfig:
    config = {
        "configurable": {"thread_id": thread_id},
        "model_config": {
            "default_model": {
                "provider": "OpenAI",
                "model": "deepseek-chat",
                "params": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": APIKeys.DEEPSEEK_API_KEY_ALT,
                    "temperature": 0.2,
                    "extra_body": {"enable_thinking": False},
                },
            },
            "embedding_model": {
                "provider": "OpenAI",
                "model": "text-embedding-v4",
                "params": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": APIKeys.QWEN_API_KEY_ALT,
                    "temperature": 0.2,
                    "extra_body": {"enable_thinking": False},
                },
            },
            "summarize_model": {
                "provider": "OpenAI",
                "model": "qwen3-8b",
                "params": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": APIKeys.QWEN_API_KEY_ALT,
                    "temperature": 0.2,
                    "extra_body": {"enable_thinking": False},
                },
            },
            "reasoning_model": {
                "provider": "OpenAI",
                "model": "qwen3-32b",
                "params": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": APIKeys.QWEN_API_KEY_ALT,
                    "temperature": 0.2,
                    "extra_body": {"enable_thinking": False},
                },
            },
        },
    }
    return config


def get_immunology_model_config() -> RunnableConfig:
    """
    获取免疫学模型配置
    返回符合框架要求的 RunnableConfig 对象，包含所有必要的模型配置
    """
    return {
        "configurable": {
            "model_config": {
                "default_model": {
                    "provider": "OpenAI",
                    "model": "deepseek-chat",
                    "params": {
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key": APIKeys.DEEPSEEK_API_KEY_ALT,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "embedding_model": {
                    "provider": "OpenAI",
                    "model": "text-embedding-3-small",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY_ALT,
                    },
                },
                "reasoning_model": {
                    "provider": "OpenAI",
                    "model": "deepseek-chat",
                    "params": {
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key": APIKeys.DEEPSEEK_API_KEY_ALT,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
            }
        }
    }
