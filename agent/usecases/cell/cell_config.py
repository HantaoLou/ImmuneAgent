from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from common.constants import (
    QWEN_BASE_URL,
    QWEN_MODEL_OLLAMA,
    QWEN_MODEL_VLLM,
    REASONING_MODEL,
)
from config.api_keys import APIKeys


def get_cell_runnable_config(thread_id=None, work_directory=None) -> RunnableConfig:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "work_directory": work_directory,
            "mcp_config": {"service_ids": ["metabcr", "bcell_analysis"]},
            "model_config": {
                "default_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
                "embedding_model": {
                    "provider": "OpenAI",
                    "model": "text-embedding-3-small",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                    },
                },
                "summarize_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
                "reasoning_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
                "deep_research_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
                "hypothesis_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
                "planning_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                        "temperature": 0.2,
                        # "extra_body": {"enable_thinking": False}
                    },
                },
            },
        }
    }
    return config
