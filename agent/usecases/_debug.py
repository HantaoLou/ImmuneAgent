import uuid

from langchain_core.runnables import RunnableConfig


def get_debug_runnable_config(thread_id=None) -> RunnableConfig:
    """
    只允许在 __main__ 中引入。
    """
    return {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4()),
            "model_config": {
                "summarize_model": {
                    "provider": "Ollama",
                    "model": "qwen3:8b",
                },
                "reasoning_model": {
                    "provider": "Ollama",
                    "model": "qwen3:8b",
                },
            },
            "mcp_config": {"service_ids": []},
        }
    }
