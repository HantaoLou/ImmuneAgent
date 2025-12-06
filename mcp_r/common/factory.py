from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel
from typing import Literal
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from functools import cache as fc
from config.config import ApplicationConfig

class ModelConfig(BaseModel):
    provider: Literal["Ollama", "OpenAI"] = "Ollama"
    model: str
    params: dict = {}

class ModelSetConfig(BaseModel):
    embedding_model: ModelConfig
    default_model: ModelConfig
    reasoning_model: ModelConfig

def _get_model_set_config(config: RunnableConfig) -> ModelSetConfig:
    model_config = config["configurable"].get("model_config", {})
    return ModelSetConfig.model_validate(model_config)

def _get_model(model_config: ModelConfig):
    if model_config.provider == "Ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_config.model, **model_config.params)
    elif model_config.provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_config.model, **model_config.params)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")

def get_embedding_model(config: RunnableConfig):
    return _get_model(_get_model_set_config(config).embedding_model)

def get_reasoning_model(config: RunnableConfig):
    return _get_model(_get_model_set_config(config).reasoning_model)

def get_default_model(config: RunnableConfig):
    return _get_model(_get_model_set_config(config).default_model)

class MCPConfig(BaseModel):
    service_ids: list[str]

def _get_mcp_config(config: RunnableConfig) -> MCPConfig:
    return MCPConfig.model_validate(config["configurable"].get("mcp_config", {}))

@fc
def get_all_mcp_servers() -> dict[str, Connection]:
    return ApplicationConfig.get_instance().get_mcp_servers()


async def get_mcp_client(c: RunnableConfig) -> MultiServerMCPClient:
    server_ids = _get_mcp_config(c).service_ids
    return MultiServerMCPClient(
        connections={
            k: v for k, v in get_all_mcp_servers().items() if k in server_ids
        }
    )

if __name__ == '__main__':
    c = RunnableConfig(configurable={
        "model_config": {
            "default_model": {
                "provider": "Ollama",
                "model": "gpt-3.5-turbo",
            },
            "embedding_model": {
                "provider": "Ollama",
                "model": "text-embedding-ada-002",
            },
            "reasoning_model": {
                "provider": "Ollama",
                "model": "gpt-4",
            },
        },
        "mcp_config": {
            "service_ids": ["mcp1", "mcp2"]
        }
        }
    )
    print(get_default_model(c))
    print(get_embedding_model(c))
    import asyncio
    print(_get_mcp_config(c))
    print(asyncio.run(get_mcp_client(c)))
