import os
from functools import cache as fc
from typing import Literal, Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from pydantic import BaseModel

from config.config import ApplicationConfig
from usecases.immunity.common.constants import MODEL_TIER, OPENAI_API_KEY

# Set OpenAI API key from constants if not in environment
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ModelConfig(BaseModel):
    provider: Literal["Ollama", "OpenAI"] = "Ollama"
    model: str
    params: dict = {}


class ModelSetConfig(BaseModel):
    embedding_model: Optional[ModelConfig] = None
    default_model: Optional[ModelConfig] = None
    reasoning_model: Optional[ModelConfig] = None
    summarize_model: Optional[ModelConfig] = None
    deep_research_model: Optional[ModelConfig] = None
    hypothesis_model: Optional[ModelConfig] = None
    planning_model: Optional[ModelConfig] = None


def _get_model_set_config(config: Optional[RunnableConfig]) -> ModelSetConfig:
    # Handle None config or missing configurable key
    if config is None or "configurable" not in config:
        return ModelSetConfig()  # Return defaults
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


def get_embedding_model(config: Optional[RunnableConfig] = None):
    model_config = _get_model_set_config(config).embedding_model
    if model_config is None:
        # Provide default configuration if not specified
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    if model_config.provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_config.model, **model_config.params)
    elif model_config.provider == "Ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_config.model, **model_config.params)
    else:
        raise ValueError(f"Unknown provider for embeddings: {model_config.provider}")


def get_reasoning_model(config: Optional[RunnableConfig] = None):
    model_config = _get_model_set_config(config).reasoning_model
    if model_config is None:
        # Use upgraded model for better performance
        import os

        from langchain_openai import ChatOpenAI

        # Check for tier preference
        tier = os.environ.get("MODEL_TIER", MODEL_TIER)

        if tier == "TIER4":
            # Ultra-performance with O1-preview for maximum reasoning
            # WARNING: Very high cost ($15/1M input tokens)
            model = ChatOpenAI(
                model="o1-preview",
                temperature=1.0,  # O1 models use fixed temperature
                max_tokens=32768,  # O1 supports much longer outputs
            )
        elif tier == "TIER3":
            # Maximum performance with GPT-4o
            model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=8000)
        elif tier == "TIER2":
            # Balanced performance with gpt-4o
            model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=4000)
        else:  # TIER1
            # Cost-effective with gpt-4o-mini
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4000)

        return _wrap_model_for_structured_output(model)
    model = _get_model(model_config)
    return _wrap_model_for_structured_output(model)


def get_default_model(config: Optional[RunnableConfig] = None):
    model_config = _get_model_set_config(config).default_model
    if model_config is None:
        # Use upgraded model for better performance
        import os

        from langchain_openai import ChatOpenAI

        # Check for tier preference
        tier = os.environ.get("MODEL_TIER", MODEL_TIER)

        if tier == "TIER3":
            # Maximum performance with gpt-4o
            model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=8000)
        elif tier == "TIER2":
            # Balanced performance with gpt-4o
            model = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=4000)
        else:  # TIER1
            # Cost-effective with gpt-4o-mini
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=3000)

        return _wrap_model_for_structured_output(model)
    model = _get_model(model_config)
    return _wrap_model_for_structured_output(model)


def get_summarize_model(config: Optional[RunnableConfig] = None):
    model_config = _get_model_set_config(config).summarize_model
    if model_config is None:
        # Provide default configuration if not specified
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
        return _wrap_model_for_structured_output(model)
    model = _get_model(model_config)
    return _wrap_model_for_structured_output(model)


def get_hypothesis_model(config: Optional[RunnableConfig] = None):
    """Get specialized model for hypothesis generation.

    Hypothesis generation requires creative reasoning with structured output.
    Uses lower temperature for consistent scientific hypotheses.

    Args:
        config: Optional runtime configuration

    Returns:
        Wrapped model optimized for hypothesis generation
    """
    # Check if a custom hypothesis model is provided
    model_config = _get_model_set_config(config)
    # Try to use reasoning_model config if available, otherwise use defaults
    if hasattr(model_config, "hypothesis_model") and model_config.hypothesis_model:
        model = _get_model(model_config.hypothesis_model)
        return model

    import os

    from langchain_openai import ChatOpenAI

    # Check for tier preference
    tier = os.environ.get("MODEL_TIER", MODEL_TIER)

    if tier == "TIER4":
        # Ultra-performance with O1-preview for complex hypotheses
        model = ChatOpenAI(
            model="o1-preview",
            temperature=1.0,  # O1 models use fixed temperature
            max_tokens=16384,  # Longer outputs for detailed hypotheses
        )
    elif tier == "TIER3":
        # Maximum performance with GPT-4o for sophisticated hypotheses
        model = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=6000)
    elif tier == "TIER2":
        # Balanced performance with gpt-4o
        model = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=4000)
    else:  # TIER1
        # Cost-effective with gpt-4o-mini
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=3000)

    return _wrap_model_for_structured_output(model)


def get_planning_model(config: Optional[RunnableConfig] = None):
    """Get specialized model for experimental planning.

    Planning requires methodical reasoning with comprehensive outputs.
    Uses very low temperature for consistent, detailed plans.

    Args:
        config: Optional runtime configuration

    Returns:
        Wrapped model optimized for experimental planning
    """
    # Check if a custom planning model is provided
    model_config = _get_model_set_config(config)
    if hasattr(model_config, "planning_model") and model_config.planning_model:
        model = _get_model(model_config.planning_model)
        return model

    import os

    from langchain_openai import ChatOpenAI

    # Check for tier preference
    tier = os.environ.get("MODEL_TIER", MODEL_TIER)

    if tier == "TIER4":
        # Ultra-performance with O1-preview for complex planning
        model = ChatOpenAI(
            model="o1-preview",
            temperature=1.0,  # O1 models use fixed temperature
            max_tokens=32768,  # Maximum output for comprehensive plans
        )
    elif tier == "TIER3":
        # Maximum performance with GPT-4o for detailed planning
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=8000)
    elif tier == "TIER2":
        # Balanced performance with gpt-4o
        model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=6000)
    else:  # TIER1
        # Cost-effective with gpt-4o-mini
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=4000)

    return _wrap_model_for_structured_output(model)


def get_tool_selection_model(config: Optional[RunnableConfig] = None):
    """Get specialized model for tool and method selection.

    Tool selection requires analytical reasoning to match tools to tasks.
    Uses low temperature for consistent, accurate tool matching.

    Args:
        config: Optional runtime configuration

    Returns:
        Wrapped model optimized for tool selection
    """
    # Check if a custom tool selection model is provided
    model_config = _get_model_set_config(config)
    if (
        hasattr(model_config, "tool_selection_model")
        and model_config.tool_selection_model
    ):
        model = _get_model(model_config.tool_selection_model)
        return _wrap_model_for_structured_output(model)

    import os

    from langchain_openai import ChatOpenAI

    # Check for tier preference
    tier = os.environ.get("MODEL_TIER", MODEL_TIER)

    if tier == "TIER3" or tier == "TIER4":
        # High performance for accurate tool selection
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)
    elif tier == "TIER2":
        # Balanced performance
        model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=3000)
    else:  # TIER1
        # Cost-effective
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2500)

    return _wrap_model_for_structured_output(model)


def get_deep_research_model(config: Optional[RunnableConfig] = None):
    """Get specialized model for deep research analysis.

    Deep research requires thorough analysis with citations and evidence.
    Uses very low temperature for factual, evidence-based research.

    Args:
        config: Optional runtime configuration

    Returns:
        Wrapped model optimized for deep research
    """
    # Check if a custom deep research model is provided
    model_config = _get_model_set_config(config)
    if (
        hasattr(model_config, "deep_research_model")
        and model_config.deep_research_model
    ):
        model = _get_model(model_config.deep_research_model)
        return model

    import os

    from langchain_openai import ChatOpenAI

    # Check for tier preference
    tier = os.environ.get("MODEL_TIER", MODEL_TIER)

    if tier == "TIER4":
        # Ultra-performance with O1-preview for deep analysis
        model = ChatOpenAI(
            model="o1-preview",
            temperature=1.0,  # O1 models use fixed temperature
            max_tokens=32768,  # Maximum output for comprehensive research
        )
    elif tier == "TIER3":
        # Maximum performance with GPT-4o for thorough research
        model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=10000)
    elif tier == "TIER2":
        # Balanced performance with gpt-4o
        model = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=6000)
    else:  # TIER1
        # Cost-effective with gpt-4o-mini but still good for research
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=4000)

    return _wrap_model_for_structured_output(model)


class StructuredOutputWrapper:
    """Wrapper class to handle structured output method selection automatically"""

    def __init__(self, model):
        self._model = model

    def with_structured_output(self, schema, **kwargs):
        """Automatically select the appropriate method based on model type"""
        # Check if method is not specified
        if "method" not in kwargs:
            # Determine the appropriate method based on model name
            model_name = getattr(self._model, "model_name", "") or getattr(
                self._model, "model", ""
            )

            # GPT-3.5-turbo doesn't support json_schema, use function_calling instead
            if "gpt-3.5" in str(model_name).lower():
                kwargs["method"] = "function_calling"
            # For models that support json_schema (GPT-4, GPT-4o, etc.), use json_schema
            elif any(x in str(model_name).lower() for x in ["gpt-4", "o1"]):
                kwargs["method"] = "json_schema"
            # Default to function_calling for compatibility
            else:
                kwargs["method"] = "function_calling"

        return self._model.with_structured_output(schema, **kwargs)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped model"""
        return getattr(self._model, name)

    def invoke(self, *args, **kwargs):
        """Invoke the wrapped model"""
        return self._model.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        """Async invoke the wrapped model"""
        return await self._model.ainvoke(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Make the wrapper callable - redirects to invoke for compatibility"""
        return self.invoke(*args, **kwargs)


def _wrap_model_for_structured_output(model):
    """Wrap model to handle structured output method selection"""
    return StructuredOutputWrapper(model)


class MCPConfig(BaseModel):
    service_ids: list[str]


def _get_mcp_config(config: Optional[RunnableConfig]) -> MCPConfig:
    # Handle None config or missing configurable key
    if config is None or "configurable" not in config:
        return MCPConfig(service_ids=[])  # Return default empty list
    return MCPConfig.model_validate(config["configurable"].get("mcp_config", {}))


@fc
def get_all_mcp_servers() -> dict[str, Connection]:
    return ApplicationConfig.get_instance().get_mcp_servers()


async def get_mcp_client(c: Optional[RunnableConfig] = None) -> MultiServerMCPClient:
    server_ids = _get_mcp_config(c).service_ids
    return MultiServerMCPClient(
        connections={k: v for k, v in get_all_mcp_servers().items() if k in server_ids}
    )


if __name__ == "__main__":
    c = RunnableConfig(
        configurable={
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
            "mcp_config": {"service_ids": ["mcp1", "mcp2"]},
        }
    )
    print(get_default_model(c))
    print(get_embedding_model(c))
    import asyncio

    print(_get_mcp_config(c))
    print(asyncio.run(get_mcp_client(c)))
