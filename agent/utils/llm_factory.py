"""
LLM Factory Module - Provides unified LLM instance creation functionality

Categorized by model purpose, supports the following types:
- Reasoning models (reasoning): For logical reasoning, task classification, decision-making, etc.
- Bioinformatics models (bioinformatics): For bioinformatics-related tasks
- Advanced reasoning models: Specifically for complex reasoning tasks
- Code models: For code generation and analysis

Usage:
    from agent.utils.llm_factory import create_reasoning_llm, create_bioinformatics_llm, create_code_llm

    # Create reasoning model
    llm = create_reasoning_llm()

    # Create bioinformatics model
    llm = create_bioinformatics_llm()

    # Create code model
    llm = create_code_llm()
"""

from typing import Optional, Any, Dict, List, Tuple, Callable
from enum import Enum
import os
from datetime import datetime

from langchain_core.tools import BaseTool

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

if load_dotenv is not None and find_dotenv is not None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

try:
    from zhipuai import ZhipuAI as ZhipuAiClient
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ZhipuAiClient = None
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None

if not LLM_AVAILABLE:
    print(
        "Warning: langchain-related libraries not installed, LLM functionality will be unavailable"
    )

# 🔥 Import progress context
try:
    from utils.progress_context import (
        get_progress_callback,
        set_progress_callback,
        ProgressCallbackContext,
    )

    PROGRESS_CONTEXT_AVAILABLE = True
except ImportError:
    PROGRESS_CONTEXT_AVAILABLE = False
    set_progress_callback = None
    get_progress_callback = None
    ProgressCallbackContext = None
    print("Warning: progress_context not available, SSE streaming will be disabled")


# 🔥 Auto-detect progress callback from context
def _get_auto_progress_callback() -> Optional[Callable]:
    """
    Automatically get progress callback from context (ContextVar)

    This allows all LLM creation functions to automatically use SSE
    when a callback is set in the context, without requiring explicit passing.

    Returns:
        Progress callback function if available in context, None otherwise
    """
    if PROGRESS_CONTEXT_AVAILABLE and get_progress_callback is not None:
        return get_progress_callback()
    return None


def _bind_session_to_callback(
    callback: Callable, session_id: str, node_name: Optional[str] = None
) -> Callable:
    """
    Bind session_id and node_name to callback function

    This wraps a callback to automatically inject session_id and node_name
    into all callback invocations.

    Args:
        callback: Original callback function
        session_id: Session ID to bind
        node_name: Node name to bind

    Returns:
        Wrapped callback with session_id and node_name bound
    """
    _node = node_name or "unknown"

    def wrapped(
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if details is None:
            details = {}
        details["session_id"] = session_id
        details["node_name"] = _node
        details["timestamp"] = datetime.now().isoformat()
        return callback(event_type, message, details, **kwargs)

    return wrapped


# ===================== Model Purpose Enum =====================
class ModelPurpose(str, Enum):
    """Model purpose enumeration"""

    REASONING = "reasoning"  # Reasoning model: For logical reasoning, task classification, decision-making, etc.
    BIOINFORMATICS = (
        "bioinformatics"  # Bioinformatics model: For bioinformatics-related tasks
    )
    REASONING_ADVANCED = "reasoning_advanced"  # Advanced reasoning model: Specifically for complex reasoning tasks
    CODE = "code"  # Code model: For code generation and analysis


# ===================== Model Configuration Mapping =====================
MODEL_CONFIGS: Dict[ModelPurpose, List[Tuple[str, str, float]]] = {
    ModelPurpose.REASONING: [
        ("zhipu", "glm-4.5", 0.2),
        ("dashscope", "qwen-max", 0.2),  # Qwen Max - preferred
        ("dashscope", "qwen-turbo", 0.1),  # Qwen Turbo - preferred
    ],
    # Bioinformatics model: Prefer Qwen, then use models with good scientific literature understanding
    ModelPurpose.BIOINFORMATICS: [
        ("zhipu", "glm-4.5", 0.2),
        ("dashscope", "qwen-max", 0.2),  # Qwen Max - preferred
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.2),
    ],
    # Advanced reasoning model: Prefer Qwen for complex reasoning tasks
    ModelPurpose.REASONING_ADVANCED: [
        ("zhipu", "glm-4.5", 0.1),
        ("dashscope", "qwen-max", 0.1),
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.1),  # Zhipu AI
    ],
    # Code model: Prefer Qwen, then use models with strong code generation capabilities
    ModelPurpose.CODE: [
        ("zhipu", "glm-4.5", 0.1),
        ("dashscope", "qwen-max", 0.1),  # Qwen Max - preferred
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.1),
    ],
}


# ===================== Low-level Provider Creation Functions =====================
def _mask_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def _create_openai_llm(
    model: str,
    temperature: float = 0.1,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """Internal function: Create OpenAI GPT LLM instance with optional SSE callback"""
    if not LLM_AVAILABLE or ChatOpenAI is None:
        return None

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    # Get timeout setting (default 120 seconds)
    timeout = int(os.getenv("LLM_TIMEOUT", "120"))
    max_retries = 2
    masked_key = _mask_api_key(api_key)

    # Auto-detect callback if not provided
    if progress_callback is None:
        progress_callback = _get_auto_progress_callback()

    streaming_enabled = bool(progress_callback)

    print(
        "Creating OpenAI-compatible LLM instance: "
        f"model={model}, temperature={temperature}, base_url={base_url}, "
        f"timeout={timeout}, max_retries={max_retries}, api_key={masked_key}, "
        f"streaming={streaming_enabled}"
    )

    try:
        # Prepare callbacks if progress_callback is available
        callbacks = None
        if progress_callback:
            try:
                from agent.utils.sse_callback_handler import SSECallbackHandler

                callbacks = [SSECallbackHandler(progress_callback=progress_callback)]
            except ImportError:
                try:
                    from utils.sse_callback_handler import SSECallbackHandler

                    callbacks = [
                        SSECallbackHandler(progress_callback=progress_callback)
                    ]
                except ImportError:
                    print(
                        "Warning: SSECallbackHandler not available, streaming disabled"
                    )
                    streaming_enabled = False

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            streaming=streaming_enabled,
            callbacks=callbacks,
        )
        return llm
    except Exception as e:
        print(f"Error: Failed to create OpenAI LLM ({model}): {e}")
        return None


def _create_zhipu_llm(
    model: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """Internal function: Create Zhipu AI LLM instance with optional SSE callback"""
    if not LLM_AVAILABLE or ZhipuAiClient is None:
        return None

    if api_key is None:
        api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("Warning: ZHIPU_API_KEY or ZHIPUAI_API_KEY not found")
        return None

    if not os.getenv("ZHIPUAI_API_KEY"):
        os.environ["ZHIPUAI_API_KEY"] = api_key

    # Auto-detect callback from context if not provided
    if progress_callback is None:
        progress_callback = _get_auto_progress_callback()
        if progress_callback:
            print(
                f"[LLM Factory] Auto-detected progress_callback from context for Zhipu"
            )

    try:
        from agent.utils.sse_callback_handler import (
            create_llm_with_sse,
            attach_sse_to_llm,
        )

        llm = create_llm_with_sse(
            model=model,
            temperature=temperature,
            progress_callback=progress_callback,
            streaming=bool(progress_callback),
        )

        return llm
    except ImportError:
        try:
            from utils.sse_callback_handler import (
                create_llm_with_sse,
                attach_sse_to_llm,
            )

            llm = create_llm_with_sse(
                model=model,
                temperature=temperature,
                progress_callback=progress_callback,
                streaming=bool(progress_callback),
            )
            return llm
        except Exception as e:
            print(f"Error: Failed to create Zhipu AI LLM ({model}): {e}")
            return None
    except Exception as e:
        print(f"Error: Failed to create Zhipu AI LLM ({model}): {e}")
        return None


def _create_llm_with_tools(
    model: str = "qwen-max",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature: float = 0.1,
    tools: List[BaseTool] = None,
) -> Optional[Any]:
    """Internal function: Create LLM instance with tools"""
    if not LLM_AVAILABLE or ChatOpenAI is None:
        return None
    return ChatOpenAI(
        model=model, base_url=base_url, temperature=temperature
    ).bind_tools(tools)


# ===================== Create LLM by Purpose =====================
def create_reasoning_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create reasoning model instance

    For logical reasoning, task classification, decision-making, etc. Prefer models with good reasoning performance.

    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
        progress_callback: Optional progress callback for SSE streaming (auto-detected from context if not provided)

    Returns:
        LLM instance, returns None if creation fails

    Examples:
        >>> llm = create_reasoning_llm()
        >>> llm = create_reasoning_llm(temperature=0.2)
        >>> llm = create_reasoning_llm(custom_model="anthropic:claude-3-opus-20240229")
    """
    return _create_llm_by_purpose(
        ModelPurpose.REASONING,
        temperature=temperature,
        custom_model=custom_model,
        progress_callback=progress_callback,
    )


def create_bioinformatics_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create bioinformatics model instance

    For bioinformatics-related tasks such as literature analysis, data interpretation, etc. Prefer models with good scientific literature understanding.

    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
        progress_callback: Optional progress callback for SSE streaming (auto-detected from context if not provided)

    Returns:
        LLM instance, returns None if creation fails

    Examples:
        >>> llm = create_bioinformatics_llm()
        >>> llm = create_bioinformatics_llm(temperature=0.3)
    """
    return _create_llm_by_purpose(
        ModelPurpose.BIOINFORMATICS,
        temperature=temperature,
        custom_model=custom_model,
        progress_callback=progress_callback,
    )


def create_reasoning_advanced_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create advanced reasoning model instance

    Specifically for complex reasoning tasks, prefer models with the strongest reasoning capabilities.

    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
        progress_callback: Optional progress callback for SSE streaming (auto-detected from context if not provided)

    Returns:
        LLM instance, returns None if creation fails

    Examples:
        >>> llm = create_reasoning_advanced_llm()
    """
    return _create_llm_by_purpose(
        ModelPurpose.REASONING_ADVANCED,
        temperature=temperature,
        custom_model=custom_model,
        progress_callback=progress_callback,
    )


def create_code_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create code model instance

    For code generation, code analysis, code review, etc. Prefer models with strong code generation capabilities.

    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "openai:gpt-4o")
        progress_callback: Optional progress callback for SSE streaming (auto-detected from context if not provided)

    Returns:
        LLM instance, returns None if creation fails

    Examples:
        >>> llm = create_code_llm()
        >>> llm = create_code_llm(temperature=0.2)
    """
    return _create_llm_by_purpose(
        ModelPurpose.CODE,
        temperature=temperature,
        custom_model=custom_model,
        progress_callback=progress_callback,
    )


# ===================== Core Creation Function =====================
def _create_llm_by_purpose(
    purpose: ModelPurpose,
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create LLM instance by purpose (core function)

    Args:
        purpose: Model purpose
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model)
        progress_callback: Optional progress callback for SSE streaming

    Returns:
        LLM instance, returns None if creation fails
    """
    if not LLM_AVAILABLE:
        print(
            "Warning: LLM-related libraries not installed, cannot create LLM instance"
        )
        return None

    # Auto-detect callback if not provided
    if progress_callback is None:
        progress_callback = _get_auto_progress_callback()

    # If custom model is specified, use it first
    if custom_model:
        try:
            provider, model = custom_model.split(":", 1)
            temp = temperature if temperature is not None else 0.1

            if provider == "openai":
                llm = _create_openai_llm(
                    model,
                    temp,
                    base_url="https://xiaoai.plus/v1",
                    progress_callback=progress_callback,
                )
            elif provider == "dashscope":
                dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv(
                    "QIANFAN_API_KEY"
                )
                if dashscope_key:
                    llm = _create_openai_llm(
                        model,
                        temp,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        api_key=dashscope_key,
                        progress_callback=progress_callback,
                    )
                else:
                    llm = None
            elif provider == "zhipu":
                zhipu_key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
                if zhipu_key:
                    llm = _create_zhipu_llm(
                        model,
                        temp,
                        api_key=zhipu_key,
                        progress_callback=progress_callback,
                    )
                else:
                    llm = None
            else:
                print(f"Warning: Unknown provider '{provider}', ignoring custom model")
                llm = None

            if llm is not None:
                print(f"Successfully created custom {provider} LLM instance ({model})")
                return llm
        except ValueError:
            print(
                f"Warning: Custom model format error '{custom_model}', should be 'provider:model'"
            )

    # Try to create model by priority
    configs = MODEL_CONFIGS.get(purpose, [])
    if not configs:
        print(f"Warning: No model configuration found for purpose '{purpose.value}'")
        return None

    failed_attempts = []
    for provider, model, default_temp in configs:
        temp = temperature if temperature is not None else default_temp

        if provider == "openai":
            llm = _create_openai_llm(
                model,
                temp,
                base_url="https://xiaoai.plus/v1",
                progress_callback=progress_callback,
            )
            if llm is None:
                failed_attempts.append(
                    f"{provider}:{model} (missing OPENAI_API_KEY or creation failed)"
                )
        elif provider == "dashscope":
            dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv(
                "QIANFAN_API_KEY"
            )
            if dashscope_key:
                llm = _create_openai_llm(
                    model,
                    temp,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key=dashscope_key,
                    progress_callback=progress_callback,
                )
                if llm is None:
                    failed_attempts.append(f"{provider}:{model} (creation failed)")
            else:
                llm = None
                failed_attempts.append(
                    f"{provider}:{model}:{dashscope_key} (missing DASHSCOPE_API_KEY or QIANFAN_API_KEY)"
                )
        elif provider == "zhipu":
            zhipu_key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
            if zhipu_key:
                llm = _create_zhipu_llm(
                    model, temp, api_key=zhipu_key, progress_callback=progress_callback
                )
                if llm is None:
                    failed_attempts.append(
                        f"{provider}:{model} (creation failed or adapter unavailable)"
                    )
            else:
                llm = None
                failed_attempts.append(f"{provider}:{model} (missing ZHIPU_API_KEY)")
        else:
            failed_attempts.append(f"{provider}:{model} (unknown provider)")
            continue

        if llm is not None:
            print(
                f"Successfully created {provider} {purpose.value} model instance ({model})"
            )
            return llm

    # Provide detailed error information
    error_msg = f"Warning: Cannot create {purpose.value} model, all configured models are unavailable"
    if failed_attempts:
        error_msg += f"\n  Failed attempts: {', '.join(failed_attempts)}"
    error_msg += f"\n  Please check your API keys or model availability"
    print(error_msg)
    return None


# ===================== Generic Creation Function (Backward Compatible) =====================
def create_llm(
    purpose: str = "reasoning",
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Generic LLM creation function (backward compatible)

    Args:
        purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
        temperature: Temperature parameter
        custom_model: Custom model name
        progress_callback: Optional progress callback for SSE streaming (auto-detected from context if not provided)

    Returns:
        LLM instance, returns None if creation fails

    Examples:
        >>> llm = create_llm("reasoning")
        >>> llm = create_llm("bioinformatics", temperature=0.2)
    """
    try:
        purpose_enum = ModelPurpose(purpose)
        return _create_llm_by_purpose(
            purpose_enum, temperature, custom_model, progress_callback
        )
    except ValueError:
        print(
            f"Warning: Unknown model purpose '{purpose}', using default reasoning model"
        )
        return _create_llm_by_purpose(
            ModelPurpose.REASONING, temperature, custom_model, progress_callback
        )


# ===================== Helper Functions =====================
def is_llm_available() -> bool:
    """
    Check if LLM functionality is available

    Returns:
        True if LLM-related libraries are installed, False otherwise
    """
    return LLM_AVAILABLE


def get_available_providers() -> List[str]:
    """
    Get list of available LLM providers (based on environment variables)

    Returns:
        List of available providers, e.g., ["openai", "dashscope"]
    """
    providers = []

    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")

    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY"):
        providers.append("dashscope")

    if os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY"):
        providers.append("zhipu")

    return providers


def get_model_configs(purpose: Optional[ModelPurpose] = None) -> Dict:
    """
    Get model configuration information

    Args:
        purpose: Model purpose, if None returns all configurations

    Returns:
        Model configuration dictionary
    """
    if purpose is None:
        return {
            purpose.value: [
                {"provider": provider, "model": model, "temperature": temp}
                for provider, model, temp in configs
            ]
            for purpose, configs in MODEL_CONFIGS.items()
        }
    else:
        configs = MODEL_CONFIGS.get(purpose, [])
        return {
            purpose.value: [
                {"provider": provider, "model": model, "temperature": temp}
                for provider, model, temp in configs
            ]
        }


def get_current_llm_config() -> Dict[str, str]:
    """
    Get current LLM configuration

    Returns the currently available LLM configuration info including model name and provider

    Returns:
        Dict containing 'model' and 'provider' fields
    """
    if os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY"):
        return {"model": "glm-4.5", "provider": "zhipu"}
    elif os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY"):
        return {"model": "qwen-max", "provider": "dashscope"}
    elif os.getenv("OPENAI_API_KEY"):
        return {"model": "gpt-4", "provider": "openai"}
    else:
        return {"model": "", "provider": ""}


def create_llm_with_callback(
    purpose: str = "reasoning",
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Any]:
    """
    Create LLM instance with progress callback for thinking capture

    This is a convenience function that creates an LLM with thinking capture support.
    The progress_callback will be called during LLM invocation to report thinking process.

    Args:
        purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
        temperature: Temperature parameter
        custom_model: Custom model name
        progress_callback: Callback function for reporting thinking process

    Returns:
        LLM instance with thinking capture, returns None if creation fails

    Examples:
        >>> from agent.state import GlobalState
        >>> state = GlobalState(user_input="test", progress_callback=my_callback)
        >>> llm = create_llm_with_callback("reasoning", progress_callback=state.progress_callback)
        >>> response = llm.invoke([HumanMessage(content="Hello")])
    """
    try:
        purpose_enum = ModelPurpose(purpose)
    except ValueError:
        print(
            f"Warning: Unknown model purpose '{purpose}', using default reasoning model"
        )
        purpose_enum = ModelPurpose.REASONING

    # Get model configuration
    configs = MODEL_CONFIGS.get(purpose_enum, [])
    if not configs:
        print(
            f"Warning: No model configuration found for purpose '{purpose_enum.value}'"
        )
        return None

    # Try to create model with progress_callback
    for provider, model, default_temp in configs:
        temp = temperature if temperature is not None else default_temp

        if provider == "zhipu":
            zhipu_key = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
            if zhipu_key:
                llm = _create_zhipu_llm(
                    model, temp, api_key=zhipu_key, progress_callback=progress_callback
                )
                if llm is not None:
                    print(
                        f"Successfully created {provider} {purpose_enum.value} model with callback ({model})"
                    )
                    return llm
        # 其他provider暂时不支持progress_callback，使用标准创建方式
        elif provider == "dashscope":
            dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv(
                "QIANFAN_API_KEY"
            )
            if dashscope_key:
                llm = _create_openai_llm(
                    model,
                    temp,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    api_key=dashscope_key,
                )
                if llm is not None:
                    print(
                        f"Successfully created {provider} {purpose_enum.value} model ({model}) without callback"
                    )
                    return llm
        elif provider == "openai":
            llm = _create_openai_llm(model, temp, base_url="https://xiaoai.plus/v1")
            if llm is not None:
                print(
                    f"Successfully created {provider} {purpose_enum.value} model ({model}) without callback"
                )
                return llm

    # Fallback to standard creation without callback
    print(
        f"Warning: Could not create LLM with callback, falling back to standard creation"
    )
    return _create_llm_by_purpose(purpose_enum, temperature, custom_model)


def create_llm_with_thinking(
    purpose: str = "reasoning",
    progress_callback: Optional[Callable] = None,
    session_id: Optional[str] = None,
    node_name: Optional[str] = None,
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None,
    **kwargs,
) -> Optional[Any]:
    """
    Create LLM instance with thinking capture and automatic callback binding

    This is the recommended way to create LLM instances. It automatically:
    1. Detects progress callback from context if not provided
    2. Binds session_id and node_name to the callback
    3. Enables streaming for thinking capture
    4. Reports LLM thinking to the frontend

    Args:
        purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
        progress_callback: Optional progress callback (auto-detected from context if not provided)
        session_id: Optional session ID for multi-session isolation
        node_name: Optional node name for tracking
        temperature: Temperature parameter
        custom_model: Custom model name (format: provider:model)
        **kwargs: Additional arguments passed to LLM creation

    Returns:
        LLM instance with thinking capture, returns None if creation fails

    Examples:
        >>> # Simple usage with auto-detection
        >>> llm = create_llm_with_thinking("reasoning")

        >>> # With explicit session tracking
        >>> llm = create_llm_with_thinking(
        ...     purpose="reasoning",
        ...     session_id="session_123",
        ...     node_name="supervisor"
        ... )

        >>> # From GlobalState
        >>> llm = state.get_llm(purpose="reasoning", node_name="general_qa")
    """
    # 1. Auto-detect callback if not provided
    if progress_callback is None:
        progress_callback = _get_auto_progress_callback()

    # 2. Bind session_id to callback if both are available
    if progress_callback and session_id:
        progress_callback = _bind_session_to_callback(
            progress_callback, session_id, node_name
        )

    # 3. Create LLM using standard factory
    try:
        purpose_enum = ModelPurpose(purpose)
    except ValueError:
        print(
            f"Warning: Unknown model purpose '{purpose}', using default reasoning model"
        )
        purpose_enum = ModelPurpose.REASONING

    llm = _create_llm_by_purpose(
        purpose_enum,
        temperature=temperature,
        custom_model=custom_model,
        progress_callback=progress_callback,
    )

    # 4. Ensure streaming is enabled
    if llm and hasattr(llm, "streaming"):
        llm.streaming = True

    return llm
