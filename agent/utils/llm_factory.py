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

from typing import Optional, Any, Dict, List, Tuple
from enum import Enum
import os

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
    from zai import ZhipuAiClient
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
    print("Warning: langchain-related libraries not installed, LLM functionality will be unavailable")


# ===================== Model Purpose Enum =====================
class ModelPurpose(str, Enum):
    """Model purpose enumeration"""
    REASONING = "reasoning"  # Reasoning model: For logical reasoning, task classification, decision-making, etc.
    BIOINFORMATICS = "bioinformatics"  # Bioinformatics model: For bioinformatics-related tasks
    REASONING_ADVANCED = "reasoning_advanced"  # Advanced reasoning model: Specifically for complex reasoning tasks
    CODE = "code"  # Code model: For code generation and analysis


# ===================== Model Configuration Mapping =====================
MODEL_CONFIGS: Dict[ModelPurpose, List[Tuple[str, str, float]]] = {
    ModelPurpose.REASONING: [
        ("dashscope", "qwen-turbo", 0.1),  # Qwen Turbo - preferred
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.1),
    ],
    
    # Bioinformatics model: Prefer Qwen, then use models with good scientific literature understanding
    ModelPurpose.BIOINFORMATICS: [
        ("dashscope", "qwen-max", 0.2),  # Qwen Max - preferred
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.2),
    ],
    
    # Advanced reasoning model: Prefer Qwen for complex reasoning tasks
    ModelPurpose.REASONING_ADVANCED: [
        ("dashscope", "qwen-max", 0.1),
        ("zhipu", "glm-4.5-air:1131206110::21rbvay4", 0.1),  # Zhipu AI
    ],
    
    # Code model: Prefer Qwen, then use models with strong code generation capabilities
    ModelPurpose.CODE: [
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
    api_key: Optional[str] = None
) -> Optional[Any]:
    """Internal function: Create OpenAI GPT LLM instance"""
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
    print(
        "Creating OpenAI-compatible LLM instance: "
        f"model={model}, temperature={temperature}, base_url={base_url}, "
        f"timeout={timeout}, max_retries={max_retries}, api_key={masked_key}"
    )
    
    try:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,  # Add timeout setting
            max_retries=max_retries  # Maximum retry count
        )
    except Exception as e:
        print(f"Error: Failed to create OpenAI LLM ({model}): {e}")
        return None

def _create_zhipu_llm(
    model: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None
) -> Optional[Any]:
    """Internal function: Create Zhipu AI LLM instance (using adapter)"""
    if not LLM_AVAILABLE or ZhipuAiClient is None:
        return None
    
    if api_key is None:
        api_key = os.getenv("ZHIPU_API_KEY")
    
    if not api_key:
        return None
    
    print(f"Creating Zhipu AI LLM instance: {model}, {temperature}")
    try:
        # Use adapter to create instance compatible with OpenAI interface
        from agent.utils.zhipu_adapter import ZhipuAIAdapter
        return ZhipuAIAdapter(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    except ImportError as e:
        print(f"Error: Failed to import ZhipuAI adapter: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to create Zhipu AI LLM ({model}): {e}")
        return None

# ===================== Create LLM by Purpose =====================
def create_reasoning_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Create reasoning model instance
    
    For logical reasoning, task classification, decision-making, etc. Prefer models with good reasoning performance.
    
    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
    
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
        custom_model=custom_model
    )


def create_bioinformatics_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Create bioinformatics model instance
    
    For bioinformatics-related tasks such as literature analysis, data interpretation, etc. Prefer models with good scientific literature understanding.
    
    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
    
    Returns:
        LLM instance, returns None if creation fails
    
    Examples:
        >>> llm = create_bioinformatics_llm()
        >>> llm = create_bioinformatics_llm(temperature=0.3)
    """
    return _create_llm_by_purpose(
        ModelPurpose.BIOINFORMATICS,
        temperature=temperature,
        custom_model=custom_model
    )


def create_reasoning_advanced_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Create advanced reasoning model instance
    
    Specifically for complex reasoning tasks, prefer models with the strongest reasoning capabilities.
    
    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "anthropic:claude-3-opus-20240229")
    
    Returns:
        LLM instance, returns None if creation fails
    
    Examples:
        >>> llm = create_reasoning_advanced_llm()
    """
    return _create_llm_by_purpose(
        ModelPurpose.REASONING_ADVANCED,
        temperature=temperature,
        custom_model=custom_model
    )


def create_code_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Create code model instance
    
    For code generation, code analysis, code review, etc. Prefer models with strong code generation capabilities.
    
    Args:
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model, e.g., "openai:gpt-4o")
    
    Returns:
        LLM instance, returns None if creation fails
    
    Examples:
        >>> llm = create_code_llm()
        >>> llm = create_code_llm(temperature=0.2)
    """
    return _create_llm_by_purpose(
        ModelPurpose.CODE,
        temperature=temperature,
        custom_model=custom_model
    )


# ===================== Core Creation Function =====================
def _create_llm_by_purpose(
    purpose: ModelPurpose,
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Create LLM instance by purpose (core function)
    
    Args:
        purpose: Model purpose
        temperature: Temperature parameter, if None uses default configuration
        custom_model: Custom model name (format: provider:model)
    
    Returns:
        LLM instance, returns None if creation fails
    """
    if not LLM_AVAILABLE:
        print("Warning: LLM-related libraries not installed, cannot create LLM instance")
        return None
    
    # If custom model is specified, use it first
    if custom_model:
        try:
            provider, model = custom_model.split(":", 1)
            temp = temperature if temperature is not None else 0.1
            
            if provider == "openai":
                llm = _create_openai_llm(model, temp, base_url="https://xiaoai.plus/v1")
            elif provider == "dashscope":
                # DashScope uses DASHSCOPE_API_KEY
                dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY")
                if dashscope_key:
                    llm = _create_openai_llm(model, temp, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=dashscope_key)
                else:
                    llm = None
            elif provider == "zhipu":
                zhipu_key = os.getenv("ZHIPU_API_KEY")
                if zhipu_key:
                    llm = _create_zhipu_llm(model, temp, api_key=zhipu_key)
                else:
                    llm = None
            else:
                print(f"Warning: Unknown provider '{provider}', ignoring custom model")
                llm = None
            
            if llm is not None:
                print(f"Successfully created custom {provider} LLM instance ({model})")
                return llm
        except ValueError:
            print(f"Warning: Custom model format error '{custom_model}', should be 'provider:model'")
    
    # Try to create model by priority
    configs = MODEL_CONFIGS.get(purpose, [])
    if not configs:
        print(f"Warning: No model configuration found for purpose '{purpose.value}'")
        return None
    
    failed_attempts = []
    for provider, model, default_temp in configs:
        temp = temperature if temperature is not None else default_temp
        
        if provider == "openai":
            llm = _create_openai_llm(model, temp, base_url="https://xiaoai.plus/v1")
            if llm is None:
                failed_attempts.append(f"{provider}:{model} (missing OPENAI_API_KEY or creation failed)")
        elif provider == "dashscope":
            # DashScope uses DASHSCOPE_API_KEY
            dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY")
            if dashscope_key:
                llm = _create_openai_llm(model, temp, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=dashscope_key)
                if llm is None:
                    failed_attempts.append(f"{provider}:{model} (creation failed)")
            else:
                llm = None
                failed_attempts.append(f"{provider}:{model}:{dashscope_key} (missing DASHSCOPE_API_KEY or QIANFAN_API_KEY)")
        elif provider == "zhipu":
            # Zhipu uses ZHIPU_API_KEY
            zhipu_key = os.getenv("ZHIPU_API_KEY")
            if zhipu_key:
                llm = _create_zhipu_llm(model, temp, api_key=zhipu_key)
                if llm is None:
                    failed_attempts.append(f"{provider}:{model} (creation failed or adapter unavailable)")
            else:
                llm = None
                failed_attempts.append(f"{provider}:{model} (missing ZHIPU_API_KEY)")
        else:
            failed_attempts.append(f"{provider}:{model} (unknown provider)")
            continue
        
        if llm is not None:
            print(f"Successfully created {provider} {purpose.value} model instance ({model})")
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
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    Generic LLM creation function (backward compatible)
    
    Args:
        purpose: Model purpose, options: "reasoning", "bioinformatics", "reasoning_advanced", "code"
        temperature: Temperature parameter
        custom_model: Custom model name
    
    Returns:
        LLM instance, returns None if creation fails
    
    Examples:
        >>> llm = create_llm("reasoning")
        >>> llm = create_llm("bioinformatics", temperature=0.2)
    """
    try:
        purpose_enum = ModelPurpose(purpose)
        return _create_llm_by_purpose(purpose_enum, temperature, custom_model)
    except ValueError:
        print(f"Warning: Unknown model purpose '{purpose}', using default reasoning model")
        return _create_llm_by_purpose(ModelPurpose.REASONING, temperature, custom_model)


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
