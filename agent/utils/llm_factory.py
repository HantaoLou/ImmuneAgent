"""
LLM工厂模块 - 提供统一的LLM实例创建功能

按模型作用分类，支持以下类型：
- 普通推理模型（reasoning）：用于逻辑推理、任务分类、决策等
- 生信专项模型（bioinformatics）：用于生物信息学相关任务
- 高级推理模型：专门用于复杂推理任务
- 代码模型：用于代码生成和分析

使用方式：
    from agent.utils.llm_factory import create_reasoning_llm, create_bioinformatics_llm, create_code_llm
    
    # 创建推理模型
    llm = create_reasoning_llm()
    
    # 创建生信专项模型
    llm = create_bioinformatics_llm()
    
    # 创建代码模型
    llm = create_code_llm()
"""

from typing import Optional, Any, Dict, List, Tuple
from enum import Enum
import os

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ChatAnthropic = None
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None
    print("警告：langchain相关库未安装，LLM功能将不可用")


# ===================== 模型用途枚举 =====================
class ModelPurpose(str, Enum):
    """模型用途枚举"""
    REASONING = "reasoning"  # 推理模型：用于逻辑推理、任务分类、决策等
    BIOINFORMATICS = "bioinformatics"  # 生信专项模型：用于生物信息学相关任务
    REASONING_ADVANCED = "reasoning_advanced"  # 高级推理模型：专门用于复杂推理任务
    CODE = "code"  # 代码模型：用于代码生成和分析


# ===================== 模型配置映射 =====================
MODEL_CONFIGS: Dict[ModelPurpose, List[Tuple[str, str, float]]] = {
    ModelPurpose.REASONING: [
        ("openai", "qwen-turbo", 0.1),  # 通义千问 Max - 优先使用
    ],
    
    # 生信专项模型：优先使用通义千问，其次使用对科学文献理解好的模型
    ModelPurpose.BIOINFORMATICS: [
        ("openai", "qwen-max", 0.2),  # 通义千问 Max - 优先使用
    ],
    
    # 高级推理模型：优先使用通义千问，用于复杂推理任务
    ModelPurpose.REASONING_ADVANCED: [
        ("openai", "qwen-max", 0.1),  # 通义千问 Max - 优先使用
    ],
    
    # 代码模型：优先使用通义千问，其次使用代码生成能力强的模型
    ModelPurpose.CODE: [
        ("openai", "qwen-max", 0.1),  # 通义千问 Max - 优先使用
    ],
}


# ===================== 底层供应商创建函数 =====================
def _create_openai_llm(
    model: str,
    temperature: float = 0.1,
    api_key: Optional[str] = None
) -> Optional[Any]:
    """内部函数：创建OpenAI GPT LLM实例"""
    if not LLM_AVAILABLE or ChatOpenAI is None:
        return None
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    except Exception as e:
        print(f"错误：创建OpenAI LLM失败 ({model}): {e}")
        return None


# ===================== 按用途创建LLM =====================
def create_reasoning_llm(
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    创建推理模型实例
    
    用于逻辑推理、任务分类、决策等任务。优先使用推理性能好的模型。
    
    Args:
        temperature: 温度参数，如果为None则使用默认配置
        custom_model: 自定义模型名称（格式：provider:model，如 "anthropic:claude-3-opus-20240229"）
    
    Returns:
        LLM实例，如果创建失败则返回None
    
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
    创建生信专项模型实例
    
    用于生物信息学相关任务，如文献分析、数据解读等。优先使用对科学文献理解好的模型。
    
    Args:
        temperature: 温度参数，如果为None则使用默认配置
        custom_model: 自定义模型名称（格式：provider:model，如 "anthropic:claude-3-opus-20240229"）
    
    Returns:
        LLM实例，如果创建失败则返回None
    
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
    创建高级推理模型实例
    
    专门用于复杂推理任务，优先使用推理能力最强的模型。
    
    Args:
        temperature: 温度参数，如果为None则使用默认配置
        custom_model: 自定义模型名称（格式：provider:model，如 "anthropic:claude-3-opus-20240229"）
    
    Returns:
        LLM实例，如果创建失败则返回None
    
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
    创建代码模型实例
    
    用于代码生成、代码分析、代码审查等任务。优先使用代码生成能力强的模型。
    
    Args:
        temperature: 温度参数，如果为None则使用默认配置
        custom_model: 自定义模型名称（格式：provider:model，如 "openai:gpt-4o"）
    
    Returns:
        LLM实例，如果创建失败则返回None
    
    Examples:
        >>> llm = create_code_llm()
        >>> llm = create_code_llm(temperature=0.2)
    """
    return _create_llm_by_purpose(
        ModelPurpose.CODE,
        temperature=temperature,
        custom_model=custom_model
    )


# ===================== 核心创建函数 =====================
def _create_llm_by_purpose(
    purpose: ModelPurpose,
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    根据用途创建LLM实例（核心函数）
    
    Args:
        purpose: 模型用途
        temperature: 温度参数，如果为None则使用默认配置
        custom_model: 自定义模型名称（格式：provider:model）
    
    Returns:
        LLM实例，如果创建失败则返回None
    """
    if not LLM_AVAILABLE:
        print("警告：LLM相关库未安装，无法创建LLM实例")
        return None
    
    # 如果指定了自定义模型，优先使用
    if custom_model:
        try:
            provider, model = custom_model.split(":", 1)
            temp = temperature if temperature is not None else 0.1
            
            if provider == "openai":
                llm = _create_openai_llm(model, temp)
            else:
                print(f"警告：未知的提供者 '{provider}'，忽略自定义模型")
                llm = None
            
            if llm is not None:
                print(f"成功创建自定义{provider} LLM实例 ({model})")
                return llm
        except ValueError:
            print(f"警告：自定义模型格式错误 '{custom_model}'，应为 'provider:model'")
    
    # 按优先级尝试创建模型
    configs = MODEL_CONFIGS.get(purpose, [])
    if not configs:
        print(f"警告：未找到用途 '{purpose.value}' 的模型配置")
        return None
    
    for provider, model, default_temp in configs:
        temp = temperature if temperature is not None else default_temp
        
        if provider == "openai":
            llm = _create_openai_llm(model, temp)
        else:
            continue
        
        if llm is not None:
            print(f"成功创建{provider} {purpose.value}模型实例 ({model})")
            return llm
    
    print(f"警告：无法创建{purpose.value}模型，所有配置的模型都不可用")
    return None


# ===================== 通用创建函数（向后兼容） =====================
def create_llm(
    purpose: str = "reasoning",
    temperature: Optional[float] = None,
    custom_model: Optional[str] = None
) -> Optional[Any]:
    """
    通用LLM创建函数（向后兼容）
    
    Args:
        purpose: 模型用途，可选 "reasoning", "bioinformatics", "reasoning_advanced", "code"
        temperature: 温度参数
        custom_model: 自定义模型名称
    
    Returns:
        LLM实例，如果创建失败则返回None
    
    Examples:
        >>> llm = create_llm("reasoning")
        >>> llm = create_llm("bioinformatics", temperature=0.2)
    """
    try:
        purpose_enum = ModelPurpose(purpose)
        return _create_llm_by_purpose(purpose_enum, temperature, custom_model)
    except ValueError:
        print(f"警告：未知的模型用途 '{purpose}'，使用默认推理模型")
        return _create_llm_by_purpose(ModelPurpose.REASONING, temperature, custom_model)


# ===================== 辅助函数 =====================
def is_llm_available() -> bool:
    """
    检查LLM功能是否可用
    
    Returns:
        如果LLM相关库已安装则返回True，否则返回False
    """
    return LLM_AVAILABLE


def get_available_providers() -> List[str]:
    """
    获取可用的LLM提供者列表（基于环境变量）
    
    Returns:
        可用提供者的列表，如 ["openai"]
    """
    providers = []

    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    
    return providers


def get_model_configs(purpose: Optional[ModelPurpose] = None) -> Dict:
    """
    获取模型配置信息
    
    Args:
        purpose: 模型用途，如果为None则返回所有配置
    
    Returns:
        模型配置字典
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
