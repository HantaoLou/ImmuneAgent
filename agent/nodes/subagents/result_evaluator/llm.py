"""
LLM Module
Extracted from Biomni framework (biomni/llm.py)

MODIFIED: 集成 llm_factory.py 的模型配置，支持 DashScope 和 Zhipu
"""

import os
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel

# 扩展支持 DashScope 和 Zhipu
SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom", "DashScope", "Zhipu"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    purpose: str = "reasoning_advanced",  # 新增: 模型用途，默认使用高级推理模型
) -> BaseChatModel:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    
    MODIFIED: 始终优先使用 llm_factory.py 的配置
    
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom", "DashScope", "Zhipu"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        purpose (str): Model purpose for llm_factory: "reasoning", "bioinformatics", "reasoning_advanced", "code"
    """
    # ============================================================
    # 始终优先使用 llm_factory.py 的配置（除非用户明确指定了所有参数）
    # ============================================================
    # 如果只指定了 source 但没有指定 model，说明用户想用 llm_factory 的配置
    use_llm_factory = (model is None) or (model is None and source is None)
    
    if use_llm_factory:
        try:
            from agent.utils.llm_factory import create_llm
            llm = create_llm(purpose=purpose, temperature=temperature)
            if llm is not None:
                print(f"[result_evaluator/llm] Using llm_factory config for purpose={purpose}")
                # 如果需要 stop_sequences，需要重新包装
                if stop_sequences and hasattr(llm, 'bind'):
                    llm = llm.bind(stop=stop_sequences)
                return llm
        except ImportError as e:
            print(f"[result_evaluator/llm] llm_factory not available: {e}")
        except Exception as e:
            print(f"[result_evaluator/llm] llm_factory failed: {e}")

    # Use defaults if not specified
    if model is None:
        model = "claude-3-5-sonnet-20241022"
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = "EMPTY"

    # Auto-detect source from model name if not specified
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            if model[:7] == "claude-":
                source = "Anthropic"
            elif model[:7] == "gpt-oss":
                source = "Ollama"
            elif model[:4] == "gpt-":
                source = "OpenAI"
            elif model.startswith("azure-"):
                source = "AzureOpenAI"
            elif model[:7] == "gemini-":
                source = "Gemini"
            elif "groq" in model.lower():
                source = "Groq"
            elif base_url is not None:
                source = "Custom"
            # 新增: 检测 DashScope 和 Zhipu 模型
            elif model.startswith(("qwen-", "qwen2-", "qwen2.5-", "deepseek-")):
                source = "DashScope"
            elif model.startswith(("glm-", "chatglm-")):
                source = "Zhipu"
            elif "/" in model or any(
                name in model.lower()
                for name in [
                    "llama",
                    "mistral",
                    "qwen",
                    "gemma",
                    "phi",
                    "dolphin",
                    "orca",
                    "vicuna",
                    "deepseek",
                ]
            ):
                source = "Ollama"
            elif model.startswith(
                ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
            ):
                source = "Bedrock"
            else:
                raise ValueError("Unable to determine model source. Please specify 'source' parameter.")

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        API_VERSION = "2024-12-01-preview"
        model = model.replace("azure-", "")
        return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=API_VERSION,
            temperature=temperature,
        )

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # Ensure ANTHROPIC_API_KEY is loaded from bash_profile if not in environment
        if not os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import subprocess

                result = subprocess.run(
                    ["bash", "-c", "source ~/.bash_profile 2>/dev/null && echo $ANTHROPIC_API_KEY"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    os.environ["ANTHROPIC_API_KEY"] = result.stdout.strip()
                    print("✓ Loaded ANTHROPIC_API_KEY from ~/.bash_profile")
            except Exception as e:
                print(f"Note: Could not load ANTHROPIC_API_KEY from bash_profile: {e}")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
        )

    elif source == "Gemini":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        return ChatBedrock(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
        )
        return llm

    elif source == "DashScope":
        # 阿里云 DashScope (Qwen 系列模型)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for DashScope models. Install with: pip install langchain-openai"
            )
        dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY")
        if not dashscope_key:
            raise ValueError("DASHSCOPE_API_KEY or QIANFAN_API_KEY is required for DashScope models")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=dashscope_key,
        )

    elif source == "Zhipu":
        # 智谱 AI (GLM 系列模型)
        try:
            from agent.utils.zhipu_adapter import ZhipuAIAdapter
        except ImportError:
            # Fallback to OpenAI-compatible API if adapter not available
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(  # noqa: B904
                    "langchain-openai package is required for Zhipu models. Install with: pip install langchain-openai"
                )
            zhipu_key = os.getenv("ZHIPU_API_KEY")
            if not zhipu_key:
                raise ValueError("ZHIPU_API_KEY is required for Zhipu models")
            # 使用智谱的 OpenAI 兼容接口
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=8192,
                stop_sequences=stop_sequences,
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                api_key=zhipu_key,
            )
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if not zhipu_key:
            raise ValueError("ZHIPU_API_KEY is required for Zhipu models")
        return ZhipuAIAdapter(
            model=model,
            temperature=temperature,
            api_key=zhipu_key,
            timeout=120,
        )

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', 'Ollama', 'DashScope', or 'Zhipu'"
        )
