"""
统一的 API Key 配置管理

所有 API key 都集中在这里管理，支持从环境变量读取。
如果环境变量不存在，则使用默认值（开发/测试用）。

使用方法：
1. 设置环境变量（推荐）：
   export OPENAI_API_KEY="your-key"
   export TAVILY_API_KEY="your-key"
   export QWEN_API_KEY="your-key"
   export DEEPSEEK_API_KEY="your-key"
   export QDRANT_API_KEY="your-key"

2. 或者直接修改此文件中的默认值

注意：生产环境请务必使用环境变量，不要将真实 API key 提交到代码仓库。
"""

import os
from typing import Optional


class APIKeys:
    """统一的 API Key 管理类"""
    
    # OpenAI API Keys
    # 默认值：从环境变量读取，如果没有则使用硬编码的默认值（仅用于开发测试）
    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        "your openai api key"
    )
    
    # Qwen API Keys (阿里云通义千问)
    # 用于 DashScope API (https://dashscope.aliyuncs.com)
    QWEN_API_KEY: str = os.getenv(
        "QWEN_API_KEY",
        "your qwen api key"
    )
    
    # Qwen API Key (另一个，用于某些配置)
    QWEN_API_KEY_ALT: str = os.getenv(
        "QWEN_API_KEY_ALT",
        "your qwen api key alt"
    )
    
    # XiaoAI Plus API Key (小爱AI Plus)
    # 用于 xiaoai.plus API (https://xiaoai.plus/v1)
    XIAOAI_API_KEY: str = os.getenv(
        "XIAOAI_API_KEY",
        "your xiaoai api key"
    )
    
    # DeepSeek API Key
    # 用于 api.deepseek.com
    DEEPSEEK_API_KEY: str = os.getenv(
        "DEEPSEEK_API_KEY",
        "your deepseek api key"
    )
    
    # DeepSeek API Key (另一个，用于 immunology 配置)
    DEEPSEEK_API_KEY_ALT: str = os.getenv(
        "DEEPSEEK_API_KEY_ALT",
        "your deepseek api key alt"
    )
    
    # XiaoAI Plus API Key (另一个，用于某些配置)
    XIAOAI_API_KEY_ALT: str = os.getenv(
        "XIAOAI_API_KEY_ALT",
        "your xiaoai api key alt"
    )
    
    # Tavily API Key (用于网络搜索)
    TAVILY_API_KEY: str = os.getenv(
        "TAVILY_API_KEY",
        "your tavily api key"
    )
    
    # Qdrant API Key (向量数据库)
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    
    # Anthropic API Key (Claude, 如果使用)
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    
    # 其他可能的 API Keys
    # 如果项目中有其他 API key，可以在这里添加
    
    @classmethod
    def get_openai_key(cls) -> str:
        """获取 OpenAI API Key"""
        return cls.OPENAI_API_KEY
    
    @classmethod
    def get_qwen_key(cls) -> str:
        """获取 Qwen API Key"""
        return cls.QWEN_API_KEY
    
    @classmethod
    def get_xiaoai_key(cls) -> str:
        """获取 XiaoAI Plus API Key"""
        return cls.XIAOAI_API_KEY
    
    @classmethod
    def get_deepseek_key(cls) -> str:
        """获取 DeepSeek API Key"""
        return cls.DEEPSEEK_API_KEY
    
    @classmethod
    def get_tavily_key(cls) -> str:
        """获取 Tavily API Key"""
        return cls.TAVILY_API_KEY
    
    @classmethod
    def get_qdrant_key(cls) -> Optional[str]:
        """获取 Qdrant API Key"""
        return cls.QDRANT_API_KEY
    
    @classmethod
    def get_anthropic_key(cls) -> Optional[str]:
        """获取 Anthropic API Key"""
        return cls.ANTHROPIC_API_KEY


# 为了向后兼容，导出常用的 key
OPENAI_API_KEY = APIKeys.OPENAI_API_KEY
QWEN_API_KEY = APIKeys.QWEN_API_KEY
XIAOAI_API_KEY = APIKeys.XIAOAI_API_KEY
DEEPSEEK_API_KEY = APIKeys.DEEPSEEK_API_KEY
TAVILY_API_KEY = APIKeys.TAVILY_API_KEY
QDRANT_API_KEY = APIKeys.QDRANT_API_KEY

