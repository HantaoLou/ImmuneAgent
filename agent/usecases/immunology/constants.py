"""
Constants for ImmuneAgent system.
"""

import os
from pathlib import Path

# API Keys - 从统一的 API keys 配置导入
import sys
from pathlib import Path
# 添加 agent 目录到路径，以便导入 config.api_keys
agent_root = Path(__file__).parent.parent.parent
if str(agent_root) not in sys.path:
    sys.path.insert(0, str(agent_root))

from config.api_keys import APIKeys
OPENAI_API_KEY = APIKeys.OPENAI_API_KEY

# Set environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Paths
IMMUNOLOGY_ROOT = Path(__file__).parent
AGENT_ROOT = IMMUNOLOGY_ROOT.parent.parent
PROJECT_ROOT = AGENT_ROOT.parent

# Model configurations
DEFAULT_LLM_MODEL = "gpt-4"
DEFAULT_FAST_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 10
RERANK_TOP_K = 5

# Tool execution settings
TOOL_TIMEOUT = 300  # seconds
MAX_RETRIES = 3
BATCH_SIZE = 100
