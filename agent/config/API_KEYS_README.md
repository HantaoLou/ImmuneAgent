# API Keys 统一配置说明

## 概述

所有 API key 现在都统一在 `agent/config/api_keys.py` 中管理。您只需要在一个地方修改，就能改变全局的 API key。

## 使用方法

### 方法 1：使用环境变量（推荐）

在运行程序前设置环境变量：

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:QWEN_API_KEY="your-qwen-key"
$env:XIAOAI_API_KEY="your-xiaoai-key"
$env:DEEPSEEK_API_KEY="your-deepseek-key"
$env:TAVILY_API_KEY="your-tavily-key"
$env:QDRANT_API_KEY="your-qdrant-key"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-openai-key
set QWEN_API_KEY=your-qwen-key
set XIAOAI_API_KEY=your-xiaoai-key
set DEEPSEEK_API_KEY=your-deepseek-key
set TAVILY_API_KEY=your-tavily-key
set QDRANT_API_KEY=your-qdrant-key
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-key"
export QWEN_API_KEY="your-qwen-key"
export XIAOAI_API_KEY="your-xiaoai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export TAVILY_API_KEY="your-tavily-key"
export QDRANT_API_KEY="your-qdrant-key"
```

### 方法 2：直接修改配置文件

编辑 `agent/config/api_keys.py` 文件，修改默认值：

```python
OPENAI_API_KEY: str = os.getenv(
    "OPENAI_API_KEY",
    "your-default-key-here"  # 修改这里
)
```

## 支持的 API Keys

- `OPENAI_API_KEY` - OpenAI API key
- `QWEN_API_KEY` - 阿里云通义千问 API key (DashScope)
- `QWEN_API_KEY_ALT` - 另一个 Qwen API key（用于某些配置）
- `XIAOAI_API_KEY` - 小爱AI Plus API key
- `XIAOAI_API_KEY_ALT` - 另一个 XiaoAI API key（用于某些配置）
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `DEEPSEEK_API_KEY_ALT` - 另一个 DeepSeek API key（用于 immunology 配置）
- `TAVILY_API_KEY` - Tavily 搜索 API key
- `QDRANT_API_KEY` - Qdrant 向量数据库 API key（可选）
- `ANTHROPIC_API_KEY` - Anthropic Claude API key（可选）

## 在代码中使用

所有配置文件现在都从统一的配置导入：

```python
from config.api_keys import APIKeys

# 使用 API key
api_key = APIKeys.OPENAI_API_KEY
# 或
api_key = APIKeys.QWEN_API_KEY
# 或
api_key = APIKeys.XIAOAI_API_KEY
```

## 已更新的文件

以下文件已经更新为使用统一的 API key 配置：

- `agent/usecases/immunity/common/constants.py`
- `agent/usecases/immunity/config/immunity_config.py`
- `agent/usecases/immunity/config/immunity_config_qwen.py`
- `agent/usecases/immunity/config/immunity_config_gpt4.1.py`
- `agent/usecases/immunity/config/immunity_config_deepseek.py`
- `agent/usecases/cell/cell_config.py`
- `agent/web/session/usecases.py`
- `agent/usecases/immunology/constants.py`
- `agent/usecases/immunology/immunology_config.py`
- `agent/usecases/immunity/tools/retrieve_tools.py`
- `agent/usecases/cell/tool/retrieval_tools.py`
- `agent/usecases/retrieval/tools.py`
- `agent/usecases/immunology/test_planning_demo.py`
- `agent/usecases/immunology/test_10_questions.py`

## 安全建议

1. **生产环境**：务必使用环境变量，不要将真实 API key 提交到代码仓库
2. **开发环境**：可以使用配置文件中的默认值（仅用于测试）
3. **版本控制**：确保 `.env` 文件已添加到 `.gitignore` 中
4. **定期轮换**：定期更换 API key 以提高安全性

## 注意事项

- 环境变量的优先级高于配置文件中的默认值
- 如果某个 API key 未设置环境变量，将使用配置文件中的默认值
- 所有硬编码的 API key 已被替换为从统一配置读取

