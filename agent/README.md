# Agent

本项目实现了若干 Agent，每个 Agent 在 usecases 中作为一个单独的目录。

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Required Files](#required-files)
- [Common Issues](#common-issues)
- [Developer Guide](#developer-guide)

---

## Quick Start

```bash
# 1. Install uv
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
cd agent
uv sync

# 3. Configure API keys (see Configuration section)
export OPENAI_API_KEY="your-key"

# 4. Run the workflow
python usecases/immunity/start_improved_workflow.py \
  --query "please design a computational method to identifiy broadly neutralizing antibodies against H5N1" \
  --file_url "https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx"
```

---

## System Requirements

- **Python**: >= 3.12
- **Operating System**: Windows 10/11, Linux, or macOS

---

## Installation

### Install uv

#### Windows (PowerShell)

```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Verify installation
uv --version
```

#### Linux/Mac

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**Note**: After installation, you may need to restart your terminal or run `source ~/.bashrc` (Linux/Mac)

### Install Project Dependencies

```bash
# Navigate to agent directory
cd agent

# Install all dependencies
uv sync

# Verify installation
python -c "import langchain; import langgraph; print('Dependencies installed successfully')"
```

---

## Configuration

### Configure API Keys

#### Method 1: Using Environment Variables (Recommended)

**Windows (PowerShell):**

```powershell
# Set API Keys (current session)
$env:OPENAI_API_KEY="your-openai-api-key"
$env:QWEN_API_KEY="your-qwen-api-key"
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"

# Permanent setup (requires terminal restart)
setx OPENAI_API_KEY "your-openai-api-key"
setx QWEN_API_KEY "your-qwen-api-key"
```

**Linux/Mac:**

```bash
# Temporary setup (current session)
export OPENAI_API_KEY="your-openai-api-key"
export QWEN_API_KEY="your-qwen-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# Permanent setup (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.bashrc
echo 'export QWEN_API_KEY="your-qwen-api-key"' >> ~/.bashrc
source ~/.bashrc
```

#### Method 2: Edit Configuration File

Edit `agent/config/api_keys.py` and modify the default values:

```python
OPENAI_API_KEY: str = os.getenv(
    "OPENAI_API_KEY",
    "your-actual-api-key-here"  # Modify here
)
```

**⚠️ Note**: For production environments, use environment variables. Do not commit real API keys to the code repository.

---

## Usage

### Verify Environment

```bash
cd agent
python verify_setup.py
```

### Basic Usage

```bash
# Query only
python usecases/immunity/start_improved_workflow.py --query "Your query question"

# Query + initial file URL
python usecases/immunity/start_improved_workflow.py \
  --query "Analyze BCR data" \
  --file_url "https://example.com/data.xlsx"
```

### Complete Example

```bash
# Example: Design computational method for H5N1 broadly neutralizing antibodies
python usecases/immunity/start_improved_workflow.py \
  --query "please design a computational method to identifiy broadly neutralizing antibodies against H5N1" \
  --file_url "https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx"
```

**Note**: During execution, the console may prompt you to:
- Provide RDS files for single-cell analysis
- Provide Excel files for experimental data integration
- Modify tool parameters
- Provide additional input files as needed

**For testing purposes, we provide the following sample data:**
1. Initial CSV: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu-simple.xlsx
2. RDS file: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/fluBcells.rds
3. Experimental data file 1: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_simple%28origin_flu-binding_neutralizations%29.xlsx
4. Experimental data file 2: https://immunity-test.oss-cn-beijing.aliyuncs.com/demo/flu_second_simple.xlsx

### Parameter Description

- `--query`: (Required) Your immunology research question or task description
- `--file_url`: (Optional) HTTP/HTTPS URL of the initial file. The script will automatically download and convert to CSV if it's an Excel file

---

## Required Files

The workflow may require different types of files depending on your analysis task. Here's what you need to know:

### Initial File (Optional)

You can provide an initial file via `--file_url` parameter:

- **Supported formats**: CSV (.csv), Excel (.xlsx, .xls)
- **Automatic conversion**: Excel files are automatically converted to CSV
- **Common use cases**: BCR prediction data, initial dataset

### Additional Files (May be required during workflow)

During workflow execution, the console may prompt you to provide additional files or modify tool parameters. Be prepared to provide:

#### 1. **RDS Files** (Seurat single-cell RNA-seq data)
- **Format**: `.rds` (R data format)
- **Use case**: Single-cell analysis, UMAP extraction, cell type annotation
- **Example**: `20240923_flu_B_annotation.rds`
- **When needed**: For tools like `extract_seurat_umap_metadata` or `integrate_bcr_data_complete`

#### 2. **Excel Files** (Experimental data)
- **Format**: `.xlsx` or `.xls`
- **Use case**: Binding assays, neutralization assays, experimental measurements
- **Example**: Binding assay results, neutralization data
- **When needed**: For tools like `integrate_binding_neutralization_experiments`

#### 3. **CSV Files** (BCR data, processed data)
- **Format**: `.csv`
- **Use case**: BCR sequence data, processed analysis results
- **Example**: Integrated BCR dataset, prediction results
- **When needed**: For data integration and analysis tools

#### 4. **FASTQ Files** (Bulk sequencing data)
- **Format**: `.fastq` or `.fastq.gz`
- **Use case**: Bulk BCR sequencing data
- **When needed**: For tools like `integrate_scbcr_bulk_bcr_data`
- **Note**: Usually provided as a directory containing multiple FASTQ files

### File Preparation Tips

1. **Check console output**: The workflow will prompt you when additional files are needed
2. **Have files ready**: Prepare RDS, Excel, or other required files before starting
3. **File paths**: You can provide local file paths or URLs when prompted
4. **File format**: Ensure files are in the correct format (RDS for Seurat, Excel for experimental data)

---

## Common Issues

### 1. ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'langchain_mcp_adapters'`

**Solution**:
```bash
cd agent
uv sync
```

### 2. API Key Error

**Error**: `Invalid API key` or `API key not found`

**Solution**:
1. Check environment variables:
   ```powershell
   # Windows PowerShell
   echo $env:OPENAI_API_KEY
   
   # Linux/Mac
   echo $OPENAI_API_KEY
   ```
2. Ensure environment variables are set in the correct terminal session
3. Or directly edit `agent/config/api_keys.py`

### 3. Python Version Error

**Error**: `requires-python = ">=3.12"`

**Solution**:
```bash
# Check version
python --version

# If version is below 3.12, please upgrade Python
```

### 4. uv Command Not Found

**Solution**:
- Windows: Restart PowerShell or run `refreshenv`
- Linux/Mac: Run `source ~/.bashrc` or restart terminal

### 5. File Download Failed

**Solution**:
1. Check network connection
2. Verify the file URL is accessible
3. Check error logs for detailed information

---

## Developer Guide

### Prerequisites

请先熟练掌握以下基本概念：
- [uv](https://docs.astral.sh/uv/)
  - 基本命令
  - 项目结构
  - 环境管理
- [langgraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
  - Graph
  - State
  - Stream
  - Interrupt

### Code Formatting

```sh
uvx ruff check --select I --fix
uvx ruff format
```

### Run Agent Code

```sh
uv run usecases/antibody/graph/retrieval_graph.py
```

### Run Server

运行后端服务之前，先编译并部署前端静态文件，见 `ui` 项目

```sh
mkdir /opt/antibody_gen/

# 数据库 schema 迁移，第一次运行时需要
# 如果改了 schema 需要重新迁移，只需删除原数据库文件，重新运行
alembic upgrade head

# run server
uv run main.py

# 按照控制台的指示访问服务器，并输入 access token
```

### How to Build an Agent

构建一个新的 Agent 需要遵循以下步骤和结构：

#### 1. Create Directory Structure

在 `usecases/` 目录下创建新的 Agent 目录，建议的目录结构：

```
usecases/your_agent/
├── __init__.py
├── your_agent_config.py      # Agent 配置
├── state/
│   ├── __init__.py
│   └── state.py             # 状态定义
├── graph/
│   ├── __init__.py
│   └── main_graph.py        # 主图定义
├── tool/
│   ├── __init__.py
│   └── tools.py             # 工具定义
└── start_your_agent.py      # 启动脚本
```

#### 2. Define State Class

在 `state/state.py` 中定义 Agent 的状态类，继承自 `pydantic.BaseModel`：

```python
from typing import Dict, List, Optional
from pydantic import BaseModel

class YourAgentState(BaseModel):
    """Your Agent 状态类"""
    input_query: str
    intermediate_results: List[str] = []
    final_output: str = ""
    # 其他需要的状态字段
```

#### 3. Create Configuration

在 `your_agent_config.py` 中定义配置函数：

```python
from common.constants import REASONING_MODEL
from langchain_core.runnables import RunnableConfig

def get_your_agent_runnable_config(thread_id=None) -> RunnableConfig:
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "model_config": {
            "default_model": {
                "provider": "Ollama",
                "model": REASONING_MODEL,
                "params": {"temperature": 0.2}
            },
            "reasoning_model": {
                "provider": "Ollama",
                "model": REASONING_MODEL,
                "params": {"temperature": 0.2}
            }
        }
    }
    return config
```

#### 4. Implement Graph Nodes

在 `graph/main_graph.py` 中实现各个节点函数：

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from usecases.your_agent.state.state import YourAgentState

def process_input(state: YourAgentState, config: RunnableConfig):
    """处理输入节点"""
    # 实现逻辑
    return state

def generate_output(state: YourAgentState, config: RunnableConfig):
    """生成输出节点"""
    # 实现逻辑
    return state

def create_your_agent_graph():
    """创建 Agent 图"""
    workflow = StateGraph(YourAgentState)
    
    # 添加节点
    workflow.add_node("process_input", process_input)
    workflow.add_node("generate_output", generate_output)
    
    # 设置边
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "generate_output")
    workflow.add_edge("generate_output", END)
    
    return workflow.compile()
```

#### 5. Create Startup Script

在 `start_your_agent.py` 中创建启动逻辑：

```python
from usecases.your_agent.graph.main_graph import create_your_agent_graph
from usecases.your_agent.your_agent_config import get_your_agent_runnable_config

if __name__ == "__main__":
    graph = create_your_agent_graph()
    config = get_your_agent_runnable_config()
    
    # 运行图
    result = graph.invoke({
        "input_query": "你的查询"
    }, config=config)
    
    print(result)
```

### How to Integrate MCP with Agent

#### 1. Configure MCP Server

在 `config/config.py` 中添加新的 MCP 服务器配置：

```python
class ApplicationConfig(BaseModel):
    mcp_servers: dict[str, dict] = {
        # 现有配置...
        "your_mcp_service": {
            "transport": "sse",  # 或 "stdio", "streamable_http"
            "url": "http://localhost:8080/sse",
            "timeout": 120,
            "sse_read_timeout": 120,
            "session_kwargs": {},
        }
    }
```

#### 2. Use MCP in Agent

在 Agent 的节点函数中通过 factory 获取 MCP 客户端：

```python
from common.factory import get_mcp_client

async def mcp_node(state: YourAgentState, config: RunnableConfig):
    """使用 MCP 的节点"""
    mcp_client = await get_mcp_client(config)
```

#### 3. Configure MCP Service ID

在 Agent 配置中指定需要的 MCP 服务：

```python
def get_your_agent_runnable_config(thread_id=None) -> RunnableConfig:
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "model_config": {
            # 模型配置...
        },
        "mcp_config": {
            "service_ids": ["your_mcp_service"]  # 指定需要的 MCP 服务
        }
    }
    return config
```

### How to Register Agent to Usecase

#### Register Agent in usecases.py

参考 `web/session/usecases.py` 的方式，在 `Usecases` 类中添加新的 Agent：
提供
- 默认配置
- 初始状态
- 工厂方法，输出为 `CompiledStateGraph`
- 结果工厂方法，接受最终的 state 作为输入，生成 str 类型的输出，作为 agent 运行的总结

如下

```python
from web.session.usecases import Usecase, Usecases
from usecases.your_agent.graph.main_graph import create_your_agent_graph

class Usecases:
    # 现有的 usecase...
    
    YOUR_AGENT = Usecase(
        name="your_agent",
        default_configuration={
            "model_config": {
                "default_model": {
                    "provider": "Ollama",
                    "model": "qwen2.5:7b",
                    "params": {"temperature": 0.2}
                },
                "reasoning_model": {
                    "provider": "Ollama", 
                    "model": "qwen2.5:7b",
                    "params": {"temperature": 0.2}
                }
            },
            "mcp_config": {"service_ids": ["your_mcp_service"]},
            # 其他配置参数...
        },
        init_state_factory=lambda user_message: {
            "input_query": user_message,
            "intermediate_results": [],
            "final_output": "",
            # 其他初始状态字段...
        },
        graph_factory=create_your_agent_graph,
        result_factory=lambda s: s['final_report']
    )
    
    @classmethod
    def get_usecase(cls, name: str) -> Usecase:
        for usecase in cls.__dict__.values():
            if isinstance(usecase, Usecase):
                if usecase.name == name:
                    return usecase
        raise ValueError(f"Usecase {name} not found")
```

### Best Practices

- 使用 `interrupt` 来实现人机交互
- 模型，mcp 都使用 `common/factory` 中的工厂方法，从 `RunnableConfig` 中获取
