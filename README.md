# Immune Agent - Bioinformatics Intelligent Agent System

## Project Overview

Immune Agent is an intelligent agent system built on LangGraph, designed specifically for bioinformatics-related tasks. The system supports task classification, task decomposition, parallel execution, and more.

## Features

- 🤖 **Multi-LLM Support**: Supports various large language models including Qwen (Tongyi Qianwen), Anthropic Claude, OpenAI GPT, and more
- 📊 **Task Classification**: Automatically identifies user task types (general Q&A, execution plans, immunology tasks, etc.)
- 🔄 **State Management**: LangGraph-based state graph management with state persistence support
- 🧩 **Modular Design**: Modular sub-graph design with Supervisor, Executor, Parallel, and other components
- 📁 **File Management**: Automatic handling of uploaded files with sandbox directory management

## Installation

### Basic Installation (Core Features Only)

```bash
cd agent
pip install -e .
```

### Install All LLM Providers

```bash
pip install -e ".[all]"
```

### Install Development Dependencies (Testing Tools)

```bash
pip install -e ".[dev]"
```

## Environment Variable Configuration

### Method 1: Using .env File (Recommended)

1. Create a `.env` file in the project root directory (at the same level as the `agent` directory):

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

2. Edit the `.env` file and fill in your API Keys:

```env
# Qwen / DashScope (Recommended)
DASHSCOPE_API_KEY=sk-your-dashscope-api-key

# Or use Qianfan
# QIANFAN_API_KEY=your-qianfan-api-key

# Anthropic Claude (Optional)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key

# OpenAI GPT (Optional)
OPENAI_API_KEY=sk-your-openai-api-key
```

3. The project will automatically load environment variables from the `.env` file (using `python-dotenv`)

### Method 2: Windows System Environment Variables

#### Via Graphical Interface:

1. Right-click "This PC" → "Properties"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Click "New" under "User variables" or "System variables"
5. Add the variable name and value:
   - Variable name: `DASHSCOPE_API_KEY`
   - Variable value: `your-api-key-here`
6. Repeat steps 4-5 to add other API Keys

#### Via PowerShell (Current User):

```powershell
# Set Qwen / DashScope API Key
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your-api-key-here', 'User')

# Set Anthropic API Key
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'your-api-key-here', 'User')

# Set OpenAI API Key
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-api-key-here', 'User')
```

**Note**: You need to reopen the terminal/PowerShell for changes to take effect.

### Method 3: PowerShell Temporary Setup (Current Session Only)

```powershell
# Set environment variables (valid for current PowerShell session only)
$env:DASHSCOPE_API_KEY = "your-api-key-here"
$env:ANTHROPIC_API_KEY = "your-api-key-here"
$env:OPENAI_API_KEY = "your-api-key-here"
```

### Method 4: Command Line Temporary Setup (Current Session Only)

```cmd
# Windows CMD
set DASHSCOPE_API_KEY=your-api-key-here
set ANTHROPIC_API_KEY=your-api-key-here
set OPENAI_API_KEY=your-api-key-here
```

## Getting API Keys

### Qwen / DashScope

1. Visit the [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)
2. Register/log in to your account
3. Create an API Key
4. Copy the API Key to your environment variables

### Anthropic Claude

1. Visit the [Anthropic Website](https://www.anthropic.com/)
2. Register/log in to your account
3. Create an API Key in the console
4. Copy the API Key to your environment variables

### OpenAI GPT

1. Visit the [OpenAI Platform](https://platform.openai.com/)
2. Register/log in to your account
3. Create a new API Key on the API Keys page
4. Copy the API Key to your environment variables

## Quick Start

```python
from main_graph import build_main_graph

# Build the main graph
main_graph = build_main_graph()

# Run the agent
result = main_graph.invoke({
    "user_input": "Analyze this immunology-related task",
    "sandbox_dir": "./sandbox"
})

print("Task type:", result.user_task_type)
print("Execution result:", result.merged_result)
```

## Running Tests

```bash
cd agent

# Run all tests
pytest tests/test_agent.py -v

# Or use the quick script
python run_tests.py

# Or use the original test script
python test_agent.py
```

## Project Structure

```
agent/
├── main_graph.py          # Main graph definition
├── state.py               # Global state definition
├── nodes/
│   └── subagents/
│       ├── supervisor/    # Supervisor sub-graph (task classification)
│       ├── executor/      # Executor sub-graph
│       └── parallel/      # Parallel task sub-graph
├── utils/
│   ├── llm_factory.py     # LLM factory (unified LLM creation)
│   └── code_cache_manager.py  # Code cache management
├── tests/                 # Test cases
└── pyproject.toml         # Project configuration and dependencies
```

## Dependencies

### Core Dependencies
- `langgraph` - State graph framework
- `langchain-core` - LangChain core library
- `pydantic` - Data model validation
- `python-dotenv` - Environment variable management

### Optional Dependencies (LLM Providers)
- `langchain-qianfan` - Qwen / Tongyi Qianwen (Recommended)
- `langchain-anthropic` - Anthropic Claude
- `langchain-openai` - OpenAI GPT

### Development Dependencies
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support

## Notes

1. **API Key Security**:
   - Do not commit the `.env` file to Git
   - The `.env` file is included in `.gitignore` (if configured)
   - Use `.env.example` as a template

2. **Priority**:
   - System environment variables > `.env` file
   - If both `DASHSCOPE_API_KEY` and `QIANFAN_API_KEY` are set, `DASHSCOPE_API_KEY` takes precedence

3. **Fallback**:
   - If no API Key is configured, the system will use keyword matching as a fallback
   - Functionality will be limited, but basic task classification will still work

## License

MIT License
