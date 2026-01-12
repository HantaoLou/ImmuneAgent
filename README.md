# Bio Agent - 生物信息学智能Agent系统

## 项目简介

Bio Agent 是一个基于 LangGraph 构建的智能Agent系统，专门用于生物信息学相关任务。系统支持任务分类、任务分解、并行执行等功能。

## 功能特性

- 🤖 **多LLM支持**：支持通义千问、Anthropic Claude、OpenAI GPT等多种大语言模型
- 📊 **任务分类**：自动识别用户任务类型（普通问答、执行计划、免疫学任务等）
- 🔄 **状态管理**：基于LangGraph的状态图管理，支持状态持久化
- 🧩 **模块化设计**：Supervisor、Executor、Parallel等子图模块化设计
- 📁 **文件管理**：自动处理上传文件，支持沙盒目录管理

## 安装依赖

### 基础安装（仅核心功能）

```bash
cd agent
pip install -e .
```

### 安装所有LLM提供者

```bash
pip install -e ".[all]"
```

### 安装开发依赖（测试工具）

```bash
pip install -e ".[dev]"
```

## 环境变量配置

### 方法1：使用 .env 文件（推荐）

1. 在项目根目录（与 `agent` 目录同级）创建 `.env` 文件：

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API Key：

```env
# 通义千问（推荐）
DASHSCOPE_API_KEY=sk-your-dashscope-api-key

# 或使用千帆
# QIANFAN_API_KEY=your-qianfan-api-key

# Anthropic Claude（可选）
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key

# OpenAI GPT（可选）
OPENAI_API_KEY=sk-your-openai-api-key
```

3. 项目会自动加载 `.env` 文件中的环境变量（使用 `python-dotenv`）

### 方法2：Windows 系统环境变量

#### 通过图形界面设置：

1. 右键"此电脑" → "属性"
2. 点击"高级系统设置"
3. 点击"环境变量"
4. 在"用户变量"或"系统变量"中点击"新建"
5. 添加变量名和值：
   - 变量名：`DASHSCOPE_API_KEY`
   - 变量值：`your-api-key-here`
6. 重复步骤4-5添加其他 API Key

#### 通过 PowerShell 设置（当前用户）：

```powershell
# 设置通义千问 API Key
[System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'your-api-key-here', 'User')

# 设置 Anthropic API Key
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'your-api-key-here', 'User')

# 设置 OpenAI API Key
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your-api-key-here', 'User')
```

**注意**：设置后需要重新打开终端/PowerShell 才能生效。

### 方法3：PowerShell 临时设置（仅当前会话）

```powershell
# 设置环境变量（仅当前 PowerShell 会话有效）
$env:DASHSCOPE_API_KEY = "your-api-key-here"
$env:ANTHROPIC_API_KEY = "your-api-key-here"
$env:OPENAI_API_KEY = "your-api-key-here"
```

### 方法4：命令行临时设置（仅当前会话）

```cmd
# Windows CMD
set DASHSCOPE_API_KEY=your-api-key-here
set ANTHROPIC_API_KEY=your-api-key-here
set OPENAI_API_KEY=your-api-key-here
```

## 获取 API Key

### 通义千问（DashScope）

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key
4. 复制 API Key 到环境变量

### Anthropic Claude

1. 访问 [Anthropic 官网](https://www.anthropic.com/)
2. 注册/登录账号
3. 在控制台创建 API Key
4. 复制 API Key 到环境变量

### OpenAI GPT

1. 访问 [OpenAI 官网](https://platform.openai.com/)
2. 注册/登录账号
3. 在 API Keys 页面创建新的 API Key
4. 复制 API Key 到环境变量

## 快速开始

```python
from main_graph import build_main_graph

# 构建主图
main_graph = build_main_graph()

# 运行Agent
result = main_graph.invoke({
    "user_input": "分析这个免疫学相关的任务",
    "sandbox_dir": "./sandbox"
})

print("任务类型:", result.user_task_type)
print("执行结果:", result.merged_result)
```

## 运行测试

```bash
cd agent

# 运行所有测试
pytest tests/test_agent.py -v

# 或使用快速脚本
python run_tests.py

# 或使用原来的测试脚本
python test_agent.py
```

## 项目结构

```
agent/
├── main_graph.py          # 主图定义
├── state.py               # 全局状态定义
├── nodes/
│   └── subagents/
│       ├── supervisor/    # 监督者子图（任务分类）
│       ├── executor/      # 执行者子图
│       └── parallel/       # 并行任务子图
├── utils/
│   ├── llm_factory.py     # LLM工厂（统一LLM创建）
│   └── code_cache_manager.py  # 代码缓存管理
├── tests/                 # 测试用例
└── pyproject.toml         # 项目配置和依赖
```

## 依赖说明

### 核心依赖
- `langgraph` - 状态图框架
- `langchain-core` - LangChain核心库
- `pydantic` - 数据模型验证
- `python-dotenv` - 环境变量管理

### 可选依赖（LLM提供者）
- `langchain-qianfan` - 通义千问（推荐）
- `langchain-anthropic` - Anthropic Claude
- `langchain-openai` - OpenAI GPT

### 开发依赖
- `pytest` - 测试框架
- `pytest-asyncio` - 异步测试支持

## 注意事项

1. **API Key 安全**：
   - 不要将 `.env` 文件提交到 Git
   - `.env` 文件已在 `.gitignore` 中（如果配置了的话）
   - 使用 `.env.example` 作为模板

2. **优先级**：
   - 系统环境变量 > `.env` 文件
   - 如果同时设置了 `DASHSCOPE_API_KEY` 和 `QIANFAN_API_KEY`，优先使用 `DASHSCOPE_API_KEY`

3. **降级方案**：
   - 如果未配置任何 API Key，系统会使用关键字匹配作为降级方案
   - 功能会受限，但基本任务分类仍可工作

## 许可证

MIT License
