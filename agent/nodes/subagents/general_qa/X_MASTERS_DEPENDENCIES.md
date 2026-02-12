# X-Masters 融合优化 - 依赖安装说明

## 新增依赖

为了将 X-Masters 的多解验证机制融合进 general_qa，需要安装以下新依赖：

### 必需依赖

1. **tavily-python** - 用于 X-Masters 的 web_search 工具
   ```bash
   pip install tavily-python
   ```

2. **python-dotenv** - 用于加载环境变量配置（如果尚未安装）
   ```bash
   pip install python-dotenv
   ```

### 环境变量配置

需要在 `.env` 文件中配置以下变量：

1. **TAVILY_API_KEY** - Tavily API 密钥（用于 web_search）
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

2. **QDRANT_HOST** 和 **QDRANT_PORT** - Qdrant 向量数据库配置（用于 knowledge_search）
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

3. **QDRANT_COLLECTIONS** - Qdrant 集合名称（多个集合用逗号分隔）
   ```
   QDRANT_COLLECTIONS=collection1,collection2
   ```

### 已存在的依赖（无需额外安装）

以下依赖应该已经在项目中存在，无需额外安装：

- `langgraph` - 图工作流框架
- `langchain-core` - LangChain 核心库
- `pydantic` - 数据验证库
- 其他生物医学数据库工具依赖（通过 `data_lake` 模块提供）

### 安装命令汇总

```bash
# 安装新依赖
pip install tavily-python python-dotenv

# 如果使用 requirements.txt，添加以下行：
# tavily-python>=0.3.0
# python-dotenv>=1.0.0
```

### 验证安装

安装完成后，可以通过以下方式验证：

```python
# 验证 tavily
try:
    from tavily import TavilyClient
    print("✓ tavily-python installed")
except ImportError:
    print("✗ tavily-python not installed")

# 验证 dotenv
try:
    from dotenv import load_dotenv
    print("✓ python-dotenv installed")
except ImportError:
    print("✗ python-dotenv not installed")
```

### 注意事项

1. **Tavily API Key**: 需要注册 Tavily 账号获取 API 密钥
   - 访问: https://tavily.com/
   - 注册账号并获取 API key

2. **Qdrant 配置**: 如果项目已经配置了 Qdrant，则无需修改
   - 确保 Qdrant 服务正在运行
   - 确保集合已创建并包含数据

3. **可选功能**: 如果不需要 web_search 功能，可以跳过 tavily-python 的安装
   - 但 X-Masters 的 Critic/Rewriter/Selector 仍会尝试使用 web_search
   - 如果未配置，会返回错误信息但不会中断流程

### 故障排除

如果遇到导入错误：

1. **ImportError: No module named 'tavily'**
   - 运行: `pip install tavily-python`

2. **ImportError: No module named 'dotenv'**
   - 运行: `pip install python-dotenv`

3. **CodeActAgent 导入错误**
   - 确保 `agent/result_evaluator/` 目录存在
   - 确保 `agent.py` 和 `executor.py` 在正确位置

4. **Qdrant 连接错误**
   - 检查 QDRANT_HOST 和 QDRANT_PORT 配置
   - 确保 Qdrant 服务正在运行
   - 检查网络连接

