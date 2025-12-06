# Immunology Module Code Tree

本文档展示了 `agent/usecases/immunology` 目录下的完整代码结构树。

## 目录结构

```
immunology/
├── README.md                           # 模块说明文档
├── __init__.py                         # 模块初始化文件
├── constants.py                        # 常量定义
├── deploy_qdrant_papers.py            # Qdrant论文部署脚本
├── download_papers_locally.py         # 本地论文下载脚本
├── enhanced_immune_agent.py           # 增强免疫智能体主文件
├── example_usage.py                   # 使用示例
├── immunology_config.py               # 免疫学配置文件
├── load_local_papers_to_qdrant.py     # 本地论文加载到Qdrant
├── qdrant_deployment.py               # Qdrant部署脚本
├── unified_agent.py                   # 统一智能体
├── test_10_questions.py               # 10个问题测试
├── test_complete_immuneagent.py       # 完整免疫智能体测试
├── test_full_immuneagent.py           # 全功能免疫智能体测试
├── test_performance.py                # 性能测试
├── test_planning_demo.py              # 规划演示测试
├── test_qdrant_complete.py            # Qdrant完整测试
│
├── graph/                              # 图结构模块
│   ├── __init__.py                     # 图模块初始化
│   ├── planning_graph.py               # 规划图
│   └── retrieval_graph.py              # 检索图
│
├── prompts/                            # 提示词模块
│   ├── __init__.py                     # 提示词模块初始化
│   └── immunology_prompts.py           # 免疫学提示词
│
├── state/                              # 状态管理模块
│   ├── __init__.py                     # 状态模块初始化
│   └── state.py                        # 状态定义
│
├── tools/                              # 工具模块
│   ├── __init__.py                     # 工具模块初始化
│   ├── execution_tools.py              # 执行工具
│   ├── full_tool_registry.py           # 完整工具注册表
│   ├── hypothesis_tools.py             # 假设生成工具
│   ├── mcp_tools.py                    # MCP工具
│   ├── planning_tools.py               # 规划工具
│   ├── qdrant_integration.py           # Qdrant集成
│   ├── retrieval_tools.py              # 检索工具
│   └── scanpy_tools.py                 # Scanpy分析工具
│
└── utils/                              # 工具函数模块
    ├── __init__.py                     # 工具函数模块初始化
    └── helpers.py                      # 辅助函数
```

## 模块说明

### 核心文件
- **enhanced_immune_agent.py**: 增强免疫智能体的主要实现，包含完整的分析流程
- **unified_agent.py**: 统一智能体接口
- **immunology_config.py**: 免疫学相关的配置参数

### 图结构 (graph/)
- **planning_graph.py**: 实现研究规划的图结构
- **retrieval_graph.py**: 实现信息检索的图结构

### 工具集 (tools/)
- **hypothesis_tools.py**: 科学假设生成和管理工具
- **planning_tools.py**: 研究计划制定工具
- **execution_tools.py**: 实验执行工具
- **retrieval_tools.py**: 文献检索工具
- **qdrant_integration.py**: 向量数据库集成
- **scanpy_tools.py**: 单细胞数据分析工具
- **mcp_tools.py**: MCP协议工具

### 状态管理 (state/)
- **state.py**: 定义智能体的状态结构和状态转换

### 提示词 (prompts/)
- **immunology_prompts.py**: 免疫学领域专用的提示词模板

### 测试文件
- **test_*.py**: 各种功能测试脚本，包括性能测试、完整性测试等

### 部署和数据管理
- **deploy_qdrant_papers.py**: 部署论文到Qdrant向量数据库
- **download_papers_locally.py**: 下载论文到本地
- **load_local_papers_to_qdrant.py**: 将本地论文加载到Qdrant

## 文件统计

- **总文件数**: 29个文件
- **Python文件**: 26个 (.py)
- **文档文件**: 1个 (.md)
- **目录数**: 5个子目录

---

*生成时间: 2025年*
*目录路径: D:\PartTimeJob\agent\antibody_gen\agent\usecases\immunology*