# Deep Research & PaperQA 集成总结

## 集成日期
2026-02-09

## 概述
在N3知识检索节点中集成了两个强大的知识检索增强功能：
1. **Deep Research子图**: 提供深度研究和综合分析能力
2. **PaperQA论文检索**: 提供科学文献检索和证据收集能力

---

## 1. PaperQA集成

### 功能描述
PaperQA是一个科学文献检索和证据收集系统，通过以下方式增强知识检索：
- **Tavily搜索**: 从学术网站（PubMed, ArXiv, Nature等）检索相关论文
- **Qdrant向量数据库**: 从本地知识库检索相关领域知识
- **paper-qa处理**: 使用LLM对检索到的文献进行评分和证据提取

### 集成位置
- **文件**: `agent/nodes/subagents/paper_qa/`
- **主要函数**: `safe_paper_pipeline(question, max_papers=8, timeout=120.0)`

### 返回数据
```python
{
    "evidence_text_block": str,  # 格式化的证据文本，可直接注入prompt
    "confidence": float,          # 证据置信度 (0.0-1.0)
    "papers_discovered": int,    # 发现的论文数量
    "papers_indexed": int,       # 成功索引的论文数量
    "sources": List[str],        # 数据源列表，如 ["tavily", "qdrant"]
    "evidence_items": List[dict], # 详细的证据项列表
    "answer": str,               # 初步答案（如果paper-qa可用）
    "references": str,           # 参考文献列表
    "cost": float                # 处理成本
}
```

### 使用场景
- 需要最新研究文献的问题
- 需要多源证据支持的问题
- 复杂机制解释问题
- 专业算法问题

---

## 2. Deep Research集成

### 功能描述
Deep Research是一个深度研究分析系统，通过以下方式增强知识检索：
- **研究规划**: 将复杂问题分解为多个子研究任务
- **并行研究**: 使用多个研究者并行执行研究任务
- **综合分析**: 整合所有研究结果生成综合报告

### 集成位置
- **文件**: `agent/nodes/subagents/deep_research/deep_researcher.py`
- **主要函数**: `run_deep_research(question, return_full_state=False, **config_overrides)`

### 返回数据
```python
{
    "final_report": str,      # 最终研究报告
    "research_brief": str,    # 研究摘要
    "message_count": int,    # 消息数量
    "thread_id": str         # 线程ID
}
```

### 触发条件
Deep Research在以下情况下会被触发：
1. PaperQA置信度 < 0.5（证据不足）
2. 核心领域数量 > 2（多领域复杂问题）
3. 问题类型为 "Mechanism Explanation" 或 "Professional Algorithm"

### 使用场景
- 需要深度分析的问题
- 多领域交叉问题
- 需要综合多个研究视角的问题

---

## 3. N3节点增强

### 工作流程

```
N3: Cross-Domain Knowledge Retrieval
├── Step 1: PaperQA Literature Retrieval
│   ├── 从Tavily和Qdrant检索相关文献
│   ├── 使用paper-qa进行证据评分
│   └── 生成格式化的证据文本
│
├── Step 2: Deep Research (条件触发)
│   ├── 判断是否需要深度研究
│   ├── 执行深度研究分析
│   └── 生成综合研究报告
│
├── Step 3: Tool Selection
│   ├── 基于关键词选择工具
│   ├── 基于领域选择工具
│   └── 合并并去重
│
├── Step 4: Build Enhanced Prompt
│   ├── 基础prompt
│   ├── PaperQA证据
│   ├── Deep Research报告
│   └── 工具使用指令
│
└── Step 5: Execute with Tools
    ├── LLM调用
    ├── 工具执行
    └── 结果整合
```

### 代码实现

```python
def n3_knowledge_retrieval_node(state: GeneralQAState) -> GeneralQAState:
    # Step 1: PaperQA Literature Retrieval
    paper_evidence = ""
    paper_confidence = 0.0
    # ... PaperQA调用 ...
    
    # Step 2: Deep Research (if needed)
    deep_research_result = ""
    # ... Deep Research调用 ...
    
    # Step 3: Load and select tools
    tools = []
    # ... 工具选择逻辑 ...
    
    # Step 4: Build enhanced prompt
    prompt = base_prompt + external_knowledge + tool_instruction
    
    # Step 5: Execute with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=5)
    # ... 结果处理 ...
```

---

## 4. 异步处理

### 问题
N3节点函数是同步的，但PaperQA和Deep Research都是异步函数。

### 解决方案
使用 `ThreadPoolExecutor` 在新的事件循环中运行异步函数：

```python
def run_paper_pipeline():
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        return new_loop.run_until_complete(safe_paper_pipeline(...))
    finally:
        new_loop.close()

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(run_paper_pipeline)
    paper_result = future.result(timeout=130.0)
```

### 超时设置
- **PaperQA**: 130秒超时（内部120秒）
- **Deep Research**: 300秒超时

---

## 5. 知识有效性增强

### 逻辑
如果PaperQA或Deep Research返回了有效结果，即使工具调用返回"Missing"，也会将知识有效性标记为"Valid"：

```python
if paper_evidence or deep_research_result:
    if state.knowledge_validity_label == "Missing":
        state.knowledge_validity_label = "Valid"
        print(f"  ✓ External knowledge sources improved knowledge validity")
```

---

## 6. 协同工作策略

### 优先级
1. **PaperQA** (快速，针对性强)
   - 首先执行，获取文献证据
   - 如果置信度高，可能不需要Deep Research

2. **Deep Research** (深度，全面)
   - 条件触发，用于复杂问题
   - 提供更全面的研究视角

3. **工具调用** (精确，结构化)
   - 与外部知识源协同工作
   - 提供结构化的数据库查询结果

### 信息整合
所有三个来源的信息都会被整合到prompt中：
- PaperQA证据 → `paper_evidence`
- Deep Research报告 → `deep_research_result`
- 工具调用结果 → 通过LLM工具调用机制

---

## 7. 配置要求

### 环境变量

#### PaperQA
- `TAVILY_API_KEY`: Tavily搜索API密钥
- `QDRANT_HOST`: Qdrant向量数据库主机
- `QDRANT_PORT`: Qdrant端口（默认6333）
- `QDRANT_COLLECTION`: 集合名称（默认"Immunology"）
- `EMBEDDING_PROVIDER`: 嵌入模型提供商（"openai"或"ollama"）
- `EMBEDDING_MODEL`: 嵌入模型名称
- `EMBEDDING_API_KEY`: 嵌入API密钥

#### Deep Research
- `RESEARCH_MODEL`: 研究模型（默认"deepseek:deepseek-chat"）
- `FINAL_REPORT_MODEL`: 最终报告模型（默认"deepseek:deepseek-reasoner"）
- `SEARCH_API`: 搜索API（"tavily"或其他）
- `MAX_RESEARCHER_ITERATIONS`: 最大研究迭代次数（默认4）
- `MAX_CONCURRENT_RESEARCH_UNITS`: 最大并发研究单元（默认3）

### 依赖包
```bash
# PaperQA
pip install tavily-python
pip install qdrant-client
pip install langchain-qdrant
pip install paperqa  # 可选，用于增强证据处理

# Deep Research
# 依赖已在deep_research模块中定义
```

---

## 8. 性能考虑

### 执行时间
- **PaperQA**: ~30-120秒（取决于网络和论文数量）
- **Deep Research**: ~60-300秒（取决于问题复杂度和迭代次数）
- **工具调用**: ~5-30秒（取决于工具数量和调用次数）

### 优化策略
1. **并行执行**: PaperQA和Deep Research可以并行执行（如果都触发）
2. **条件触发**: Deep Research只在需要时触发
3. **超时控制**: 设置合理的超时时间，避免长时间等待
4. **缓存**: PaperQA支持缓存，避免重复索引相同论文

---

## 9. 错误处理

### 优雅降级
- 如果PaperQA失败，继续使用工具调用
- 如果Deep Research失败，继续使用PaperQA和工具调用
- 如果两者都失败，仅使用工具调用

### 错误日志
所有错误都会被记录但不中断流程：
```python
except Exception as e:
    print(f"  ⚠ PaperQA retrieval failed: {e}")
    # 继续执行，不影响其他功能
```

---

## 10. 测试建议

### 测试场景
1. **简单问题**（仅工具调用）
   - 验证PaperQA和Deep Research不会被不必要地触发

2. **中等复杂问题**（PaperQA + 工具调用）
   - 验证PaperQA能正确检索文献
   - 验证证据能正确整合到prompt中

3. **复杂问题**（PaperQA + Deep Research + 工具调用）
   - 验证Deep Research能被正确触发
   - 验证所有三个来源的信息能正确整合

4. **超时场景**
   - 验证超时机制能正常工作
   - 验证超时后系统能继续执行

### 验证点
- PaperQA返回的证据格式正确
- Deep Research返回的报告格式正确
- 外部知识源能提升知识有效性
- 异步调用不会阻塞主流程
- 错误处理不会中断整个流程

---

## 11. 未来优化方向

1. **智能触发**
   - 基于问题复杂度自动决定是否触发Deep Research
   - 基于PaperQA置信度动态调整Deep Research参数

2. **结果缓存**
   - 缓存PaperQA结果，避免重复检索
   - 缓存Deep Research结果，提高响应速度

3. **并行优化**
   - 真正并行执行PaperQA和Deep Research
   - 优化工具调用的并行度

4. **结果融合**
   - 更智能的证据融合策略
   - 基于置信度的加权融合

---

## 总结

本次集成成功地将PaperQA和Deep Research两个强大的知识检索系统整合到N3节点中，显著增强了系统的知识检索能力：

✅ **PaperQA**: 提供快速、精准的文献检索和证据收集
✅ **Deep Research**: 提供深度、全面的研究分析
✅ **工具调用**: 提供结构化、精确的数据库查询
✅ **协同工作**: 三个来源的信息无缝整合，相互补充

系统现在能够：
- 从多个来源获取知识（文献、研究、数据库）
- 根据问题复杂度智能选择检索策略
- 优雅处理错误和超时
- 提供更准确、更全面的知识检索结果

