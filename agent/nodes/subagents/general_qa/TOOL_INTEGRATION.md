# General QA 工具集成文档

## 概述

本文档说明如何将 `tools/` 目录下的所有生物医学工具集成到 General QA 的各个节点中,使用 LangChain 1.0 和 LangGraph 1.0 的最新特性。

## 工具分类

### 1. 核心查询工具 (Core Query Tools)
- `query_knowledge_graph`: 生物医学知识图谱查询
- `query_tcr_mcpas`: T细胞受体序列查询
- `query_mirdb`: miRNA目标预测查询
- `query_mirtarbase`: 实验验证的miRNA-目标查询
- `query_bindingdb`: 药物-靶点结合亲和力查询
- `query_gtex_expression`: 组织基因表达查询
- `query_sgrna_human/mouse`: CRISPR sgRNA设计查询
- `query_genetic_interaction`: 遗传相互作用查询
- `query_variant`: 遗传变异查询

### 2. 疾病/基因工具 (Disease/Gene Tools)
- `query_disgenet`: 疾病-基因关联查询
- `query_omim`: 孟德尔遗传疾病查询
- `query_proteinatlas`: 蛋白质图谱查询
- `query_gene_info`: 基因基本信息查询

### 3. 药物工具 (Drug Tools)
- `query_drug_interaction`: 药物-药物相互作用查询

### 4. 通路工具 (Pathway Tools)
- `query_msigdb`: MSigDB人类基因集查询
- `query_mousemine`: MouseMine小鼠基因集查询

### 5. 本体工具 (Ontology Tools)
- `query_go_term/hierarchy/relations`: 基因本体查询
- `query_hpo_term/hierarchy/xref`: 人类表型本体查询

### 6. 遗传工具 (Genetic Tools)
- `query_gwas_catalog`: GWAS目录查询
- `query_genebass`: 基因负担分析查询

### 7. 相互作用工具 (Interaction Tools)
- `query_ppi`: 蛋白质-蛋白质相互作用查询
- `query_synthetic_interaction`: 合成致死性查询

### 8. 表达工具 (Expression Tools)
- `query_depmap`: DepMap癌症依赖性查询
- `query_celltype_marker`: 细胞类型标记查询
- `query_czi_census`: CZI单细胞数据集查询

### 9. 重定位工具 (Repurposing Tools)
- `query_drug_for_disease`: AI预测的药物-疾病关联
- `query_disease_for_drug`: AI预测的疾病-药物关联

## 节点工具分配

### N0: 输入预处理 (Input Preprocessing)
- **工具**: 无
- **说明**: 此节点仅进行问题分类,不需要工具

### N1: 问题分解 (Question Decomposition)
- **工具**: 疾病/基因工具 + 本体工具(部分)
- **说明**: 用于实体识别和领域定位
- **工具列表**:
  - `query_disgenet`
  - `query_omim`
  - `query_proteinatlas`
  - `query_gene_info`
  - `query_go_term`
  - `query_hpo_term`

### N2: 计算/算法识别 (Calculation/Algorithm Recognition)
- **工具**: 无
- **说明**: 此节点仅识别计算需求,不需要工具

### N3: 知识检索 (Knowledge Retrieval)
- **工具**: **所有工具** (35个工具)
- **说明**: 这是最重要的知识检索节点,需要访问所有生物医学数据库
- **工具列表**: 所有可用工具

### N4: 计算分解 (Calculation Decomposition)
- **工具**: 表达工具 + 遗传工具 + 核心查询工具(部分)
- **说明**: 用于获取计算所需的数据
- **工具列表**:
  - `query_gtex_expression`
  - `query_variant`
  - `query_gwas_catalog`
  - `query_genebass`
  - `query_knowledge_graph`
  - `query_tcr_mcpas`
  - `query_mirdb`

### N5: 算法验证 (Algorithm Validation)
- **工具**: 通路工具 + 相互作用工具 + 疾病/基因工具
- **说明**: 用于验证算法的适用性和参数提取
- **工具列表**:
  - `query_msigdb`
  - `query_mousemine`
  - `query_ppi`
  - `query_synthetic_interaction`
  - `query_disgenet`
  - `query_omim`
  - `query_proteinatlas`
  - `query_gene_info`

### N6: 初始推理 (Initial Inference)
- **工具**: 核心查询工具 + 疾病/基因工具 + 相互作用工具 + 遗传工具
- **说明**: 用于初始关联推理
- **工具列表**:
  - 所有核心查询工具
  - 所有疾病/基因工具
  - 所有相互作用工具
  - 所有遗传工具

### N7: 完整推理 (Complete Inference)
- **工具**: **所有工具** (35个工具)
- **说明**: 完整逻辑推理需要访问所有知识源
- **工具列表**: 所有可用工具

### N8: 答案生成 (Answer Generation)
- **工具**: 疾病/基因工具 + 本体工具 + 药物工具
- **说明**: 用于答案细化和验证
- **工具列表**:
  - `query_disgenet`
  - `query_omim`
  - `query_proteinatlas`
  - `query_gene_info`
  - `query_go_term`
  - `query_hpo_term`
  - `query_drug_interaction`

### N9: 结果验证 (Result Validation)
- **工具**: 疾病/基因工具(部分) + 本体工具(部分)
- **说明**: 用于结果验证
- **工具列表**:
  - `query_disgenet`
  - `query_omim`
  - `query_go_term`
  - `query_hpo_term`

### N10: 异常处理 (Exception Handling)
- **工具**: **所有工具** (35个工具)
- **说明**: 当出现异常时,需要访问所有工具寻找替代方案
- **工具列表**: 所有可用工具

### N11: 人工干预 (Manual Intervention)
- **工具**: 无
- **说明**: 此节点生成人工干预指南,不需要工具

## 技术实现

### 1. 工具加载器 (`tools/tool_loader.py`)

创建了统一的工具加载器,提供以下功能:

- `load_all_tools()`: 加载所有工具为 LangChain StructuredTool 对象
- `get_tools_by_category()`: 按类别组织工具
- `get_tools_for_node(node_name)`: 根据节点名称获取合适的工具列表

### 2. LLM工具调用 (`graph.py`)

修改了 `_call_llm()` 函数,支持工具调用:

- 使用 `.bind_tools()` 绑定工具到 LLM
- 自动检测和处理 `tool_calls`
- 执行工具调用并创建 `ToolMessage`
- 支持多轮工具调用迭代(最多5轮)
- 自动处理工具执行错误

### 3. LangChain 1.0 特性

使用了以下 LangChain 1.0 最新特性:

- **StructuredTool**: 使用结构化工具定义,支持 Pydantic 模式验证
- **bind_tools()**: 使用新的工具绑定方法
- **ToolMessage**: 使用标准化的工具消息格式
- **自动工具调用**: LLM 自动决定何时调用工具

### 4. LangGraph 1.0 特性

- **状态管理**: 工具调用结果自动集成到状态中
- **节点路由**: 根据工具调用结果决定下一步流程
- **错误处理**: 工具执行错误不会中断整个流程

## 使用示例

### 基本使用

工具会自动绑定到相应的节点,无需手动配置:

```python
# 节点会自动加载和使用工具
state = n3_knowledge_retrieval_node(state)
# LLM 会自动调用相关工具进行知识检索
```

### 工具调用流程

1. LLM 接收提示和可用工具列表
2. LLM 决定是否需要调用工具
3. 如果调用工具,执行工具并获取结果
4. 将工具结果添加到消息历史
5. LLM 基于工具结果生成最终响应
6. 重复步骤2-5直到不再需要工具调用

## 工具调用日志

工具调用会输出详细日志:

```
N3: Cross-Domain Knowledge Retrieval
  📚 Loaded 35 tool(s) for knowledge retrieval
  🔧 Bound 35 tool(s) to LLM
  🔧 Executing 2 tool call(s) (iteration 1/5)
    ✓ query_disgenet executed successfully
    ✓ query_gwas_catalog executed successfully
```

## 注意事项

1. **工具可用性**: 确保数据库连接正常,工具才能正常工作
2. **工具调用限制**: 每个节点最多进行5轮工具调用迭代,防止无限循环
3. **错误处理**: 工具执行失败不会中断整个流程,错误信息会传递给 LLM
4. **性能考虑**: 工具调用会增加响应时间,但提供更准确的结果

## 未来改进

1. **工具选择优化**: 根据问题类型智能选择最相关的工具子集
2. **并行工具调用**: 支持并行执行多个独立的工具调用
3. **工具结果缓存**: 缓存常用查询结果以提高性能
4. **工具使用统计**: 跟踪工具使用情况以优化工具分配

