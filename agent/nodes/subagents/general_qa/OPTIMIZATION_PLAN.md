# General QA Prompt 优化方案

## 一、优化目标

1. **领域适配的 Prompt**：不同领域使用专门的 prompt，承载领域独属的规则和优化
2. **Prompt 模块化**：将 `prompt.py` 拆分为 `prompts/` 文件夹，按领域组织
3. **工具智能分配**：基于领域和问题类型，更精准地分配工具
4. **保持节点结构**：沿用现有的12节点架构，通过领域路由实现适配

## 二、领域划分

基于 `csv_questions_data.json` 中的 `raw_subject` 和 `question_type`，识别出以下领域：

### 2.1 主要领域（基于 raw_subject）
- **Genetics** - 遗传学
- **Genomics** - 基因组学
- **Immunology** - 免疫学
- **Biochemistry** - 生物化学
- **Molecular Biology** - 分子生物学
- **Bioinformatics** - 生物信息学
- **Computational Biology** - 计算生物学
- **Clinical Medicine** - 临床医学
- **Microbiology** - 微生物学
- **Biophysics** - 生物物理学
- **Neuroscience** - 神经科学
- **Pathology** - 病理学
- **Pharmacy** - 药学
- **Physiology** - 生理学
- **Anatomy** - 解剖学
- **Ecology** - 生态学
- **Public Health** - 公共卫生
- **Bioengineering** - 生物工程
- **Biology** - 生物学（通用）

### 2.2 问题类型（基于 question_type）
- **genetics_genomics** - 遗传学/基因组学
- **bioinformatics** - 生物信息学
- **clinical_medicine** - 临床医学
- **protein_structure** - 蛋白质结构
- **signaling_pathway** - 信号通路
- **vdj_bcr_tcr** - V(D)J/BCR/TCR
- **immune_cells** - 免疫细胞
- **antibody** - 抗体
- **mhc_binding** - MHC结合
- **microbiology** - 微生物学
- **mechanistic_reasoning** - 机制推理
- **general_biomedical** - 通用生物医学

## 三、架构设计

### 3.1 目录结构

```
agent/nodes/subagents/general_qa/
├── prompt.py                    # 保留为统一入口，负责路由
├── prompts/                     # 新建：领域特定的prompt文件夹
│   ├── __init__.py             # 导出所有prompt模块
│   ├── base.py                 # 基础prompt模板和通用函数
│   ├── prompt-genetics.py      # 遗传学领域
│   ├── prompt-genomics.py      # 基因组学领域
│   ├── prompt-immunology.py    # 免疫学领域
│   ├── prompt-biochemistry.py  # 生物化学领域
│   ├── prompt-molecular_biology.py  # 分子生物学
│   ├── prompt-bioinformatics.py     # 生物信息学
│   ├── prompt-clinical_medicine.py  # 临床医学
│   ├── prompt-microbiology.py       # 微生物学
│   ├── prompt-biophysics.py         # 生物物理学
│   ├── prompt-neuroscience.py       # 神经科学
│   ├── prompt-pathology.py          # 病理学
│   ├── prompt-pharmacy.py           # 药学
│   ├── prompt-physiology.py         # 生理学
│   ├── prompt-general.py            # 通用领域（默认）
│   └── domain_mapper.py              # 领域映射和路由逻辑
├── tools/
│   ├── tool_loader.py          # 工具加载（需增强领域适配）
│   ├── tool_trigger.py         # 工具触发（需增强领域适配）
│   └── ...
└── graph.py                     # 节点实现（需适配新的prompt路由）
```

### 3.2 Prompt 路由机制

**设计思路**：
1. `prompt.py` 作为统一入口，根据 `state.core_domains` 或 `state.question_type_label` 路由到对应的领域prompt模块
2. 每个领域prompt模块实现所有12个节点的prompt函数
3. 如果领域未匹配，使用 `prompt-general.py` 作为默认

**路由逻辑**：
```python
# prompt.py (统一入口)
def get_input_preprocessing_prompt(user_input: str, domain: str = None) -> str:
    """根据领域路由到对应的prompt"""
    domain_module = domain_mapper.get_prompt_module(domain)
    return domain_module.get_input_preprocessing_prompt(user_input)
```

### 3.3 领域特定 Prompt 结构

每个领域prompt文件（如 `prompt-genetics.py`）包含：

```python
"""
Genetics Domain-Specific Prompts
Contains all 12 node prompts optimized for genetics questions
"""

# 领域特定规则和约束
DOMAIN_SPECIFIC_RULES = {
    "calculation_focus": ["Hardy-Weinberg equilibrium", "genetic linkage", "recombination frequency"],
    "knowledge_sources": ["OMIM", "GWAS Catalog", "Genebass"],
    "common_entities": ["allele", "genotype", "phenotype", "locus", "haplotype"],
    "validation_criteria": ["Must verify against population genetics principles", "Check HWE assumptions"]
}

# 领域特定工具偏好
DOMAIN_TOOLS = [
    "query_gwas_catalog",
    "query_genebass", 
    "query_variant",
    "query_omim",
    "query_disgenet"
]

# 实现所有12个节点的prompt函数
def get_input_preprocessing_prompt(user_input: str) -> str:
    """N0 prompt with genetics-specific enhancements"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    genetics_enhancements = """
    **Genetics-Specific Rules:**
    - Pay special attention to inheritance patterns (autosomal, X-linked, etc.)
    - Extract genotype/phenotype relationships explicitly
    - Identify population genetics parameters (HWE, Fst, etc.)
    """
    return base_prompt + genetics_enhancements

# ... 其他11个节点的prompt函数
```

## 四、具体优化点

### 4.1 N0: Input Preprocessing
**领域特定优化**：
- **Genetics**: 强调基因型/表型提取、遗传模式识别、群体遗传参数
- **Immunology**: 强调细胞类型、受体、抗原-抗体关系
- **Clinical Medicine**: 强调患者特征、诊断标准、治疗指南
- **Bioinformatics**: 强调算法参数、数据格式、计算约束

### 4.2 N1: Question Decomposition
**领域特定优化**：
- **Genetics**: 分解为遗传模式分析、连锁分析、群体遗传计算
- **Immunology**: 分解为免疫细胞功能、信号转导、抗原识别
- **Clinical Medicine**: 分解为诊断流程、治疗方案选择、药物相互作用

### 4.3 N3: Knowledge Retrieval
**领域特定优化**：
- **Genetics**: 优先使用 GWAS Catalog, Genebass, OMIM
- **Immunology**: 优先使用 TCR/BCR工具、细胞标记工具
- **Clinical Medicine**: 优先使用药物工具、疾病-基因关联工具

### 4.4 N6/N7: Inference
**领域特定优化**：
- **Genetics**: 强调遗传逻辑链、孟德尔定律应用
- **Immunology**: 强调免疫机制推理、细胞-分子相互作用
- **Clinical Medicine**: 强调临床决策树、指南遵循

### 4.5 N8: Answer Generation
**领域特定优化**：
- **Genetics**: 答案格式包含基因型表示、遗传概率
- **Immunology**: 答案格式包含细胞类型、受体名称
- **Clinical Medicine**: 答案格式包含药物名称、剂量、治疗方案

## 五、工具分配优化

### 5.1 领域到工具的映射增强

在 `tool_trigger.py` 中增强 `DOMAIN_TO_TOOLS`：

```python
DOMAIN_TO_TOOLS = {
    # 现有映射...
    
    # 新增：基于题目数据的领域映射
    "Genetics": [
        "query_gwas_catalog", "query_genebass", "query_variant",
        "query_omim", "query_disgenet", "query_gene_info"
    ],
    "Genomics": [
        "query_variant", "query_gwas_catalog", "query_genebass",
        "query_gene_info", "query_knowledge_graph"
    ],
    "Immunology": [
        "query_tcr_mcpas", "query_celltype_marker", "query_ppi",
        "query_knowledge_graph", "query_proteinatlas"
    ],
    "Clinical Medicine": [
        "query_drug_interaction", "query_drug_for_disease",
        "query_disease_for_drug", "query_omim", "query_disgenet",
        "query_hpo_term"
    ],
    "Bioinformatics": [
        "query_variant", "query_gwas_catalog", "query_knowledge_graph",
        "query_gene_info", "query_go_term"
    ],
    # ... 其他领域
}
```

### 5.2 节点级别的工具分配增强

在 `tool_loader.py` 的 `get_tools_for_node` 中，根据领域动态调整：

```python
def get_tools_for_node(node_name: str, domain: str = None) -> List[StructuredTool]:
    """根据节点和领域返回工具"""
    base_tools = get_base_tools_for_node(node_name)
    
    if domain:
        domain_tools = DOMAIN_TO_TOOLS.get(domain, [])
        # 合并并去重
        return merge_tools(base_tools, domain_tools)
    
    return base_tools
```

## 六、实施步骤

### Phase 1: 基础架构搭建
1. 创建 `prompts/` 文件夹
2. 创建 `prompts/base.py` 包含通用prompt模板
3. 创建 `prompts/domain_mapper.py` 实现领域路由
4. 创建 `prompts/prompt-general.py` 作为默认实现

### Phase 2: 核心领域实现
优先实现高频领域（基于题目数据统计）：
1. **Genetics** - 最高优先级
2. **Immunology** - 高优先级
3. **Clinical Medicine** - 高优先级
4. **Bioinformatics** - 中优先级
5. **Biochemistry** - 中优先级

### Phase 3: 工具分配优化
1. 增强 `tool_trigger.py` 的领域映射
2. 修改 `tool_loader.py` 支持领域参数
3. 在 `graph.py` 中传递领域信息到工具分配

### Phase 4: 其他领域实现
逐步实现剩余领域，每个领域独立测试

### Phase 5: 测试和优化
1. 使用 `csv_questions_data.json` 进行测试
2. 对比优化前后的效果
3. 根据测试结果调整prompt和工具分配

## 七、注意事项

1. **向后兼容**：保持 `prompt.py` 作为统一入口，现有代码无需大幅修改
2. **默认行为**：未匹配领域时使用 `prompt-general.py`，确保系统稳定
3. **渐进式迁移**：先实现核心领域，再逐步扩展
4. **英文提示词**：所有prompt和文案必须使用英文
5. **不引用答案**：prompt中不能包含题目集锦中的参考答案

## 八、跨领域问题处理机制

### 8.1 跨领域识别

当问题涉及多个领域时（如 `state.core_domains` 包含多个领域），系统应：

1. **识别跨领域场景**：
   - 检测 `core_domains` 是否包含多个领域
   - 检测 `question_type` 是否映射到多个领域
   - 检测问题文本中是否包含多个领域的核心实体

2. **触发跨领域处理**：
   - 使用 `prompt-cross_domain.py` 模块
   - 融合所有涉及领域的规则和工具
   - 合并各领域的实体提取规则

### 8.2 跨领域Prompt设计

```python
# prompts/prompt-cross_domain.py
def get_input_preprocessing_prompt(user_input: str, domains: List[str]) -> str:
    """跨领域问题的N0 prompt，融合多个领域的规则"""
    base_prompt = get_base_input_preprocessing_prompt(user_input)
    
    # 收集所有涉及领域的增强规则
    domain_enhancements = []
    for domain in domains:
        domain_module = get_prompt_module(domain=domain)
        if hasattr(domain_module, 'get_domain_extraction_rules'):
            domain_enhancements.append(domain_module.get_domain_extraction_rules())
    
    # 合并所有领域的规则
    merged_enhancements = "\n\n".join(domain_enhancements)
    return base_prompt + "\n\n**Multi-Domain Rules (merged):**\n" + merged_enhancements
```

### 8.3 跨领域工具分配

- 合并所有涉及领域的优先级工具
- 去重后按领域优先级排序
- 确保工具调用覆盖所有相关领域

## 九、计算类问题优化

### 9.1 计算类通用模板

在 `prompts/base.py` 中新增计算类问题的通用指导：

```python
def get_calculation_guide() -> str:
    """通用计算步骤指导，所有领域复用"""
    return """
**Calculation Step Guide (Universal):**
1. **Parameter Extraction**: Extract all key parameters (e.g., allele frequency, sample size, concentration) and confirm their validity
   - Verify parameter units and ranges (e.g., frequency ∈ [0,1], concentration > 0)
   - Identify missing parameters and mark them explicitly

2. **Formula Selection**: Select appropriate formula based on domain and question type
   - For population genetics: Hardy-Weinberg equation (p²+2pq+q²=1)
   - For concentration: C1V1 = C2V2
   - For statistical tests: Chi-square, t-test formulas
   - Verify formula applicability conditions

3. **Step-by-Step Calculation**: Perform calculation showing each intermediate step
   - Show substitution of values into formula
   - Calculate intermediate results
   - Track units throughout calculation

4. **Result Verification**: Verify result conforms to domain constraints
   - Check result is within expected range (e.g., probability ≤ 1, frequency ∈ [0,1])
   - Verify units are correct
   - Check result makes biological sense
   - Compare with critical/reference values if applicable
"""
```

### 9.2 领域特定计算规则

各领域在对应节点Prompt中引入该模板，并添加领域特定的计算规则：

- **Genetics**: HWE计算、重组频率、遗传概率
- **Biochemistry**: 浓度计算、分子量、反应平衡
- **Bioinformatics**: 算法参数、统计检验、序列分析

## 十、新增领域支持体系

### 10.1 领域Prompt模板生成工具

创建 `scripts/generate_domain_prompt.py`：

```python
"""
自动生成领域Prompt模板
用法: python generate_domain_prompt.py --domain Immunology --entities "T cell,B cell,antibody" --tools "query_tcr_mcpas,query_celltype_marker"
"""

def generate_domain_prompt_template(domain: str, entities: List[str], tools: List[str], calculation_focus: List[str] = None):
    """生成包含12个节点函数框架的prompt文件"""
    # 读取模板
    template = read_template("domain_prompt_template.py.j2")
    
    # 填充领域特定信息
    content = template.render(
        domain=domain,
        entities=entities,
        tools=tools,
        calculation_focus=calculation_focus or []
    )
    
    # 写入文件
    write_file(f"prompts/prompt-{domain.lower()}.py", content)
```

### 10.2 领域模块验收清单

制定《领域模块验收清单》，要求新增领域必须满足：

1. **完整性要求**：
   - ✅ 实现所有12个节点的Prompt函数
   - ✅ 包含 `DOMAIN_CONFIG` 配置（工具列表、实体列表、验证标准）
   - ✅ 实现 `get_domain_tools()` 函数

2. **质量要求**：
   - ✅ 包含领域专属规则（至少3个节点的详细增强）
   - ✅ 包含计算类问题的领域特定规则（如适用）
   - ✅ 包含验证标准（validation_criteria）

3. **测试要求**：
   - ✅ 配套≥5个测试用例（覆盖不同问题类型）
   - ✅ 测试用例覆盖该领域的主要场景
   - ✅ 通过单元测试和集成测试

## 十一、节点-工具调用规则

### 11.1 节点工具使用映射

在 `tools/tool_loader.py` 中明确节点-工具的调用规则：

```python
# 节点是否支持工具调用
NODE_TOOL_USAGE = {
    "n0_input_preprocessing": False,  # 不调用工具，纯文本处理
    "n1_question_decomposition": False,  # 不调用工具，问题分解
    "n2_calculation_algorithm_recognition": False,  # 不调用工具，算法识别
    "n3_knowledge_retrieval": True,  # 必须调用工具，知识检索
    "n4_calculation_decomposition": True,  # 可选调用工具，获取计算参数
    "n5_algorithm_validation": True,  # 可选调用工具，验证算法适用性
    "n6_initial_inference": False,  # 不调用工具，基于已有知识推理
    "n7_complete_inference": False,  # 不调用工具，完整逻辑推理
    "n8_answer_generation": False,  # 不调用工具，答案生成
    "n9_result_validation": True,  # 可选调用工具，验证结果
    "n10_exception_handling": True,  # 可选调用工具，异常处理
    "n11_manual_intervention": False,  # 不调用工具，人工介入
}

def get_tools_for_node(node_name: str, domain: Optional[str] = None, question_type: Optional[str] = None) -> List[StructuredTool]:
    """根据节点和领域返回工具，遵循节点工具使用规则"""
    # 检查节点是否支持工具调用
    if not NODE_TOOL_USAGE.get(node_name, False):
        return []  # 不支持工具调用的节点返回空列表
    
    # 获取基础工具
    base_tools = get_base_tools_for_node(node_name)
    
    # 如果指定了领域，获取领域特定的工具
    if domain or question_type:
        domain_module = domain_mapper.get_prompt_module(domain=domain, question_type=question_type)
        if hasattr(domain_module, 'get_domain_tools'):
            domain_tool_names = domain_module.get_domain_tools()
            all_tools = load_all_tools()
            tool_map = {tool.name: tool for tool in all_tools}
            domain_tools = [tool_map[name] for name in domain_tool_names if name in tool_map]
            
            # 合并工具，领域工具优先
            merged_tools = domain_tools + [t for t in base_tools if t not in domain_tools]
            return merged_tools
    
    return base_tools
```

## 十二、测试策略（量化+全覆盖）

### 12.1 量化测试指标

1. **准确率指标**：
   - 核心领域问题答案正确率 ≥ 90%
   - 跨领域问题答案正确率 ≥ 80%
   - 计算类问题计算准确率 ≥ 95%

2. **工具调用指标**：
   - 领域工具调用准确率 ≥ 95%（正确识别需要调用的工具）
   - 无效工具调用率 ≤ 5%（不应调用工具时未调用）
   - 工具调用覆盖率 ≥ 90%（需要工具时成功调用）

3. **性能指标**：
   - 单问题处理延迟 ≤ 500ms（缓存生效后）
   - 领域路由延迟 ≤ 10ms
   - 模块加载延迟 ≤ 50ms（首次加载）

### 12.2 测试覆盖范围

1. **正常场景**：
   - 核心领域问题（Genetics, Immunology, Clinical Medicine）
   - 跨领域问题（Genetics + Bioinformatics）
   - 计算类问题（HWE, 浓度计算）
   - 不同问题类型（Multiple Choice, Text Matching, Calculation）

2. **边缘场景**：
   - 低资源领域（Ecology, Public Health）
   - 复杂计算问题（多步骤遗传计算）
   - 罕见问题类型（Code-Command, Procedure）

3. **异常场景**：
   - 领域模块缺失（优雅降级到general）
   - 工具调用失败（错误处理机制）
   - 跨领域问题（多领域规则融合）
   - 领域识别失败（使用默认general）

### 12.3 测试用例要求

每个领域模块必须包含：
- ≥5个测试用例
- 覆盖该领域的主要问题类型
- 包含至少1个计算类问题（如适用）
- 包含至少1个跨领域问题（如适用）

## 十三、预期效果

1. **准确性提升**：领域特定prompt能更精准地提取信息和推理
   - 核心领域问题准确率提升 ≥ 15%
   - 计算类问题准确率提升 ≥ 20%

2. **工具使用优化**：领域适配的工具分配减少无关工具调用
   - 无效工具调用率降低 ≥ 50%
   - 工具调用准确率提升 ≥ 20%

3. **可维护性提升**：模块化的prompt结构便于维护和扩展
   - 新增领域开发时间减少 ≥ 60%
   - 领域模块质量统一性提升

4. **扩展性增强**：新增领域只需添加新的prompt文件
   - 支持跨领域问题处理
   - 支持计算类问题统一优化

