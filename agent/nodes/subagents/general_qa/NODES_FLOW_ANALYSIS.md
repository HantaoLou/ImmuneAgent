# GeneralQA 子图节点流程分析文档

## 整体架构

GeneralQA 子图包含 12 个节点，分为以下流程路径：

### 主流程路径
1. **推理路径**（非计算题）：N0 → N1 → N3 → N6 → N7 → N8 → N9 → END
2. **计算路径**（数值计算题）：N0 → N2 → N3 → N6 → N4 → N7 → N8 → N9 → END
3. **算法路径**（算法题）：N0 → N2 → N3 → N6 → N5 → N7 → N8 → N9 → END
4. **异常处理路径**：任何节点 → N10 → (重试/人工干预) → N11 → END

---

## 节点详细分析

### N0: Input Preprocessing & Question Classification（输入预处理与问题分类）

#### 功能描述
- 清理和标准化用户输入
- 分类问题类型（Multiple Choice/Text Matching/Mechanism Explanation/Numerical Calculation/Logical Calculation/Professional Algorithm）
- **标准化题型分类**：Calculation-[subcategory] / ClinicalDecision-[subcategory] / ProfessionalKnowledge-[subcategory]
- 提取选项（如果有）
- 确定答案格式（Single Choice/Multi-Select/Numeric/Short Text/Long Text/Sequence/Formula/List/Procedure/Code-Command）
- 评估数据完整性（Complete/Partial Missing/Severe Missing）
- **提取结构化三维度信息**：Subject（主体）、Condition（条件）、Goal（目标）
- **提取核心关键词**和**选项特征**
- **提取约束信息**：negative_constraints（否定约束）、exclusive_constraints（专属约束）、strong_restrictions（强限制）、key_constraints（关键约束）

#### 输入要求
- `user_input`: 用户原始输入（必需）

#### 处理逻辑
1. 验证输入非空
2. 调用 LLM 进行预处理和分类
3. 解析 JSON 响应，提取所有字段
4. 自动检测约束（如果 LLM 未提取）
5. 验证结构化三维度信息完整性
6. 如果缺少任何维度或子字段，标记为 "Severe Missing"

#### 输出结果
- `cleaned_text`: 清理后的文本
- `question_type_label`: 问题类型标签
- `question_category_standard`: 标准化题型分类（新增）
- `category_specific_constraints`: 题型专属约束（新增）
- `data_completeness_label`: 数据完整性标签
- `question_options`: 选项列表
- `answer_format_label`: 答案格式标签
- `core_keywords`: 核心关键词列表（新增）
- `option_features`: 选项特征字典（新增）
- `structured_subject`: 结构化主体信息
- `structured_condition`: 结构化条件信息（包含 hard_constraints）
- `structured_goal`: 结构化目标信息
- `negative_constraints`: 否定约束列表
- `exclusive_constraints`: 专属约束列表
- `strong_restrictions`: 强限制列表
- `key_constraints`: 关键约束列表

#### 路由逻辑（route_after_n0）
- 如果 `data_completeness_label == "Severe Missing"` → N10（异常处理）
- 如果 `question_type_label == "Numerical Calculation"` → N2（计算识别）
- 否则 → N1（问题拆解）

#### 工具绑定
- **无工具绑定**

#### 关键优化点
- ✅ 标准化题型分类（三类：计算题/临床决策题/专业知识题）
- ✅ 结构化三维度信息提取（Subject/Condition/Goal）
- ✅ 核心关键词和选项特征提取
- ✅ 约束信息自动检测和分类
- ✅ 数据完整性严格验证

---

### N1: Question Decomposition & Domain Localization（问题拆解与领域定位）

#### 功能描述
- 拆解问题为可执行的子目标
- 识别核心领域（细化到细分领域，而非大类）
- 提取关键实体（细化到选项级）
- 提取答案约束（单位、精度、方向、数量）
- **绑定题型专属解题步骤模板**（category_specific_solution_steps）
- **提取推理核心限制**（inference_core_restrictions）：合并 negative/exclusive/strong/key 约束

#### 输入要求
- `cleaned_text`: 清理后的文本（必需）
- `structured_subject`, `structured_condition`, `structured_goal`: 结构化信息（可选但推荐）
- `question_category_standard`: 标准化题型分类（用于绑定解题步骤）

#### 处理逻辑
1. 验证 `cleaned_text` 存在
2. 加载基础实体查询工具（disease_gene + ontology）
3. 调用 LLM 进行问题拆解
4. 解析响应，提取所有字段
5. 合并约束信息到 `inference_core_restrictions`
6. 从 `structured_conditions` 中提取关键约束（critical_constraints）

#### 输出结果
- `structured_conditions`: 结构化条件字典
- `core_domains`: 核心领域列表（细化到细分领域）
- `research_objective`: 研究目标（包含题型专属解题步骤）
- `key_entities`: 关键实体列表（细化到选项级）
- `answer_constraints`: 答案约束列表
- `category_specific_solution_steps`: 题型专属解题步骤模板（新增）
- `critical_constraints`: 关键约束列表
- `inference_core_restrictions`: 推理核心限制（合并所有约束）

#### 路由逻辑
- 固定路由到 N3（知识检索）

#### 工具绑定
- **基础工具**：disease_gene 工具 + ontology 工具（GO、HPO）

#### 关键优化点
- ✅ 领域细化到细分领域（如 "Single-cell TCR Sequencing" 而非 "Immunology"）
- ✅ 实体细化到选项级（如 "Drosophila plasma protein"）
- ✅ 题型专属解题步骤绑定
- ✅ 推理核心限制合并和传递

---

### N2: Calculation/Algorithm Requirement Recognition（计算/算法需求识别）

#### 功能描述
- 识别问题是否需要计算或算法
- 分类计算类型：Numerical（数值计算）/ Logical Calculation（逻辑计算）/ Algorithm（算法）
- 提取关键参数和公式线索
- **关键**：正确识别 "最小集合" 类问题为 Logical Calculation，而非 Numerical

#### 输入要求
- `cleaned_text`: 清理后的文本（必需）
- `question_type_label`: 问题类型标签（用于上下文）

#### 处理逻辑
1. 验证 `cleaned_text` 存在
2. 调用 LLM 进行识别
3. 解析响应，提取计算类型和关键参数

#### 输出结果
- `calculation_type_label`: 计算类型标签（Numerical/Logical Calculation/Algorithm）
- `key_parameters`: 关键参数字典（包含 formula_clues、parameters、algorithm_name 等）

#### 路由逻辑
- 固定路由到 N3（知识检索）

#### 工具绑定
- **无工具绑定**

#### 关键优化点
- ✅ 正确区分 Numerical 和 Logical Calculation
- ✅ 提取算法名称（algorithm_name）用于后续算法验证

---

### N3: Cross-Domain Knowledge Retrieval（跨领域知识检索）

#### 功能描述
- **综合知识检索节点**：使用所有可用工具进行知识检索
- 检索领域知识（domain_knowledge_map）
- **双层知识校验**：相关性校验（≥80%匹配）+ 事实性校验（权威来源）
- **题型专属核心知识召回**：
  - Calculation：公式 + 参数定义 + 临界值
  - ClinicalDecision：最新指南 + 药物禁忌 + 药物类别共性
  - ProfessionalKnowledge：教材级核心概念 + 因果逻辑
- **选项特征检索**：为每个选项检索特定特征知识
- **隐含矛盾解决方案检索**：如果 N0 提取了隐含矛盾，检索解决方案
- **PaperQA 文献检索**（辅助功能，30分钟超时）
- **Deep Research 深度研究**（辅助功能，33分钟超时）

#### 输入要求
- `core_domains`: 核心领域列表（必需）
- `calculation_type_label`: 计算类型（可选）
- `key_entities`: 关键实体列表
- `structured_subject`, `structured_condition`, `structured_goal`: 结构化信息
- `question_category_standard`: 标准化题型分类

#### 处理逻辑
1. 验证 `core_domains` 或 `calculation_type_label` 存在
2. **Step 1**: 运行 PaperQA 文献检索（异步，30分钟超时）
3. **Step 2**: 运行 Deep Research（如果满足条件：paper_confidence < 0.5 或 domains > 2 或复杂问题类型）
4. **Step 3**: 加载所有工具（all_tools）
5. **Step 4**: 基于关键词和领域选择工具（get_tools_by_keywords）
6. **Step 5**: 构建增强 prompt（包含工具使用强制指令）
7. **Step 6**: 调用 LLM（绑定所有工具，max_iterations=5）
8. **Step 7**: 解析响应，提取知识映射

#### 输出结果
- `domain_knowledge_map`: 领域知识映射表
- `knowledge_validity_label`: 知识有效性标签（Valid/Invalid/Missing）
- `knowledge_authority_source`: 知识权威来源标注（新增）
- `paperqa_result`: PaperQA 检索结果（新增）
- `deep_research_result`: Deep Research 结果（新增）

#### 路由逻辑（route_after_n3）
- 如果 `knowledge_validity_label == "Missing"` → N10（异常处理）
- 否则 → N6（初步推理）

#### 工具绑定
- **所有工具**（all_tools）
- **关键词触发工具选择**（get_tools_by_keywords）
- **强制工具使用指令**（在 prompt 中）

#### 关键优化点
- ✅ 双层知识校验（相关性 + 事实性）
- ✅ 题型专属核心知识召回
- ✅ 药物类别共性检索（ClinicalDecision）
- ✅ 选项特征检索
- ✅ PaperQA 和 Deep Research 集成
- ✅ 强制工具使用指令（防止 LLM 不使用工具）

---

### N4: Calculation Step Decomposition & Formula Matching（计算步骤拆解与公式匹配）

#### 功能描述
- 拆解计算步骤（calculation_steps）
- 匹配公式（matched_formula）
- 提取单位转换规则（unit_conversion_rules）
- 验证公式匹配结果（formula_match_result）
- **关键**：对于结合亲和力计算，验证公式不是过于简单的线性模型

#### 输入要求
- `cleaned_text`: 清理后的文本（必需）
- `key_parameters`: 关键参数字典（必需）
- `calculation_type_label == "Numerical"`: 必须是数值计算类型
- `domain_knowledge_map`: 领域知识映射（用于公式匹配）

#### 处理逻辑
1. 验证输入条件
2. 加载计算支持工具（expression + genetic + core_query + binding tools）
3. 包含关键约束（critical_constraints）到 prompt
4. 调用 LLM 进行拆解和匹配
5. 如果解析失败，创建 fallback calculation_steps
6. 验证公式（检查是否过于简单）

#### 输出结果
- `calculation_steps`: 计算步骤列表
- `matched_formula`: 匹配的公式字典
- `unit_conversion_rules`: 单位转换规则列表
- `formula_match_result`: 公式匹配结果（Match Success/Match Failed）

#### 路由逻辑（route_after_n4）
- 如果 `formula_match_result == "Match Failed"` 且无 calculation_steps：
  - 如果 `auto_retry_count < 1` → 重试 N2（最多1次）
  - 否则 → N10（异常处理）
- 否则 → N7（完整推理，即使公式匹配失败也允许继续）

#### 工具绑定
- **计算支持工具**：expression + genetic + core_query + binding tools

#### 关键优化点
- ✅ Fallback 机制（如果 LLM 解析失败，从 key_parameters 创建基本步骤）
- ✅ 公式验证（检查是否过于简单）
- ✅ 自动重试机制（最多1次）

---

### N5: Algorithm Parameter Extraction & Applicability Validation（算法参数提取与适用性验证）

#### 功能描述
- 提取算法参数（algorithm_parameters）
- 验证算法适用性（applicability_result）
- 提供替代算法建议（alternative_algorithms）

#### 输入要求
- `cleaned_text`: 清理后的文本（必需）
- `key_parameters`: 关键参数字典（必需，包含 algorithm_name）
- `calculation_type_label == "Algorithm"`: 必须是算法类型

#### 处理逻辑
1. 验证输入条件
2. 加载算法验证工具（pathway + interaction + disease_gene）
3. 调用 LLM 进行参数提取和验证
4. 解析响应

#### 输出结果
- `algorithm_parameters`: 算法参数字典
- `applicability_result`: 适用性结果（Applicable/Not Applicable）
- `alternative_algorithms`: 替代算法建议列表

#### 路由逻辑（route_after_n5）
- 如果 `applicability_result == "Not Applicable"` → N10（异常处理）
- 否则 → N7（完整推理）

#### 工具绑定
- **算法验证工具**：pathway + interaction + disease_gene

#### 关键优化点
- ✅ 容错处理（放宽输入验证条件）

---

### N6: Initial Association Inference（初步关联推理）

#### 功能描述
- 建立现象-知识匹配表（phenomenon_knowledge_match_table）
- 评估匹配置信度（match_confidence_label）
- **关键**：匹配置信度必须绑定到知识有效性（只有 Valid 知识才能是 High 置信度）
- **题型逻辑检查**（category_logic_check）：验证推理逻辑是否符合题型范式

#### 输入要求
- `structured_conditions`: 结构化条件（必需，如果缺失会创建 fallback）
- `domain_knowledge_map`: 领域知识映射（必需，如果缺失会从 core_domains 创建 fallback）
- `research_objective`: 研究目标
- `key_entities`: 关键实体列表
- `inference_core_restrictions`: 推理核心限制（用于逻辑验证）

#### 处理逻辑
1. 验证输入（如果缺失，创建 fallback）
2. 加载知识检索工具（core_query + disease_gene + interaction + genetic）
3. 添加约束信息到 prompt（用于逻辑验证）
4. 调用 LLM 进行匹配
5. 解析响应

#### 输出结果
- `phenomenon_knowledge_match_table`: 现象-知识匹配表
- `match_confidence_label`: 匹配置信度标签（High/Medium/Low）
- `category_logic_check`: 题型逻辑检查结果（新增）

#### 路由逻辑（route_after_n6）
- 如果存在 `calculation_type_label`：
  - 如果是 "Numerical" 或 "Logical Calculation" → N4（计算拆解）
  - 如果是 "Algorithm" → N5（算法验证）
- 如果不存在 `phenomenon_knowledge_match_table`：
  - 如果存在 `domain_knowledge_map` → 创建 fallback → N7
  - 如果 `n1_visits >= 2` → N10（异常处理）
  - 如果 `auto_retry_count < 1` → 重试 N1（最多1次）
  - 否则 → N10（异常处理）
- 否则 → N7（完整推理）

#### 工具绑定
- **知识检索工具**：core_query + disease_gene + interaction + genetic

#### 关键优化点
- ✅ Fallback 机制（从 domain_knowledge_map 创建匹配表）
- ✅ 匹配置信度绑定到知识有效性
- ✅ 题型逻辑检查
- ✅ 自动重试机制（最多1次）

---

### N7: Complete Logical Inference（完整逻辑推理）

#### 功能描述
- 构建完整推理路径（closed_inference_path）
- 生成核心结论（core_conclusion）
- **关键**：推理路径必须覆盖所有 `category_specific_solution_steps`
- **事实验证步骤**（fact_verification）：在推理路径末尾添加事实验证
- **禁止主观结论**：结论必须基于事实/计算/指南，而非主观判断

#### 输入要求
- 至少满足以下之一：
  - `phenomenon_knowledge_match_table`: 现象-知识匹配表
  - `calculation_steps`: 计算步骤列表
  - `algorithm_parameters`: 算法参数字典
- 如果都缺失，会尝试从 `domain_knowledge_map` 或 `key_parameters` 创建 fallback

#### 处理逻辑
1. 验证输入（如果缺失，创建 fallback）
2. 加载所有工具（all_tools）
3. 构建增强 prompt：
   - 包含关键约束（critical_constraints）
   - 包含硬约束（hard_constraints）
   - 包含原始问题文本（用于严格锚定）
   - 包含关键约束（key_constraints）
   - 自动检测并强调否定约束和专属约束
4. 调用 LLM 进行推理（max_iterations=5）
5. 解析响应
6. 验证数值精度和单位一致性（对于计算题）

#### 输出结果
- `closed_inference_path`: 完整推理路径（必须包含 fact_verification 步骤）
- `core_conclusion`: 核心结论（必须基于事实，非主观）
- `fact_verification_result`: 事实验证结果（新增）

#### 路由逻辑（route_after_n7）
- 如果 `closed_inference_path` 或 `core_conclusion` 缺失：
  - 如果 `n6_visits >= 2` → N10（异常处理）
  - 如果 `auto_retry_count < 1` → 重试 N6（最多1次）
  - 否则 → N10（异常处理）
- 如果来自 N10 重试且重试失败 → 标记失败 → N10
- 如果重试成功 → 重置 retry_count → N8
- 否则 → N8（答案生成）

#### 工具绑定
- **所有工具**（all_tools）

#### 关键优化点
- ✅ Fallback 机制（从 domain_knowledge_map 或 key_parameters 创建推理基础）
- ✅ 强制步骤完整性（覆盖所有 category_specific_solution_steps）
- ✅ 事实验证步骤（在推理路径末尾）
- ✅ 约束自动检测和强调（否定约束、专属约束）
- ✅ 数值精度和单位一致性验证
- ✅ 自动重试机制（最多1次）

---

### N8: Multi-Type Answer Generation（多类型答案生成）

#### 功能描述
- 生成结构化答案（structured_answer）
- 匹配选项（option_matching_table）
- 生成最终答案（final_answer）
- **关键**：单选题禁止输出 "Cannot generate"，必须选择一个选项
- **格式转换**：自动处理 Sequence 格式的 rank 序列转换
- **合理性校验**：检查数值量级、序列长度、选项一致性

#### 输入要求
- `core_conclusion`: 核心结论（必需）
- `question_options`: 选项列表（如果是选择题）
- `answer_format_label`: 答案格式标签
- `closed_inference_path`: 推理路径（用于提取计算结果）

#### 处理逻辑
1. 验证 `core_conclusion` 存在
2. 加载答案精炼工具（disease_gene + ontology + drug）
3. 从推理路径提取计算结果（如果有）
4. 构建 prompt：
   - 包含原始问题文本（用于选项检查）
   - 包含格式控制指令
5. 调用 LLM 生成答案（max_iterations=3）
6. 解析响应
7. **强制选项选择**：如果是单选题且所有选项被排除，强制选择最佳匹配选项
8. **选项匹配验证**：
   - 单选题必须只有1个匹配
   - 如果0个匹配，强制语义匹配重试
9. **格式转换**：Sequence 格式的 rank 序列转换
10. **合理性校验**：检查数值量级、序列长度、选项一致性

#### 输出结果
- `structured_answer`: 结构化答案字典
  - `final_answer`: 最终答案
  - `answer_content`: 答案内容
    - `option_matching_table`: 选项匹配表（如果是选择题）
- `final_answer`: 最终答案字符串

#### 路由逻辑（route_after_n8）
- 如果 `structured_answer` 或 `final_answer` 缺失：
  - 如果 `n7_visits >= 2` 或 `n3_visits >= 2` → N10（异常处理）
  - 如果是事实题且 `auto_retry_count < 1` 且 `n3_visits < 2` → 重试 N3（最多1次）
  - 如果 `auto_retry_count < 1` 且 `n7_visits < 2` → 重试 N7（最多1次）
  - 否则 → N10（异常处理）
- 如果来自 N10 重试且重试失败 → 标记失败 → N10
- 如果重试成功 → 重置 retry_count → N9
- 否则 → N9（结果验证）

#### 工具绑定
- **答案精炼工具**：disease_gene + ontology + drug

#### 关键优化点
- ✅ 单选题强制选择（禁止 "Cannot generate"）
- ✅ 选项匹配验证和自动修正
- ✅ 语义匹配重试机制
- ✅ 格式自动转换（rank 序列）
- ✅ 合理性校验（数值量级、序列长度、选项一致性）
- ✅ 自动重试机制（最多1次）

---

### N9: Result Validation & Consistency Judgment（结果验证与一致性判断）

#### 功能描述
- 验证答案格式（format_valid_label）
- 验证步骤完整性（step_complete_label）：检查 n7 步骤是否覆盖所有 category_specific_solution_steps
- 验证事实正确性（fact_correct_label）：检查答案是否匹配权威事实/计算/指南
- 判断一致性（consistency_label）：只有 format_valid + step_complete + fact_correct 全部 Valid 才是 Consistent
- 计算可靠性分数（reliability_score）：基于扣分系统（事实错误扣5分，格式/步骤问题扣2分）
- **强制拦截**：如果 fact_correct_label=Invalid 或 step_complete_label=Invalid，拦截答案输出

#### 输入要求
- `structured_answer`: 结构化答案（必需）
- `closed_inference_path`: 推理路径（必需）
- `answer_format_label`: 答案格式标签
- `question_options`: 选项列表
- `core_keywords`: 核心关键词（用于验证）
- `option_features`: 选项特征（用于验证）

#### 处理逻辑
1. 验证输入
2. 加载验证工具（disease_gene[:2] + ontology[:2]）
3. 检查推理路径一致性（critical_constraints 是否被考虑）
4. 构建增强 prompt：
   - 包含原始问题文本（用于事实正确性检查）
   - 包含核心关键词和选项特征
   - 添加两个核心验证检查：
     - 检查1：答案与约束匹配性（negative/exclusive/hard 约束）
     - 检查2：答案逻辑合理性（数值量级、序列长度、选项一致性）
5. 调用 LLM 进行验证（max_iterations=2）
6. 解析响应
7. **代码级验证**：
   - 检查约束违反（negative/exclusive 约束）
   - 检查逻辑合理性（数值量级、序列长度）
8. 应用验证结果（如果检测到问题，覆盖 LLM 结果）

#### 输出结果
- `consistency_label`: 一致性标签（Consistent/Inconsistent）
- `reliability_score`: 可靠性分数（1-5）
- `format_valid_label`: 格式有效性标签（Valid/Invalid）
- `format_issues`: 格式问题列表
- `step_complete_label`: 步骤完整性标签（新增）
- `fact_correct_label`: 事实正确性标签（新增）

#### 路由逻辑（route_after_n9）
- 如果 `consistency_label == "Inconsistent"` 或 `reliability_score <= 3` → N10（异常处理）
- 否则 → END（成功）

#### 工具绑定
- **验证工具**：disease_gene[:2] + ontology[:2]

#### 关键优化点
- ✅ 三维验证（格式 + 步骤 + 事实）
- ✅ 代码级验证（约束违反、逻辑合理性）
- ✅ 强制拦截机制（事实错误或步骤不完整时拦截）
- ✅ 扣分系统（事实错误扣5分，格式/步骤问题扣2分）

---

### N10: Knowledge/Calculation Exception Handling（知识/计算异常处理）

#### 功能描述
- 识别异常类型（exception_type_label）
- 提供解决方案建议（solution_suggestion）
- **自动重试机制**：如果建议 "Retry"，自动重试目标节点（最多1次）
- **循环检测**：如果目标节点已访问2次以上，停止重试并进入人工干预

#### 输入要求
- 异常上下文（从 state 中提取）

#### 处理逻辑
1. 确定异常类型（从 state.exception_type_label 或上下文推断）
2. 加载所有工具（all_tools，用于寻找替代方案）
3. 构建异常处理 prompt
4. 调用 LLM 分析异常并提供建议（max_iterations=4）
5. 解析响应

#### 输出结果
- `exception_type_label`: 异常类型标签（精确识别根因）
- `solution_suggestion`: 解决方案建议（Retry/Retry-N7/Retry-N8/Manual Intervention）

#### 路由逻辑（route_after_n10）
- 如果 `solution_suggestion == "Manual Intervention"` → N11（人工干预）
- 如果 `solution_suggestion == "Retry"` 或包含 "Retry-N7"/"Retry-N8"：
  - 确定重试目标节点（基于 exception_type_label）
  - 如果目标节点已访问 `>= 2` 次 → N11（人工干预，防止循环）
  - 如果 `retry_count >= 1` → N11（人工干预，最多重试1次）
  - 否则 → 重试目标节点（n7 或 n8）
- 否则 → END

#### 工具绑定
- **所有工具**（all_tools）

#### 关键优化点
- ✅ 精确异常类型识别（根因分析）
- ✅ 自动重试机制（最多1次）
- ✅ 循环检测和防止（节点访问次数限制）
- ✅ 重试成功时重置 retry_count

---

### N11: Manual Intervention Trigger（人工干预触发）

#### 功能描述
- 生成人工干预指南（manual_intervention_guide）
- 生成中间结果快照（intermediate_result_snapshot）
- **关键**：指南必须包含核心问题、具体可执行步骤、预期结果

#### 输入要求
- `exception_type_label`: 异常类型标签（必需）

#### 处理逻辑
1. 验证 `exception_type_label` 存在
2. 收集中间结果快照
3. 构建人工干预 prompt
4. 调用 LLM 生成指南
5. 解析响应

#### 输出结果
- `manual_intervention_guide`: 人工干预指南（包含核心问题、具体步骤、预期结果）
- `intermediate_result_snapshot`: 中间结果快照

#### 路由逻辑
- 固定路由到 END

#### 工具绑定
- **无工具绑定**

#### 关键优化点
- ✅ 详细的干预指南（核心问题 + 具体步骤 + 预期结果）

---

## 流程优化总结

### 已实现的优化

1. **题型分类标准化**：三类题型（计算题/临床决策题/专业知识题）及其专属约束和解题步骤
2. **结构化信息提取**：三维度信息（Subject/Condition/Goal）确保信息完整性
3. **双层知识校验**：相关性 + 事实性校验，确保知识质量
4. **强制工具使用**：N3 节点强制使用数据库查询工具
5. **步骤完整性强制**：N7 推理路径必须覆盖所有题型专属步骤
6. **事实验证步骤**：在推理路径末尾添加事实验证
7. **自动重试机制**：节点失败时自动重试（最多1次）
8. **循环检测和防止**：节点访问次数限制，防止无限循环
9. **答案合理性校验**：数值量级、序列长度、选项一致性检查
10. **三维验证**：格式 + 步骤 + 事实验证，确保答案质量

### 潜在优化方向

1. **工具使用监控**：确保 LLM 实际使用了工具，而非仅依赖训练数据
2. **知识检索精度**：提高选项特征检索和药物类别共性检索的准确性
3. **推理路径质量**：确保推理路径不是形式化步骤，而是实际推理过程
4. **事实验证落地**：确保 fact_verification 步骤不是占位符，而是实际验证
5. **约束落地精度**：确保所有约束（negative/exclusive/hard）都被正确应用

---

## 节点执行流程图

```
START
  ↓
N0 (输入预处理)
  ↓
  ├─→ [Numerical Calculation] → N2 (计算识别) → N3 (知识检索)
  │                                                      ↓
  └─→ [Other Types] → N1 (问题拆解) → N3 (知识检索) → N6 (初步推理)
                                                              ↓
                                                      ├─→ [Numerical/Logical] → N4 (计算拆解) → N7 (完整推理)
                                                      ├─→ [Algorithm] → N5 (算法验证) → N7 (完整推理)
                                                      └─→ [Other] → N7 (完整推理)
                                                                      ↓
                                                              N8 (答案生成)
                                                                      ↓
                                                              N9 (结果验证)
                                                                      ↓
                                                              ├─→ [Consistent] → END
                                                              └─→ [Inconsistent] → N10 (异常处理)
                                                                                      ↓
                                                                              ├─→ [Retry] → N7/N8 (重试)
                                                                              └─→ [Manual] → N11 (人工干预) → END
```

---

## 状态字段完整列表

详见 `agent/nodes/subagents/general_qa/state.py`，包含所有节点的输入输出字段定义。

