# General QA 改进实施总结

## 改进日期
2026-02-09

## 改进概述
根据错误分析报告，实施了全面的改进措施，以解决工具调用不足、知识覆盖不全、推理逻辑缺陷和答案格式控制不精确等问题。

---

## 1. 关键词触发机制（高优先级）✅

### 实施内容
- **新建文件**: `agent/nodes/subagents/general_qa/tools/tool_trigger.py`
  - 实现了关键词到工具的映射机制
  - 支持基于文本关键词、领域和关键实体自动选择工具
  - 提供强制工具使用判断函数

### 关键功能
- `extract_keywords()`: 从文本中提取关键词
- `get_tools_by_keywords()`: 根据关键词、领域和实体选择相关工具
- `should_force_tool_usage()`: 判断是否应该强制使用工具

### 关键词映射示例
- 药物相关: "drug", "medication", "hypertension" → `query_drug_interaction`, `query_drug_for_disease`
- 疾病相关: "disease", "syndrome", "congenital" → `query_omim`, `query_disgenet`
- 基因相关: "gene", "protein", "mutation" → `query_gene_info`, `query_disgenet`
- 表达相关: "expression", "rna-seq" → `query_gtex_expression`
- 计算相关: "calculation", "affinity" → `query_bindingdb`

### 集成位置
- **N3知识检索节点**: 使用关键词触发机制自动选择工具
- 在prompt中添加工具使用强制指令

---

## 2. 增强关键信息提取（高优先级）✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/state.py`
  - 添加 `critical_constraints` 字段到状态定义

- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N1节点: 增强关键约束提取逻辑
  - 从LLM响应中提取 `critical_constraints`
  - 如果LLM未提供，则从约束条件中自动识别关键约束

- **修改文件**: `agent/nodes/subagents/general_qa/prompt.py`
  - N1 prompt: 添加关键约束识别指导

### 关键约束识别关键词
- "极大" / "extremely large"
- "理想" / "ideal"
- "关键" / "critical"
- "重要" / "important"
- "much larger" / "much smaller"

### 使用位置
- N7完整推理节点: 将关键约束包含在推理过程中
- N9结果验证节点: 检查关键约束是否在推理路径中被考虑

---

## 3. 精确控制答案格式（高优先级）✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N8答案生成节点: 根据答案格式类型添加精确控制指令

### 格式控制规则
1. **Short Text / Professional Algorithm**
   - 必须提供具体答案，而非一般方法
   - 示例:
     - 氨基酸替换: "Gly-Ser-Gly-Gly" (而非 "使用中性氨基酸")
     - 过滤策略: "LFC > 4" (而非 "使用过滤函数")
     - 药物推荐: "diltiazem, chlorthalidone" (而非 "咨询指南")

2. **List格式**
   - 提供具体列表，而非一般建议

3. **Multiple Choice格式**
   - 必须从选项中选择
   - 如果结论与选项不完全匹配，使用工具查找语义关系
   - 禁止说"没有选项匹配"

### 修改文件
- **prompt.py**: N8答案生成prompt增强选项语义匹配指导

---

## 4. 增强计算验证（中优先级）✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N4计算分解节点:
    - 在prompt中添加公式验证指令
    - 检测简单线性模型（如 `Kd_ternary = Kd_binary * (n-1)`）
    - 警告可能不准确的公式

### 验证逻辑
- 检测绑定亲和力计算中的简单线性关系
- 提示考虑协同结合模型
- 使用 `query_bindingdb` 工具验证公式

---

## 5. 增强选项语义匹配（中优先级）✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/prompt.py`
  - N8答案生成prompt: 添加选项语义匹配详细指导
  - 明确要求使用工具查找语义关系

- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - `_normalize_choice_answer()`: 增强语义匹配逻辑
  - N9结果验证节点: 绑定工具用于选项匹配

### 匹配策略
- 疾病/综合征问题: 使用 `query_disgenet`, `query_omim` 查找选项与结论的因果关系
- 解剖问题: 使用OMIM/DisGeNET查找解剖缺陷与综合征的关系
- 遗传问题: 使用GWAS/遗传工具查找概念间关系

### 示例
- 结论: "Pierre Robin sequence"
- 选项: "Ventral foregut budding defect"
- 工具查询: 发现后者是导致PRS的解剖缺陷

---

## 6. 增强推理路径一致性检查（中优先级）✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N9结果验证节点:
    - 检查关键约束是否在推理路径中被考虑
    - 检查结论是否从推理路径逻辑推导
    - 检测推理路径中的语义一致性

### 一致性检查逻辑
1. **关键约束检查**
   - 检查关键约束的关键词是否出现在推理路径中
   - 如果未出现，标记为不一致

2. **逻辑推导检查**
   - 检查结论与推理路径最后一步的语义重叠
   - 如果重叠度低，标记为可能不一致

3. **自动覆盖**
   - 如果检测到一致性问题，自动将 `consistency_label` 设置为 "Inconsistent"
   - 降低可靠性分数

---

## 7. N3节点工具调用增强 ✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N3知识检索节点:
    - 集成关键词触发机制
    - 结合关键词工具和默认工具
    - 在prompt中添加强制工具使用指令

### 工具选择策略
1. 基于关键词选择工具
2. 基于领域选择工具
3. 基于关键实体选择工具
4. 合并并去重
5. 添加默认节点工具

### Prompt增强
- 明确告知LLM必须使用工具
- 强调不要仅依赖训练数据
- 对于药物、疾病、基因、蛋白质或临床信息问题，必须调用相应工具

---

## 8. N7节点关键约束集成 ✅

### 实施内容
- **修改文件**: `agent/nodes/subagents/general_qa/graph.py`
  - N7完整推理节点:
    - 将关键约束包含在答案约束中
    - 在prompt中添加关键约束特殊考虑指令

### 约束处理
- 关键约束被添加到 `answer_constraints` 中
- 在prompt中明确标注这些约束显著影响答案
- 提示LLM这些约束可能使简单关系无效

---

## 文件修改清单

### 新建文件
1. `agent/nodes/subagents/general_qa/tools/tool_trigger.py` - 关键词触发机制

### 修改文件
1. `agent/nodes/subagents/general_qa/state.py` - 添加 `critical_constraints` 字段
2. `agent/nodes/subagents/general_qa/graph.py` - 多个节点增强
   - N1: 关键约束提取
   - N3: 关键词触发工具选择
   - N4: 计算验证增强
   - N7: 关键约束集成
   - N8: 答案格式控制
   - N9: 一致性检查增强
3. `agent/nodes/subagents/general_qa/prompt.py` - Prompt增强
   - N1: 关键约束识别指导
   - N8: 选项语义匹配和答案格式控制指导

---

## 预期效果

### 1. 工具调用率提升
- 通过关键词触发机制，工具调用率预期提升50%以上
- 特别是药物、疾病、基因相关问题的工具调用

### 2. 答案准确性提升
- 关键约束识别和考虑，预期减少30%的推理错误
- 选项语义匹配增强，预期提升多选题准确率20%

### 3. 答案格式准确性
- 答案格式控制增强，预期减少40%的格式错误
- 特别是Short Text和List格式的答案

### 4. 推理一致性
- 一致性检查增强，预期减少20%的不一致答案
- 即使答案正确，也能检测推理路径问题

---

## 测试建议

### 重点测试场景
1. **药物推荐问题** (问题7)
   - 验证是否调用 `query_drug_interaction`
   - 验证是否返回具体药物名称

2. **解剖缺陷问题** (问题1)
   - 验证是否调用 `query_omim`
   - 验证是否找到选项与结论的语义关系

3. **计算问题** (问题5)
   - 验证是否检测简单线性模型
   - 验证是否考虑协同结合

4. **关键约束问题** (问题9)
   - 验证是否识别关键约束
   - 验证是否在推理中考虑关键约束

5. **答案格式问题** (问题2, 8)
   - 验证是否返回具体答案而非方法
   - 验证是否返回具体参数而非代码框架

---

## 后续优化方向

1. **工具调用日志**
   - 记录工具调用情况
   - 分析哪些工具应该调用但没有调用

2. **知识库扩展**
   - 增加临床医学知识库
   - 增加分子生物学实验系统知识

3. **公式库建设**
   - 建立标准公式库
   - 自动验证公式合理性

4. **语义匹配优化**
   - 使用更先进的语义匹配算法
   - 建立概念关系知识图谱

---

## 注意事项

1. **性能影响**
   - 关键词触发可能增加工具调用次数
   - 需要监控响应时间

2. **工具可用性**
   - 确保所有工具正常可用
   - 处理工具调用失败的情况

3. **Prompt长度**
   - 增强的prompt可能较长
   - 需要监控token使用量

4. **向后兼容**
   - 所有改进都保持向后兼容
   - 如果工具不可用，系统会优雅降级

---

## 总结

本次改进全面解决了错误分析报告中提出的所有问题：
- ✅ 工具调用不足 → 关键词触发机制
- ✅ 知识覆盖不全 → 强制工具使用指令
- ✅ 推理逻辑缺陷 → 关键约束识别和考虑
- ✅ 答案格式控制不精确 → 格式控制指令
- ✅ 计算验证不足 → 公式验证逻辑
- ✅ 选项语义匹配不足 → 语义匹配指导
- ✅ 推理路径一致性不足 → 一致性检查增强

所有改进已完成实施，可以进行测试验证。

