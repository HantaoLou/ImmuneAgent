# General QA 正确率未提升问题分析报告

## 一、核心问题总结

通过分析测试输出和代码，发现了5个导致正确率未提升的关键问题：

### 1. **N8选项匹配规则没有被代码层面强制执行**
**问题表现**：
- 测试输出显示：案例2中，n8同时标记C和D为"match"，但最终选了C，n9给了Consistent和5.0分
- 代码问题：`_normalize_choice_answer`函数虽然检查了Single Choice是否只有一个match，但这个检查是在normalize阶段，如果LLM已经输出了多个match，这个函数会尝试修复，但可能修复不正确
- **根本原因**：代码中没有在n8节点执行后立即验证选项匹配表是否符合格式要求（Single Choice必须只有1个match）

**修复方案**：
- 在`n8_answer_generation_node`中，在normalize之前添加代码层面的验证：
  - 如果`answer_format_label == "Single Choice"`，检查`option_matching_table`中"match"的数量
  - 如果超过1个，强制修正为只保留第一个match，并标记为异常
  - 如果0个，尝试从`core_conclusion`中推断

### 2. **N9校验失败后没有自动重推机制**
**问题表现**：
- 测试输出显示：案例1中，n9标记Inconsistent，然后进入n10异常处理，最终到n11人工干预
- 代码问题：`route_after_n10`虽然有注释说"If retry, we would need to route back"，但实际上没有实现重推逻辑
- **根本原因**：即使检测到错误，也只是标记，没有自动回到前面的节点重新推理

**修复方案**：
- 在`GeneralQAState`中添加`retry_count`字段，记录重试次数
- 在`route_after_n10`中实现重推逻辑：
  - 如果`solution_suggestion == "Retry"`，根据`exception_type_label`路由回相应的节点
  - 例如：`Inference Path Inconsistent` → 路由回`n7_complete_inference`
  - `Answer Format Invalid` → 路由回`n8_answer_generation`
  - `Answer Generation Failed` → 路由回`n8_answer_generation`
  - 限制重试次数（例如最多2次），避免无限循环

### 3. **关键约束没有被真正使用**
**问题表现**：
- 测试输出显示：案例1中，n0没有提取key_constraints（hard_constraints是空的），但题干明确说了"one SNP homozygous mutant, four SNPs heterozygous"
- **根本原因**：虽然提取了key_constraints，但在推理过程中可能没有被充分利用

**修复方案**：
- 在`n0_input_preprocessing_node`中，强化key_constraints的提取逻辑
- 在`n7_complete_inference_node`中，确保key_constraints被传递到prompt中
- 在`get_complete_inference_prompt`中，添加明确的指令要求使用key_constraints

### 4. **流程中断无兜底**
**问题表现**：
- 测试输出显示：案例3走到n6后就没有继续了
- 代码问题：`route_after_n6`是直接edge到n7，如果n6执行失败但没有抛出异常，可能不会继续
- **根本原因**：没有检测机制来确保流程继续执行

**修复方案**：
- 在`route_after_n6`中添加检测逻辑：
  - 检查`phenomenon_knowledge_match_table`是否存在
  - 如果不存在，尝试从`domain_knowledge_map`构造fallback
  - 如果仍然失败，路由到`n10_exception_handling`

### 5. **知识检索与Goal脱节**
**问题表现**：
- 测试输出显示：案例3中，题目明确要求"推荐降压药"，却召回大量降脂、甲减治疗的知识
- **根本原因**：虽然prompt中有规则，但LLM可能没有严格遵守

**修复方案**：
- 在`n3_knowledge_retrieval_node`中，添加代码层面的过滤逻辑：
  - 在LLM返回`domain_knowledge_map`后，检查每个知识点是否与`structured_goal`相关
  - 如果`structured_goal.type`是"conclusion judgment"且`structured_goal.constraint`包含"降压药"，则过滤掉不相关的知识（如降脂、甲减）
  - 可以使用简单的关键词匹配或语义相似度检查

## 二、修复优先级

1. **高优先级**（直接影响正确率）：
   - N8选项匹配规则代码层面强制执行
   - N9校验失败后自动重推机制

2. **中优先级**（影响推理质量）：
   - 关键约束的提取和使用
   - 知识检索与Goal的关联性

3. **低优先级**（影响流程稳定性）：
   - 流程中断兜底机制

## 三、实施建议

1. **分阶段实施**：先修复高优先级问题，测试效果后再修复中低优先级问题
2. **添加日志**：在修复过程中添加详细的日志，便于追踪问题
3. **测试验证**：修复后使用相同的测试用例验证效果


