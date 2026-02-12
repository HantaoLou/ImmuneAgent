# 领域模块验收清单

本文档定义了新增领域Prompt模块的验收标准，确保所有领域模块的质量和一致性。

## 一、必需实现项（Mandatory）

### 1.1 文件结构
- [ ] 文件命名：`prompt-{domain_name}.py`（小写，下划线分隔）
- [ ] 文件位置：`agent/nodes/subagents/general_qa/prompts/`
- [ ] 文件编码：UTF-8

### 1.2 函数实现（12个节点）
- [ ] `get_input_preprocessing_prompt(user_input: str) -> str` - N0
- [ ] `get_question_decomposition_prompt(...) -> str` - N1
- [ ] `get_calculation_algorithm_recognition_prompt(...) -> str` - N2
- [ ] `get_knowledge_retrieval_prompt(...) -> str` - N3
- [ ] `get_calculation_decomposition_prompt(...) -> str` - N4
- [ ] `get_algorithm_validation_prompt(...) -> str` - N5
- [ ] `get_initial_inference_prompt(...) -> str` - N6
- [ ] `get_complete_inference_prompt(...) -> str` - N7
- [ ] `get_answer_generation_prompt(...) -> str` - N8
- [ ] `get_result_validation_prompt(...) -> str` - N9
- [ ] `get_exception_handling_prompt(...) -> str` - N10
- [ ] `get_manual_intervention_prompt(...) -> str` - N11

### 1.3 领域特定函数
- [ ] `get_domain_tools() -> List[str]` - 返回领域优先级工具列表
- [ ] `get_domain_extraction_rules() -> str` - 返回领域提取规则（用于跨领域合并）

### 1.4 DOMAIN_CONFIG配置
- [ ] `name`: 领域名称
- [ ] `priority_tools`: 优先级工具列表（至少3个）
- [ ] `tool_priority`: 工具优先级映射（可选）
- [ ] `fallback_tools`: 降级工具映射（可选）
- [ ] `common_entities`: 常见实体列表
- [ ] `calculation_focus`: 计算焦点列表（如果有计算类问题）
- [ ] `validation_criteria`: 验证标准列表
- [ ] `extraction_rules`: 提取规则字符串

## 二、Prompt增强要求

### 2.1 N0: Input Preprocessing增强
- [ ] 至少3条领域特定的提取规则
- [ ] 领域特定的类别约束（至少2个类别）
- [ ] 实体提取规则（如基因型、表型、细胞类型等）

### 2.2 N1: Question Decomposition增强
- [ ] 至少2种领域特定的分解模式
- [ ] 领域特定的域识别规则
- [ ] 子目标分解指导

### 2.3 N3: Knowledge Retrieval增强
- [ ] 领域特定工具使用策略
- [ ] 工具调用优先级
- [ ] 知识检索焦点

### 2.4 N4-N11: 可选增强
- [ ] 计算类问题：N4需要领域特定的计算规则
- [ ] 推理类问题：N6/N7需要领域特定的推理逻辑
- [ ] 答案生成：N8需要领域特定的答案格式

## 三、集成要求

### 3.1 domain_mapper.py集成
- [ ] 在`DOMAIN_MAPPING`中添加领域映射
  - `raw_subject` → `prompt-{domain_name}`
  - `question_type` → `prompt-{domain_name}`（如果适用）

### 3.2 tool_trigger.py集成
- [ ] 在`DOMAIN_TO_TOOLS`中添加领域到工具的映射
- [ ] 确保工具列表与`DOMAIN_CONFIG["priority_tools"]`一致

### 3.3 向后兼容性
- [ ] 所有函数签名与base模板一致
- [ ] 所有函数返回字符串类型
- [ ] 不破坏现有调用

## 四、测试要求

### 4.1 单元测试
- [ ] 测试所有12个节点函数可以正常调用
- [ ] 测试`get_domain_tools()`返回有效工具列表
- [ ] 测试`get_domain_extraction_rules()`返回非空字符串

### 4.2 集成测试
- [ ] 测试领域路由：`get_prompt_module(domain="DomainName")`返回正确模块
- [ ] 测试工具分配：`get_tools_for_node("n3_knowledge_retrieval", domain="DomainName")`返回领域工具
- [ ] 测试跨领域检测：多领域问题正确路由到`prompt-cross_domain`

### 4.3 功能测试
- [ ] 至少5个领域特定问题的测试用例
- [ ] 覆盖不同问题类型（Multiple Choice, Calculation, Mechanism Explanation等）
- [ ] 验证领域特定规则被正确应用

## 五、文档要求

### 5.1 代码文档
- [ ] 模块级docstring说明领域和用途
- [ ] 所有函数有docstring
- [ ] DOMAIN_CONFIG有注释说明

### 5.2 使用文档
- [ ] 在`OPTIMIZATION_PLAN.md`或相关文档中记录新领域
- [ ] 提供领域特定问题的示例

## 六、质量检查

### 6.1 代码质量
- [ ] 通过linter检查（无错误）
- [ ] 遵循PEP 8代码风格
- [ ] 所有TODO项已处理或标记为未来工作

### 6.2 Prompt质量
- [ ] 所有prompt文本为英文
- [ ] 无语法错误
- [ ] 格式清晰，易于LLM理解

### 6.3 性能考虑
- [ ] 模块导入时间合理（<100ms）
- [ ] 无循环依赖
- [ ] 缓存机制正常工作

## 七、验收流程

1. **代码审查**：检查所有必需实现项
2. **单元测试**：运行单元测试，确保通过
3. **集成测试**：运行集成测试，验证路由和工具分配
4. **功能测试**：使用测试用例验证领域特定功能
5. **性能测试**：验证模块加载和运行性能
6. **文档审查**：检查文档完整性

## 八、验收标准

### 通过标准
- ✅ 所有必需实现项（一、二、三）完成
- ✅ 所有测试（四）通过
- ✅ 代码质量（六）达标
- ✅ 至少5个测试用例通过

### 优秀标准（额外加分）
- ⭐ 超过5个测试用例
- ⭐ 提供详细的使用示例
- ⭐ 包含性能优化
- ⭐ 支持跨领域问题

## 九、示例验收

### 示例：Genetics领域
- ✅ 12个节点函数全部实现
- ✅ DOMAIN_CONFIG完整配置
- ✅ N0包含4条提取规则（继承模式、基因型/表型、群体遗传参数、变异符号）
- ✅ N1包含3种分解模式（继承模式、群体遗传、遗传连锁）
- ✅ N3包含6个优先级工具
- ✅ 集成到domain_mapper.py和tool_trigger.py
- ✅ 通过所有测试

## 十、常见问题

### Q: 如果领域没有计算类问题，N4需要实现吗？
A: 是的，但可以直接使用base模板，无需领域特定增强。

### Q: 工具列表必须与现有工具完全匹配吗？
A: 是的，工具名称必须与`tool_loader.py`中定义的工具名称完全一致。

### Q: 可以复用其他领域的实现吗？
A: 可以，但需要根据领域特点进行调整，不能直接复制。

### Q: 测试用例从哪里来？
A: 可以从`csv_questions_data.json`中筛选领域特定问题，或创建新的测试用例。

