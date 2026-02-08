# General QA 通用化架构设计文档

## 核心设计理念

实现全节点通用化升级，所有节点基于3个领域/知识点无关的通用概念设计，不引入任何具体知识点术语。

## 三大通用概念

### 1. 约束优先级 (Constraint Priority)

**定义**：将场景约束分为「核心约束（C1）/ 次要约束（C2）」

- **C1（核心约束）**：决定知识点结论的必要且充分条件
  - 如果C1不满足，结论必然不成立
  - 如果C1满足，结论可以成立（还需要其他条件）
  - C1是判断匹配度的关键依据

- **C2（次要约束）**：仅影响结论的细节，不改变核心结论
  - C2影响结论的具体数值、表达方式、适用范围等
  - 但不影响结论的核心判断（是/否、有/无、方向等）

**作用**：让系统能够区分哪些约束是决定性的，哪些只是修饰性的。

### 2. 条件化知识 (Conditionalized Knowledge)

**定义**：知识点按「条件集（Kc）→ 结论（Kr）」标准化表达

- **Kc（条件集）**：知识点结论成立的约束条件集合
  - 必须是C1类型的约束（核心约束）
  - 以集合形式表达，便于进行集合匹配
  - 例如：Kc = {"large_sample", "random_filtering", "no_global_missing_snp"}

- **Kr（结论）**：在Kc条件下成立的结论
  - 结论可以是任何形式：判断、数值、方向、机制等
  - 结论必须与Kc绑定，不能独立存在

**作用**：让Agent可做精准的集合匹配（C1是否等于/包含Kc），而非自然语言泛化推理。

### 3. 匹配度判定 (Match Degree Judgment)

**定义**：通用规则：若场景C1 ⊇ 知识Kc → 匹配成功，触发对应Kr；若不匹配 → 触发泛化知识

**匹配规则**：
- **完全匹配**：C1 ⊇ Kc 且 C1 = Kc（或C1包含Kc的所有元素）
  - 触发精准结论Kr
  - 使用知识库中的精确结论，不做泛化推理

- **部分匹配**：C1 ⊇ Kc 但C1包含Kc之外的元素
  - 仍触发精准结论Kr（因为核心条件满足）
  - 但需要验证C1的额外元素是否与Kr冲突

- **不匹配**：C1不包含Kc，或C1与Kc有冲突
  - 不触发Kr
  - 退居泛化推理（使用LLM进行自然语言推理）

**作用**：核心约束完全匹配才会得到精准结论，否则退居泛化推理，是实质验证的核心。

## 节点通用化设计

### Node 1: Question Parsing Node
**职责**：提取并分类约束为C1和C2
- 从问题中提取所有约束
- 判断每个约束是C1（核心）还是C2（次要）
- 输出：`core_constraints (C1)`, `secondary_constraints (C2)`

### Node 2: Knowledge Activation Node
**职责**：将知识标准化为Kc→Kr格式
- 激活领域知识
- 将知识转换为条件化格式：每个知识点 = {Kc: [...], Kr: "..."}
- 输出：`conditionalized_knowledge: List[{Kc: Set[str], Kr: str}]`

### Node 3: Data Processing Node
**职责**：执行C1 ⊇ Kc匹配度判定
- 对每个知识点的Kc，检查C1是否包含Kc
- 计算匹配度：完全匹配/部分匹配/不匹配
- 输出：`match_results: List[{knowledge_id, match_type, matched_Kr}]`

### Node 4: Reasoning Engine Node
**职责**：基于匹配结果选择推理策略
- 如果有完全匹配：直接使用Kr作为精准结论
- 如果有部分匹配：使用Kr但验证额外约束
- 如果无匹配：使用泛化推理（LLM自然语言推理）
- 输出：`reasoning_strategy`, `precise_conclusion (Kr)`, `generalized_reasoning`

### Node 5: Conclusion Validation Node
**职责**：验证匹配度判定结果的正确性
- 验证使用的Kr是否与C1匹配
- 验证C2是否与Kr兼容
- 输出：`validation_result`, `final_answer`

### Node 6: Final Answer Node
**职责**：生成最终答案
- 整合精准结论和泛化推理
- 生成用户友好的答案

## 数据流

```
问题输入
  ↓
Node 1: 提取C1, C2
  ↓
Node 2: 生成Kc→Kr知识库
  ↓
Node 3: C1 ⊇ Kc匹配判定
  ↓
Node 4: 选择精准结论(Kr)或泛化推理
  ↓
Node 5: 验证匹配正确性
  ↓
Node 6: 生成最终答案
```

## 优势

1. **领域无关**：所有概念都是抽象的，不依赖具体知识点
2. **精准匹配**：通过集合匹配实现精准结论，避免泛化错误
3. **可扩展**：新领域只需定义Kc→Kr映射，无需修改节点逻辑
4. **可验证**：匹配度判定是可验证的，不依赖黑盒推理

