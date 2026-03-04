# CodeAct 任务生成指南

本指南介绍如何创建高质量的 CodeAct 任务，以确保代码生成和执行的准确性。

## 目录

1. [任务结构](#任务结构)
2. [任务类型](#任务类型)
3. [参数规范](#参数规范)
4. [输出约束](#输出约束)
5. [最佳实践](#最佳实践)
6. [示例模板](#示例模板)

---

## 任务结构

一个完整的 CodeAct 任务应包含以下要素：

```python
from nodes.subagents.code_act.todo_list import TodoTask, TodoTaskType, TodoTaskStatus

task = TodoTask(
    id="task_001",                    # 唯一任务ID
    type=TodoTaskType.GENERAL,        # 任务类型
    status=TodoTaskStatus.PENDING,    # 任务状态
    priority=1,                       # 优先级 (1-10, 数字越小优先级越高)
    description="任务描述",            # 清晰的任务描述
    parameters={                      # 任务参数
        # 输入参数
        "input_file": "/path/to/input.csv",
        
        # 输出约束 (重要！)
        "output_constraints": {
            "f1_score": {"min": 0.01, "max": 1.0, "description": "F1分数应大于0"},
        }
    }
)
```

---

## 任务类型

### 1. GENERAL (通用任务)

用于一般性的数据处理、计算、分析任务。

```python
task = TodoTask(
    type=TodoTaskType.GENERAL,
    description="计算CSV文件中两列的相关系数",
    parameters={
        "input_file": "/data/input.csv",
        "column_a": "feature_1",
        "column_b": "feature_2",
    }
)
```

### 2. MCP_TOOL (MCP工具调用)

用于调用外部 MCP 服务。

```python
task = TodoTask(
    type=TodoTaskType.MCP_TOOL,
    description="使用 NetTCR 预测 TCR 结合亲和力",
    parameters={
        "tool_name": "analyze_vdj_batch",
        "input_file": "/data/sequences.fasta",
        "output_dir": "/data/output/"
    }
)
```

### 3. FILE_CONVERT (文件格式转换)

用于文件格式转换任务。

```python
task = TodoTask(
    type=TodoTaskType.FILE_CONVERT,
    description="将CSV文件转换为FASTA格式",
    parameters={
        "input_file": "/data/sequences.csv",
        "output_file": "/data/sequences.fasta",
        "sequence_column": "tcr_sequence"
    }
)
```

---

## 参数规范

### 输入参数命名规范

| 参数类型 | 推荐命名 | 说明 |
|---------|---------|------|
| 输入文件 | `input_file`, `prediction_file`, `data_file` | 自动触发数据探索 |
| 输出文件 | `output_file`, `output_dir` | 指定输出位置 |
| 数据列 | `column_name`, `target_column` | 指定处理列 |
| 文件格式 | `format`, `output_format` | 指定格式 |

### 参数值规范

```python
parameters = {
    # ✅ 推荐：使用绝对路径
    "input_file": "/data/sessions/session_id/input/data.csv",
    
    # ❌ 不推荐：使用相对路径
    # "input_file": "./data.csv",
    
    # ✅ 推荐：明确指定文件格式
    "file_format": "csv",
    
    # ✅ 推荐：提供示例值或说明
    "score_type": "F1",  # 可选值: F1, Precision, Recall, Accuracy
}
```

---

## 输出约束

### 为什么需要输出约束？

输出约束帮助 CodeAct：
1. **验证结果合理性** - 自动检测异常输出
2. **触发重试机制** - 当结果不满足约束时自动重试
3. **提供调试信息** - 帮助定位问题

### 约束格式

```python
"output_constraints": {
    "字段名": {
        "min": 最小值,           # 可选
        "max": 最大值,           # 可选
        "non_zero": True/False,  # 可选，是否必须非零
        "description": "说明"    # 可选，约束说明
    }
}
```

### 常见约束示例

#### F1/Accuracy 评分任务
```python
"output_constraints": {
    "f1_score": {"min": 0.01, "max": 1.0, "non_zero": True, 
                 "description": "F1分数应在0-1之间且大于0"},
    "precision": {"min": 0.0, "max": 1.0},
    "recall": {"min": 0.0, "max": 1.0}
}
```

#### 计数/统计任务
```python
"output_constraints": {
    "total_count": {"min": 0, "description": "计数应非负"},
    "match_count": {"min": 0, "non_zero": True, 
                    "description": "匹配数应大于0"}
}
```

#### 百分比任务
```python
"output_constraints": {
    "percentage": {"min": 0.0, "max": 100.0},
    "ratio": {"min": 0.0, "max": 1.0}
}
```

---

## 最佳实践

### 1. 描述清晰具体

```python
# ❌ 不好的描述
description = "分析数据"

# ✅ 好的描述
description = """
分析 /data/predictions.csv 文件：
1. 读取 CSV 文件
2. 计算 precision, recall, f1_score
3. 将结果保存到 result 字典中
"""
```

### 2. 提供文件路径时触发数据探索

CodeAct 会自动探索以下参数指定的文件：
- `prediction_file`
- `input_file`
- `data_file`
- `file_path`
- `csv_file`
- `input_path`

### 3. 为计算任务添加约束

```python
# 任何涉及数值计算的任务都应添加约束
parameters = {
    "input_file": "/data/scores.csv",
    "output_constraints": {
        # 防止代码选错列导致结果为0
        "score": {"min": 0.01, "non_zero": True}
    }
}
```

### 4. 指定列名时提供多个候选

```python
parameters = {
    "input_file": "/data/tcr.csv",
    # 提供可能的列名，帮助代码识别
    "possible_sequence_columns": ["tcr_sequence", "sequence", "seq", "CDR3"],
    "possible_label_columns": ["label", "target", "binding", "class"]
}
```

### 5. 复杂任务拆分

```python
# ❌ 不推荐：一个任务完成所有事情
description = "读取文件、处理数据、训练模型、评估、保存结果"

# ✅ 推荐：拆分为多个子任务
tasks = [
    TodoTask(id="task_1", description="读取并预处理数据", ...),
    TodoTask(id="task_2", description="训练模型", depends_on=["task_1"], ...),
    TodoTask(id="task_3", description="评估模型性能", depends_on=["task_2"], ...),
]
```

---

## 示例模板

### F1 评分任务模板

```python
f1_scoring_task = TodoTask(
    id="f1_scoring_001",
    type=TodoTaskType.GENERAL,
    status=TodoTaskStatus.PENDING,
    priority=1,
    description="""
对预测结果进行 F1 评分分析：
1. 读取预测文件
2. 识别真实标签列和预测标签列
3. 计算 precision, recall, f1_score
4. 返回评分结果
""",
    parameters={
        "prediction_file": "/data/sessions/session_id/predictions.csv",
        "score_type": "F1",
        "output_file": "/data/sessions/session_id/output/scores.json",
        
        # 输出约束
        "output_constraints": {
            "f1_score": {
                "min": 0.01, 
                "max": 1.0, 
                "non_zero": True,
                "description": "F1分数应大于0，否则可能选择了错误的列"
            },
            "precision": {"min": 0.0, "max": 1.0},
            "recall": {"min": 0.0, "max": 1.0}
        },
        
        # 可选：提供列名提示
        "column_hints": {
            "possible_true_labels": ["true_label", "actual", "y_true", "label"],
            "possible_predictions": ["predicted", "y_pred", "prediction", "pred"]
        }
    }
)
```

### 文件处理任务模板

```python
file_process_task = TodoTask(
    id="process_001",
    type=TodoTaskType.GENERAL,
    status=TodoTaskStatus.PENDING,
    priority=1,
    description="处理输入文件并生成统计报告",
    parameters={
        "input_file": "/data/input.csv",
        "output_file": "/data/output/report.json",
        
        # 输出约束
        "output_constraints": {
            "row_count": {"min": 1, "description": "至少应有一行数据"},
            "column_count": {"min": 1, "description": "至少应有一列"}
        }
    }
)
```

### MCP 工具调用模板

```python
mcp_tool_task = TodoTask(
    id="mcp_001",
    type=TodoTaskType.MCP_TOOL,
    status=TodoTaskStatus.PENDING,
    priority=1,
    description="调用 NetTCR 进行 TCR 结合预测",
    parameters={
        "tool_name": "analyze_vdj_batch",
        "input_file": "/data/sequences.fasta",
        "output_dir": "/data/output/",
        
        # 输出约束
        "output_constraints": {
            "predictions_count": {"min": 1, "description": "应产生预测结果"}
        }
    }
)
```

---

## 执行流程

新的 CodeAct 执行流程包含以下阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CodeAct 执行流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. read_todo          - 读取任务列表                                 │
│  2. select_next_task   - 选择下一个待执行任务                          │
│  3. explore_data       - 数据探索（自动识别输入文件）                   │
│  4. generate_code      - 生成代码                                     │
│  5. execute_code       - 执行代码                                     │
│  6. validate_output    - 验证输出（检查约束）                          │
│  7. update_todo        - 更新任务状态                                  │
│  8. check_pending      - 检查是否有待处理任务                          │
│                                                                      │
│  失败时: validate_output → revision → generate_code (重试)           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据探索阶段 (explore_data)

自动探索任务参数中指定的数据文件，获取：
- 列名列表
- 数据类型
- 样本数据
- 空值统计

### 输出验证阶段 (validate_output)

验证执行结果是否满足 `output_constraints`：
- 检查数值范围
- 检查非零约束
- 收集警告信息

---

## 常见问题

### Q: 为什么 F1 分数为 0？

A: 通常是因为代码选择了错误的列。解决方案：
1. 添加 `output_constraints` 约束
2. 在 `column_hints` 中提供可能的列名
3. 数据探索阶段会自动打印列名，帮助调试

### Q: 如何让任务自动重试？

A: 添加 `output_constraints`，当结果不满足约束时会自动触发 Revision 机制重试。

### Q: 如何查看数据探索结果？

A: 数据探索结果会打印在日志中，格式为：
```
[Data Exploration] Columns: ['col1', 'col2', ...]
[Data Exploration] Shape: [100, 5]
[Data Exploration] Dtypes: {...}
```

---

## 更新日志

- **v1.0** - 初始版本
  - 添加数据探索节点
  - 添加输出验证节点
  - 支持 output_constraints 参数
  - 编写任务生成指南

