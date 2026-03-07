# -*- coding: utf-8 -*-
"""
Iterative Executor 提示词模块

定义生成 tasks.md 所需的提示词模板。
"""

from typing import Dict, List, Any, Optional


# ============================================================================
# Tasks.md 生成提示词
# ============================================================================

TASK_GENERATION_SYSTEM_PROMPT = """你是一个专业的生物医药任务规划专家。

你的职责是根据用户输入和实验计划，生成结构化的任务列表（tasks.md 格式）。

## 输出要求

1. **格式**: 输出必须是有效的 Markdown 格式
2. **结构**: 每个任务包含编号、标题、描述和具体步骤
3. **可执行性**: 每个任务步骤必须是可以用 MCP 工具或代码实现的
4. **依赖关系**: 任务之间可以有依赖关系，需要按顺序执行

## 可用的 MCP 服务

{mcp_services_info}

## 注意事项

- 优先使用 MCP 工具进行数据分析
- 生成的代码要考虑错误处理
- 输出文件保存到沙盒的 `/data/sessions/{session_id}/output/` 目录
"""

TASK_GENERATION_USER_PROMPT = """请根据以下信息生成任务列表：

## 用户输入
{user_input}

## 实验计划
{execution_plan}

## 可用参数
{extracted_parameters}

## 输入文件
{file_paths}

## 要求

1. 生成详细的任务步骤
2. 每个任务明确输入和输出
3. 使用可用的 MCP 服务
4. 输出文件保存到正确路径

请生成 tasks.md 内容：
"""

# ============================================================================
# 文件处理任务模板
# ============================================================================

FILE_PROCESSING_TASKS_TEMPLATE = """# 文件处理任务

## 任务目标
处理用户上传的文件，提取参数表。

## 输入文件
{file_list}

## 执行步骤

### 任务 1: 上传文件到沙盒
将本地文件上传到沙盒 `/data/sessions/{session_id}/input/` 目录。

### 任务 2: 分析文件内容
对每个文件进行内容分析：
- 识别文件类型 (CSV, FASTA, JSON, etc.)
- 提取列名/字段名
- 统计行数/记录数
- 检测数据类型

### 任务 3: 生成参数表
将分析结果整理成参数表 JSON 格式，保存到 `/data/sessions/{session_id}/.agent/file_params.json`

输出格式:
```json
{{
  "files": {{
    "metadata.csv": {{
      "path": "/data/sessions/{session_id}/input/metadata.csv",
      "type": "csv",
      "columns": ["col1", "col2", ...],
      "rows": 100,
      "detected_data_type": "antibody_metadata"
    }}
  }},
  "inferred_params": {{
    "target_organism": "H1N1",
    "analysis_type": "antibody_prediction"
  }}
}}
```
"""

# ============================================================================
# NetTCR 分析任务模板
# ============================================================================

NETTCR_ANALYSIS_TASKS_TEMPLATE = """# NetTCR 肽段-TCR 结合预测任务

## 任务目标
使用 NetTCR MCP 服务预测肽段与 TCR 的结合亲和力。

## 输入参数
{params_info}

## 执行步骤

### 任务 1: 检查肽段支持
使用 `check_peptide_support` 工具检查肽段是否支持预测。

输入参数:
- peptides: 肽段序列列表

### 任务 2: 执行结合预测
使用 `predict_tcr_binding_complete` 工具进行完整预测。

输入参数:
- peptides: 肽段序列
- tcr_sequences: TCR 序列（可选）
- output_file: 输出文件路径

### 任务 3: 分析预测结果
- 读取预测结果文件
- 筛选高亲和力结合
- 生成分析报告

### 任务 4: 保存结果
将预测结果保存到:
- `/data/sessions/{session_id}/output/nettcr_predictions.csv`
- `/data/sessions/{session_id}/output/reports/nettcr_analysis_report.md`
"""

# ============================================================================
# IgBLAST 分析任务模板
# ============================================================================

IGBLAST_ANALYSIS_TASKS_TEMPLATE = """# IgBLAST 抗体序列分析任务

## 任务目标
使用 IgBLAST MCP 服务分析抗体序列的 V(D)J 基因使用情况。

## 输入文件
{file_paths}

## 执行步骤

### 任务 1: 准备输入数据
- 检查输入文件格式
- 如需转换（如 CSV 转 FASTA），在沙盒中执行转换

### 任务 2: 执行 V(D)J 分析
使用 `analyze_vdj_batch` 工具批量分析序列。

输入参数:
- input_file: 输入 FASTA 文件路径
- output_dir: 输出目录

### 任务 3: 解析分析结果
- 读取 AIRR 格式输出
- 提取 V/D/J 基因使用统计
- 识别 CDR3 序列

### 任务 4: 保存结果
将分析结果保存到:
- `/data/sessions/{session_id}/output/igblast_results.csv`
- `/data/sessions/{session_id}/output/reports/igblast_analysis_report.md`
"""

# ============================================================================
# 评估提示词
# ============================================================================

EVALUATION_PROMPT = """请评估以下任务执行结果：

## 原始任务
{tasks_content}

## 执行输出
{execution_output}

## 评估标准
{evaluation_criteria}

## 评估维度

1. **完整性** (0-100): 是否完成了所有任务步骤
2. **正确性** (0-100): 输出结果是否正确
3. **质量** (0-100): 输出文件质量如何
4. **错误处理** (0-100): 是否有未处理的错误

## 输出格式

```json
{{
  "completeness_score": 85,
  "correctness_score": 90,
  "quality_score": 80,
  "error_handling_score": 95,
  "overall_score": 87.5,
  "issues": ["问题1", "问题2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "status": "success" | "needs_improvement" | "failed"
}}
```

请评估并输出结果：
"""

# ============================================================================
# 任务优化提示词
# ============================================================================

TASK_OPTIMIZATION_PROMPT = """根据评估结果优化任务列表：

## 原始任务
{original_tasks}

## 评估结果
{evaluation_result}

## 执行日志
{execution_log}

## 优化要求

1. 修复导致失败的任务步骤
2. 添加缺失的错误处理
3. 优化参数推断逻辑
4. 改进输出文件处理

请生成优化后的 tasks.md：
"""


# ============================================================================
# 辅助函数
# ============================================================================

def format_mcp_services_info(services: List[str]) -> str:
    """
    格式化 MCP 服务信息
    
    Args:
        services: MCP 服务名称列表
        
    Returns:
        str: 格式化的服务信息
    """
    if not services:
        return "- 暂无可用的 MCP 服务"
    
    service_descriptions = {
        "nettcr": "NetTCR - 肽段-TCR 结合预测",
        "igblast": "IgBLAST - 抗体/TCR 序列分析",
        "blast": "BLAST - 序列比对",
    }
    
    lines = []
    for service in services:
        desc = service_descriptions.get(service, service)
        lines.append(f"- {desc}")
    
    return "\n".join(lines)


def format_params_info(params: Dict[str, Any]) -> str:
    """
    格式化参数信息
    
    Args:
        params: 参数字典
        
    Returns:
        str: 格式化的参数信息
    """
    if not params:
        return "- 无"
    
    lines = []
    for key, value in params.items():
        if isinstance(value, (list, dict)):
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        else:
            value_str = str(value)
        lines.append(f"- {key}: {value_str}")
    
    return "\n".join(lines)


def format_file_paths_info(file_paths: Dict[str, str]) -> str:
    """
    格式化文件路径信息
    
    Args:
        file_paths: 文件路径字典
        
    Returns:
        str: 格式化的文件路径信息
    """
    if not file_paths:
        return "- 无"
    
    lines = []
    for name, path in file_paths.items():
        lines.append(f"- {name}: {path}")
    
    return "\n".join(lines)

