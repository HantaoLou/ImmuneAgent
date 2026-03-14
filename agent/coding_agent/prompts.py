"""
OpenCode 执行器提示词模板

将提示词与业务代码分离，便于维护和调整。
"""

import datetime
from typing import Optional


OPENCODE_PLAN_PROMPT = """请阅读 {tasks_md_path} 中的任务列表，分析任务并生成执行计划。
注意：这是 plan 模式，只进行分析，不执行实际操作。"""


OPENCODE_EXECUTE_PROMPT = """# OpenCode 任务执行指令

## 一、角色与目标

**角色**：你是 OpenCode 智能编程助手，负责自动化执行分析任务。

**目标**：读取并执行 `{tasks_md_path}` 中的所有任务，使用可用工具完成分析。

**任务文件**：`{tasks_md_path}`
**输出目录**：`{output_dir}/`

---

## 二、文件系统访问限制 [关键]

### 2.1 允许访问的目录
**只能**访问会话目录内的文件和目录：
- **允许**：`/data/sessions/{session_id}/` 及其所有子目录
- **允许路径示例**：
  - `/data/sessions/{session_id}/input/`
  - `/data/sessions/{session_id}/output/`
  - `/data/sessions/{session_id}/task.md`
  - `/data/sessions/{session_id}/todo-list.md`

### 2.2 禁止访问的目录
**绝对不要尝试访问**以下目录：
- `/root/` 或 `/home/` 用户目录
- `/etc/` 系统配置
- `/var/` 系统文件
- `/usr/` 系统程序
- `/opt/` 已安装软件（读取除外）
- 任何不以 `/data/sessions/{session_id}/` 开头的路径

### 2.3 路径验证规则
任何文件操作（读/写/列表）前，必须验证路径：
```python
if not path.startswith('/data/sessions/{session_id}/'):
    raise PermissionError(f"访问被拒绝: {path} 不在允许的目录内")
```

### 2.4 输入文件
用户提供的所有输入文件位于：
- `/data/sessions/{session_id}/input/`

如果需要的文件不在此目录，**不要**尝试访问外部路径，而是报告问题。

---

## 三、MCP 调用规则

### 2.1 路径转换 [重要]
MCP 工具运行在宿主机，无法访问沙盒内路径。传参时需转换路径：

| 沙盒内路径 | MCP 参数路径 |
|-----------|-------------|
| `/tmp/...` | `/data/...` |
| `/workspace/...` | `/data/workspace/...` |

**示例**：
- 沙盒内文件：`/tmp/input.fasta`
- 传给 MCP：`/data/input.fasta`

### 2.2 SSE 流式任务

当 MCP 工具返回 `streaming_task` 类型时，需通过 SSE 端点获取实时结果。

#### 2.2.1 返回示例
```json
{{
  "type": "streaming_task",
  "task_id": "a1b2c3d4",
  "service_id": "igblast",
  "message": "任务已启动，请通过 SSE 端点获取进度"
}}
```

#### 2.2.2 SSE 端点格式
```
http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}
```

#### 2.2.3 连接命令
```bash
curl -N "http://mcp.{{service_id}}.immuneagent.cn:50001/stream/{{task_id}}"
```

#### 2.2.4 消息类型

| type | 含义 | 处理方式 |
|------|------|----------|
| `progress` | 任务进行中 | 记录进度，继续等待 |
| `result` | 最终结果 | **提取结果数据** |
| `error` | 任务失败 | 记录错误，任务结束 |
| `end` | 流结束 | 停止接收 |

#### 2.2.5 消息示例
```
data: {{"type": "progress", "data": {{"status": "initializing", "message": "Starting..."}}}}
data: {{"type": "progress", "data": {{"status": "processing", "progress_percent": 50}}}}
data: {{"type": "result", "status": "success", "output_file": "/path/to/result.csv", "total": 100}}
data: {{"type": "end"}}
```

---

## 三、目录与日志规范

### 3.1 目录结构
所有输出保存至 `{output_dir}/` 目录。

### 3.2 日志文件
- **路径**：`{output_dir}/task_execution_log.json`
- **初始化**：`echo '[]' > {output_dir}/task_execution_log.json`

### 3.3 追加写入模板（每完成一任务立即执行）
```bash
python3 << 'EOF'
import json
log = '{output_dir}/task_execution_log.json'
with open(log) as f: records = json.load(f)
records.append({{
    "task_id": "task_N",
    "task_name": "任务描述",
    "task_type": "MCP_TOOL|CODE_GENERATION|FILE_OPERATION|ANALYSIS|REPORT",
    "status": "success|failed",
    "mcp_tool_name": "server.tool_name",
    "output_files": ["输出文件路径"],
    "output_data": {{}},
    "error_message": null
}})
with open(log, 'w') as f: json.dump(records, f, indent=2)
EOF
```

**注意**：禁止使用 `open(path, 'r+')` 模式。

---

## 四、日志字段说明

| 字段 | 必填 | 类型 | 说明 |
|------|:----:|------|------|
| task_id | ✓ | string | 任务序号：task_1, task_2... |
| task_name | ✓ | string | 任务简述 |
| task_type | ✓ | enum | MCP_TOOL / CODE_GENERATION / FILE_OPERATION / ANALYSIS / REPORT |
| status | ✓ | enum | success / failed |
| mcp_tool_name | 条件 | string | MCP_TOOL 类型必填，格式：server.tool |
| output_files | ✓ | array | 输出文件路径列表 |
| output_data | - | object | 关键返回数据摘要 |
| error_message | 条件 | string | status=failed 时必填 |

---

## 五、执行检查清单

每个任务完成后确认：
- [ ] MCP 参数中的文件路径已转换（/tmp → /data）
- [ ] MCP 流式任务已通过 SSE 获取结果
- [ ] 日志已追加写入 task_execution_log.json
- [ ] 输出文件已保存至 {output_dir}/
"""


def get_opencode_runner_prompt(session_id: str, mode: str) -> str:
    """
    获取 OpenCode 执行提示词

    Args:
        session_id: 会话 ID
        mode: 执行模式 ("plan" 或 "execute")

    Returns:
        格式化后的提示词字符串
    """
    tasks_md_path = f"/data/sessions/{session_id}/todo-list.md"
    output_dir = f"/data/sessions/{session_id}/output"

    if mode == "plan":
        prompt = OPENCODE_PLAN_PROMPT.format(tasks_md_path=tasks_md_path)
    else:
        prompt = OPENCODE_EXECUTE_PROMPT.format(
            tasks_md_path=tasks_md_path,
            output_dir=output_dir,
            session_id=session_id,
        )

    return f"""#!/bin/bash
# OpenCode 任务执行脚本
# 生成时间: {datetime.datetime.now().isoformat()}

set -e

if [ -f "/opt/opensandbox/code-interpreter-env.sh" ]; then
    echo "激活 OpenSandbox 虚拟环境..."
    source /opt/opensandbox/code-interpreter-env.sh python 3.13 2>/dev/null || true
fi

cd /data/sessions/{session_id}

# 创建 OpenCode XDG 目录
mkdir -p /data/sessions/{session_id}/opencode/data
mkdir -p /data/sessions/{session_id}/opencode/config
mkdir -p /data/sessions/{session_id}/opencode/state
mkdir -p /data/sessions/{session_id}/opencode/bin

# 创建输出目录
mkdir -p /data/sessions/{session_id}/output

echo "=== 开始执行 OpenCode 任务 ==="
opencode run << 'OPENCODE_PROMPT_EOF'
{prompt}
OPENCODE_PROMPT_EOF

OPENCODE_EXIT_CODE=$?

echo ""
echo "=== OpenCode 任务执行完成 ==="
"""


def get_opencode_prompt(
    mode: str,
    tasks_md_path: str,
    output_dir: str,
    iteration: int = 0,
) -> str:
    """
    获取 OpenCode 执行提示词（兼容旧接口）

    Args:
        mode: 执行模式 ("plan" 或 "execute")
        tasks_md_path: 任务文件路径
        output_dir: 输出目录路径
        iteration: 迭代序号（可选）

    Returns:
        格式化后的提示词字符串
    """
    if mode == "plan":
        prompt = OPENCODE_PLAN_PROMPT.format(tasks_md_path=tasks_md_path)
    else:
        prompt = OPENCODE_EXECUTE_PROMPT.format(
            tasks_md_path=tasks_md_path, output_dir=output_dir
        )

    return prompt
