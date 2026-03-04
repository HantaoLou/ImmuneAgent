"""
CodeAct Executor - 统一的沙盒代码执行接口

这是其他子图与 OpenSandbox 沟通的唯一入口。
所有沙盒操作都应该通过此接口执行，而不是直接调用 opensandbox_executor。

架构原则：
- 其他子图（executor, result_evaluator, immunity, supervisor）不直接调用 OpenSandbox
- 所有代码执行请求发送给 CodeAct 子图
- CodeAct 负责：生成代码 → 执行代码 → 返回结果
- OpenSandbox 是远程沙盒，只有 CodeAct 子图与之直接交互

使用示例：
    from utils.codeact_executor import execute_code_via_codeact

    # 场景1: CSV 转 FASTA
    result = execute_code_via_codeact(
        task_description="将 /data/sessions/xxx/input/data.csv 转换为 FASTA 格式",
        sandbox_id=existing_sandbox_id,
        keep_alive=True
    )

    # 场景2: 读取远程文件
    result = execute_code_via_codeact(
        task_description="读取 /data/sessions/xxx/output 目录下所有 .csv 文件",
        sandbox_id=opensandbox_id
    )
"""

from __future__ import annotations

import os
import json
import asyncio
import time
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import sys

# 获取 agent 目录
AGENT_DIR = Path(__file__).parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


class CodeActExecutionStatus(str, Enum):
    """执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"


@dataclass
class CodeActResult:
    """CodeAct 执行结果"""
    status: CodeActExecutionStatus
    output: str = ""
    error: str = ""
    sandbox_id: Optional[str] = None
    returncode: int = 0
    parsed_result: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0

    def is_success(self) -> bool:
        return self.status == CodeActExecutionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "sandbox_id": self.sandbox_id,
            "returncode": self.returncode,
            "parsed_result": self.parsed_result,
            "execution_time_ms": self.execution_time_ms
        }


def is_codeact_available() -> bool:
    """检查 CodeAct 是否可用"""
    try:
        from utils.opensandbox_executor import is_opensandbox_enabled
        return is_opensandbox_enabled()
    except ImportError:
        return False


def execute_code_via_codeact(
    task_description: str,
    code_template: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    timeout_seconds: int = 120,
    keep_alive: bool = True,
    env: Optional[Dict[str, str]] = None,
    auto_parse_json: bool = True
) -> CodeActResult:
    """
    通过 CodeAct 在沙盒中执行代码（同步接口）

    这是其他子图与 OpenSandbox 沟通的唯一入口。

    Args:
        task_description: 任务描述（自然语言，描述要执行的任务）
        code_template: 可选的代码模板（如果提供，直接执行此代码，不经过 LLM 生成）
        sandbox_id: 现有的沙盒ID（复用沙盒）
        timeout_seconds: 超时时间（秒）
        keep_alive: 是否保持沙盒存活
        env: 环境变量
        auto_parse_json: 是否自动解析输出中的 JSON

    Returns:
        CodeActResult: 包含执行结果的对象

    Example:
        # 简单任务 - 让 CodeAct 生成代码
        result = execute_code_via_codeact(
            task_description="读取 /data/sessions/xxx/output/result.csv 的前5行"
        )

        # 指定代码 - 直接执行
        result = execute_code_via_codeact(
            task_description="执行文件复制",
            code_template=\'\'\'
import shutil
shutil.copy("/data/source.csv", "/data/target.csv")
print("Done")
\'\'\'
        )
    """
    start_time = time.time()

    try:
        # 如果提供了代码模板，直接执行
        if code_template:
            return _execute_code_direct(
                code=code_template,
                task_description=task_description,
                sandbox_id=sandbox_id,
                timeout_seconds=timeout_seconds,
                keep_alive=keep_alive,
                env=env,
                auto_parse_json=auto_parse_json
            )

        # 否则，通过 CodeAct 子图生成代码并执行
        return _execute_via_codeact_subgraph(
            task_description=task_description,
            sandbox_id=sandbox_id,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            auto_parse_json=auto_parse_json
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"CodeAct execution failed: {str(e)}",
            execution_time_ms=execution_time
        )


async def execute_code_via_codeact_async(
    task_description: str,
    code_template: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    timeout_seconds: int = 120,
    keep_alive: bool = True,
    env: Optional[Dict[str, str]] = None,
    auto_parse_json: bool = True
) -> CodeActResult:
    """
    异步版本的 execute_code_via_codeact
    """
    # 在事件循环中运行同步版本
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: execute_code_via_codeact(
            task_description=task_description,
            code_template=code_template,
            sandbox_id=sandbox_id,
            timeout_seconds=timeout_seconds,
            keep_alive=keep_alive,
            env=env,
            auto_parse_json=auto_parse_json
        )
    )


def _execute_code_direct(
    code: str,
    task_description: str,
    sandbox_id: Optional[str],
    timeout_seconds: int,
    keep_alive: bool,
    env: Optional[Dict[str, str]],
    auto_parse_json: bool
) -> CodeActResult:
    """
    直接执行代码（不经过 LLM 生成）

    当调用者已经知道要执行什么代码时使用此方法。
    """
    start_time = time.time()

    try:
        from utils.opensandbox_executor import run_code_in_opensandbox_sync, is_opensandbox_enabled

        if not is_opensandbox_enabled():
            return CodeActResult(
                status=CodeActExecutionStatus.ERROR,
                error="OpenSandbox not enabled. Set CODEACT_SANDBOX_PROVIDER=opensandbox or OPENSANDBOX_ENABLED=true"
            )

        # 准备环境变量
        exec_env = env or {}
        if "OPENSANDBOX_SKIP_MCP_INSTALL" not in exec_env:
            exec_env["OPENSANDBOX_SKIP_MCP_INSTALL"] = "true"

        # 执行代码
        result = run_code_in_opensandbox_sync(
            code=code,
            task_id=f"codeact_direct_{int(time.time())}",
            timeout_seconds=timeout_seconds,
            existing_sandbox_id=sandbox_id,
            keep_alive=keep_alive,
            env=exec_env
        )

        execution_time = int((time.time() - start_time) * 1000)

        if result is None:
            return CodeActResult(
                status=CodeActExecutionStatus.ERROR,
                error="Sandbox returned empty result",
                execution_time_ms=execution_time
            )

        # 解析结果
        stdout = result.get("stdout", "") + result.get("formatted_output", "")
        stderr = result.get("stderr", "")
        error_msg = result.get("error", "")
        returncode = result.get("returncode", 0)
        result_sandbox_id = result.get("sandbox_id", sandbox_id)

        # 确定状态
        # 注意: returncode 为 None 时，不应视为错误（OpenSandbox SDK 可能不返回 returncode）
        # 只有当 returncode 明确非 None 且非 0 时，才是执行错误
        if error_msg or (returncode is not None and returncode != 0):
            status = CodeActExecutionStatus.ERROR
        else:
            status = CodeActExecutionStatus.SUCCESS

        # 自动解析 JSON
        parsed_result = None
        if auto_parse_json and stdout:
            parsed_result = _extract_json_from_output(stdout)

        return CodeActResult(
            status=status,
            output=stdout,
            error=stderr or error_msg,
            sandbox_id=result_sandbox_id,
            returncode=returncode,
            parsed_result=parsed_result,
            execution_time_ms=execution_time
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"Direct execution failed: {str(e)}",
            execution_time_ms=execution_time
        )


def _execute_via_codeact_subgraph(
    task_description: str,
    sandbox_id: Optional[str],
    timeout_seconds: int,
    keep_alive: bool,
    auto_parse_json: bool
) -> CodeActResult:
    """
    通过 CodeAct 子图执行任务（LLM 生成代码）

    当任务需要 LLM 理解并生成代码时使用此方法。
    """
    start_time = time.time()

    try:
        # 导入 CodeAct 子图组件
        from nodes.subagents.code_act.graph import build_codeact_subgraph, CodeActState, CodeActExecutionMode
        from state import SubTask

        # 创建临时任务
        temp_task = SubTask(
            id=f"codeact_exec_{int(time.time())}",
            description=task_description,
            service_id="codeact",
            tool_name="execute_code",
            status="pending",
            result={}
        )

        # 构建 CodeAct 状态
        codeact_state = CodeActState(
            task=temp_task,
            task_description=task_description,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT
        )

        # 如果有现有的 sandbox_id，注入到状态中
        if sandbox_id:
            codeact_state.existing_sandbox_id = sandbox_id

        # 构建并执行子图
        subgraph = build_codeact_subgraph()
        result_state = subgraph.invoke(codeact_state.model_dump())

        execution_time = int((time.time() - start_time) * 1000)

        # 解析结果
        if isinstance(result_state, dict):
            exec_result = result_state.get("execution_result", {})
            generated_code = result_state.get("generated_code", "")
            result_sandbox_id = result_state.get("sandbox_id") or exec_result.get("sandbox_id") or sandbox_id
        else:
            exec_result = getattr(result_state, "execution_result", {}) or {}
            generated_code = getattr(result_state, "generated_code", "")
            result_sandbox_id = getattr(result_state, "sandbox_id", None) or sandbox_id

        # 提取输出
        output = exec_result.get("output", "") or exec_result.get("stdout", "")
        error = exec_result.get("error", "") or exec_result.get("stderr", "")
        status_str = exec_result.get("status", "unknown")

        # 确定状态
        if status_str == "success":
            status = CodeActExecutionStatus.SUCCESS
        elif status_str == "timeout":
            status = CodeActExecutionStatus.TIMEOUT
        else:
            status = CodeActExecutionStatus.ERROR if error else CodeActExecutionStatus.SUCCESS

        # 自动解析 JSON
        parsed_result = None
        if auto_parse_json and output:
            parsed_result = _extract_json_from_output(output)

        return CodeActResult(
            status=status,
            output=output,
            error=error,
            sandbox_id=result_sandbox_id,
            parsed_result=parsed_result,
            execution_time_ms=execution_time
        )

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        # 如果子图调用失败，回退到直接执行
        print(f"[CodeActExecutor] Subgraph execution failed, falling back to direct: {e}")
        return CodeActResult(
            status=CodeActExecutionStatus.ERROR,
            error=f"CodeAct subgraph failed: {str(e)}",
            execution_time_ms=execution_time
        )


def _extract_json_from_output(output: str) -> Optional[Dict[str, Any]]:
    """
    从输出中提取 JSON 结果

    支持多种格式：
    1. 标记格式: __JSON_START__ ... __JSON_END__
    2. 直接 JSON: {...}
    3. JSON 数组: [...]
    """
    import re

    # 尝试提取标记格式的 JSON
    marker_patterns = [
        r'__JSON_START__\s*(.*?)\s*__JSON_END__',
        r'__OUTPUT_FILES_JSON_START__\s*(.*?)\s*__OUTPUT_FILES_JSON_END__',
        r'__RESULT_JSON__\s*(.*?)\s*__END_RESULT_JSON__',
    ]

    for pattern in marker_patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # 尝试直接提取 JSON 对象
    json_object_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_object_pattern, output, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            continue

    # 尝试提取 JSON 数组
    json_array_pattern = r'\[[^\]]*\]'
    matches = re.findall(json_array_pattern, output, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list) and len(result) > 0:
                return {"items": result}
        except json.JSONDecodeError:
            continue

    return None


# ==================== 便捷函数 ====================

def read_remote_file(file_path: str, sandbox_id: Optional[str] = None, max_lines: int = 1000) -> CodeActResult:
    """
    读取远程沙盒中的文件

    Args:
        file_path: 远程文件路径
        sandbox_id: 沙盒 ID
        max_lines: 最大读取行数

    Returns:
        CodeActResult，output 包含文件内容
    """
    code = f'''
import os

file_path = "{file_path}"
max_lines = {max_lines}

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.rstrip('\\n'))
        print(f"__FILE_LINES__:{{len(lines)}}")
        for line in lines:
            print(line)
else:
    print(f"__FILE_NOT_FOUND__:{{file_path}}")
'''

    return execute_code_via_codeact(
        task_description=f"读取远程文件 {file_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )


def list_remote_directory(dir_path: str, sandbox_id: Optional[str] = None, pattern: str = "*") -> CodeActResult:
    """
    列出远程沙盒目录中的文件

    Args:
        dir_path: 远程目录路径
        sandbox_id: 沙盒 ID
        pattern: 文件匹配模式（glob 格式）

    Returns:
        CodeActResult，parsed_result 包含文件列表
    """
    code = f'''
import os
import json
from pathlib import Path

dir_path = "{dir_path}"
pattern = "{pattern}"

results = []
path = Path(dir_path)

if path.exists() and path.is_dir():
    for file_path in path.glob(pattern):
        if file_path.is_file():
            results.append({{
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "ext": file_path.suffix
            }})

print("__DIR_LIST_START__")
print(json.dumps(results, ensure_ascii=False))
print("__DIR_LIST_END__")
'''

    result = execute_code_via_codeact(
        task_description=f"列出远程目录 {dir_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )

    return result


def copy_file_in_sandbox(
    source_path: str,
    target_path: str,
    sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    在沙盒内复制文件

    Args:
        source_path: 源文件路径
        target_path: 目标文件路径
        sandbox_id: 沙盒 ID

    Returns:
        CodeActResult
    """
    code = f'''
import os
import shutil

source = "{source_path}"
target = "{target_path}"

try:
    if os.path.exists(source):
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source, target)
        print(f"__FILE_COPIED__:{{target}}")
    else:
        print(f"__FILE_NOT_FOUND__:{{source}}")
except Exception as e:
    print(f"__COPY_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"复制文件 {source_path} 到 {target_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )


def convert_csv_to_fasta(
    csv_path: str,
    output_path: str,
    sequence_columns: Optional[List[str]] = None,
    sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    将 CSV 文件转换为 FASTA 格式

    Args:
        csv_path: CSV 文件路径
        output_path: 输出 FASTA 文件路径
        sequence_columns: 序列列名列表（如果为 None，自动检测）
        sandbox_id: 沙盒 ID

    Returns:
        CodeActResult
    """
    seq_cols_str = str(sequence_columns) if sequence_columns else "None"

    code = f'''
import os
import csv

csv_path = "{csv_path}"
output_path = "{output_path}"
sequence_columns = {seq_cols_str}

# 支持的序列列名
SEQ_COLUMN_PATTERNS = [
    'sequence', 'seq', 'cdr3', 'CDR3',
    'heavy_dna', 'light_dna', 'Heavy_DNA', 'Light_DNA',
    'vdj_sequence', 'nucleotide_sequence'
]

def find_sequence_columns(headers):
    """自动检测序列列"""
    found = []
    for header in headers:
        header_lower = header.lower()
        for pattern in SEQ_COLUMN_PATTERNS:
            if pattern.lower() in header_lower:
                found.append(header)
                break
    return found

try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if sequence_columns is None:
            sequence_columns = find_sequence_columns(headers)

        if not sequence_columns:
            print("__CSV_NO_SEQ_COLUMNS__")
        else:
            # 转换路径（容器内）
            container_output = output_path.replace("/data/sessions/", "/tmp/sessions/", 1)
            os.makedirs(os.path.dirname(container_output), exist_ok=True)

            with open(container_output, 'w') as out_f:
                count = 0
                for row_idx, row in enumerate(reader):
                    for seq_col in sequence_columns:
                        seq = row.get(seq_col, '').strip()
                        if seq and len(seq) > 0:
                            header_id = row.get('id', row.get('ID', f'seq_{{row_idx}}'))
                            out_f.write(f"> {{header_id}}_{{seq_col}}\\n")
                            out_f.write(f"{{seq}}\\n")
                            count += 1

                print(f"__CSV_TO_FASTA_SUCCESS__:{{container_output}}:{{count}}")
except FileNotFoundError:
    print(f"__CSV_NOT_FOUND__:{{csv_path}}")
except Exception as e:
    print(f"__CSV_TO_FASTA_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"将 CSV {csv_path} 转换为 FASTA 格式",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )


# ==================== 更多便捷函数 ====================

def convert_rds_to_csv(
    rds_path: str,
    output_csv_path: Optional[str] = None,
    sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    将 RDS 文件转换为 CSV 格式

    Args:
        rds_path: RDS 文件路径
        output_csv_path: 输出 CSV 文件路径（可选，默认使用相同名称）
        sandbox_id: 沙盒 ID

    Returns:
        CodeActResult
    """
    code = f'''
import pyreadr
import pandas as pd
import os

rds_path = "{rds_path}"
csv_path = "{output_csv_path or rds_path.replace('.rds', '.csv').replace('.RDS', '.csv')}"

# Ensure output directory exists
output_dir = os.path.dirname(csv_path)
os.makedirs(output_dir, exist_ok=True)
try:
    os.chmod(output_dir, 0o777)
except Exception:
    pass

try:
    # Read RDS file
    result = pyreadr.read_r(rds_path)
    
    if not result:
        print("[ERROR] Failed to read RDS file or file is empty", flush=True)
        print("__RDS_CSV_FAILED__", flush=True)
    else:
        # Get the first data.frame or compatible object
        df = None
        for key, data in result.items():
            if isinstance(data, pd.DataFrame):
                df = data
                print(f"Found DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                break
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                try:
                    df = pd.DataFrame(data)
                    print(f"Converted dict to DataFrame: {{key}} with {{len(df)}} rows", flush=True)
                    break
                except Exception as e:
                    print(f"Could not convert {{key}} to DataFrame: {{e}}", flush=True)
                    continue
        
        if df is None:
            print("[ERROR] No compatible data object found in RDS file", flush=True)
            print("__RDS_CSV_FAILED__", flush=True)
        else:
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"[OK] Generated CSV with {{len(df)}} rows: {{csv_path}}", flush=True)
            print(f"__RDS_CSV_SUCCESS__:{{csv_path}}:{{len(df)}}", flush=True)
            
except ImportError:
    print("[ERROR] pyreadr library not available. Install with: pip install pyreadr", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
except Exception as e:
    print(f"[ERROR] Error converting RDS to CSV: {{e}}", flush=True)
    print("__RDS_CSV_FAILED__", flush=True)
'''

    return execute_code_via_codeact(
        task_description=f"将 RDS {rds_path} 转换为 CSV 格式",
        code_template=code,
        sandbox_id=sandbox_id,
        timeout_seconds=120,
        keep_alive=True
    )


def analyze_file_structure(
    file_path: str,
    sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    分析文件结构（列名、行数、数据类型等）

    Args:
        file_path: 文件路径
        sandbox_id: 沙盒 ID

    Returns:
        CodeActResult，parsed_result 包含文件结构信息
    """
    code = f'''
import os
import json

file_path = "{file_path}"

result = {{
    "path": file_path,
    "exists": False,
    "file_type": None,
    "row_count": 0,
    "column_names": [],
    "columns_info": {{}}
}}

if not os.path.exists(file_path):
    print(f"__FILE_NOT_FOUND__:{{file_path}}")
else:
    result["exists"] = True
    ext = os.path.splitext(file_path)[1].lower()
    result["file_type"] = ext
    
    try:
        if ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                result["column_names"] = reader.fieldnames or []
                rows = list(reader)
                result["row_count"] = len(rows)
                
                # Sample first few rows to infer types
                if rows and result["column_names"]:
                    for col in result["column_names"]:
                        sample_values = [row.get(col, '') for row in rows[:5]]
                        result["columns_info"][col] = {{
                            "sample_values": sample_values,
                            "type_guess": "string"
                        }}
                        # Check if numeric
                        try:
                            [float(v) for v in sample_values if v]
                            result["columns_info"][col]["type_guess"] = "numeric"
                        except ValueError:
                            pass
                        
        elif ext in ['.fasta', '.fa']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                seq_count = 0
                for line in f:
                    if line.startswith('>'):
                        seq_count += 1
                result["row_count"] = seq_count
                result["column_names"] = ["sequence_id", "sequence"]
                
        elif ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    result["row_count"] = len(data)
                    if data and isinstance(data[0], dict):
                        result["column_names"] = list(data[0].keys())
                elif isinstance(data, dict):
                    result["column_names"] = list(data.keys())
                    
        print("__FILE_ANALYSIS_START__")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("__FILE_ANALYSIS_END__")
        
    except Exception as e:
        print(f"__ANALYSIS_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"分析文件结构 {file_path}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )


def prepare_nettcr_input(
    input_csv: str,
    output_path: str,
    tcr_columns: Optional[List[str]] = None,
    sandbox_id: Optional[str] = None
) -> CodeActResult:
    """
    准备 NetTCR 输入文件

    Args:
        input_csv: 输入 CSV 文件路径
        output_path: 输出文件路径
        tcr_columns: TCR 序列列名列表
        sandbox_id: 沙盒 ID

    Returns:
        CodeActResult
    """
    tcr_cols_str = str(tcr_columns) if tcr_columns else "None"
    
    code = f'''
import os
import csv
import json

input_csv = "{input_csv}"
output_path = "{output_path}"
tcr_columns = {tcr_cols_str}

# NetTCR 标准列名映射
NETTCR_COLUMN_PATTERNS = [
    'tcr_sequence', 'CDR3', 'cdr3', 'CDR3_beta', 'cdr3_beta',
    'tcr_seq', 'tcr', 'TCR_sequence'
]

def find_tcr_columns(headers):
    """自动检测 TCR 序序列"""
    found = []
    for header in headers:
        header_lower = header.lower()
        for pattern in NETTCR_COLUMN_PATTERNS:
            if pattern.lower() in header_lower:
                found.append(header)
                break
    return found

try:
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        
        if tcr_columns is None:
            tcr_columns = find_tcr_columns(headers)
        
        if not tcr_columns:
            print("__NETTCR_NO_TCR_COLUMNS__")
        else:
            # 准备输出目录
            container_output = output_path.replace("/data/sessions/", "/tmp/sessions/", 1)
            os.makedirs(os.path.dirname(container_output), exist_ok=True)
            
            results = []
            for row in reader:
                for tcr_col in tcr_columns:
                    tcr_seq = row.get(tcr_col, '').strip()
                    if tcr_seq and len(tcr_seq) > 0:
                        results.append({{
                            "tcr_sequence": tcr_seq,
                            "original_column": tcr_col
                        }})
            
            # 写入输出
            with open(container_output, 'w', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=["tcr_sequence", "original_column"])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"__NETTCR_PREP_SUCCESS__:{{container_output}}:{{len(results)}}")
            
except FileNotFoundError:
    print(f"__NETTCR_INPUT_NOT_FOUND__:{{input_csv}}")
except Exception as e:
    print(f"__NETTCR_PREP_ERROR__:{{str(e)}}")
'''

    return execute_code_via_codeact(
        task_description=f"准备 NetTCR 输入文件 {input_csv}",
        code_template=code,
        sandbox_id=sandbox_id,
        keep_alive=True
    )


# ==================== 导出 ====================

__all__ = [
    # 主要接口
    "execute_code_via_codeact",
    "execute_code_via_codeact_async",
    "is_codeact_available",
    # 结果类
    "CodeActResult",
    "CodeActExecutionStatus",
    # 便捷函数
    "read_remote_file",
    "list_remote_directory",
    "copy_file_in_sandbox",
    "convert_csv_to_fasta",
    "convert_rds_to_csv",
    "analyze_file_structure",
    "prepare_nettcr_input",
]

