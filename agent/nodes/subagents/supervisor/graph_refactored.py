"""
Supervisor Agent Subgraph (Refactored)

重构版本 - 从"自己干活"模式改为"发布任务"模式
- supervisor 不再直接执行文件上传/分析代码
- 通过调用 codeact 子图来完成文件处理任务
- 代码量目标：从 2249 行减少到约 600 行

重构日期: 2026-03-03
"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
import sys
import os
import json
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path

# ==================== 日志配置 ====================
# 创建 supervisor 专用日志记录器
logger = logging.getLogger("supervisor_refactored")
logger.setLevel(logging.DEBUG)

# 确保日志目录存在
LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 文件处理器 - 详细日志
log_file = LOG_DIR / f"supervisor_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# 控制台处理器 - 关键信息
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# 避免重复添加处理器
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info(f"[Supervisor] 日志文件: {log_file}")

# ==================== 本地模块导入 ====================
from .prompt import TASK_CLASSIFICATION_SYSTEM_PROMPT, get_task_classification_user_prompt

# 添加 agent 目录到路径
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType, SubTask

# CodeAct 子图导入
from nodes.subagents.code_act.graph import (
    build_codeact_subgraph,
    CodeActState,
    CodeActExecutionMode,
    codeact_input_mapper,
    codeact_output_mapper
)

# LLM 相关导入
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from utils.llm_factory import create_reasoning_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    create_reasoning_llm = None
    HumanMessage = None
    SystemMessage = None
    print("Warning: langchain-related libraries not installed, will use keyword matching as fallback")


# ==================== 常量定义 ====================

class FileSourceType(str, Enum):
    """文件来源类型"""
    LOCAL = "local"     # 本地文件，需要上传到沙箱
    REMOTE = "remote"   # 沙箱服务器上的文件，无需上传
    URL = "url"         # 下载链接，需要在沙箱中下载


class TaskMode:
    """任务执行模式"""
    SYNC = "sync"       # 同步任务：等待执行完成后返回结果
    ASYNC = "async"     # 异步任务：发布任务后立即返回


# ==================== 数据模型 ====================

class DetectedFile(BaseModel):
    """检测到的文件信息"""
    path: str = Field(description="文件路径或 URL")
    source_type: FileSourceType = Field(description="文件来源类型")
    suggested_name: Optional[str] = Field(default=None, description="建议的文件名（用于 URL）")


class ExtractedFile(BaseModel):
    """LLM 提取的文件信息"""
    path: str = Field(description="文件路径或 URL")
    purpose: str = Field(description="文件用途，如 metadata, antigen_sequence, antibody_data")
    format: str = Field(default="unknown", description="文件格式: csv, fasta, pdb, rds, json, txt")
    source: str = Field(default="remote", description="文件来源: local / remote / url")


class ExtractedParam(BaseModel):
    """LLM 提取的参数"""
    name: str = Field(description="参数名 (snake_case 格式)")
    value: Any = Field(description="参数值")
    description: Optional[str] = Field(default=None, description="参数描述")


class LLMExtractionResult(BaseModel):
    """LLM 结构化提取结果"""
    task_description: str = Field(default="", description="用户任务的一句话描述")
    target_organism: Optional[str] = Field(default=None, description="目标生物/病原体")
    mcp_services: List[str] = Field(default_factory=list, description="需要使用的 MCP 服务列表")
    files: List[ExtractedFile] = Field(default_factory=list, description="检测到的文件列表")
    parameters: List[ExtractedParam] = Field(default_factory=list, description="提取的参数列表")
    notes: List[str] = Field(default_factory=list, description="用户的特殊要求或备注")
    analysis_type: str = Field(default="other", description="分析类型")


class FileAnalysis(BaseModel):
    """文件分析结果"""
    original_path: str = Field(description="原始文件路径")
    sandbox_path: str = Field(description="沙箱中的路径")
    file_type: str = Field(description="文件类型 (csv, fasta, pdb, etc.)")
    column_names: List[str] = Field(default_factory=list, description="列名（如果是表格文件）")
    detected_data_type: Optional[str] = Field(default=None, description="检测到的数据类型")
    content_summary: Optional[str] = Field(default=None, description="内容摘要")
    row_count: Optional[int] = Field(default=None, description="行数（如果是表格文件）")


class InputPreprocessResult(BaseModel):
    """输入预处理结果"""
    detected_files: List[DetectedFile] = Field(default_factory=list, description="检测到的文件")
    extracted_params: Dict[str, Any] = Field(default_factory=dict, description="提取的参数")
    task_description: str = Field(default="", description="任务描述")
    mcp_services: List[str] = Field(default_factory=list, description="MCP 服务列表")


# ==================== SupervisorState 定义 ====================

class SupervisorState(BaseModel):
    """Supervisor 子图状态"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        from_attributes=True
    )
    
    # 输入字段
    user_input: str = Field(description="用户原始输入")
    user_task_type: Optional[UserTaskType] = Field(default=None, description="用户任务类型")
    uploaded_files: List[str] = Field(default_factory=list, description="已上传的文件路径列表")
    sandbox_file_paths: Dict[str, str] = Field(default_factory=dict, description="沙箱文件路径映射")
    sandbox_dir: str = Field(description="沙箱目录路径")
    execution_plan: Optional[str] = Field(default=None, description="执行计划")
    
    # 预处理结果字段
    session_id: Optional[str] = Field(default=None, description="唯一会话 ID")
    preprocess_result: Optional[InputPreprocessResult] = Field(default=None, description="输入预处理结果")
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict, description="提取的参数表")
    file_analyses: List[FileAnalysis] = Field(default_factory=list, description="文件分析结果")
    sandbox_data_dir: Optional[str] = Field(default=None, description="沙箱数据目录路径")
    opensandbox_id: Optional[str] = Field(default=None, description="OpenSandbox 实例 ID")
    
    # CodeAct 调用结果
    codeact_upload_result: Optional[Dict[str, Any]] = Field(default=None, description="CodeAct 上传结果")
    codeact_analysis_result: Optional[Dict[str, Any]] = Field(default=None, description="CodeAct 分析结果")


# ==================== 核心辅助函数 ====================

def _generate_session_id() -> str:
    """生成唯一会话 ID，格式: 日期_时间_短UUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def _get_llm():
    """
    获取推理模型实例（用于任务分类和参数提取）
    
    使用通用 LLM 工厂创建推理模型，优先选择推理性能好的模型。
    
    Returns:
        LLM 实例，如果不可用则返回 None
    """
    if not LLM_AVAILABLE or create_reasoning_llm is None:
        return None
    
    return create_reasoning_llm(temperature=0.1)


def _classify_file_source(file_path: str) -> FileSourceType:
    """
    分类文件来源类型
    
    Args:
        file_path: 文件路径或 URL
        
    Returns:
        FileSourceType: 文件来源类型
    """
    if file_path.startswith(("http://", "https://")):
        return FileSourceType.URL
    
    # 沙箱服务器路径（容器外/服务器端）
    if file_path.startswith(("/data/", "/home/sandbox/", "/opt/", "/mnt/", "/shared/")):
        return FileSourceType.REMOTE
    
    # Windows 本地路径或相对路径
    if re.match(r'^[A-Za-z]:', file_path) or file_path.startswith(('./', '../')):
        return FileSourceType.LOCAL
    
    # 默认当作远程路径
    return FileSourceType.REMOTE


def _is_valid_file_path(path: str) -> bool:
    """
    判断是否为有效的文件路径（而非字典 key 或其他无效字符串）
    
    Args:
        path: 待检查的字符串
        
    Returns:
        bool: 是否为有效文件路径
    """
    if not path or len(path) < 3:
        return False
    
    # 检查是否包含文件扩展名（有效路径通常有）
    if '.' not in Path(path).name:
        return False
    
    # 检查是否为常见的字典 key 模式（如 h5ad_file, meta_csv_file）
    if '_' in path and not any(c in path for c in ['/', '\\', ':']):
        # 不包含路径分隔符的带下划线字符串可能是字典 key
        # 但需要排除一些特殊情况
        if path.replace('_', '').isalnum():
            return False
    
    # 检查是否为有效路径格式
    path_obj = Path(path)
    
    # URL 格式
    if path.startswith(('http://', 'https://')):
        return True
    
    # Unix 绝对路径
    if path.startswith('/'):
        return True
    
    # Windows 路径
    if re.match(r'^[A-Za-z]:', path):
        return True
    
    # 相对路径
    if path.startswith(('./', '../')):
        return True
    
    return False


def _detect_file_paths(text: str) -> List[DetectedFile]:
    """
    从文本中检测文件路径
    
    Args:
        text: 用户输入文本
        
    Returns:
        检测到的文件列表
    """
    detected = []
    seen = set()
    
    # 匹配各种文件路径模式
    patterns = [
        r'/data/[^\s,\)\]\}]+',                    # /data/ 开头的路径
        r'/home/[^\s,\)\]\}]+',                   # /home/ 开头的路径
        r'/tmp/[^\s,\)\]\}]+',                    # /tmp/ 开头的路径
        r'[A-Za-z]:\\[^\s,\)\]\}]+',              # Windows 路径
        r'https?://[^\s,\)\]\}]+',                # URL
        r'\./[^\s,\)\]\}]+',                      # 相对路径 ./
        r'(?<!\w)/[a-zA-Z0-9_\-./]+\.[a-zA-Z]{2,4}(?!\w)',  # 带扩展名的 Unix 路径
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            # 去除各种包裹符号，包括 markdown 反引号
            path = match.group(0).strip('.,;:\'\"`')
            if path and path not in seen and len(path) > 3:
                seen.add(path)
                source_type = _classify_file_source(path)
                detected.append(DetectedFile(
                    path=path,
                    source_type=source_type
                ))
    
    return detected


def _build_file_upload_task(detected_files: List[DetectedFile], session_id: str) -> str:
    """
    构建文件上传/转存任务描述
    
    重要原则：所有文件都统一转存到沙盒会话目录，方便统一管理
    
    Args:
        detected_files: 检测到的文件列表
        session_id: 会话 ID
        
    Returns:
        格式化的任务描述
    """
    sandbox_input_dir = f"/data/sessions/{session_id}/input"
    container_input_dir = f"/tmp/sessions/{session_id}/input"
    
    # 去重
    unique_files = {}
    for f in detected_files:
        if f.path not in unique_files:
            unique_files[f.path] = f
    
    # 按来源类型分组
    local_files = [f for f in unique_files.values() if f.source_type == FileSourceType.LOCAL]
    remote_files = [f for f in unique_files.values() if f.source_type == FileSourceType.REMOTE]
    url_files = [f for f in unique_files.values() if f.source_type == FileSourceType.URL]
    
    # 生成文件列表 JSON
    files_json = []
    for f in unique_files.values():
        file_name = Path(f.path).name
        files_json.append({
            "original_path": f.path,
            "source_type": f.source_type.value,
            "target_name": file_name
        })
    
    import json
    files_json_str = json.dumps(files_json, ensure_ascii=False, indent=2)
    
    task_desc = f"""## 任务：将所有文件统一转存到沙盒会话目录

**重要原则**：所有文件都必须转存到会话目录，统一管理！

### 目标目录
- 数据目录: {sandbox_input_dir}
- 容器目录: {container_input_dir}

### 待处理文件（共 {len(unique_files)} 个）

```json
{files_json_str}
```

### 文件来源分类
- LOCAL 文件（{len(local_files)} 个）: 需要从本地路径读取并保存
- REMOTE 文件（{len(remote_files)} 个）: 服务器上已存在的文件，需要复制到会话目录
- URL 文件（{len(url_files)} 个）: 需要下载

### 执行要求

1. **创建目标目录**
   ```python
   import os
   os.makedirs("{container_input_dir}", exist_ok=True)
   ```

2. **处理所有文件** - 根据来源类型采取不同策略：

   - **REMOTE 文件**：使用 shutil.copy2 复制到目标目录
   - **URL 文件**：使用 requests 或 wget 下载到目标目录
   - **LOCAL 文件**：尝试读取并保存（如果是本地测试环境）

3. **生成文件映射** - 返回原始路径到沙盒路径的映射

### 返回格式（必须包含 JSON 标记）
```python
print("__FILE_TRANSFER_START__")
print(json.dumps({{
    "status": "success",
    "uploaded_files": {{
        "/data/xxx/original.csv": "{sandbox_input_dir}/original.csv",
        ...
    }},
    "total_files": {len(unique_files)}
}}))
print("__FILE_TRANSFER_END__")
```

如果某个文件不存在或处理失败，在 error 字段中说明，但继续处理其他文件。
"""
    return task_desc


def _build_file_analysis_task(sandbox_file_paths: Dict[str, str]) -> str:
    """
    构建文件分析任务描述
    
    Args:
        sandbox_file_paths: 沙箱文件路径映射 {原始路径: 沙箱路径}
        
    Returns:
        格式化的任务描述
    """
    task_desc = """## 任务：分析文件内容

### 待分析文件

"""
    for original_path, sandbox_path in sandbox_file_paths.items():
        ext = Path(sandbox_path).suffix.lower()
        task_desc += f"- `{sandbox_path}` (格式: {ext})\n"
    
    task_desc += """
### 分析要求

1. **CSV/TSV 文件**：
   - 读取前 10 行预览
   - 获取列名和数据类型
   - 统计行数
   - 识别可能的生物学数据类型（抗体序列、元数据等）

2. **FASTA 文件**：
   - 统计序列数量
   - 获取序列 ID 列表
   - 检测序列类型（DNA/蛋白质）

3. **JSON 文件**：
   - 解析 JSON 结构
   - 获取主要字段

4. **PDB 文件**：
   - 获取结构信息
   - 统计原子/链数量

### 返回格式
返回 JSON 格式的分析结果：
{
    "files": [
        {
            "path": "沙箱路径",
            "file_type": "csv|fasta|json|pdb|...",
            "row_count": 行数,
            "column_names": ["列名1", "列名2"],
            "detected_data_type": "antibody|antigen|metadata|...",
            "summary": "内容摘要"
        }
    ]
}
"""
    return task_desc


def _call_codeact(
    state: SupervisorState,
    task_description: str,
    task_type: str = "general",
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    调用 CodeAct 子图执行任务
    
    这是重构的核心函数：supervisor 不再直接执行代码，
    而是通过 CodeAct 子图来执行。
    
    Args:
        state: 当前 supervisor 状态
        task_description: 任务描述（自然语言）
        task_type: 任务类型 ("file_upload", "file_analysis", "general")
        parameters: 任务参数
        
    Returns:
        Dict: CodeAct 执行结果
            - status: "success" | "failed"
            - output: 输出内容
            - error: 错误信息（如有）
    """
    logger.info(f"  [_call_codeact] 开始执行任务: {task_type}")
    
    try:
        # 创建 SubTask - 注意字段名和类型必须匹配 SubTask 模型定义
        # task_type 必须是 UserTaskType 枚举，不能是字符串
        task = SubTask(
            task_id=str(uuid.uuid4())[:8],  # 字段名是 task_id，不是 id
            task_type=UserTaskType.EXECUTE_PLAN,  # 使用 EXECUTE_PLAN 作为通用任务类型
            content=task_description,  # 字段名是 content
            result={"tools": [], "inputs": []}
        )
        
        # 创建一个最小化的 GlobalState 用于传递 opensandbox_id 和 session_id
        # 必须在构造时传入，避免 Pydantic validate_assignment 验证失败
        temp_parent_state = GlobalState(
            user_input="",
            sandbox_dir="",
            session_id=state.session_id,  # 传递 session_id 以便 codeact 能找到正确的沙盒目录
            merged_result={"opensandbox_id": state.opensandbox_id} if state.opensandbox_id else {}
        )
        
        # 构建 CodeAct 状态（在构造时传入 parent_state）
        codeact_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters=parameters or {},
            parent_state=temp_parent_state
        )
        
        # 获取 CodeAct 子图
        # 注意：supervisor 阶段的临时任务（文件上传、分析）使用 legacy 模式，不需要 todo-list.md
        # 只有 executor 阶段才使用 todo 模式
        codeact_graph = build_codeact_subgraph(use_todo_mode=False)
        
        # 执行子图
        logger.debug(f"  [_call_codeact] 调用 CodeAct 子图...")
        result_state = codeact_graph.invoke(codeact_state)
        
        # 调试：检查 result_state 类型和内容
        logger.debug(f"  [_call_codeact] result_state type: {type(result_state)}")
        if result_state:
            if isinstance(result_state, dict):
                logger.debug(f"  [_call_codeact] result_state keys: {list(result_state.keys())}")
                er = result_state.get("execution_result")
                logger.debug(f"  [_call_codeact] execution_result type: {type(er)}, value: {er}")
            else:
                er = getattr(result_state, 'execution_result', None)
                logger.debug(f"  [_call_codeact] execution_result type: {type(er)}")
        
        # 提取结果
        result = codeact_output_mapper(result_state)
        logger.debug(f"  [_call_codeact] mapped result: {result}")
        
        # 如果 CodeAct 返回了新的 sandbox_id，更新到 supervisor state
        if result_state:
            if isinstance(result_state, dict):
                er = result_state.get("execution_result")
                new_sandbox_id = er.get("sandbox_id") if er else None
                if not new_sandbox_id and result_state.get("parent_state"):
                    ps = result_state.get("parent_state")
                    if hasattr(ps, 'merged_result'):
                        mr = ps.merged_result
                        new_sandbox_id = mr.get('opensandbox_id') if mr else None
            else:
                er = getattr(result_state, 'execution_result', None)
                new_sandbox_id = er.get("sandbox_id") if er else None
            
            if new_sandbox_id and new_sandbox_id != state.opensandbox_id:
                state.opensandbox_id = new_sandbox_id
                logger.debug(f"  [_call_codeact] 更新 opensandbox_id: {new_sandbox_id}")
        
        # 详细记录执行结果
        status = result.get('status', 'unknown')
        if status == 'success':
            logger.info(f"  [_call_codeact] 执行完成: {status}")
            output_preview = str(result.get('output', ''))[:200]
            if output_preview:
                logger.debug(f"  [_call_codeact] 输出预览: {output_preview}...")
        else:
            # 失败时详细记录所有错误信息
            logger.error(f"  [_call_codeact] 执行完成: {status}")
            logger.error(f"  [_call_codeact] 错误信息: {result.get('error')}")
            logger.error(f"  [_call_codeact] 错误类型: {result.get('error_type')}")
            logger.error(f"  [_call_codeact] 错误类别: {result.get('error_category')}")
            if result.get('output'):
                logger.error(f"  [_call_codeact] 输出: {str(result.get('output'))[:500]}")
        return result
        
    except Exception as e:
        logger.error(f"  [_call_codeact] 执行失败: {e}")
        logger.error(f"  [_call_codeact] 异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"  [_call_codeact] 堆栈跟踪:\n{traceback.format_exc()}")
        return {
            "status": "failed",
            "output": None,
            "error": str(e),
            "error_type": type(e).__name__,
            "error_category": "exception"
        }


# ==================== LLM 结构化提取函数 ====================

PARAMETER_EXTRACTION_SYSTEM_PROMPT = """You are a bioinformatics task analysis expert. Your task is to extract structured information from user input.

## Output Format Requirements
Output strictly in the following JSON format, do not include any other text:

```json
{
  "task_description": "A one-sentence description of the task the user wants to accomplish",
  "target_organism": "Target organism/pathogen (e.g., H5N1, SARS-CoV-2, flu), null if not specified",
  "mcp_services": ["service_name_1", "service_name_2"],
  "files": [
    {
      "path": "Full file path or URL",
      "purpose": "File purpose (e.g., metadata, antigen_sequence, antibody_data)",
      "format": "File format (csv, fasta, pdb, rds, json, txt)",
      "source": "Source type: local / remote / url"
    }
  ],
  "parameters": [
    {
      "name": "Parameter name (use snake_case format)",
      "value": "Parameter value",
      "description": "Parameter description"
    }
  ],
  "notes": ["User's special requirements or notes"],
  "analysis_type": "Analysis type (antibody_discovery / structure_prediction / sequence_analysis / data_integration / other)"
}
```

## File Source Classification Rules
- Paths starting with /data/, /home/sandbox/, /opt/, /mnt/, /shared/ → "remote" (files on the server)
- Starting with http:// or https:// → "url" (download links)
- Windows paths (e.g., C:\\, D:/) or relative paths (./, ../) → "local" (local files)

## Parameter Name Normalization Rules
- Use snake_case format (lowercase letters, underscore separated)
- Common mappings:
  - "antigen file" → "antigen_file"
  - "metadata" → "metadata_file"
  - "RDS file" → "rds_file"
  - "output directory" → "output_dir"
"""

PARAMETER_EXTRACTION_USER_PROMPT = """Please extract structured information from the following user input:

---
{user_input}
---

Output strictly in JSON format, do not include any explanation or other text."""


def _extract_json_from_response(text: str) -> Optional[str]:
    """从 LLM 响应中提取 JSON 字符串"""
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    
    # 从 markdown 代码块提取
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    # 查找 JSON 对象
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[brace_start:i+1]
    
    return None


def _llm_extract_structured_input(user_input: str) -> LLMExtractionResult:
    """
    使用 LLM 从用户输入中提取结构化信息
    
    Args:
        user_input: 用户原始输入文本
        
    Returns:
        LLMExtractionResult: 结构化提取结果
    """
    llm = _get_llm()
    
    if not llm:
        logger.warning("  [WARN] LLM 不可用，使用正则回退")
        return _fallback_regex_extraction(user_input)
    
    try:
        messages = [
            SystemMessage(content=PARAMETER_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=PARAMETER_EXTRACTION_USER_PROMPT.format(user_input=user_input))
        ]
        
        logger.info("  [LLM] 正在提取结构化信息...")
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        json_str = _extract_json_from_response(result_text)
        if json_str:
            try:
                result_dict = json.loads(json_str)
                result = LLMExtractionResult.model_validate(result_dict)
                logger.info(f"  [OK] LLM 提取成功: {len(result.files)} 文件, {len(result.parameters)} 参数, {len(result.mcp_services)} 服务")
                return result
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"  [WARN] JSON 解析失败: {e}")
        
        logger.warning("  [WARN] 无法解析 LLM 响应，使用正则回退")
        return _fallback_regex_extraction(user_input)
        
    except Exception as e:
        logger.warning(f"  [WARN] LLM 提取失败: {e}")
        return _fallback_regex_extraction(user_input)


def _fallback_regex_extraction(user_input: str) -> LLMExtractionResult:
    """正则回退提取"""
    result = LLMExtractionResult(task_description="")
    
    # 提取文件路径
    detected_files = _detect_file_paths(user_input)
    for f in detected_files:
        ext = Path(f.path).suffix.lower().lstrip('.') or 'unknown'
        result.files.append(ExtractedFile(
            path=f.path,
            purpose="unknown",
            format=ext,
            source=f.source_type.value
        ))
    
    # 提取 MCP 服务
    service_pattern = r'(?:use|using|mcp\s*services?)[:\s]*[-\s]*(\w+)'
    for match in re.finditer(service_pattern, user_input, re.IGNORECASE):
        service = match.group(1).lower()
        if service not in ['the', 'following', 'these']:
            result.mcp_services.append(service)
    
    return result


def _convert_llm_files_to_detected(llm_result: LLMExtractionResult) -> List[DetectedFile]:
    """将 LLM 提取的文件转换为 DetectedFile 列表"""
    detected = []
    for f in llm_result.files:
        source_type = FileSourceType.REMOTE
        if f.source == "local":
            source_type = FileSourceType.LOCAL
        elif f.source == "url":
            source_type = FileSourceType.URL
        
        detected.append(DetectedFile(
            path=f.path,
            source_type=source_type
        ))
    return detected


def _build_parameter_table(
    llm_result: LLMExtractionResult,
    sandbox_file_paths: Dict[str, str],
    file_analyses: List[FileAnalysis],
    session_id: str
) -> Dict[str, Any]:
    """
    构建参数表
    
    Args:
        llm_result: LLM 提取结果
        sandbox_file_paths: 沙箱文件路径映射
        file_analyses: 文件分析结果
        session_id: 会话 ID
        
    Returns:
        参数表字典
    """
    param_table = {
        "session_id": session_id,
        "task_description": llm_result.task_description,
        "target_organism": llm_result.target_organism,
        "mcp_services": llm_result.mcp_services,
        "analysis_type": llm_result.analysis_type,
        "notes": llm_result.notes,
        "files": {},
        "params": {}
    }
    
    # 添加文件信息
    for original_path, sandbox_path in sandbox_file_paths.items():
        ext = Path(sandbox_path).suffix.lower().lstrip('.')
        param_table["files"][original_path] = {
            "sandbox_path": sandbox_path,
            "format": ext
        }
    
    # 添加分析结果
    for fa in file_analyses:
        if fa.original_path in param_table["files"]:
            param_table["files"][fa.original_path]["row_count"] = fa.row_count
            param_table["files"][fa.original_path]["columns"] = fa.column_names
            param_table["files"][fa.original_path]["data_type"] = fa.detected_data_type
    
    # 添加提取的参数
    for param in llm_result.parameters:
        param_table["params"][param.name] = param.value
    
    return param_table


# ==================== 节点函数 ====================

def preprocess_user_input_node(state: SupervisorState) -> SupervisorState:
    """
    轻量版输入预处理节点（重构后）
    
    职责：
    1. 生成唯一会话 ID
    2. 使用 LLM 进行结构化提取（文件、参数、服务）
    3. 在远程沙盒中创建会话目录结构 ⭐ 新增
    
    不再负责：
    - 文件上传（移至 upload_files_node）
    - 文件分析（移至 analyze_files_node）
    - 参数表构建（移至 build_params_node）
    """
    logger.info("=" * 60)
    logger.info("Preprocess Node (轻量版)")
    logger.info("=" * 60)
    
    user_input = state.user_input
    logger.debug(f"用户输入: {user_input[:200]}...")
    
    # 1. 生成或复用会话 ID
    session_id = state.session_id or _generate_session_id()
    logger.info(f"  Session ID: {session_id}")
    logger.debug(f"  Sandbox 数据目录: /data/sessions/{session_id}")
    
    # 2. 在远程沙盒中创建目录结构
    sandbox_data_dir = f"/data/sessions/{session_id}"
    sandbox_container_dir = f"/tmp/sessions/{session_id}"
    
    # 使用 CodeAct 统一接口创建目录（遵循架构原则）
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        
        if is_codeact_available():
            create_dirs_code = f'''
import os

# 沙盒目录结构
dirs = [
    "{sandbox_container_dir}",
    "{sandbox_container_dir}/input",
    "{sandbox_container_dir}/output",
    "{sandbox_container_dir}/output/reports",
    "{sandbox_container_dir}/workspace"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {{d}}")

print("__SANDBOX_DIRS_CREATED__:success")
'''
            
            result = execute_code_via_codeact(
                task_description=f"创建会话目录结构: {session_id}",
                code_template=create_dirs_code,
                sandbox_id=state.opensandbox_id,  # 使用状态中的 sandbox_id
                timeout_seconds=30,
                keep_alive=True
            )
            
            if result.is_success() and "__SANDBOX_DIRS_CREATED__" in result.output:
                logger.info(f"  ✅ 沙盒目录结构已创建: {sandbox_data_dir}")
                # 保存 sandbox_id 到状态
                if result.sandbox_id:
                    state.opensandbox_id = result.sandbox_id
                    logger.debug(f"  OpenSandbox ID: {result.sandbox_id}")
            else:
                logger.warning(f"  ⚠️ 沙盒目录创建可能失败: {result.error}")
        else:
            logger.warning("  ⚠️ CodeAct 不可用，跳过沙盒目录创建")
            
    except Exception as e:
        logger.warning(f"  ⚠️ 创建沙盒目录失败: {e}")
    
    # 3. LLM 结构化提取
    logger.info("  正在执行 LLM 结构化提取...")
    llm_result = _llm_extract_structured_input(user_input)
    
    # 4. 打印提取结果
    logger.info(f"  [Task] {llm_result.task_description[:50]}..." if llm_result.task_description else "  [Task] (未提取到任务描述)")
    if llm_result.target_organism:
        logger.info(f"  [Target] {llm_result.target_organism}")
    if llm_result.mcp_services:
        logger.info(f"  [Services] {llm_result.mcp_services}")
    if llm_result.files:
        logger.info(f"  [Files] {len(llm_result.files)} 个文件")
        for f in llm_result.files:
            logger.debug(f"    - {f.path} ({f.format}, {f.source})")
    
    # 5. 更新状态
    state.session_id = session_id
    state.sandbox_data_dir = sandbox_data_dir
    state.preprocess_result = InputPreprocessResult(
        detected_files=_convert_llm_files_to_detected(llm_result),
        extracted_params={p.name: p.value for p in llm_result.parameters},
        task_description=llm_result.task_description,
        mcp_services=llm_result.mcp_services
    )
    
    logger.info("=" * 60)
    return state


def detect_files_node(state: SupervisorState) -> SupervisorState:
    """
    检测文件节点
    
    从用户输入和预处理结果中检测文件路径，分类为 LOCAL/REMOTE/URL
    """
    logger.info("=" * 60)
    logger.info("Detect Files Node")
    logger.info("=" * 60)
    
    detected_files = []
    seen_paths = set()
    
    # 1. 从预处理结果获取文件
    if state.preprocess_result:
        for f in state.preprocess_result.detected_files:
            if f.path not in seen_paths:
                detected_files.append(f)
                seen_paths.add(f.path)
    
    # 2. 从用户输入中额外检测文件路径
    raw_detected = _detect_file_paths(state.user_input)
    for f in raw_detected:
        if f.path not in seen_paths:
            detected_files.append(f)
            seen_paths.add(f.path)
    
    # 3. 添加 GlobalState.file_paths 中的文件（重要：使用 values 而非 keys）
    # 注意：state.uploaded_files 可能包含字典的 keys，需要正确处理
    for uploaded_file in state.uploaded_files:
        # 跳过无效路径（可能是字典 key 而非实际路径）
        if not uploaded_file or not _is_valid_file_path(uploaded_file):
            continue
        if uploaded_file not in seen_paths:
            source_type = _classify_file_source(uploaded_file)
            detected_files.append(DetectedFile(
                path=uploaded_file,
                source_type=source_type
            ))
            seen_paths.add(uploaded_file)
    
    # 4. 分类统计
    local_count = sum(1 for f in detected_files if f.source_type == FileSourceType.LOCAL)
    remote_count = sum(1 for f in detected_files if f.source_type == FileSourceType.REMOTE)
    url_count = sum(1 for f in detected_files if f.source_type == FileSourceType.URL)
    
    logger.info(f"  检测到 {len(detected_files)} 个文件:")
    logger.info(f"    - LOCAL: {local_count}")
    logger.info(f"    - REMOTE: {remote_count}")
    logger.info(f"    - URL: {url_count}")
    
    # 详细日志：文件列表
    if detected_files:
        logger.debug("【检测到的文件列表】")
        for f in detected_files:
            logger.debug(f"  - {f.path} ({f.source_type.value})")
    
    # 5. 更新状态
    if not state.preprocess_result:
        state.preprocess_result = InputPreprocessResult()
    state.preprocess_result.detected_files = detected_files
    
    logger.info("=" * 60)
    return state


def upload_files_node(state: SupervisorState) -> SupervisorState:
    """
    文件转存节点（重构后）
    
    重要原则：所有文件都统一转存到沙盒会话目录！
    - LOCAL 文件：上传到沙盒
    - REMOTE 文件：复制到会话目录
    - URL 文件：下载到会话目录
    
    通过 execute_code_via_codeact 直接执行转存代码
    """
    logger.info("=" * 60)
    logger.info("Upload Files Node (调用 CodeAct)")
    logger.info("=" * 60)
    
    if not state.preprocess_result or not state.preprocess_result.detected_files:
        logger.info("  没有需要转存的文件")
        state.codeact_upload_result = {"status": "skipped", "uploaded_files": {}}
        return state
    
    # 去重
    unique_files = {}
    for f in state.preprocess_result.detected_files:
        if f.path not in unique_files:
            unique_files[f.path] = f
    
    logger.info(f"  需要转存 {len(unique_files)} 个文件")
    
    # 1. 使用 execute_code_via_codeact 直接执行文件转存
    try:
        from utils.codeact_executor import execute_code_via_codeact, is_codeact_available
        
        if not is_codeact_available():
            logger.warning("  CodeAct 不可用，跳过文件转存")
            state.codeact_upload_result = {"status": "skipped", "error": "CodeAct not available"}
            return state
        
        # 构建文件列表
        files_json = []
        for path, f in unique_files.items():
            files_json.append({
                "original_path": path,
                "source_type": f.source_type.value,
                "target_name": Path(path).name
            })
        
        import json
        files_json_str = json.dumps(files_json, ensure_ascii=False)
        
        sandbox_input_dir = f"/data/sessions/{state.session_id}/input"
        container_input_dir = f"/tmp/sessions/{state.session_id}/input"
        
        # 构建转存代码 - 使用字符串拼接避免 f-string 转义问题
        transfer_code = '''
import os
import json
import shutil

files_json = ''' + files_json_str + '''
container_input_dir = "''' + container_input_dir + '''"
sandbox_input_dir = "''' + sandbox_input_dir + '''"

# 创建目标目录
os.makedirs(container_input_dir, exist_ok=True)

uploaded_files = dict()
errors = []

for f in files_json:
    original_path = f["original_path"]
    source_type = f["source_type"]
    target_name = f["target_name"]
    
    # 计算容器内源路径（REMOTE 文件在沙盒中的路径）
    if original_path.startswith("/data/"):
        container_source_path = original_path.replace("/data/", "/tmp/", 1)
    else:
        container_source_path = original_path
    
    target_path = os.path.join(container_input_dir, target_name)
    sandbox_target_path = os.path.join(sandbox_input_dir, target_name)
    
    try:
        if source_type == "remote":
            # REMOTE 文件：复制到会话目录
            if os.path.exists(container_source_path):
                shutil.copy2(container_source_path, target_path)
                uploaded_files[original_path] = sandbox_target_path
                print("COPIED: " + original_path + " -> " + target_path)
            elif os.path.exists(original_path):
                # 尝试原始路径
                shutil.copy2(original_path, target_path)
                uploaded_files[original_path] = sandbox_target_path
                print("COPIED: " + original_path + " -> " + target_path)
            else:
                errors.append("File not found: " + container_source_path)
                print("ERROR: File not found: " + container_source_path)
        
        elif source_type == "local":
            # LOCAL 文件：在沙盒环境中通常不存在，跳过
            errors.append("Local file not accessible in sandbox: " + original_path)
            print("SKIP: Local file: " + original_path)
        
        elif source_type == "url":
            # URL 文件：需要下载（暂时跳过）
            errors.append("URL download not implemented: " + original_path)
            print("SKIP: URL: " + original_path)
    
    except Exception as e:
        errors.append("Error processing " + original_path + ": " + str(e))
        print("ERROR: " + str(e))

print("__FILE_TRANSFER_START__")
result = {
    "status": "success" if len(errors) == 0 else "partial",
    "uploaded_files": uploaded_files,
    "errors": errors,
    "total_files": len(files_json),
    "transferred_files": len(uploaded_files)
}
print(json.dumps(result))
print("__FILE_TRANSFER_END__")
'''
        
        logger.info("  调用 CodeAct 执行文件转存...")
        logger.debug(f"  源文件: {list(unique_files.keys())}")
        logger.debug(f"  目标目录: {sandbox_input_dir}")
        
        result = execute_code_via_codeact(
            task_description=f"转存 {len(unique_files)} 个文件到沙盒会话目录",
            code_template=transfer_code,
            sandbox_id=state.opensandbox_id,
            timeout_seconds=120,
            keep_alive=True
        )
        
        logger.info(f"  CodeAct 执行结果: {result.status}")
        logger.debug(f"  输出: {result.output[:500] if result.output else 'N/A'}...")
        
        # 解析结果
        import re
        uploaded_files = {}
        if result.output and "__FILE_TRANSFER_START__" in result.output:
            match = re.search(r'__FILE_TRANSFER_START__\s*(.*?)\s*__FILE_TRANSFER_END__', result.output, re.DOTALL)
            if match:
                try:
                    transfer_result = json.loads(match.group(1))
                    uploaded_files = transfer_result.get("uploaded_files", {})
                    errors = transfer_result.get("errors", [])
                    if errors:
                        logger.warning(f"  部分文件转存失败: {errors}")
                except json.JSONDecodeError as e:
                    logger.warning(f"  解析转存结果失败: {e}")
        
        # 更新 sandbox_id
        if result.sandbox_id:
            state.opensandbox_id = result.sandbox_id
            logger.debug(f"  更新 opensandbox_id: {result.sandbox_id}")
        
        # 更新状态
        state.codeact_upload_result = {
            "status": "success" if result.is_success() else "failed",
            "output": {"uploaded_files": uploaded_files},
            "error": result.error
        }
        state.sandbox_file_paths.update(uploaded_files)
        
        logger.info(f"  转存完成: {len(uploaded_files)}/{len(unique_files)} 个文件")
        
        # 详细日志
        logger.debug("【文件转存映射】")
        for orig, sandbox in uploaded_files.items():
            logger.debug(f"  {orig} -> {sandbox}")
        
    except Exception as e:
        logger.error(f"  文件转存失败: {e}")
        import traceback
        traceback.print_exc()
        state.codeact_upload_result = {"status": "failed", "error": str(e)}
    
    logger.info("=" * 60)
    return state


def analyze_files_node(state: SupervisorState) -> SupervisorState:
    """
    分析文件节点（重构后）
    
    通过调用 CodeAct 子图来完成文件分析任务
    """
    logger.info("=" * 60)
    logger.info("Analyze Files Node (调用 CodeAct)")
    logger.info("=" * 60)
    
    if not state.sandbox_file_paths:
        logger.info("  没有需要分析的文件")
        state.codeact_analysis_result = {"status": "skipped", "files": []}
        return state
    
    # 1. 构建分析任务描述
    logger.debug("  构建分析任务描述...")
    task_description = _build_file_analysis_task(state.sandbox_file_paths)
    logger.debug(f"  任务描述长度: {len(task_description)} 字符")
    
    # 2. 调用 CodeAct 执行分析
    logger.info("  调用 CodeAct 执行文件分析...")
    result = _call_codeact(
        state=state,
        task_description=task_description,
        task_type="file_analysis",
        parameters={
            "session_id": state.session_id,
            "sandbox_file_paths": state.sandbox_file_paths
        }
    )
    
    # 3. 更新状态
    state.codeact_analysis_result = result
    
    if result.get("status") == "success":
        analysis_output = result.get("output", {}).get("files", [])
        # 转换为 FileAnalysis 对象
        for file_info in analysis_output:
            original_path = None
            for orig, sandbox in state.sandbox_file_paths.items():
                if sandbox == file_info.get("path"):
                    original_path = orig
                    break
            
            if original_path:
                state.file_analyses.append(FileAnalysis(
                    original_path=original_path,
                    sandbox_path=file_info.get("path", ""),
                    file_type=file_info.get("file_type", "unknown"),
                    column_names=file_info.get("column_names", []),
                    detected_data_type=file_info.get("detected_data_type"),
                    content_summary=file_info.get("summary"),
                    row_count=file_info.get("row_count")
                ))
        
        logger.info(f"  分析完成: {len(state.file_analyses)} 个文件")
        
        # 详细日志：文件分析结果
        logger.debug("【文件分析结果】")
        for fa in state.file_analyses:
            logger.debug(f"  文件: {fa.original_path}")
            logger.debug(f"    沙箱路径: {fa.sandbox_path}")
            logger.debug(f"    类型: {fa.file_type}")
            logger.debug(f"    数据类型: {fa.detected_data_type}")
            logger.debug(f"    行数: {fa.row_count}")
            logger.debug(f"    列名: {fa.column_names[:5] if fa.column_names else []}...")
    else:
        # 失败时详细记录错误信息
        error_msg = result.get('error', 'Unknown error')
        error_type = result.get('error_type', 'Unknown')
        error_category = result.get('error_category', 'Unknown')
        
        logger.error(f"  ❌ 分析失败!")
        logger.error(f"     错误信息: {error_msg}")
        logger.error(f"     错误类型: {error_type}")
        logger.error(f"     错误类别: {error_category}")
        
        # 如果有输出，也记录下来
        if result.get('output'):
            logger.error(f"     输出内容: {str(result.get('output'))[:500]}")
        
        # 记录任务描述（便于调试）
        logger.error(f"     任务描述: {task_description[:200]}...")
        
        state.file_analyses_failed = True
    
    logger.info("=" * 60)
    return state


def build_params_node(state: SupervisorState) -> SupervisorState:
    """
    构建参数表节点
    
    整合 LLM 提取结果和文件分析结果，生成完整的参数表
    """
    logger.info("=" * 60)
    logger.info("Build Params Node")
    logger.info("=" * 60)
    
    # 1. 重新获取 LLM 提取结果
    logger.debug("  重新获取 LLM 提取结果...")
    llm_result = _llm_extract_structured_input(state.user_input)
    
    # 2. 构建参数表
    logger.debug("  构建参数表...")
    param_table = _build_parameter_table(
        llm_result=llm_result,
        sandbox_file_paths=state.sandbox_file_paths,
        file_analyses=state.file_analyses,
        session_id=state.session_id
    )
    
    # 3. 更新状态
    state.extracted_parameters = param_table
    
    # 4. 记录参数表到日志（完整 JSON）
    logger.info("  参数表构建完成:")
    logger.info(f"    - Session ID: {param_table.get('session_id')}")
    logger.info(f"    - 任务描述: {param_table.get('task_description', '')[:80]}...")
    logger.info(f"    - 目标生物: {param_table.get('target_organism')}")
    logger.info(f"    - MCP 服务: {param_table.get('mcp_services', [])}")
    logger.info(f"    - 文件数量: {len(param_table.get('files', {}))}")
    logger.info(f"    - 参数数量: {len(param_table.get('params', {}))}")
    logger.info(f"    - 备注: {param_table.get('notes', [])}")
    
    # 详细日志：完整参数表 JSON
    logger.debug("=" * 40)
    logger.debug("【完整参数表 JSON】")
    logger.debug(json.dumps(param_table, ensure_ascii=False, indent=2))
    logger.debug("=" * 40)
    
    # 详细日志：文件信息
    if param_table.get('files'):
        logger.debug("【文件信息】")
        for orig_path, file_info in param_table['files'].items():
            logger.debug(f"  原始路径: {orig_path}")
            logger.debug(f"    沙箱路径: {file_info.get('sandbox_path')}")
            logger.debug(f"    格式: {file_info.get('format')}")
            logger.debug(f"    数据类型: {file_info.get('data_type')}")
    
    logger.info("=" * 60)
    return state


def classify_user_description_node(state: SupervisorState) -> SupervisorState:
    """
    任务分类节点
    
    根据用户输入和提取的信息，分类任务类型
    """
    logger.info("=" * 60)
    logger.info("Classify User Description Node")
    logger.info("=" * 60)
    
    # 使用 LLM 进行分类（如果有）
    llm = _get_llm()
    
    if llm:
        try:
            messages = [
                SystemMessage(content=TASK_CLASSIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=get_task_classification_user_prompt(state.user_input))
            ]
            
            response = llm.invoke(messages)
            result_text = response.content.strip().lower()
            
            # 解析分类结果
            if "execute_plan" in result_text:
                state.user_task_type = UserTaskType.EXECUTE_PLAN
            elif "immunology" in result_text or "antibody" in result_text or "antigen" in result_text:
                state.user_task_type = UserTaskType.IMMUNOLOGY_TASK
            else:
                state.user_task_type = UserTaskType.GENERAL_QA
            
            # 安全访问 .value 属性
            task_type_str = state.user_task_type.value if hasattr(state.user_task_type, 'value') else str(state.user_task_type)
            logger.info(f"  LLM 分类结果: {task_type_str}")
            
        except Exception as e:
            logger.warning(f"  [WARN] LLM 分类失败，使用关键词匹配: {e}")
            state.user_task_type = _classify_by_keywords(state.user_input)
            # 安全访问 .value 属性
            task_type_str = state.user_task_type.value if hasattr(state.user_task_type, 'value') else str(state.user_task_type)
            logger.info(f"  关键词分类结果: {task_type_str}")
    else:
        state.user_task_type = _classify_by_keywords(state.user_input)
        # 安全访问 .value 属性
        task_type_str = state.user_task_type.value if hasattr(state.user_task_type, 'value') else str(state.user_task_type)
        logger.info(f"  关键词分类结果: {task_type_str}")
    
    logger.info("=" * 60)
    return state


def _classify_by_keywords(user_input: str) -> UserTaskType:
    """关键词分类回退函数"""
    user_input_lower = user_input.lower()
    
    if any(keyword in user_input_lower for keyword in [
        "execute", "plan", "step", "follow", "according to",
        "执行", "计划", "步骤", "按照"
    ]):
        return UserTaskType.EXECUTE_PLAN
    
    if any(keyword in user_input_lower for keyword in [
        "immun", "antigen", "antibody", "vaccine", "immune",
        "免疫", "抗原", "抗体", "疫苗"
    ]):
        return UserTaskType.IMMUNOLOGY_TASK
    
    return UserTaskType.GENERAL_QA


# ==================== 条件路由函数 ====================

def _should_upload_files(state: SupervisorState) -> str:
    """
    判断是否需要上传/转存文件
    
    重要原则：所有文件都需要转存到沙盒会话目录，统一管理！
    - LOCAL 文件：上传到沙盒
    - REMOTE 文件：复制到会话目录
    - URL 文件：下载到会话目录
    
    Returns:
        "upload": 需要上传/转存文件
        "skip": 跳过（没有文件）
    """
    if not state.preprocess_result or not state.preprocess_result.detected_files:
        return "skip"
    
    # 有任何文件都需要转存（包括 REMOTE 文件）
    # 修改：不再区分文件类型，所有文件统一转存
    if len(state.preprocess_result.detected_files) > 0:
        logger.info(f"  检测到 {len(state.preprocess_result.detected_files)} 个文件，准备转存到沙盒会话目录")
        return "upload"
    
    return "skip"


def _should_analyze_files(state: SupervisorState) -> str:
    """
    判断是否需要分析文件
    
    Returns:
        "analyze": 需要分析文件
        "skip": 跳过分析
    """
    if not state.sandbox_file_paths:
        return "skip"
    
    return "analyze"


# ==================== 状态映射函数 ====================

def supervisor_input_mapper(global_state: GlobalState) -> SupervisorState:
    """
    主图 → 子图状态映射
    
    将主图的 GlobalState 映射为 SupervisorState，提取子图需要的信息。
    
    Args:
        global_state: 主图的全局状态
        
    Returns:
        SupervisorState: 子图状态
    """
    # 修复：使用 file_paths 的 values（实际路径）而非 keys（如 h5ad_file 这样的名称）
    # 去重并过滤无效路径
    uploaded_files = []
    if global_state.file_paths:
        for key, path in global_state.file_paths.items():
            if path and path not in uploaded_files:
                uploaded_files.append(path)
    
    # 获取已有的文件分析结果
    existing_file_analyses = []
    if global_state.file_analyses:
        for fa in global_state.file_analyses:
            if isinstance(fa, dict):
                existing_file_analyses.append(FileAnalysis(**fa))
            elif isinstance(fa, FileAnalysis):
                existing_file_analyses.append(fa)
    
    return SupervisorState(
        user_input=global_state.user_input,
        user_task_type=None,  # 将在子图中确定
        uploaded_files=uploaded_files,
        sandbox_file_paths=dict(global_state.file_paths) if global_state.file_paths else {},
        sandbox_dir=global_state.sandbox_dir,
        execution_plan=global_state.execution_plan,
        # 传递会话相关字段以避免重复创建
        session_id=global_state.session_id,
        sandbox_data_dir=global_state.sandbox_data_dir,
        opensandbox_id=global_state.opensandbox_id,  # 传递 OpenSandbox ID
        extracted_parameters=global_state.extracted_parameters,
        file_analyses=existing_file_analyses,
    )


def supervisor_output_mapper(subgraph_output: Union[SupervisorState, Dict[str, Any]], global_state: GlobalState) -> GlobalState:
    """
    子图 → 主图状态映射
    
    将子图的 SupervisorState 结果同步回主图的 GlobalState。
    
    Args:
        subgraph_output: 子图输出状态（可能是 SupervisorState 对象或 dict）
        global_state: 主图的全局状态（将被更新）
        
    Returns:
        GlobalState: 更新后的主图状态
    """
    # 处理 dict 格式状态
    if isinstance(subgraph_output, dict):
        subgraph_output = SupervisorState(**subgraph_output)
    
    # 1. 存储任务类型分类结果
    if subgraph_output.user_task_type:
        task_type = subgraph_output.user_task_type
        if isinstance(task_type, str):
            try:
                task_type = UserTaskType(task_type)
            except (ValueError, KeyError):
                pass
        global_state.user_task_type = task_type
    
    # 2. 同步执行计划
    if subgraph_output.execution_plan:
        global_state.execution_plan = subgraph_output.execution_plan
    
    # 3. 同步沙箱文件路径
    if subgraph_output.sandbox_file_paths:
        global_state.file_paths = subgraph_output.sandbox_file_paths
    
    # 4. 同步预处理结果（参数表、文件分析）
    if subgraph_output.extracted_parameters:
        # 直接存储到 GlobalState.extracted_parameters 字段（供 executor/codeact 使用）
        global_state.extracted_parameters = subgraph_output.extracted_parameters
        
        # 同时存储到 merged_result（向后兼容）
        if global_state.merged_result is None:
            global_state.merged_result = {}
        global_state.merged_result["extracted_parameters"] = subgraph_output.extracted_parameters
        global_state.merged_result["file_analyses"] = [
            {
                "sandbox_path": fa.sandbox_path,
                "file_type": fa.file_type,
                "data_type": fa.detected_data_type,
                "columns": fa.column_names,
                "summary": fa.content_summary,
            }
            for fa in subgraph_output.file_analyses
        ] if subgraph_output.file_analyses else []
    
    # 5. 同步会话 ID 和沙箱目录
    if global_state.merged_result is None:
        global_state.merged_result = {}
    
    if subgraph_output.session_id:
        global_state.merged_result["session_id"] = subgraph_output.session_id
        global_state.session_id = subgraph_output.session_id
    
    if subgraph_output.sandbox_data_dir:
        global_state.merged_result["sandbox_data_dir"] = subgraph_output.sandbox_data_dir
        global_state.merged_result["sandbox_input_dir"] = f"{subgraph_output.sandbox_data_dir}/input"
        global_state.merged_result["sandbox_output_dir"] = f"{subgraph_output.sandbox_data_dir}/output"
        global_state.sandbox_data_dir = subgraph_output.sandbox_data_dir
    
    # 6. 同步 OpenSandbox ID
    if subgraph_output.opensandbox_id:
        global_state.merged_result["opensandbox_id"] = subgraph_output.opensandbox_id
        global_state.opensandbox_id = subgraph_output.opensandbox_id
    
    return global_state


# ==================== 子图构建函数 ====================

def build_supervisor_subgraph():
    """
    构建 Supervisor Agent 子图（重构后）
    
    新流程：
    1. preprocess - 轻量预处理（生成 session ID，LLM 提取）
    2. detect_files - 检测文件（分类 LOCAL/REMOTE/URL）
    3. upload_files - 上传文件（调用 CodeAct）[条件执行]
    4. analyze_files - 分析文件（调用 CodeAct）[条件执行]
    5. build_params - 构建参数表
    6. classify - 任务分类
    
    Returns:
        编译后的子图
    """
    graph = StateGraph(SupervisorState)
    
    # 添加节点
    graph.add_node("preprocess", preprocess_user_input_node)
    graph.add_node("detect_files", detect_files_node)
    graph.add_node("upload_files", upload_files_node)
    graph.add_node("analyze_files", analyze_files_node)
    graph.add_node("build_params", build_params_node)
    graph.add_node("classify", classify_user_description_node)
    
    # 添加边 - 基本流程
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "detect_files")
    
    # 条件边 - 是否需要上传文件
    graph.add_conditional_edges(
        "detect_files",
        _should_upload_files,
        {
            "upload": "upload_files",
            "skip": "build_params"
        }
    )
    
    # 条件边 - 上传后是否需要分析
    graph.add_conditional_edges(
        "upload_files",
        _should_analyze_files,
        {
            "analyze": "analyze_files",
            "skip": "build_params"
        }
    )
    
    # 分析后进入参数构建
    graph.add_edge("analyze_files", "build_params")
    
    # 参数构建后进入分类
    graph.add_edge("build_params", "classify")
    
    # 分类后结束
    graph.add_edge("classify", END)
    
    return graph.compile()


logger.info("[Supervisor] graph_refactored.py loaded - 重构版本")

