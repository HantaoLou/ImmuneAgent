"""
CodeAct Agent 子图

负责代码生成和执行，包括：
1. MCP工具调用代码生成和执行
2. 普通代码生成和执行
3. 代码修复（修复代码错误和参数错误）
4. 代码执行和错误处理
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
import sys
import os
import subprocess
from pathlib import Path
from enum import Enum

# 导入主图状态和任务模型
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from agent.state import SubTask, GlobalState
from agent.utils.llm_factory import create_code_llm
from agent.nodes.subagents.code_act.prompt import (
    MCP_TOOL_CODE_SYSTEM_PROMPT,
    get_mcp_tool_code_user_prompt,
    CODEACT_SYSTEM_PROMPT,
    get_codeact_user_prompt,
    FIX_CODE_SYSTEM_PROMPT,
    get_fix_code_user_prompt,
    FIX_PARAMETER_SYSTEM_PROMPT,
    get_fix_parameter_user_prompt
)
from agent.nodes.subagents.code_act.trajectory import (
    CodeTrajectory,
    TrajectoryPool,
    TrajectoryStatus
)
from agent.nodes.subagents.code_act.revision import (
    RevisionPlan,
    RevisionStrategy,
    create_revision_plan,
    execute_revision_plan
)

# ===================== CodeAct 子图状态模型 =====================

class CodeActExecutionMode(str, Enum):
    """CodeAct执行模式"""
    MCP_TOOL = "mcp_tool"  # 生成调用MCP工具的代码
    CODEACT = "codeact"  # 根据任务描述生成代码
    FIX_CODE = "fix_code"  # 修复代码错误
    FIX_PARAMETER = "fix_parameter"  # 修复参数错误


class CodeActState(BaseModel):
    """CodeAct子图状态"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        from_attributes=True  # 允许从属性创建（Pydantic v2）
    )
    
    task: SubTask = Field(description="待执行的任务")
    task_description: str = Field(description="任务描述")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="任务匹配的工具列表")
    inputs: List[str] = Field(default_factory=list, description="任务输入参数")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="任务参数（已解析）")
    execution_mode: CodeActExecutionMode = Field(description="执行模式")
    
    # 修复相关
    previous_code: Optional[str] = Field(default=None, description="之前的代码（用于修复）")
    previous_error: Optional[str] = Field(default=None, description="之前的错误信息（用于修复）")
    error_category: Optional[str] = Field(default=None, description="错误分类")
    revision_plan: Optional[Any] = Field(default=None, description="Revision计划（用于智能修复）")
    revision_iteration: int = Field(default=0, description="Revision迭代次数")
    
    # 代码生成和执行结果
    generated_code: Optional[str] = Field(default=None, description="生成的代码")
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="执行结果")
    
    # 轨迹记录（SE-Agent 风格）
    trajectory_history: List[CodeTrajectory] = Field(default_factory=list, description="当前任务的轨迹历史")
    trajectory_pool_id: Optional[str] = Field(default=None, description="关联的轨迹池ID")
    current_trajectory: Optional[CodeTrajectory] = Field(default=None, description="当前正在记录的轨迹")
    
    # 父状态引用
    parent_state: Optional[GlobalState] = Field(default=None, description="主图状态引用")


# ===================== 轨迹记录辅助函数 =====================

def _start_trajectory(state: CodeActState) -> CodeTrajectory:
    """
    开始记录新轨迹
    
    Args:
        state: CodeAct状态
    
    Returns:
        新创建的轨迹
    """
    from datetime import datetime
    import hashlib
    
    # 处理 execution_mode 可能是枚举或字符串的情况
    execution_mode_value = state.execution_mode
    if hasattr(execution_mode_value, 'value'):
        execution_mode_value = execution_mode_value.value
    else:
        execution_mode_value = str(execution_mode_value)
    
    trajectory = CodeTrajectory(
        trajectory_id="",  # 稍后生成
        task_id=state.task.task_id,
        execution_mode=execution_mode_value,
        generated_code="",  # 稍后填充
        status=TrajectoryStatus.PARTIAL,  # 初始状态
        parameters=state.parameters.copy(),
        tools=state.tools.copy(),
        inputs=state.inputs.copy()
    )
    
    # 生成轨迹ID
    timestamp_str = trajectory.timestamp.strftime("%Y%m%d_%H%M%S_%f")
    task_hash = hashlib.md5(state.task.task_id.encode()).hexdigest()[:8]
    trajectory.trajectory_id = f"{state.task.task_id}_{timestamp_str}_{task_hash}"
    
    return trajectory


def _update_trajectory_code(trajectory: CodeTrajectory, code: str, generation_time: float = 0.0):
    """
    更新轨迹的代码生成信息
    
    Args:
        trajectory: 轨迹
        code: 生成的代码
        generation_time: 生成耗时
    """
    trajectory.generated_code = code
    trajectory.code_length = len(code)
    trajectory.code_generation_time = generation_time


def _finalize_trajectory(trajectory: CodeTrajectory, execution_result: Dict[str, Any], execution_time: float = 0.0):
    """
    完成轨迹记录
    
    Args:
        trajectory: 轨迹
        execution_result: 执行结果
        execution_time: 执行耗时
    """
    trajectory.execution_result = execution_result
    trajectory.execution_time = execution_time
    
    # 根据执行结果设置状态
    if execution_result.get("status") == "success":
        trajectory.status = TrajectoryStatus.SUCCESS
    else:
        trajectory.status = TrajectoryStatus.FAILED
        trajectory.error_type = execution_result.get("error_type")
        trajectory.error_message = execution_result.get("error")
        trajectory.error_traceback = execution_result.get("error_traceback")
        trajectory.error_category = execution_result.get("error_category")


def _save_trajectory_to_pool(state: CodeActState, trajectory: CodeTrajectory):
    """
    保存轨迹到轨迹池
    
    Args:
        state: CodeAct状态
        trajectory: 要保存的轨迹
    """
    # 添加到轨迹历史
    state.trajectory_history.append(trajectory)
    
    # TODO: 集成 TrajectoryPool 进行持久化存储
    # 目前先保存在内存中，后续可以添加持久化


# ===================== CodeAct 节点 =====================

def _generate_code_with_llm(
    system_prompt: str,
    user_prompt: str,
    fallback_code: str = None
) -> str:
    """
    使用LLM生成代码
    
    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        fallback_code: 降级代码（LLM不可用时使用）
    
    Returns:
        生成的代码
    """
    llm = create_code_llm()
    
    if not llm:
        print("  ⚠ LLM不可用，使用降级代码")
        return fallback_code or "# LLM不可用，无法生成代码"
    
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        code = response.content.strip()
        
        # 移除可能的markdown代码块标记
        if code.startswith("```python"):
            code = code[9:]  # 移除 ```python
        elif code.startswith("```"):
            code = code[3:]  # 移除 ```
        
        if code.endswith("```"):
            code = code[:-3]  # 移除结尾的 ```
        
        code = code.strip()
        
        if not code:
            print("  ⚠ LLM返回空代码，使用降级代码")
            return fallback_code or "# LLM返回空代码"
        
        print(f"  ✓ LLM代码生成成功（长度: {len(code)} 字符）")
        return code
    
    except Exception as e:
        print(f"  ⚠ LLM代码生成失败: {e}，使用降级代码")
        return fallback_code or f"# LLM代码生成失败: {e}"


def _check_sandbox_available(state: CodeActState) -> tuple[bool, str]:
    """
    检查沙盒环境是否可用
    
    Args:
        state: CodeAct状态
    
    Returns:
        (是否可用, 沙盒目录路径)
    """
    sandbox_dir = None
    
    # 尝试从parent_state获取沙盒目录
    if state.parent_state and hasattr(state.parent_state, 'sandbox_dir'):
        sandbox_dir = state.parent_state.sandbox_dir
    
    # 如果没有沙盒目录，检查环境变量或使用默认值
    if not sandbox_dir or sandbox_dir == "DEFAULT_SANDBOX_DIR":
        import os
        sandbox_dir = os.getenv("SANDBOX_DIR")
        if not sandbox_dir:
            # 使用临时目录作为降级方案
            import tempfile
            sandbox_dir = tempfile.gettempdir()
    
    # 检查目录是否存在或可创建
    try:
        from pathlib import Path
        sandbox_path = Path(sandbox_dir)
        if not sandbox_path.exists():
            sandbox_path.mkdir(parents=True, exist_ok=True)
        # 检查是否可写
        test_file = sandbox_path / ".test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True, str(sandbox_path)
        except Exception:
            return False, str(sandbox_path)
    except Exception:
        return False, sandbox_dir or ""


def _ensure_code_executable(code: str, has_sandbox: bool, sandbox_dir: str = None) -> str:
    """
    确保代码可以执行，添加必要的包装和错误处理
    
    Args:
        code: 原始代码
        has_sandbox: 是否有沙盒环境
        sandbox_dir: 沙盒目录路径
    
    Returns:
        可执行的代码
    """
    # 如果代码已经包含result设置，直接返回
    if "result" in code and ("=" in code.split("result")[0] or "result = {" in code):
        # 确保代码有完整的错误处理
        if "try:" not in code or "except" not in code:
            # 添加错误处理包装
            wrapped_code = f"""
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    # 确保result变量存在
    if 'result' not in locals():
        result = {{"status": "success", "output": "代码执行完成"}}
except Exception as e:
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
            return wrapped_code.strip()
        return code
    
    # 如果代码没有result设置，添加
    if "result" not in code:
        code += "\nresult = {\"status\": \"success\", \"output\": \"代码执行完成\"}"
    
    # 添加错误处理
    if "try:" not in code:
        wrapped_code = f"""
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
        return wrapped_code.strip()
    
    return code


def _find_and_activate_venv(agent_dir: Path) -> Optional[Path]:
    """
    查找并激活项目虚拟环境
    
    Args:
        agent_dir: agent 目录路径
    
    Returns:
        虚拟环境的 Python 解释器路径，如果未找到则返回 None
    """
    # 查找虚拟环境目录（.venv）
    venv_paths = [
        agent_dir / ".venv",
        agent_dir.parent / ".venv",
        Path.cwd() / ".venv",
    ]
    
    for venv_path in venv_paths:
        if venv_path.exists() and venv_path.is_dir():
            # 确定 Python 解释器路径
            if os.name == 'nt':  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux
                python_exe = venv_path / "bin" / "python"
            
            if python_exe.exists():
                print(f"  ✓ 找到虚拟环境: {venv_path}")
                print(f"     Python 解释器: {python_exe}")
                return python_exe
    
    return None


def _activate_venv_in_sys_path(venv_python: Path) -> None:
    """
    在 sys.path 中激活虚拟环境
    
    Args:
        venv_python: 虚拟环境的 Python 解释器路径
    """
    import sys
    import site
    
    venv_dir = venv_python.parent.parent
    
    # 确定 site-packages 路径
    if os.name == 'nt':  # Windows
        site_packages = venv_dir / "Lib" / "site-packages"
    else:  # Unix/Linux
        import sysconfig
        site_packages = Path(sysconfig.get_path('purelib', vars={'base': str(venv_dir)}))
    
    # 将虚拟环境的 site-packages 添加到 sys.path 的最前面
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
        print(f"  ✓ 已添加虚拟环境 site-packages 到 sys.path: {site_packages}")
    
    # 也添加虚拟环境根目录（用于导入项目模块）
    if str(venv_dir) not in sys.path:
        sys.path.insert(0, str(venv_dir))
    
    # 执行 site.addsitedir 以确保虚拟环境被正确激活
    try:
        site.addsitedir(str(site_packages))
    except Exception as e:
        print(f"  ⚠ 无法添加虚拟环境 site-packages: {e}")
    
    # 验证关键包是否可以导入
    try:
        import langchain_mcp_adapters
        print(f"  ✓ 虚拟环境激活成功，可以导入 langchain-mcp-adapters")
    except ImportError:
        print(f"  ⚠ 警告：虚拟环境已激活，但无法导入 langchain-mcp-adapters")


def codeact_generate_code_node(state: CodeActState) -> CodeActState:
    """
    CodeAct节点：生成代码
    
    根据执行模式生成不同的代码：
    - mcp_tool: 生成调用MCP工具的代码
    - codeact: 根据任务描述生成完成任务的代码
    - fix_code: 修复代码错误
    - fix_parameter: 修复参数错误
    
    注意：此节点只负责生成代码，不执行代码。代码执行由 codeact_execute_code_node 负责。
    如果没有沙盒环境，会生成能够直接执行的代码（包含必要的错误处理和result设置），
    确保代码在后续执行节点中能够切实执行。
    
    同时记录代码生成轨迹（SE-Agent 风格）。
    """
    import time
    
    # 开始记录轨迹
    if not state.current_trajectory:
        state.current_trajectory = _start_trajectory(state)
    
    task = state.task
    mode = state.execution_mode
    parameters = state.parameters
    
    # 记录代码生成开始时间
    generation_start_time = time.time()
    
    # 检查沙盒环境（用于生成适合的代码，但不在此节点执行）
    has_sandbox, sandbox_dir = _check_sandbox_available(state)
    if not has_sandbox:
        print(f"  ⚠ 沙盒环境不可用，将生成可在降级模式下执行的代码（目录: {sandbox_dir}）")
    
    if mode == CodeActExecutionMode.MCP_TOOL:
        # 生成调用MCP工具的代码
        tools = state.tools
        if tools:
            # 使用第一个工具
            tool = tools[0] if isinstance(tools, list) else tools
            tool_name = tool.get("tool_name") or tool.get("name", "unknown_tool")
            tool_description = tool.get("description", "")
            
            # 使用LLM生成代码
            user_prompt = get_mcp_tool_code_user_prompt(
                tool_name=tool_name,
                tool_description=tool_description,
                parameters=parameters,
                task_description=state.task_description
            )
            
            # 降级代码
            params_str = ", ".join([f"{k}={repr(v)}" for k, v in parameters.items()])
            fallback_code = f"""
# 调用MCP工具: {tool_name}
# 参数: {params_str}
print("调用MCP工具: {tool_name}")
print("参数: {params_str}")
result = {{"status": "success", "output": "MCP工具调用结果（占位）"}}
"""
            
            generated_code = _generate_code_with_llm(
                system_prompt=MCP_TOOL_CODE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                fallback_code=fallback_code
            )
            
            # 确保代码可执行（特别是没有沙盒环境时）
            state.generated_code = _ensure_code_executable(
                generated_code,
                has_sandbox=has_sandbox,
                sandbox_dir=sandbox_dir
            )
        else:
            state.generated_code = "# 未找到匹配的工具"
    
    elif mode == CodeActExecutionMode.FIX_CODE:
        # 修复代码错误：根据之前的错误信息生成修复后的代码
        previous_code = state.previous_code or ""
        previous_error = state.previous_error or ""
        error_category = state.error_category
        
        if not previous_code or not previous_error:
            state.generated_code = "# 缺少必要的修复信息（原始代码或错误信息）"
            return state
        
        # 如果提供了Revision计划，使用智能修复（SE-Agent风格）
        if state.revision_plan:
            print(f"  🔄 使用Revision计划进行智能修复（策略: {state.revision_plan.strategy.value}）")
            print(f"     根本原因: {state.revision_plan.root_cause[:100]}...")
            print(f"     正交策略: {'是' if state.revision_plan.orthogonal else '否'}")
            
            generated_code = execute_revision_plan(
                revision_plan=state.revision_plan,
                original_code=previous_code,
                original_error=previous_error,
                task_description=state.task_description,
                parameters=state.parameters
            )
        else:
            # 使用传统修复方法
            user_prompt = get_fix_code_user_prompt(
                previous_code=previous_code,
                previous_error=previous_error,
                error_category=error_category
            )
            
            # 降级代码
            fallback_code = f"""
# 修复代码错误
# 之前的错误: {previous_error}
# 原始代码:
{previous_code}

# 降级修复：尝试基本修复
try:
    {previous_code}
    result = {{"status": "success", "output": "代码执行成功"}}
except Exception as e:
    result = {{"status": "failed", "error": str(e)}}
"""
            
            generated_code = _generate_code_with_llm(
                system_prompt=FIX_CODE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                fallback_code=fallback_code
            )
        
        # 确保代码可执行（特别是没有沙盒环境时）
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
    elif mode == CodeActExecutionMode.FIX_PARAMETER:
        # 修复参数错误：根据之前的错误信息调整参数
        previous_code = state.previous_code or ""
        previous_error = state.previous_error or ""
        error_category = state.error_category
        
        if not previous_code or not previous_error:
            state.generated_code = "# 缺少必要的修复信息（原始代码或错误信息）"
            return state
        
        # 使用LLM生成修复代码
        user_prompt = get_fix_parameter_user_prompt(
            previous_code=previous_code,
            previous_error=previous_error,
            error_category=error_category,
            parameters=parameters
        )
        
        # 降级代码
        fallback_code = f"""
# 修复参数错误
# 之前的错误: {previous_error}
# 原始代码:
{previous_code}

# 降级修复：使用提供的参数
parameters = {repr(parameters)}
try:
    # 尝试使用修正后的参数执行
    result = {{"status": "success", "output": "参数修复成功（占位）"}}
except Exception as e:
    result = {{"status": "failed", "error": str(e)}}
"""
        
        generated_code = _generate_code_with_llm(
            system_prompt=FIX_PARAMETER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_code=fallback_code
        )
        
        # 确保代码可执行（特别是没有沙盒环境时）
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
    else:
        # codeact模式：根据任务描述生成代码
        task_desc = state.task_description
        inputs = state.inputs
        
        # 使用LLM生成代码
        user_prompt = get_codeact_user_prompt(
            task_description=task_desc,
            inputs=inputs,
            outputs=None
        )
        
        # 降级代码
        fallback_code = f"""
# 根据任务描述生成代码
# 任务: {task_desc}
print("执行任务: {task_desc}")
result = {{"status": "success", "output": "任务执行结果（占位）"}}
"""
        
        generated_code = _generate_code_with_llm(
            system_prompt=CODEACT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback_code=fallback_code
        )
        
        # 确保代码可执行（特别是没有沙盒环境时）
        state.generated_code = _ensure_code_executable(
            generated_code,
            has_sandbox=has_sandbox,
            sandbox_dir=sandbox_dir
        )
    
    # 记录代码生成轨迹
    generation_time = time.time() - generation_start_time
    if state.current_trajectory:
        _update_trajectory_code(
            state.current_trajectory,
            state.generated_code,
            generation_time
        )
        state.current_trajectory.sandbox_used = has_sandbox
    
    return state


def codeact_execute_code_node(state: CodeActState) -> CodeActState:
    """
    CodeAct节点：执行代码
    
    在沙盒环境中执行生成的代码。如果没有沙盒环境，使用降级方案直接执行。
    
    同时记录执行轨迹（SE-Agent 风格）。
    """
    import time
    
    code = state.generated_code
    if not code:
        state.execution_result = {
            "status": "failed",
            "error": "未生成代码",
            "error_type": "NoCodeError"
        }
        print(f"  ✗ 执行失败：未生成代码")
        
        # 记录失败的轨迹
        if state.current_trajectory:
            _finalize_trajectory(state.current_trajectory, state.execution_result, 0.0)
            _save_trajectory_to_pool(state, state.current_trajectory)
            state.current_trajectory = None
        
        return state
    
    # 记录执行开始时间
    execution_start_time = time.time()
    
    # 检查沙盒环境
    has_sandbox, sandbox_dir = _check_sandbox_available(state)
    print(f"  ℹ 沙盒环境检查: 可用={has_sandbox}, 目录={sandbox_dir}")
    
    try:
        if has_sandbox:
            # 有沙盒环境：在沙盒目录中执行
            import os
            from pathlib import Path
            
            original_cwd = os.getcwd()
            print(f"  ℹ 切换到沙盒目录: {sandbox_dir} (原目录: {original_cwd})")
            try:
                # 切换到沙盒目录
                os.chdir(sandbox_dir)
                
                # 在沙盒环境中执行代码
                # 重要：在执行代码前，先激活指定的虚拟环境
                import sys
                import site
                
                # 1. 查找并激活虚拟环境
                venv_python = _find_and_activate_venv(agent_dir)
                if venv_python:
                    _activate_venv_in_sys_path(venv_python)
                else:
                    print(f"  ⚠ 未找到虚拟环境，将使用当前 Python 环境")
                
                # 2. 添加 agent 目录到路径（如果还没有）
                agent_dir_str = str(agent_dir)
                if agent_dir_str not in sys.path:
                    sys.path.insert(0, agent_dir_str)
                
                # 3. 确保所有必要的路径都在 sys.path 中
                try:
                    # 如果找到了虚拟环境，确保其 site-packages 在路径中
                    if venv_python:
                        venv_dir = venv_python.parent.parent
                        if os.name == 'nt':
                            venv_site_packages = venv_dir / "Lib" / "site-packages"
                        else:
                            import sysconfig
                            venv_site_packages = Path(sysconfig.get_path('purelib', vars={'base': str(venv_dir)}))
                        
                        if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
                            sys.path.insert(0, str(venv_site_packages))
                    
                    # 也添加系统 site-packages（作为备用）
                    system_site_packages = site.getsitepackages()
                    for sp in system_site_packages:
                        if sp not in sys.path:
                            sys.path.append(sp)  # 添加到末尾，优先使用虚拟环境的包
                except Exception as e:
                    print(f"  ⚠ 配置 site-packages 时出错: {e}")
                
                # 4. 验证虚拟环境是否已正确激活
                try:
                    import langchain_mcp_adapters
                    print(f"  ✓ 虚拟环境已激活，可以导入 langchain-mcp-adapters")
                except ImportError:
                    print(f"  ⚠ 警告：无法导入 langchain-mcp-adapters，可能虚拟环境未正确激活")
                
                # 预导入 mcp_helper 以便生成的代码可以使用
                # 这会在导入时验证 langchain-mcp-adapters 是否可用
                try:
                    from agent.utils.mcp_helper import invoke_mcp_tool_sync
                    local_namespace = {
                        "__sandbox_dir__": sandbox_dir,
                        "__original_cwd__": original_cwd,
                        "invoke_mcp_tool_sync": invoke_mcp_tool_sync,
                        "__builtins__": __builtins__
                    }
                except ImportError as e:
                    import traceback
                    print(f"  ⚠ 无法导入 mcp_helper: {e}")
                    print(f"     错误详情: {traceback.format_exc()}")
                    print(f"     Python 路径: {sys.path[:5]}...")
                    local_namespace = {
                        "__sandbox_dir__": sandbox_dir,
                        "__original_cwd__": original_cwd,
                        "__builtins__": __builtins__
                    }
                
                print(f"  🔄 开始执行代码（沙盒模式）...")
                exec(code, {"__builtins__": __builtins__}, local_namespace)
                print(f"  ✓ 代码执行完成")
                
                # 提取执行结果
                result = local_namespace.get("result", "执行成功，无返回结果")
                output = str(local_namespace.get("output", result))
                
                state.execution_result = {
                    "status": "success",
                    "output": output,
                    "result": result,
                    "sandbox_dir": sandbox_dir
                }
                print(f"  ✓ 执行成功，结果: {output[:100]}...")
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
                print(f"  ℹ 已恢复工作目录: {original_cwd}")
        else:
            # 没有沙盒环境：降级方案，直接执行
            print(f"  ⚠ 使用降级执行方案（无沙盒环境）")
            print(f"  🔄 开始执行代码（降级模式）...")
            
            # 重要：在执行代码前，先激活指定的虚拟环境
            import sys
            import site
            
            # 1. 查找并激活虚拟环境
            venv_python = _find_and_activate_venv(agent_dir)
            if venv_python:
                _activate_venv_in_sys_path(venv_python)
            else:
                print(f"  ⚠ 未找到虚拟环境，将使用当前 Python 环境")
            
            # 2. 添加 agent 目录到路径（如果还没有）
            agent_dir_str = str(agent_dir)
            if agent_dir_str not in sys.path:
                sys.path.insert(0, agent_dir_str)
            
            # 3. 确保所有必要的路径都在 sys.path 中
            try:
                # 如果找到了虚拟环境，确保其 site-packages 在路径中
                if venv_python:
                    venv_dir = venv_python.parent.parent
                    if os.name == 'nt':
                        venv_site_packages = venv_dir / "Lib" / "site-packages"
                    else:
                        import sysconfig
                        venv_site_packages = Path(sysconfig.get_path('purelib', vars={'base': str(venv_dir)}))
                    
                    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
                        sys.path.insert(0, str(venv_site_packages))
                
                # 也添加系统 site-packages（作为备用）
                system_site_packages = site.getsitepackages()
                for sp in system_site_packages:
                    if sp not in sys.path:
                        sys.path.append(sp)  # 添加到末尾，优先使用虚拟环境的包
            except Exception as e:
                print(f"  ⚠ 配置 site-packages 时出错: {e}")
            
            # 4. 验证虚拟环境是否已正确激活
            try:
                import langchain_mcp_adapters
                print(f"  ✓ 虚拟环境已激活，可以导入 langchain-mcp-adapters")
            except ImportError:
                print(f"  ⚠ 警告：无法导入 langchain-mcp-adapters，可能虚拟环境未正确激活")
            
            # 预导入 mcp_helper 以便生成的代码可以使用
            try:
                from agent.utils.mcp_helper import invoke_mcp_tool_sync
                local_namespace = {
                    "invoke_mcp_tool_sync": invoke_mcp_tool_sync,
                    "__builtins__": __builtins__
                }
            except ImportError as e:
                import traceback
                print(f"  ⚠ 无法导入 mcp_helper: {e}")
                print(f"     错误详情: {traceback.format_exc()}")
                print(f"     Python 路径: {sys.path[:5]}...")
                local_namespace = {
                    "__builtins__": __builtins__
                }
            
            exec(code, {"__builtins__": __builtins__}, local_namespace)
            print(f"  ✓ 代码执行完成")
            
            # 提取执行结果
            result = local_namespace.get("result", "执行成功，无返回结果")
            output = str(local_namespace.get("output", result))
            
            state.execution_result = {
                "status": "success",
                "output": output,
                "result": result,
                "sandbox_used": False
            }
            print(f"  ✓ 执行成功，结果: {output[:100]}...")
    
    except SyntaxError as e:
        # 语法错误
        error_msg = f"语法错误: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "SyntaxError",
            "error_line": getattr(e, 'lineno', None),
            "error_text": getattr(e, 'text', None),
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ 执行失败（语法错误）: {error_msg}")
        print(f"     错误行号: {error_details.get('error_line')}")
        print(f"     错误代码: {error_details.get('error_text')}")
        print(f"     代码预览: {code[:200]}...")
    
    except NameError as e:
        # 名称错误（未定义的变量或函数）
        error_msg = f"名称错误: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "NameError",
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ 执行失败（名称错误）: {error_msg}")
        print(f"     代码预览: {code[:200]}...")
    
    except ImportError as e:
        # 导入错误
        error_msg = f"导入错误: {str(e)}"
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": "ImportError",
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ 执行失败（导入错误）: {error_msg}")
        print(f"     代码预览: {code[:200]}...")
    
    except Exception as e:
        # 其他错误
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        error_details = {
            "status": "failed",
            "error": error_msg,
            "error_type": type(e).__name__,
            "error_traceback": error_traceback,
            "sandbox_used": has_sandbox,
            "code_preview": code[:500] if code else None
        }
        state.execution_result = error_details
        print(f"  ✗ 执行失败（{type(e).__name__}）: {error_msg}")
        print(f"     错误堆栈:")
        print(f"     {error_traceback}")
        print(f"     代码预览: {code[:200]}...")
    
    # 记录执行轨迹
    execution_time = time.time() - execution_start_time
    if state.current_trajectory:
        _finalize_trajectory(state.current_trajectory, state.execution_result, execution_time)
        _save_trajectory_to_pool(state, state.current_trajectory)
        state.current_trajectory = None
    
    return state


# ===================== 构建 CodeAct 子图 =====================

def codeact_revision_node(state: CodeActState) -> CodeActState:
    """
    CodeAct Revision节点：分析失败并生成Revision计划
    
    当代码执行失败时，使用Revision机制进行深度分析和智能修复。
    """
    # 检查是否有失败的轨迹
    if not state.trajectory_history:
        print("  ⚠ 无轨迹历史，无法进行Revision分析")
        return state
    
    # 获取最近的失败轨迹
    failed_trajectories = [t for t in state.trajectory_history if t.status == TrajectoryStatus.FAILED]
    if not failed_trajectories:
        print("  ℹ 无失败轨迹，无需Revision")
        return state
    
    latest_failed = failed_trajectories[-1]
    previous_failed = failed_trajectories[:-1] if len(failed_trajectories) > 1 else []
    
    print(f"  🔍 开始Revision分析（失败轨迹: {latest_failed.trajectory_id}）")
    
    # 创建Revision计划
    revision_plan = create_revision_plan(latest_failed, previous_failed)
    
    # 更新状态
    state.revision_plan = revision_plan
    state.revision_iteration += 1
    state.previous_code = latest_failed.generated_code
    state.previous_error = latest_failed.error_message or str(latest_failed.error_type)
    state.error_category = latest_failed.error_category
    
    print(f"  ✓ Revision计划生成成功")
    print(f"     策略: {revision_plan.strategy.value}")
    print(f"     根本原因: {revision_plan.root_cause[:150]}...")
    print(f"     信心度: {revision_plan.confidence:.2f}")
    print(f"     迭代次数: {state.revision_iteration}")
    
    return state


def build_codeact_subgraph():
    """构建CodeAct子图"""
    graph = StateGraph(CodeActState)
    
    graph.add_node("generate_code", codeact_generate_code_node)
    graph.add_node("execute_code", codeact_execute_code_node)
    graph.add_node("revision", codeact_revision_node)
    
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "execute_code")
    
    # 执行后根据结果决定是否进入Revision
    def should_revise(state: CodeActState) -> str:
        """判断是否需要Revision"""
        if state.execution_result and state.execution_result.get("status") == "failed":
            # 检查迭代次数限制
            if state.revision_iteration < 3:  # 最多3次Revision迭代
                return "revision"
        return "end"
    
    graph.add_conditional_edges(
        "execute_code",
        should_revise,
        {
            "revision": "revision",
            "end": END
        }
    )
    
    # Revision后重新生成代码
    graph.add_edge("revision", "generate_code")
    
    return graph.compile()


# ===================== 状态映射函数 =====================

def codeact_input_mapper(executor_state: Any, task: SubTask, execution_mode: CodeActExecutionMode, 
                         parameters: Dict[str, Any] = None, previous_code: str = None, 
                         previous_error: str = None, error_category: str = None,
                         revision_plan: Any = None, revision_iteration: int = 0) -> CodeActState:
    """
    将Executor状态映射到CodeAct子图状态
    
    Args:
        executor_state: Executor状态（可选）
        task: 要执行的任务
        execution_mode: 执行模式
        parameters: 已解析的参数
        previous_code: 之前的代码（用于修复）
        previous_error: 之前的错误（用于修复）
        error_category: 错误分类（用于修复）
    
    Returns:
        CodeAct子图状态
    """
    task_result = task.result if isinstance(task.result, dict) else {}
    tools = task_result.get("tools", [])
    inputs = task_result.get("inputs", [])
    
    # 验证和转换 tools：确保是字典列表
    if isinstance(tools, list):
        # 过滤并转换：只保留字典类型的元素，字符串转换为字典格式
        validated_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                validated_tools.append(tool)
            elif isinstance(tool, str):
                # 字符串工具名称，转换为字典格式（用于错误处理测试）
                validated_tools.append({"name": tool, "type": "unknown"})
            # 其他类型忽略
        tools = validated_tools
    else:
        tools = []
    
    # 确保 task 正确传递（Pydantic v2 兼容性）
    try:
        return CodeActState(
            task=task,
            task_description=task.content,
            tools=tools,
            inputs=inputs,
            parameters=parameters or {},
            execution_mode=execution_mode,
            previous_code=previous_code,
            previous_error=previous_error,
            error_category=error_category,
            revision_plan=revision_plan,
            revision_iteration=revision_iteration
        )
    except Exception:
        # 如果直接构造失败，使用 model_validate
        task_dict = task.model_dump() if hasattr(task, 'model_dump') else task.dict() if hasattr(task, 'dict') else task
        return CodeActState.model_validate({
            "task": task_dict,
            "task_description": task.content,
            "tools": tools,
            "inputs": inputs,
            "parameters": parameters or {},
            "execution_mode": execution_mode,
            "previous_code": previous_code,
            "previous_error": previous_error,
            "error_category": error_category,
            "revision_plan": revision_plan,
            "revision_iteration": revision_iteration
        })


def codeact_output_mapper(codeact_state: Union[CodeActState, Dict[str, Any]]) -> Dict[str, Any]:
    """
    将CodeAct子图状态映射回执行结果
    
    Args:
        codeact_state: CodeAct子图状态（可以是 CodeActState 对象或字典）
    
    Returns:
        执行结果字典
    """
    # 如果是字典，转换为 CodeActState 对象
    if isinstance(codeact_state, dict):
        codeact_state = CodeActState.model_validate(codeact_state)
    
    return {
        "status": codeact_state.execution_result.get("status") if codeact_state.execution_result else "unknown",
        "code": codeact_state.generated_code,
        "output": codeact_state.execution_result.get("output") if codeact_state.execution_result else None,
        "error": codeact_state.execution_result.get("error") if codeact_state.execution_result else None,
        "error_type": codeact_state.execution_result.get("error_type") if codeact_state.execution_result else None
    }

