from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import json
from utils.code_cache_manager import CodeCacheManager

# CodeAct子图状态模型
class CodeActState(BaseModel):
    task: StandardTask = Field(description="待执行的标准化任务")
    mcp_tool_config: Dict[str, Any] = Field(default_factory=dict, description="MCP工具配置")
    execution_result: Dict[str, Any] = Field(default_factory=dict, description="代码执行结果")
    cached: bool = Field(default=False, description="是否已缓存有效代码")

# 节点1：生成代码（优先MCP工具调用代码）
def generate_code_node(state: CodeActState) -> CodeActState:
    """生成代码：优先调用MCP工具，无工具则生成普通分析代码"""
    task = state.task
    mcp_tools = state.mcp_tool_config.get("mcp_tools", [])
    code = ""
    code_description = ""
    related_mcp_tool = None

    # 1. 优先匹配所需MCP工具，生成调用代码
    for required_tool in task.required_mcp_tools:
        mcp_tool = next((t for t in mcp_tools if t["tool_name"] == required_tool), None)
        if mcp_tool:
            related_mcp_tool = required_tool
            code_description = f"调用MCP工具{required_tool}完成{task.task_type}任务"
            # 复用配置中的示例代码，替换核心参数
            code = mcp_tool["example_code"]
            for param_name, param_value in task.core_params.items():
                if isinstance(param_value, str):
                    code = code.replace(f"{param_name}=\"\"", f"{param_name}=\"{param_value}\"")
                else:
                    code = code.replace(f"{param_name}=None", f"{param_name}={param_value}")
            break

    # 2. 无MCP工具时，生成普通分析代码（模拟LLM生成，实际可调用Claude/GPT）
    if not code:
        code_description = f"生成普通代码完成{task.task_type}任务"
        code = f"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 核心参数：{task.core_params}
def process_task():
    print(\"执行普通分析任务：{task.task_content}\")
    return \"普通分析任务执行完成\"

result = process_task()
print(result)
""".strip()

    # 3. 保存代码到状态（后续执行+缓存）
    state.execution_result["generated_code"] = code
    state.execution_result["code_description"] = code
    state.execution_result["related_mcp_tool"] = related_mcp_tool
    return state

# 节点2：执行代码（安全沙箱执行，此处简化）
def execute_code_node(state: CodeActState) -> CodeActState:
    """执行生成的代码，获取执行结果"""
    code = state.execution_result.get("generated_code", "")
    try:
        # 安全执行代码（实际场景需使用沙箱环境，如execjs、pyodide）
        local_namespace = {}
        exec(code, globals(), local_namespace)
        # 提取执行结果（默认提取result变量）
        execution_output = local_namespace.get("result", "执行成功，无返回结果")
        state.execution_result["status"] = "success"
        state.execution_result["output"] = execution_output
    except Exception as e:
        state.execution_result["status"] = "failed"
        state.execution_result["error"] = str(e)
    return state

# 节点3：缓存有效代码（仅执行成功的代码才缓存）
def cache_valid_code_node(state: CodeActState) -> CodeActState:
    """缓存执行成功的代码，供后续复用"""
    if state.execution_result.get("status") != "success":
        return state
    
    task = state.task
    code = state.execution_result.get("generated_code", "")
    code_description = state.execution_result.get("code_description", "")
    related_mcp_tool = state.execution_result.get("related_mcp_tool")

    # 调用缓存工具添加缓存
    CodeCacheManager.add_cached_code(
        task_type=task.task_type,
        core_params=task.core_params,
        executable_code=code,
        code_description=code_description,
        related_mcp_tool=related_mcp_tool,
        cache_path=task.cache_path
    )
    state.cached = True
    return state

# 构建CodeAct子图
def build_codeact_graph():
    graph = StateGraph(CodeActState)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("cache_valid_code", cache_valid_code_node)

    # 定义流转规则
    graph.add_edge(START, "generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_edge("execute_code", "cache_valid_code")
    graph.add_edge("cache_valid_code", END)

    return graph.compile()