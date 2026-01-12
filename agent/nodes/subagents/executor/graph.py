from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor
# 复用已有公共子图
from subgraphs.mcp_tool_graph import build_mcp_tool_graph
from subgraphs.codeact_graph import build_codeact_graph
from subgraphs.parallel_group_graph import build_parallel_group_graph

# ---------------------- Executor内部节点1：任务依赖解析与初始化 ----------------------
def task_dependency_initialize_node(state: ExecutorState) -> ExecutorState:
    """
    初始化任务状态：
    1. 梳理任务依赖关系
    2. 标记无依赖任务为「就绪」
    3. 标记有依赖任务为「等待依赖」
    4. 加载持久化代码缓存
    """
    # 1. 加载持久化代码缓存到内存
    state.code_cache = CodeCacheManager.load_persist_cache(state.cache_persist_path)
    print(f"Executor：加载到 {len(state.code_cache)} 条可复用代码缓存")
    
    # 2. 初始化任务状态映射
    for task in state.input_tasks:
        if not task.dependencies:
            # 无依赖任务直接就绪
            state.task_status_map[task.task_id] = TaskStatus.READY
        else:
            # 有依赖任务等待依赖完成
            state.task_status_map[task.task_id] = TaskStatus.DEPENDENCY_WAIT
    
    return state

# ---------------------- Executor内部节点2：缓存查询与工具优先判断 ----------------------
def cache_and_tool_priority_node(state: ExecutorState) -> ExecutorState:
    """
    核心逻辑：
    1. 遍历就绪任务，查询是否有代码缓存
    2. 无缓存则判断是否有可用MCP工具/Skills
    3. 标记任务执行方式（缓存复用/工具执行/CodeAct兜底）
    """
    ready_tasks = [
        task for task in state.input_tasks
        if state.task_status_map.get(task.task_id) == TaskStatus.READY
    ]
    
    for task in ready_tasks:
        # 1. 优先查询代码缓存
        cached_code = CodeCacheManager.get_cached_code(state, task)
        if cached_code:
            # 缓存复用：更新任务状态，直接使用缓存代码执行
            state.task_status_map[task.task_id] = TaskStatus.CACHE_REUSE
            # 更新缓存复用次数
            cached_code.reuse_count += 1
            state.code_cache[cached_code.cache_key] = cached_code
            continue
        
        # 2. 无缓存：判断是否有可用MCP工具/Skills
        has_available_tool = any(
            tool in state.available_tools  # available_tools可从配置加载
            for tool in task.required_tools
        )
        if has_available_tool:
            # 工具执行：标记任务状态
            state.task_status_map[task.task_id] = TaskStatus.RUNNING
            # 调用MCP工具子图（此处简化，实际调用公共工具子图）
            mcp_graph = build_mcp_tool_graph()
            mcp_result = mcp_graph.invoke({"task": task})
            state.completed_tasks[task.task_id] = {
                "task": task,
                "execution_mode": "tool",
                "result": mcp_result,
                "code": None
            }
            continue
        
        # 3. 无缓存无工具：标记为需要CodeAct生成代码
        state.task_status_map[task.task_id] = TaskStatus.RUNNING
    
    return state

# ---------------------- Executor内部节点3：CodeAct兜底与代码保存 ----------------------
def codeact_fallback_node(state: ExecutorState) -> ExecutorState:
    """
    CodeAct兜底逻辑：
    1. 遍历需要CodeAct的任务，调用CodeAct子图生成代码
    2. 执行代码，验证有效性
    3. 有效代码保存到缓存（内存+持久化）
    """
    codeact_tasks = [
        task for task in state.input_tasks
        if state.task_status_map.get(task.task_id) == TaskStatus.RUNNING
        and not any(tool in state.available_tools for tool in task.required_tools)
    ]
    
    for task in codeact_tasks:
        # 1. 调用CodeAct子图生成并执行代码
        codeact_graph = build_codeact_graph()
        codeact_result = codeact_graph.invoke({"task": task})
        
        # 2. 验证代码执行结果是否有效
        if codeact_result.get("status") == "success":
            executable_code = codeact_result.get("executable_code", "")
            code_description = codeact_result.get("code_description", "")
            
            # 3. 保存有效代码到缓存
            CodeCacheManager.add_cached_code(
                state=state,
                task=task,
                executable_code=executable_code,
                code_description=code_description
            )
            
            # 4. 记录执行结果
            state.completed_tasks[task.task_id] = {
                "task": task,
                "execution_mode": "codeact",
                "result": codeact_result,
                "code": executable_code
            }
        else:
            # 代码执行失败，记录失败结果
            state.completed_tasks[task.task_id] = {
                "task": task,
                "execution_mode": "codeact_failed",
                "result": codeact_result,
                "code": None
            }
    
    return state

# ---------------------- Executor内部节点4：并行任务调度 ----------------------
def parallel_task_schedule_node(state: ExecutorState) -> ExecutorState:
    """
    并行任务调度：
    1. 筛选出可并行执行的就绪任务（无相互依赖）
    2. 调用并行子图执行
    3. 记录并行执行结果
    """
    # 1. 筛选可并行任务（简化：所有就绪任务均可并行，无相互依赖）
    parallel_candidate_tasks = [
        task for task in state.input_tasks
        if state.task_status_map.get(task.task_id) in [TaskStatus.READY, TaskStatus.CACHE_REUSE]
    ]
    
    if not parallel_candidate_tasks:
        return state
    
    # 2. 调用并行子图执行
    parallel_graph = build_parallel_group_graph()
    parallel_result = parallel_graph.invoke({
        "parallel_tasks": parallel_candidate_tasks,
        "code_cache": state.code_cache
    })
    
    # 3. 记录并行执行结果，更新任务状态
    for task_result in parallel_result.get("completed_tasks", []):
        task_id = task_result["task"].task_id
        state.completed_tasks[task_id] = task_result
        state.task_status_map[task_id] = TaskStatus.COMPLETED
    
    return state

# ---------------------- Executor内部节点5：依赖任务激活 ----------------------
def dependency_activation_node(state: ExecutorState) -> ExecutorState:
    """
    激活后续依赖任务：
    1. 遍历所有等待依赖的任务
    2. 检查其前置依赖是否全部完成
    3. 全部完成则标记为「就绪」
    """
    waiting_tasks = [
        task for task in state.input_tasks
        if state.task_status_map.get(task.task_id) == TaskStatus.DEPENDENCY_WAIT
    ]
    
    for task in waiting_tasks:
        # 检查前置依赖是否全部完成
        all_deps_completed = all(
            dep in state.completed_tasks
            for dep in task.dependencies
        )
        
        if all_deps_completed:
            # 激活任务，标记为就绪
            state.task_status_map[task.task_id] = TaskStatus.READY
    
    return state

# ---------------------- Executor内部节点6：结果汇总与缓存持久化 ----------------------
def result_summary_node(state: ExecutorState) -> ExecutorState:
    """
    1. 汇总所有任务执行结果
    2. 持久化代码缓存（确保新增代码被保存）
    3. 标记最终执行状态
    """
    # 1. 汇总结果
    state.final_execution_result = {
        "total_tasks": len(state.input_tasks),
        "completed_tasks": len(state.completed_tasks),
        "failed_tasks": [
            task.task_id for task in state.input_tasks
            if state.task_status_map.get(task.task_id) == TaskStatus.FAILED
        ],
        "task_results": state.completed_tasks,
        "code_cache_reuse_count": sum(
            cache.reuse_count for cache in state.code_cache.values()
        )
    }
    
    # 2. 持久化代码缓存
    CodeCacheManager.save_persist_cache(state.code_cache, state.cache_persist_path)
    
    return state

# ---------------------- 构建Executor Agent子图 ----------------------
def build_executor_agent_graph():
    """构建统一执行入口的Executor Agent子图"""
    graph = StateGraph(ExecutorState)
    
    # 1. 添加所有内部节点
    graph.add_node("init_dependency", task_dependency_initialize_node)
    graph.add_node("cache_tool_check", cache_and_tool_priority_node)
    graph.add_node("codeact_fallback", codeact_fallback_node)
    graph.add_node("parallel_schedule", parallel_task_schedule_node)
    graph.add_node("activate_dependency", dependency_activation_node)
    graph.add_node("result_summary", result_summary_node)
    
    # 2. 定义节点流转规则（核心：循环执行直到所有任务完成）
    graph.add_edge(START, "init_dependency")
    
    # 初始化→缓存/工具判断→CodeAct→并行调度→依赖激活
    graph.add_edge("init_dependency", "cache_tool_check")
    graph.add_edge("cache_tool_check", "codeact_fallback")
    graph.add_edge("codeact_fallback", "parallel_schedule")
    graph.add_edge("parallel_schedule", "activate_dependency")
    
    # 依赖激活→判断是否还有未完成任务（循环/结束）
    def task_completion_router(state: ExecutorState) -> str:
        """判断是否还有未完成任务，决定是否循环执行"""
        uncompleted_task_ids = [
            task.task_id for task in state.input_tasks
            if state.task_status_map.get(task.task_id) not in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        ]
        if uncompleted_task_ids:
            # 还有未完成任务，回到缓存/工具判断，继续循环
            return "cache_tool_check"
        else:
            # 所有任务完成，进入结果汇总
            return "result_summary"
    
    graph.add_conditional_edges(
        "activate_dependency",
        task_completion_router,
        {"cache_tool_check": "cache_tool_check", "result_summary": "result_summary"}
    )
    
    # 结果汇总→结束
    graph.add_edge("result_summary", END)
    
    return graph.compile()