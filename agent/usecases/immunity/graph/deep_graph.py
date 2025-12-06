from typing import List, Dict, Any
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver  # 添加MemorySaver导入
from common.factory import get_reasoning_model
from langchain_core.output_parsers import JsonOutputParser
from usecases.immunity.schema.common_schemas import TaskExtractionResult, TaskInfo
from usecases.immunity.state.state import ImprovedCellState
from usecases.immunity.common.task_executor import TaskExecutor

# 创建全局TaskExecutor实例（将在create_deep_graph中初始化）
executor = None


async def task_decomposition_node(state: ImprovedCellState, config: RunnableConfig)  -> ImprovedCellState:
    """任务分解节点 - 兼容性函数"""
    """
    Task decomposition node - Decompose refine_plan into specific executable tasks
    
    This node receives the refine_plan from planning_graph.py and decomposes it into
    specific, executable task step lists.
    
    Args:
        state: Cell module state object containing refine_plan
        config: Runtime configuration
        
    Returns:
        ExecuteState: Updated cell state containing decomposed task list
    """
    print("[task_decomposition_node] Starting task decomposition node")
    
    try:
        # Get refine_plan
        plan = state.final_enhanced_plan
        
        if not plan or plan.strip() == "":
            print("[task_decomposition_node] No plan available for decomposition")
            return state
        from usecases.immunity.prompts.prompts import ImmunityPrompts
        from usecases.immunity.common.constants import get_tools_json
        
        # Get tools registry information
        tools_info = get_tools_json()
        
        # Create task extraction chain
        model = get_reasoning_model(config)
        output_parser = JsonOutputParser(pydantic_object=TaskExtractionResult)
        
        # Execute task extraction
        decomposed_tasks = (model | output_parser).invoke(
            ImmunityPrompts.TASK_EXTRACTION_PROMPT.format(
                plan=plan,
                tools_info=tools_info
            )
        )
         
        # JsonOutputParser returns a dictionary, not a TaskExtractionResult instance
        # Need to get the tasks list from the dictionary and convert to TaskInfo objects
        if isinstance(decomposed_tasks, dict) and "tasks" in decomposed_tasks:
            tasks_list = decomposed_tasks["tasks"]
            print(f"[task_decomposition_node] Task decomposition completed, extracted {len(tasks_list)} tasks")
            
            task_info_objects = []
            for task_dict in tasks_list:
                if isinstance(task_dict, dict):
                    try:
                        # 创建TaskInfo对象，确保所有必需字段都存在
                        task_info = TaskInfo(
                            task_id=task_dict.get("task_id", ""),
                            name=task_dict.get("name", ""),
                            description=task_dict.get("description", ""),
                            tools=task_dict.get("tools", []),
                            inputs=task_dict.get("inputs", []),
                            outputs=task_dict.get("outputs", []),
                            parameters=task_dict.get("parameters", {})
                        )
                        task_info_objects.append(task_info)
                        print(f"[task_decomposition_node] Extracted task: {task_info.task_id} - {task_info.name}")
                    except Exception as e:
                        print(f"[task_decomposition_node] Error creating TaskInfo object: {e}")
                        print(f"[task_decomposition_node] Task dict: {task_dict}")
        else:
            print(f"[task_decomposition_node] Error: Unable to get tasks list from decomposition result")
            task_info_objects = []
        
        state.decomposed_tasks = task_info_objects
        print(f"[task_decomposition_node] Successfully extracted {len(task_info_objects)} task objects")
        
        return state
        
    except Exception as e:
        import traceback
        print(f"[task_decomposition_node] Task decomposition failed: {e}")
        print(f"[task_decomposition_node] Error type: {type(e).__name__}")
        print(f"[task_decomposition_node] Detailed stack trace:")
        print(traceback.format_exc())
        return state

async def execute_task_list(state: ImprovedCellState, config: RunnableConfig) -> Dict[str, Any]:
    """执行任务列表 - 兼容性函数"""
    await executor.initialize_agent(config)
    tasks = state.decomposed_tasks
    plan = state.final_enhanced_plan
    print(f"📋 开始批量执行 {len(tasks)} 个任务")
    
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"🔄 执行任务 {i}/{len(tasks)}: {task}")
        result = await executor.execute_task(task, plan, config)
        results.append(result)
    
    print(f"✅ 批量任务执行完成，成功: {sum(1 for r in results if r['status'] == 'success')}, 失败: {sum(1 for r in results if r['status'] == 'error')}")
    
    # 返回字典格式的状态更新，LangGraph会自动合并到状态中
    return {"task_results": results}

async def create_deep_graph():
    """
    创建深度执行图 - 开始->task_decomposition_node->execute_task_list->结束的workflow
    
    Returns:
        编译后的StateGraph
    """
    global executor
    
    # 创建共享的checkpointer
    checkpointer = MemorySaver()
    
    # 创建TaskExecutor实例，传入共享的checkpointer
    executor = TaskExecutor(checkpointer=checkpointer)
    
    # 创建StateGraph，使用ImprovedCellState作为状态类型
    workflow = StateGraph(ImprovedCellState)
    
    # 添加节点
    workflow.add_node("task_decomposition_node", task_decomposition_node)
    workflow.add_node("execute_task_list", execute_task_list)
    
    # 设置入口点和边
    workflow.set_entry_point("task_decomposition_node")
    workflow.add_edge("task_decomposition_node", "execute_task_list")
    workflow.add_edge("execute_task_list", END)
    
    # 编译图，使用同一个checkpointer
    graph = workflow.compile(checkpointer=checkpointer)
    
    # 打印工作流程图
    try:
        print("\n===== Deep Executor Workflow Diagram =====")
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"生成工作流程图时出错: {str(e)}")
    
    return graph


async def run_deep_graph(planning: str, config: RunnableConfig):
    """
    运行深度执行图
    
    Args:
        decomposed_tasks: 分解后的任务列表
        config: 运行配置
        
    Returns:
        执行结果
    """
    # 创建图
    graph = await create_deep_graph()
    
    # 创建初始状态
    initial_state = ImprovedCellState(
        original_question="执行分解后的任务列表",
        final_enhanced_plan=planning
    )
    
    # 使用异步流执行图
    result = await graph.ainvoke(initial_state, config)
    # 提取执行结果 - 添加更强壮的空值检查
    task_results = []
    if result and "task_results" in result:
        task_results = result.get("task_results", [])
    else:
        print("警告: task_results 未找到")
        task_results = []
    
    print(f"\n深度执行工作流完成")
    print(f"执行结果数量: {len(task_results)}")
    
    return {
        "task_results": task_results,
        "total_tasks": len(task_results),
        "completed_tasks": len([r for r in task_results if r.get('status') == 'success']),
        "failed_tasks": len([r for r in task_results if r.get('status') == 'error'])
    }

async def complete_deep_pipeline(decomposed_tasks: List[str], config: RunnableConfig):
    """
    完整的深度执行管道 - 作为主要的执行入口
    
    Args:
        decomposed_tasks: 分解后的任务列表
        config: 运行配置
        
    Returns:
        完整的执行结果
    """
    print("=== 深度执行管道启动 ===")

    # 运行深度执行图
    result = await run_deep_graph(decomposed_tasks, config)
    
    # 添加执行统计信息
    success_rate = (result["completed_tasks"] / result["total_tasks"] * 100) if result["total_tasks"] > 0 else 0
    
    final_result = {
        **result,
        "success_rate": success_rate,
        "execution_summary": {
            "total_tasks": result["total_tasks"],
            "completed_tasks": result["completed_tasks"],
            "failed_tasks": result["failed_tasks"],
            "success_rate": f"{success_rate:.1f}%"
        }
    }
    
    print(f"\n✅ 深度执行管道完成")
    print(f"📊 执行统计: {final_result['execution_summary']}")
    
    return final_result