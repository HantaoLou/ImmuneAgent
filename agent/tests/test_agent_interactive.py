"""
Agent 交互式测试用例

这是一个交互式测试文件，用于测试 Agent 系统的 HITL（Human-in-the-Loop）功能。
测试过程中，用户需要在控制台输入参数或确认信息，以推进测试流程。

运行方式：python -m pytest tests/test_agent_interactive.py -v -s
或者直接运行：python tests/test_agent_interactive.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加 agent 目录到路径
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from main_graph import build_main_graph
from state import GlobalState, UserTaskType


def print_section(title: str, width: int = 80):
    """打印分节标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_step(step_num: int, description: str):
    """打印步骤信息"""
    print(f"\n[步骤 {step_num}] {description}")
    print("-" * 80)


def get_task_type_str(task_type):
    """安全地获取任务类型字符串"""
    if task_type is None:
        return '未知'
    if isinstance(task_type, str):
        return task_type
    if hasattr(task_type, 'value'):
        return task_type.value
    return str(task_type)


def wait_for_user_input(prompt: str = "按 Enter 继续..."):
    """等待用户输入"""
    input(f"\n{prompt}")


def test_interactive_general_qa():
    """交互式测试：普通问答流程"""
    print_section("交互式测试：普通问答流程")
    
    print("这个测试将演示普通问答的完整流程。")
    print("流程：Supervisor 分类 → General QA → 返回答案")
    
    wait_for_user_input("准备开始测试，按 Enter 继续...")
    
    # 构建主图
    print_step(1, "构建 Agent 主图")
    agent_graph = build_main_graph()
    print("✓ Agent 主图构建成功")
    
    # 用户输入
    print_step(2, "用户输入")
    user_input = input("请输入您的问题（或直接按 Enter 使用默认问题）：").strip()
    if not user_input:
        user_input = "什么是DNA？"
        print(f"使用默认问题: {user_input}")
    else:
        print(f"您的问题: {user_input}")
    
    # 执行
    print_step(3, "执行 Agent 流程")
    print("正在执行：Supervisor 分类 → General QA → 生成答案...")
    
    result = agent_graph.invoke({
        "user_input": user_input,
        "sandbox_dir": "./sandbox"
    })
    
    # 处理结果
    if isinstance(result, dict):
        result = GlobalState(**result)
    
    # 显示结果
    print_step(4, "执行结果")
    print(f"任务类型: {get_task_type_str(result.user_task_type)}")
    
    if result.merged_result and "general_qa_answer" in result.merged_result:
        answer = result.merged_result["general_qa_answer"]
        print(f"\n答案:\n{answer}\n")
        
        if "general_qa_confidence" in result.merged_result:
            print(f"置信度: {result.merged_result['general_qa_confidence']}")
        if "general_qa_related_topics" in result.merged_result:
            topics = result.merged_result["general_qa_related_topics"]
            if topics:
                print(f"相关话题: {', '.join(topics[:5])}")
    else:
        print("⚠ 未生成答案")
    
    print("\n✓ 普通问答流程测试完成")
    wait_for_user_input()


def test_interactive_execute_plan_with_hitl():
    """交互式测试：执行计划流程（包含 HITL 中断）"""
    print_section("交互式测试：执行计划流程（包含 HITL 中断）")
    
    print("这个测试将演示执行计划的完整流程，包括：")
    print("1. Supervisor 分类")
    print("2. Task Decomposition（任务分解）")
    print("3. Executor（任务执行）")
    print("4. HITL 中断（如果需要用户输入参数）")
    print("5. 恢复执行")
    
    wait_for_user_input("准备开始测试，按 Enter 继续...")
    
    # 构建主图
    print_step(1, "构建 Agent 主图")
    agent_graph = build_main_graph()
    print("✓ Agent 主图构建成功")
    
    # 用户输入
    print_step(2, "用户输入")
    print("请输入一个需要执行的任务（例如：搜索 COVID-19 相关的抗体数据）")
    user_input = input("任务描述（或直接按 Enter 使用默认任务）：").strip()
    if not user_input:
        user_input = "搜索 COVID-19 相关的抗体数据，并分析 V(D)J 重组情况"
        print(f"使用默认任务: {user_input}")
    else:
        print(f"您的任务: {user_input}")
    
    # 执行计划（可选）
    print("\n是否需要提供执行计划？")
    print("如果提供执行计划，系统将按照计划执行；否则系统会自动分解任务。")
    has_plan = input("是否提供执行计划？(y/n，默认 n): ").strip().lower() == 'y'
    
    execution_plan = None
    if has_plan:
        print("\n请输入执行计划（多行输入，输入空行结束）：")
        plan_lines = []
        while True:
            line = input()
            if not line.strip():
                break
            plan_lines.append(line)
        execution_plan = "\n".join(plan_lines)
        print(f"\n执行计划:\n{execution_plan}")
    
    # 执行
    print_step(3, "执行 Agent 流程")
    print("正在执行：Supervisor 分类 → Task Decomposition → Executor...")
    print("注意：如果任务需要参数，系统会暂停并等待您的输入。\n")
    
    initial_state = {
        "user_input": user_input,
        "sandbox_dir": "./sandbox"
    }
    if execution_plan:
        initial_state["execution_plan"] = execution_plan
    
    result = agent_graph.invoke(initial_state)
    
    # 处理结果
    if isinstance(result, dict):
        result = GlobalState(**result)
    
    # 显示结果
    print_step(4, "执行结果")
    print(f"任务类型: {get_task_type_str(result.user_task_type)}")
    
    # 显示任务分解结果
    if result.subtasks:
        print(f"\n生成了 {len(result.subtasks)} 个子任务：")
        for i, task in enumerate(result.subtasks, 1):
            print(f"  {i}. {task.task_id}: {task.content[:50]}...")
    
    if result.parallel_task_groups:
        print(f"\n生成了 {len(result.parallel_task_groups)} 个并行任务组")
    
    # 显示执行结果
    if result.merged_result and "executor_results" in result.merged_result:
        executor_results = result.merged_result["executor_results"]
        print(f"\n执行结果:")
        print(f"  总任务数: {executor_results.get('total_tasks', 0)}")
        print(f"  已完成: {executor_results.get('completed', 0)}")
        print(f"  失败: {executor_results.get('failed', 0)}")
        
        if "task_results" in executor_results:
            print(f"\n任务详情:")
            for task_id, task_result in executor_results["task_results"].items():
                status = task_result.get("status", "unknown")
                print(f"  - {task_id}: {status}")
                if task_result.get("error"):
                    print(f"    错误: {task_result['error'][:100]}...")
    
    if result.completed_tasks:
        print(f"\n已完成的任务: {len(result.completed_tasks)} 个")
        for task_id, task in result.completed_tasks.items():
            print(f"  - {task_id}: {task.content[:50]}...")
    
    print("\n✓ 执行计划流程测试完成")
    wait_for_user_input()


def test_interactive_hitl_parameter_request():
    """交互式测试：HITL 参数请求"""
    print_section("交互式测试：HITL 参数请求")
    
    print("这个测试专门测试 HITL 参数请求功能。")
    print("系统会执行一个需要参数的任务，然后暂停等待您输入参数。")
    
    wait_for_user_input("准备开始测试，按 Enter 继续...")
    
    # 构建主图
    print_step(1, "构建 Agent 主图")
    agent_graph = build_main_graph()
    print("✓ Agent 主图构建成功")
    
    # 使用一个明确需要参数的任务
    print_step(2, "用户输入")
    print("使用一个需要参数的任务：搜索抗体数据")
    user_input = "搜索 COVID-19 相关的抗体数据，疾病类型为 COVID-19，组织类型为 blood"
    
    print(f"任务: {user_input}")
    print("\n注意：如果系统无法自动推断参数，会暂停并请求您输入。")
    
    wait_for_user_input("准备执行，按 Enter 继续...")
    
    # 执行
    print_step(3, "执行 Agent 流程")
    print("正在执行任务...")
    print("如果系统需要参数，会在控制台显示提示，请按照提示输入参数。\n")
    
    result = agent_graph.invoke({
        "user_input": user_input,
        "sandbox_dir": "./sandbox"
    })
    
    # 处理结果
    if isinstance(result, dict):
        result = GlobalState(**result)
    
    # 显示结果
    print_step(4, "执行结果")
    print(f"任务类型: {get_task_type_str(result.user_task_type)}")
    
    # 检查是否有 HITL 状态
    if result.hitl_status:
        print(f"\nHITL 状态: {result.hitl_status[:100]}...")
    
    # 显示执行结果
    if result.merged_result and "executor_results" in result.merged_result:
        executor_results = result.merged_result["executor_results"]
        print(f"\n执行结果:")
        print(f"  总任务数: {executor_results.get('total_tasks', 0)}")
        print(f"  已完成: {executor_results.get('completed', 0)}")
        print(f"  失败: {executor_results.get('failed', 0)}")
    
    print("\n✓ HITL 参数请求测试完成")
    wait_for_user_input()


def test_interactive_full_workflow():
    """交互式测试：完整工作流程"""
    print_section("交互式测试：完整工作流程")
    
    print("这个测试将演示完整的 Agent 工作流程，包括所有步骤。")
    print("您可以选择不同的任务类型进行测试。")
    
    wait_for_user_input("准备开始测试，按 Enter 继续...")
    
    # 构建主图
    print_step(1, "构建 Agent 主图")
    agent_graph = build_main_graph()
    print("✓ Agent 主图构建成功")
    
    # 选择任务类型
    print_step(2, "选择任务类型")
    print("请选择要测试的任务类型：")
    print("1. 普通问答 (General QA)")
    print("2. 免疫学任务 (Immunology Task)")
    print("3. 执行计划 (Execute Plan)")
    
    choice = input("请输入选项 (1-3，默认 1): ").strip() or "1"
    
    task_examples = {
        "1": ("什么是蛋白质折叠？", UserTaskType.GENERAL_QA),
        "2": ("分析抗原抗体反应机制", UserTaskType.IMMUNOLOGY_TASK),
        "3": ("执行计划：1. 搜索抗体数据 2. 分析序列 3. 生成报告", UserTaskType.EXECUTE_PLAN),
    }
    
    if choice not in task_examples:
        choice = "1"
    
    default_input, expected_type = task_examples[choice]
    
    # 用户输入
    print(f"\n默认任务: {default_input}")
    user_input = input("请输入您的任务（或直接按 Enter 使用默认任务）：").strip()
    if not user_input:
        user_input = default_input
        print(f"使用默认任务: {user_input}")
    else:
        print(f"您的任务: {user_input}")
    
    # 执行
    print_step(3, "执行 Agent 流程")
    print("正在执行完整流程...")
    print("注意：如果任务需要参数或确认，系统会暂停并等待您的输入。\n")
    
    result = agent_graph.invoke({
        "user_input": user_input,
        "sandbox_dir": "./sandbox"
    })
    
    # 处理结果
    if isinstance(result, dict):
        result = GlobalState(**result)
    
    # 显示结果
    print_step(4, "执行结果")
    print(f"任务类型: {get_task_type_str(result.user_task_type)}")
    
    # 根据任务类型显示不同的结果
    if result.user_task_type == UserTaskType.GENERAL_QA:
        if result.merged_result and "general_qa_answer" in result.merged_result:
            answer = result.merged_result["general_qa_answer"]
            print(f"\n答案:\n{answer[:500]}...\n")
    elif result.user_task_type == UserTaskType.IMMUNOLOGY_TASK:
        if result.merged_result and "immunity_response" in result.merged_result:
            response = result.merged_result["immunity_response"]
            print(f"\n响应:\n{response}\n")
    elif result.user_task_type == UserTaskType.EXECUTE_PLAN:
        if result.subtasks:
            print(f"\n生成了 {len(result.subtasks)} 个子任务")
        if result.merged_result and "executor_results" in result.merged_result:
            executor_results = result.merged_result["executor_results"]
            print(f"执行结果: {executor_results.get('completed', 0)}/{executor_results.get('total_tasks', 0)} 任务完成")
    
    print("\n✓ 完整工作流程测试完成")
    wait_for_user_input()


def main():
    """主函数：运行所有交互式测试"""
    print_section("Agent 交互式测试套件", width=100)
    
    print("欢迎使用 Agent 交互式测试套件！")
    print("\n这个测试套件允许您交互式地测试 Agent 系统的各种功能，")
    print("特别是在需要用户输入（HITL）的场景中，您可以在控制台直接输入参数。")
    print("\n可用的测试：")
    print("1. 普通问答流程测试")
    print("2. 执行计划流程测试（包含 HITL）")
    print("3. HITL 参数请求测试")
    print("4. 完整工作流程测试")
    print("5. 运行所有测试")
    print("0. 退出")
    
    while True:
        choice = input("\n请选择要运行的测试 (0-5): ").strip()
        
        if choice == "0":
            print("\n退出测试。")
            break
        elif choice == "1":
            test_interactive_general_qa()
        elif choice == "2":
            test_interactive_execute_plan_with_hitl()
        elif choice == "3":
            test_interactive_hitl_parameter_request()
        elif choice == "4":
            test_interactive_full_workflow()
        elif choice == "5":
            print("\n运行所有测试...")
            test_interactive_general_qa()
            test_interactive_execute_plan_with_hitl()
            test_interactive_hitl_parameter_request()
            test_interactive_full_workflow()
            print_section("所有测试完成", width=100)
        else:
            print("无效的选项，请重新选择。")
    
    print("\n感谢使用 Agent 交互式测试套件！")


if __name__ == "__main__":
    main()

