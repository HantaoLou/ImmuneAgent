# 测试完整流程
import inspect
import os
import sys

# 添加agent目录到Python路径
agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, agent_dir)

from usecases.antibody.graph.plan_graph import run_planning_workflow

if __name__ == "__main__":
    query = input("Please enter your research question: ")
    from usecases._debug import get_debug_runnable_config

    rc = get_debug_runnable_config()
    run_planning_workflow(query, rc)
