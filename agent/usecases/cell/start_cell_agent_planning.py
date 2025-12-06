# 测试完整流程
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from usecases._debug import get_debug_runnable_config
from usecases.cell.graph.planning_graph import run_planning_graph
from usecases.cell.graph.retrieval_graph import complete_rag_pipeline

if __name__ == "__main__":
    print("\n" + "=" * 50)
    user_question = input("请输入您的抗体分析问题: ").strip()
    if not user_question:
        user_question = "Predict broadly neutralizing antibodies against H5N1 influenza virus using single-cell V(D)J data"
    rc = get_debug_runnable_config()
    run_planning_graph(user_question, rc)
