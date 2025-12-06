# 测试完整流程
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from usecases.cell.graph.retrieval_graph import complete_rag_pipeline

if __name__ == "__main__":
    from usecases.cell.cell_config import get_cell_runnable_config

    # 测试问题1: 流感病毒相关
    test_questions = [
        # "Predict broadly neutralizing antibodies against H5N1 influenza virus using single-cell V(D)J data, identify conserved B cell subsets, and reveal structural features of neutralization breadth",
        # # 测试问题2: SARS-CoV-2相关
        # "Analyze single-cell RNA-seq and V(D)J data to predict bnAbs targeting SARS-CoV-2 spike protein, identify memory B cell populations, and characterize epitope conservation patterns",
        # # 测试问题3: RSV相关
        # "Identify broadly neutralizing antibodies against RSV F protein from single-cell BCR repertoire data, discover protective B cell subsets, and model antibody-antigen binding interfaces",
        # # 测试问题4: 新型病毒
        "Identify antibodies in my dataset that are likely to neutralize both group 1 and group 2 influenza A viruses.",
        # 测试问题5: 综合分析
        # "Integrate single-cell V(D)J and RNA-seq data to predict broadly neutralizing antibodies, identify conserved B cell subsets across multiple pathogens, and reveal structural features of neutralization breadth using AlphaFold3 modeling",
    ]

    print("=== 测试Planning Agent ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试问题 {i} ---")
        print(f"问题: {question}")

        try:
            final_state = complete_rag_pipeline(question, get_cell_runnable_config())
            print(f"Planning生成成功")
        except Exception as e:
            print(f"Planning生成失败: {e}")
            print("\n=== 详细错误堆栈信息 ===")
            import traceback

            traceback.print_exc()
            print("=== 错误堆栈信息结束 ===\n")

        print("-" * 80)
