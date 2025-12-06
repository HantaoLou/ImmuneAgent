# 测试完整流程
import inspect
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from usecases.antibody.graph.retrieval_graph import complete_rag_pipeline

if __name__ == "__main__":
    # 打印函数签名和源码，查看参数
    print("函数签名:", inspect.signature(complete_rag_pipeline))
    print("函数源码:", inspect.getsource(complete_rag_pipeline))

    original_question = "Design antibodies targeting protein-protein interactions, specifically inhibitors of p53-MDM2 interaction"
    from usecases._debug import get_debug_runnable_config

    test_questions = [
        "Design antibodies targeting protein-protein interactions, specifically inhibitors of p53-MDM2 interaction",
        # "Design antibodies targeting protein-protein interactions, specifically inhibitors of p53-MDM2 interaction",
        # "Design antibodies targeting protein-protein interactions, specifically inhibitors of p53-MDM2 interaction",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 测试问题 {i} ---")
        print(f"问题: {question}")

        # 检查函数签名并调用正确的版本
        import inspect
        import time
        import traceback

        sig = inspect.signature(complete_rag_pipeline)
        param_count = len(sig.parameters)

        try:
            print("调用双参数版本")
            # 记录开始时间
            start_time = time.time()

            final_state = complete_rag_pipeline(question, get_debug_runnable_config())

            # 记录结束时间和执行时长
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Planning生成成功")
            print(f"执行时长: {execution_time:.2f}秒")

            # 输出详细结果信息
            if hasattr(final_state, "result"):
                print(f"结果详情: {final_state.result}")
            if hasattr(final_state, "status"):
                print(f"执行状态: {final_state.status}")

        except Exception as e:
            print(f"Planning生成失败")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("详细错误堆栈:")
            print(traceback.format_exc())

        print("-" * 80)
