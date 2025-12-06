if __name__ == "__main__":
    # 仅在作为主程序运行时添加路径配置，确保模块能正确导入
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 usecases/cell/ 向上两级到达 agent/ 目录（项目根目录）
    project_root = os.path.join(current_dir, "..", "..")
    project_root = os.path.abspath(project_root)  # 转换为绝对路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 添加 langgraph-bigtool-main 目录到 sys.path
    langgraph_bigtool_path = os.path.join(
        project_root, "usecases", "langgraph-bigtool-main"
    )
    langgraph_bigtool_path = os.path.abspath(langgraph_bigtool_path)
    if langgraph_bigtool_path not in sys.path:
        sys.path.insert(0, langgraph_bigtool_path)

    print(f"当前文件路径: {current_dir}")
    print(f"项目根目录: {project_root}")
    print(f"sys.path已更新: {project_root in sys.path}")

    # 导入必要的模块（在路径配置后）
    import asyncio

    from usecases.cell.graph.execute_graph import run_execute_graph

    def main():
        """主函数：运行执行图流程测试"""
        print("=== execute_graph.py 流程测试程序 ===")
        print("正在加载工具和配置...")
        from usecases.cell.graph.test_constant import TestConstant

        # 测试用户问题
        test_question = TestConstant.TAST

        try:
            # 运行执行图流程（启用人机交互）
            result = asyncio.run(run_execute_graph(test_question))

            if result:
                print("\n=== 流程测试完成 ===")
                print("执行图流程成功完成")
            else:
                print("\n=== 流程测试失败 ===")

        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except Exception as e:
            print(f"\n测试失败: {str(e)}")
            print("\n=== 详细错误堆栈信息 ===")
            import traceback

            traceback.print_exc()
            print("=== 错误堆栈信息结束 ===")

    # 运行主函数
    main()
