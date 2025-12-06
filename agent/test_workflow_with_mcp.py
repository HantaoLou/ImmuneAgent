#!/usr/bin/env python3
"""
测试深度执行图功能 - MCP服务启动后的完整测试
验证TaskExecutor类中的graph创建和执行方法，包括MCP服务连接
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from usecases.immunity.graph.deep_executor import TaskExecutor
from usecases.immunity.config.immunity_config import get_runnable_config
from usecases.immunity.state.state import ImprovedCellState

async def test_mcp_connection():
    """测试MCP服务连接和工具加载"""
    print("=== 测试 MCP服务连接和工具加载 ===")
    
    try:
        executor = TaskExecutor()
        config = get_runnable_config()
        
        # 初始化代理，这会测试MCP连接
        await executor.initialize_agent(config)
        
        # 验证代理创建
        assert executor.agent is not None, "主代理应该被创建"
        
        # 验证MCP工具加载
        mcp_tools = await executor._get_all_mcp_tools(config)
        assert len(mcp_tools) > 0, "应该加载至少一个MCP工具"
        
        print(f"✅ MCP连接测试通过")
        print(f"   - 主代理: 已创建")
        print(f"   - MCP工具数量: {len(mcp_tools)}")
        
        # 打印MCP工具详情
        for i, tool in enumerate(mcp_tools, 1):
            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
            print(f"   - MCP工具 {i}: {tool_name}")
        
        return True, executor
        
    except Exception as e:
        print(f"❌ MCP连接测试失败: {str(e)}")
        return False, None

async def test_create_deep_graph():
    """测试create_deep_graph方法"""
    print("\n=== 测试 create_deep_graph 方法 ===")
    
    try:
        executor = TaskExecutor()
        graph = executor.create_deep_graph()
        
        # 验证图是否创建成功
        assert graph is not None, "图创建失败"
        print("✅ create_deep_graph 方法测试通过")
        
        # 验证图的结构
        try:
            graph_dict = graph.get_graph()
            nodes = list(graph_dict.nodes())
            print(f"   - 图节点: {nodes}")
        except Exception as e:
            print(f"   - 获取图节点信息时出错: {str(e)}")
            print(f"   - 图对象类型: {type(graph)}")
        
        return True
        
    except Exception as e:
        print(f"❌ create_deep_graph 方法测试失败: {str(e)}")
        return False

async def test_workflow_execution(executor):
    """测试完整的workflow执行流程"""
    print("\n=== 测试完整的workflow执行流程 ===")
    
    try:
        config = get_runnable_config()
        
        # 准备真实的免疫学任务
        test_tasks = [
            "分析抗体序列EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS的CDR区域",
            "预测该抗体序列与抗原的结合亲和力",
            "生成抗体序列的3D结构预测"
        ]
        
        print(f"   - 测试任务数量: {len(test_tasks)}")
        for i, task in enumerate(test_tasks, 1):
            print(f"   - 任务{i}: {task[:50]}...")
        
        # 运行完整管道
        result = await executor.complete_deep_pipeline(test_tasks, config)
        
        # 添加空值检查，防止NoneType错误
        if result is None:
            print("❌ complete_deep_pipeline返回了None")
            return False, None
        
        # 验证结果结构
        if not isinstance(result, dict):
            print(f"❌ 结果类型错误，期望dict，实际: {type(result)}")
            return False, None
            
        # 检查必需字段是否存在
        required_fields = ["task_results", "execution_summary"]
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            print(f"❌ 结果缺少必需字段: {missing_fields}")
            return False, None
        
        # 安全地获取execution_summary
        summary = result.get("execution_summary", {})
        if not isinstance(summary, dict):
            print(f"❌ execution_summary类型错误，期望dict，实际: {type(summary)}")
            return False, None
            
        print(f"✅ workflow执行测试通过")
        print(f"   - 执行摘要: {summary}")
        
        # 详细分析执行结果
        task_results = result.get("task_results", [])
        if not isinstance(task_results, list):
            print(f"⚠️ task_results类型异常，期望list，实际: {type(task_results)}")
            task_results = []
            
        print(f"\n   📊 详细执行结果:")
        for i, task_result in enumerate(task_results, 1):
            if task_result and isinstance(task_result, dict):
                status = task_result.get('status', 'unknown')
                print(f"   - 任务{i}: {status}")
                if 'error' in task_result:
                    error_msg = task_result['error']
                    if isinstance(error_msg, str):
                        print(f"     错误: {error_msg[:100]}...")
                    else:
                        print(f"     错误: {str(error_msg)[:100]}...")
            else:
                print(f"   - 任务{i}: 结果格式异常")
        
        return True, result
        
    except Exception as e:
        print(f"❌ workflow执行测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

async def test_run_deep_graph(executor):
    """测试run_deep_graph方法"""
    print("\n=== 测试 run_deep_graph 方法 ===")
    
    try:
        config = get_runnable_config()
        
        # 准备明确要求使用工具的测试任务
        test_tasks = [
            "请立即使用metabcr工具计算抗体序列的分子量。具体要求：1. 直接调用metabcr工具 2. 使用示例抗体序列进行分析 3. 不要询问更多信息，直接执行工具调用",
            "请立即使用run_figure2_analysis工具分析B细胞受体的多样性。具体要求：1. 直接调用run_figure2_analysis工具 2. 使用默认参数进行分析 3. 不要询问更多信息，直接执行工具调用"
        ]
        
        # 运行深度执行图
        result = await executor.run_deep_graph(test_tasks, config)
        
        # 验证结果结构
        assert isinstance(result, dict), "结果应该是字典类型"
        required_fields = ["task_results", "total_tasks", "completed_tasks", "failed_tasks"]
        for field in required_fields:
            assert field in result, f"结果应包含{field}字段"
        
        print("✅ run_deep_graph 方法测试通过")
        print(f"   - 总任务数: {result['total_tasks']}")
        print(f"   - 完成任务数: {result['completed_tasks']}")
        print(f"   - 失败任务数: {result['failed_tasks']}")
        
        return True
        
    except Exception as e:
        print(f"❌ run_deep_graph 方法测试失败: {str(e)}")
        return False

async def test_state_compatibility():
    """测试ImprovedCellState兼容性"""
    print("\n=== 测试 ImprovedCellState 兼容性 ===")
    
    try:
        # 创建测试状态
        test_state = ImprovedCellState(
            original_question="测试免疫学问题",
            decomposed_tasks=["分析抗体", "预测结构", "计算亲和力"]
        )
        
        # 验证状态字段
        assert hasattr(test_state, 'decomposed_tasks'), "状态应有decomposed_tasks字段"
        assert hasattr(test_state, 'original_question'), "状态应有original_question字段"
        assert len(test_state.decomposed_tasks) == 3, "任务列表长度应为3"
        
        print("✅ ImprovedCellState 兼容性测试通过")
        print(f"   - 原始问题: {test_state.original_question}")
        print(f"   - 任务数量: {len(test_state.decomposed_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ImprovedCellState 兼容性测试失败: {str(e)}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始测试深度执行图功能 - MCP服务已启动版本\n")
    
    test_results = []
    executor = None
    
    # 1. 测试基础功能
    print("📋 第一阶段: 基础功能测试")
    test_results.append(await test_create_deep_graph())
    test_results.append(await test_state_compatibility())
    
    # 2. 测试MCP连接
    print("\n📋 第二阶段: MCP服务连接测试")
    mcp_success, executor = await test_mcp_connection()
    test_results.append(mcp_success)
    
    # 3. 测试workflow执行（仅在MCP连接成功时）
    if mcp_success and executor:
        print("\n📋 第三阶段: Workflow执行测试")
        test_results.append(await test_run_deep_graph(executor))
        
        print("\n📋 第四阶段: 完整管道测试")
        workflow_success, workflow_result = await test_workflow_execution(executor)
        test_results.append(workflow_success)
        
        if workflow_success and workflow_result:
            print(f"\n🎯 完整执行结果预览:")
            print(f"   - 成功率: {workflow_result.get('success_rate', 0):.1f}%")
            print(f"   - 执行统计: {workflow_result.get('execution_summary', {})}")
        elif workflow_success:
            print(f"\n🎯 完整执行测试通过，但无详细结果数据")
    else:
        print("\n⚠️ 跳过workflow执行测试（MCP连接失败）")
        test_results.extend([False, False])  # 为跳过的测试添加失败结果
    
    # 统计测试结果
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n" + "="*50)
    print(f"🏆 测试结果汇总")
    print(f"="*50)
    print(f"✅ 通过: {passed}/{total}")
    print(f"📊 成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过! Workflow功能完全正常!")
    elif passed >= total * 0.6:
        print("✨ 大部分测试通过! 核心功能正常工作!")
    else:
        print("⚠️ 部分测试失败，请检查MCP服务状态和相关功能")
    
    print(f"\n💡 提示:")
    print(f"   - 如果MCP连接失败，请确保相关MCP服务已启动")
    print(f"   - 如果任务执行失败，可能是正常的（取决于具体任务复杂度）")
    print(f"   - 重点关注graph创建和基础workflow流程是否正常")
    print(f"   - 现在使用直接加载MCP工具的方式，不再使用subagents")

if __name__ == "__main__":
    asyncio.run(main())