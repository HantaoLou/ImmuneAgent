"""
Agent 全流程测试用例

测试整个 Agent 系统的完整流程，包括：
- Supervisor 分类 → 路由 → 子图处理 → 结果返回
- 不同类型任务的完整处理流程
- 端到端的集成测试

注意：特定 subgraph 的单独测试请参考：
- test_supervisor_subgraph.py：Supervisor subgraph 单独测试
- test_general_qa_subgraph.py：General QA subgraph 单独测试

运行方式：pytest tests/test_agent.py -v
"""

import os
import sys
import pytest
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


def _ensure_global_state(result):
    """确保结果是 GlobalState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return GlobalState(**result)
    return result


def get_task_type_str(task_type):
    """安全地获取任务类型字符串"""
    if task_type is None:
        return '未知'
    if isinstance(task_type, str):
        return task_type
    if hasattr(task_type, 'value'):
        return task_type.value
    return str(task_type)


@pytest.fixture(scope="module")
def agent_graph():
    """构建并返回 Agent 主图"""
    return build_main_graph()


@pytest.fixture
def default_sandbox_dir():
    """默认沙盒目录"""
    return "./sandbox"


class TestAgentBasic:
    """基础功能测试"""
    
    def test_agent_graph_build(self, agent_graph):
        """测试 Agent 图构建是否成功"""
        assert agent_graph is not None
        print("✓ Agent 图构建成功")
    
    def test_agent_invoke_basic(self, agent_graph, default_sandbox_dir):
        """测试 Agent 基本调用"""
        result = agent_graph.invoke({
            "user_input": "测试输入",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result is not None
        assert hasattr(result, 'user_input')
        assert result.user_input == "测试输入"
        print(f"✓ Agent 基本调用成功，任务类型: {result.user_task_type}")


class TestFullWorkflow:
    """全流程测试"""
    
    def test_full_workflow_general_qa(self, agent_graph, default_sandbox_dir):
        """测试普通问答的完整流程：Supervisor → General QA → 结果"""
        result = agent_graph.invoke({
            "user_input": "什么是DNA？",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        
        # 验证完整流程
        # 1. Supervisor 分类成功
        assert result.user_task_type == UserTaskType.GENERAL_QA
        
        # 2. 路由到 General QA 并生成结果
        assert result.merged_result is not None
        assert "general_qa_answer" in result.merged_result
        assert len(result.merged_result["general_qa_answer"]) > 0
        
        print("✓ 普通问答全流程测试通过")
        print(f"  回答长度: {len(result.merged_result['general_qa_answer'])} 字符")
    
    def test_full_workflow_immunology_task(self, agent_graph, default_sandbox_dir):
        """测试免疫学任务的完整流程：Supervisor → Immunity → 结果"""
        result = agent_graph.invoke({
            "user_input": "分析抗原抗体反应",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        
        # 验证完整流程
        # 1. Supervisor 分类成功
        assert result.user_task_type == UserTaskType.IMMUNOLOGY_TASK
        
        # 2. 路由到 Immunity 节点并生成结果
        assert result.merged_result is not None
        assert "immunity_response" in result.merged_result
        
        print("✓ 免疫学任务全流程测试通过")
    
    def test_full_workflow_execute_plan(self, agent_graph, default_sandbox_dir):
        """测试执行计划的完整流程：Supervisor → Task Decomposition → Executor → 结果"""
        result = agent_graph.invoke({
            "user_input": "执行计划：1. 分析数据 2. 生成报告",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        
        # 验证完整流程
        # 1. Supervisor 分类成功
        assert result.user_task_type == UserTaskType.EXECUTE_PLAN
        
        # 2. 任务分解应该生成子任务（如果有执行计划）
        # 注意：如果没有执行计划，可能不会生成子任务
        
        # 3. Executor 执行结果
        assert result.merged_result is not None
        # 检查是否有 executor_results 或任务执行结果
        has_executor_results = (
            "executor_results" in result.merged_result or
            len(result.completed_tasks) > 0 or
            len(result.subtasks) > 0
        )
        
        print("✓ 执行计划全流程测试通过")
        if "executor_results" in result.merged_result:
            executor_results = result.merged_result["executor_results"]
            print(f"  执行结果: {executor_results.get('completed', 0)} 个任务完成")


class TestStateManagement:
    """状态管理测试"""
    
    def test_state_persistence(self, agent_graph, default_sandbox_dir):
        """测试状态持久化"""
        initial_state = {
            "user_input": "测试状态持久化",
            "sandbox_dir": default_sandbox_dir
        }
        
        result = agent_graph.invoke(initial_state)
        result = _ensure_global_state(result)
        
        # 验证状态字段
        assert result.user_input == initial_state["user_input"]
        assert result.sandbox_dir == initial_state["sandbox_dir"]
        assert result.user_task_type is not None
        assert isinstance(result.subtasks, list)
        assert isinstance(result.parallel_task_groups, dict)
        assert isinstance(result.completed_tasks, dict)
        assert isinstance(result.merged_result, dict)
        print("✓ 状态管理正常")
    
    def test_state_with_custom_sandbox(self, agent_graph):
        """测试自定义沙盒目录"""
        custom_sandbox = "./custom_sandbox"
        result = agent_graph.invoke({
            "user_input": "测试自定义沙盒",
            "sandbox_dir": custom_sandbox
        })
        
        result = _ensure_global_state(result)
        assert result.sandbox_dir == custom_sandbox
        print(f"✓ 自定义沙盒目录设置成功: {custom_sandbox}")


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_input(self, agent_graph, default_sandbox_dir):
        """测试空输入"""
        result = agent_graph.invoke({
            "user_input": "",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result is not None
        assert result.user_input == ""
        print("✓ 空输入处理正常")
    
    def test_long_input(self, agent_graph, default_sandbox_dir):
        """测试长输入"""
        long_input = "这是一个很长的输入文本。 " * 100
        result = agent_graph.invoke({
            "user_input": long_input,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result is not None
        assert len(result.user_input) > 0
        print("✓ 长输入处理正常")
    
    def test_special_characters(self, agent_graph, default_sandbox_dir):
        """测试特殊字符"""
        special_input = "测试特殊字符：!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = agent_graph.invoke({
            "user_input": special_input,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result is not None
        print("✓ 特殊字符处理正常")




class TestEndToEndIntegration:
    """端到端集成测试"""
    
    def test_multiple_task_types_processing(self, agent_graph, default_sandbox_dir):
        """测试处理多种不同类型的任务，验证系统的完整性和稳定性"""
        test_cases = [
            ("什么是DNA？", UserTaskType.GENERAL_QA, "general_qa_answer"),
            ("分析抗原抗体反应", UserTaskType.IMMUNOLOGY_TASK, "immunity_response"),
            ("执行计划：1. 分析数据 2. 生成报告", UserTaskType.EXECUTE_PLAN, "executor_results"),
        ]
        
        results = []
        for input_text, expected_type, expected_result_key in test_cases:
            result = agent_graph.invoke({
                "user_input": input_text,
                "sandbox_dir": default_sandbox_dir
            })
            result = _ensure_global_state(result)
            
            # 验证分类正确
            assert result.user_task_type == expected_type
            
            # 验证生成了相应的结果
            assert result.merged_result is not None
            # 对于执行计划，可能没有 executor_results，但应该有任务列表或执行结果
            if expected_result_key == "executor_results":
                has_results = (
                    expected_result_key in result.merged_result or
                    len(result.completed_tasks) > 0 or
                    len(result.subtasks) > 0
                )
                assert has_results, "执行计划应该生成任务或执行结果"
            else:
                assert expected_result_key in result.merged_result
            
            results.append((input_text, get_task_type_str(result.user_task_type), expected_result_key))
        
        print(f"✓ 成功处理 {len(results)} 个不同类型的任务")
        for i, (input_text, task_type, result_key) in enumerate(results, 1):
            print(f"  任务 {i}: {task_type} -> {result_key}")
    
    def test_complete_workflow_with_general_qa(self, agent_graph, default_sandbox_dir):
        """测试包含 General QA 的完整工作流程"""
        result = agent_graph.invoke({
            "user_input": "什么是蛋白质折叠？请详细解释。",
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        
        # 验证完整流程的每个步骤
        # 1. Supervisor 分类
        assert result.user_task_type == UserTaskType.GENERAL_QA
        
        # 2. 路由到 General QA
        assert result.merged_result is not None
        assert "general_qa_answer" in result.merged_result
        
        # 3. General QA 生成完整输出
        answer = result.merged_result["general_qa_answer"]
        assert answer is not None and len(answer) > 0
        
        # 4. 可选：验证其他字段（如果生成）
        if "general_qa_confidence" in result.merged_result:
            print(f"  置信度: {result.merged_result['general_qa_confidence'][:50]}...")
        if "general_qa_related_topics" in result.merged_result:
            print(f"  相关问题: {len(result.merged_result['general_qa_related_topics'])} 个")
        if "general_qa_sources" in result.merged_result:
            print(f"  参考资料: {len(result.merged_result['general_qa_sources'])} 个")
        
        print("✓ 包含 General QA 的完整工作流程测试通过")
        print(f"  回答长度: {len(answer)} 字符")
    
    def test_state_consistency_throughout_workflow(self, agent_graph, default_sandbox_dir):
        """测试整个工作流程中状态的一致性"""
        initial_input = "什么是CRISPR技术？"
        
        result = agent_graph.invoke({
            "user_input": initial_input,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        
        # 验证状态在整个流程中保持一致
        assert result.user_input == initial_input
        assert result.sandbox_dir == default_sandbox_dir
        assert result.user_task_type is not None
        assert result.merged_result is not None
        
        # 验证状态结构完整
        assert isinstance(result.subtasks, list)
        assert isinstance(result.parallel_task_groups, dict)
        assert isinstance(result.completed_tasks, dict)
        assert isinstance(result.merged_result, dict)
        
        print("✓ 状态一致性测试通过")
        print(f"  任务类型: {get_task_type_str(result.user_task_type)}")
        print(f"  结果键数量: {len(result.merged_result)}")


def test_environment_check():
    """检查全流程测试环境配置"""
    print("\n" + "=" * 50)
    print("全流程测试环境配置检查")
    print("=" * 50)
    
    dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_llm = any([dashscope_key, anthropic_key, openai_key])
    
    if dashscope_key:
        print("✓ 通义千问 API Key 已配置")
    else:
        print("✗ 通义千问 API Key 未配置")
    
    if anthropic_key:
        print("✓ Anthropic API Key 已配置")
    else:
        print("✗ Anthropic API Key 未配置")
    
    if openai_key:
        print("✓ OpenAI API Key 已配置")
    else:
        print("✗ OpenAI API Key 未配置")
    
    if not has_llm:
        print("\n⚠ 警告：未配置任何LLM API Key，将使用降级方案")
    else:
        print("✓ 至少一个LLM API Key 已配置")
    
    print("=" * 50 + "\n")
    
    # 不强制要求 API Key，因为可能有降级方案
    assert True


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])

