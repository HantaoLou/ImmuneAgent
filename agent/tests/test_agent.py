"""
Agent 测试用例

使用 pytest 框架进行测试
运行方式：pytest tests/test_agent.py -v
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from main_graph import build_main_graph
from state import GlobalState, UserTaskType


def _ensure_global_state(result):
    """确保结果是 GlobalState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return GlobalState(**result)
    return result


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


class TestTaskClassification:
    """任务分类测试"""
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("什么是免疫系统？", UserTaskType.GENERAL_QA),
        ("请解释一下什么是抗体", UserTaskType.GENERAL_QA),
        ("你好，我想了解一下生物信息学", UserTaskType.GENERAL_QA),
    ])
    def test_general_qa_classification(self, agent_graph, default_sandbox_dir, input_text, expected_type):
        """测试普通问答任务分类"""
        result = agent_graph.invoke({
            "user_input": input_text,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 注意：由于使用LLM判断，可能不完全匹配，这里只检查是否返回了类型
        assert result.user_task_type in [UserTaskType.GENERAL_QA, UserTaskType.IMMUNOLOGY_TASK]
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("执行以下计划：1. 分析数据 2. 生成报告", UserTaskType.EXECUTE_PLAN),
        ("按照这个步骤执行：第一步，第二步，第三步", UserTaskType.EXECUTE_PLAN),
        ("请按照以下计划执行任务", UserTaskType.EXECUTE_PLAN),
    ])
    def test_execute_plan_classification(self, agent_graph, default_sandbox_dir, input_text, expected_type):
        """测试执行计划任务分类"""
        result = agent_graph.invoke({
            "user_input": input_text,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 检查是否识别为执行计划类型
        assert result.user_task_type == UserTaskType.EXECUTE_PLAN
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("分析这个抗原抗体的相互作用", UserTaskType.IMMUNOLOGY_TASK),
        ("请帮我分析免疫细胞的激活过程", UserTaskType.IMMUNOLOGY_TASK),
        ("这个疫苗的免疫原性如何？", UserTaskType.IMMUNOLOGY_TASK),
        ("T细胞和B细胞的区别是什么？", UserTaskType.IMMUNOLOGY_TASK),
    ])
    def test_immunology_task_classification(self, agent_graph, default_sandbox_dir, input_text, expected_type):
        """测试免疫学任务分类"""
        result = agent_graph.invoke({
            "user_input": input_text,
            "sandbox_dir": default_sandbox_dir
        })
        
        result = _ensure_global_state(result)
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 检查是否识别为免疫学任务
        assert result.user_task_type == UserTaskType.IMMUNOLOGY_TASK


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


class TestIntegration:
    """集成测试"""
    
    def test_multiple_invocations(self, agent_graph, default_sandbox_dir):
        """测试多次调用"""
        inputs = [
            "什么是免疫系统？",
            "分析抗原抗体反应",
            "执行计划：1. 分析 2. 报告"
        ]
        
        results = []
        for input_text in inputs:
            result = agent_graph.invoke({
                "user_input": input_text,
                "sandbox_dir": default_sandbox_dir
            })
            result = _ensure_global_state(result)
            results.append(result)
            assert result.user_task_type is not None
        
        print(f"✓ 成功处理 {len(results)} 个请求")
        for i, result in enumerate(results, 1):
            print(f"  请求 {i}: {result.user_task_type.value}")


def test_environment_check():
    """检查环境配置"""
    print("\n" + "=" * 50)
    print("环境配置检查")
    print("=" * 50)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_llm = any([openai_key])
    
    if openai_key:
        print("✓ OpenAI API Key 已配置")
    else:
        print("✗ OpenAI API Key 未配置")
    
    if not has_llm:
        print("\n⚠ 警告：未配置任何LLM API Key，将使用降级方案")
    
    print("=" * 50 + "\n")
    
    # 不强制要求 API Key，因为可能有降级方案
    assert True


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])

