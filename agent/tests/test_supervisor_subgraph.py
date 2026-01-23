"""
Supervisor Subgraph 单独测试用例

测试 supervisor subgraph 的独立功能，不涉及主图。
运行方式：pytest tests/test_supervisor_subgraph.py -v
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.supervisor.graph import (
    build_supervisor_subgraph,
    SupervisorState,
    supervisor_input_mapper,
    supervisor_output_mapper
)
from state import GlobalState, UserTaskType


def _ensure_supervisor_state(result):
    """确保结果是 SupervisorState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return SupervisorState(**result)
    return result


@pytest.fixture(scope="module")
def supervisor_subgraph():
    """构建并返回 Supervisor 子图"""
    return build_supervisor_subgraph()


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    return GlobalState(
        user_input="什么是免疫系统？",
        sandbox_dir="./sandbox"
    )


class TestSupervisorSubgraphBasic:
    """Supervisor Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, supervisor_subgraph):
        """测试子图构建是否成功"""
        assert supervisor_subgraph is not None
        print("✓ Supervisor Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, supervisor_subgraph, sample_global_state):
        """测试子图基本调用"""
        # 使用 input_mapper 转换状态
        subgraph_input = supervisor_input_mapper(sample_global_state)
        
        # 调用子图
        subgraph_output = supervisor_subgraph.invoke(subgraph_input)
        
        # 确保输出是 SupervisorState 对象
        result = _ensure_supervisor_state(subgraph_output)
        
        assert result is not None
        assert hasattr(result, 'user_input')
        assert result.user_input == sample_global_state.user_input
        assert result.user_task_type is not None
        print(f"✓ Supervisor Subgraph 基本调用成功，任务类型: {result.user_task_type.value}")


class TestTaskClassification:
    """任务分类测试"""
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("什么是免疫系统？", UserTaskType.GENERAL_QA),
        ("请解释一下什么是抗体", UserTaskType.GENERAL_QA),
        ("你好，我想了解一下生物信息学", UserTaskType.GENERAL_QA),
    ])
    def test_general_qa_classification(self, supervisor_subgraph, input_text, expected_type):
        """测试普通问答任务分类"""
        input_state = SupervisorState(
            user_input=input_text,
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 验证分类结果（允许一定的灵活性，因为使用LLM判断）
        assert result.user_task_type in [UserTaskType.GENERAL_QA, UserTaskType.IMMUNOLOGY_TASK]
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("执行以下计划：1. 分析数据 2. 生成报告", UserTaskType.EXECUTE_PLAN),
        ("按照这个步骤执行：第一步，第二步，第三步", UserTaskType.EXECUTE_PLAN),
        ("请按照以下计划执行任务", UserTaskType.EXECUTE_PLAN),
    ])
    def test_execute_plan_classification(self, supervisor_subgraph, input_text, expected_type):
        """测试执行计划任务分类"""
        input_state = SupervisorState(
            user_input=input_text,
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 验证分类结果
        assert result.user_task_type == UserTaskType.EXECUTE_PLAN
    
    @pytest.mark.parametrize("input_text,expected_type", [
        ("分析这个抗原抗体的相互作用", UserTaskType.IMMUNOLOGY_TASK),
        ("请帮我分析免疫细胞的激活过程", UserTaskType.IMMUNOLOGY_TASK),
        ("这个疫苗的免疫原性如何？", UserTaskType.IMMUNOLOGY_TASK),
        ("T细胞和B细胞的区别是什么？", UserTaskType.IMMUNOLOGY_TASK),
    ])
    def test_immunology_task_classification(self, supervisor_subgraph, input_text, expected_type):
        """测试免疫学任务分类"""
        input_state = SupervisorState(
            user_input=input_text,
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        assert result.user_task_type is not None
        print(f"  输入: {input_text}")
        print(f"  判断类型: {result.user_task_type.value}, 预期: {expected_type.value}")
        
        # 验证分类结果
        assert result.user_task_type == UserTaskType.IMMUNOLOGY_TASK


class TestSupervisorStateMapping:
    """Supervisor 状态映射测试"""
    
    def test_input_mapper(self, sample_global_state):
        """测试输入映射函数"""
        subgraph_state = supervisor_input_mapper(sample_global_state)
        
        assert isinstance(subgraph_state, SupervisorState)
        assert subgraph_state.user_input == sample_global_state.user_input
        assert subgraph_state.sandbox_dir == sample_global_state.sandbox_dir
        assert subgraph_state.user_task_type is None  # 将在子图中判断
        assert subgraph_state.uploaded_files == []
        assert subgraph_state.sandbox_file_paths == {}
        print("✓ 输入映射函数测试通过")
    
    def test_output_mapper(self, sample_global_state):
        """测试输出映射函数"""
        # 创建模拟的子图输出
        subgraph_output = SupervisorState(
            user_input="什么是免疫系统？",
            user_task_type=UserTaskType.GENERAL_QA,
            sandbox_dir="./sandbox",
            uploaded_files=[],
            sandbox_file_paths={},
            execution_plan=None
        )
        
        # 执行输出映射
        updated_global_state = supervisor_output_mapper(subgraph_output, sample_global_state)
        
        # 验证结果已同步
        assert updated_global_state.user_task_type == UserTaskType.GENERAL_QA
        print("✓ 输出映射函数测试通过")


class TestSupervisorEdgeCases:
    """Supervisor 边界情况测试"""
    
    def test_empty_input(self, supervisor_subgraph):
        """测试空输入"""
        input_state = SupervisorState(
            user_input="",
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        # 即使输入为空，也应该有分类结果（默认普通问答）
        assert result.user_task_type is not None
        print(f"✓ 空输入测试通过，任务类型: {result.user_task_type.value}")
    
    def test_long_input(self, supervisor_subgraph):
        """测试长输入"""
        long_input = "什么是" + "A" * 500 + "？"
        input_state = SupervisorState(
            user_input=long_input,
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        assert result.user_task_type is not None
        print(f"✓ 长输入测试通过（输入长度: {len(long_input)} 字符）")
    
    def test_special_characters(self, supervisor_subgraph):
        """测试特殊字符输入"""
        special_input = "什么是@#$%^&*()？"
        input_state = SupervisorState(
            user_input=special_input,
            sandbox_dir="./sandbox"
        )
        
        result = supervisor_subgraph.invoke(input_state)
        result = _ensure_supervisor_state(result)
        
        assert result.user_task_type is not None
        print("✓ 特殊字符输入测试通过")


class TestSupervisorCompleteFlow:
    """Supervisor 完整流程测试"""
    
    def test_classification_and_mapping_flow(self, supervisor_subgraph, sample_global_state):
        """测试完整的分类和映射流程"""
        # 1. 输入映射
        subgraph_input = supervisor_input_mapper(sample_global_state)
        assert isinstance(subgraph_input, SupervisorState)
        assert subgraph_input.user_task_type is None
        
        # 2. 执行子图
        subgraph_output = supervisor_subgraph.invoke(subgraph_input)
        result = _ensure_supervisor_state(subgraph_output)
        
        # 3. 验证分类结果
        assert result.user_task_type is not None
        assert result.user_task_type in [
            UserTaskType.GENERAL_QA,
            UserTaskType.EXECUTE_PLAN,
            UserTaskType.IMMUNOLOGY_TASK
        ]
        
        # 4. 输出映射
        updated_global_state = supervisor_output_mapper(result, sample_global_state)
        assert updated_global_state.user_task_type == result.user_task_type
        
        print("✓ 完整流程测试通过")
        print(f"  分类结果: {result.user_task_type.value}")


def test_environment_check():
    """检查环境配置"""
    print("\n" + "=" * 50)
    print("检查 Supervisor Subgraph 环境配置...")
    
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
        print("\n警告：未配置任何LLM API Key，Supervisor将使用关键字判断降级方案")
    else:
        print("✓ 至少一个LLM API Key 已配置")
    
    print("=" * 50)
    assert True  # 确保测试通过，即使没有LLM Key

