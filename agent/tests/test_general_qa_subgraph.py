"""
General QA Subgraph 单独测试用例

测试 general_qa subgraph 的独立功能，不涉及主图。
运行方式：pytest tests/test_general_qa_subgraph.py -v
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

from nodes.subagents.general_qa.graph import (
    build_general_qa_subgraph,
    GeneralQAState,
    general_qa_input_mapper,
    general_qa_output_mapper
)
from state import GlobalState


def _ensure_general_qa_state(result):
    """确保结果是 GeneralQAState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return GeneralQAState(**result)
    return result


@pytest.fixture(scope="module")
def general_qa_subgraph():
    """构建并返回 General QA 子图"""
    return build_general_qa_subgraph()


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    return GlobalState(
        user_input="什么是免疫系统？",
        sandbox_dir="./sandbox"
    )


class TestGeneralQASubgraphBasic:
    """General QA Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, general_qa_subgraph):
        """测试子图构建是否成功"""
        assert general_qa_subgraph is not None
        print("✓ General QA Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, general_qa_subgraph, sample_global_state):
        """测试子图基本调用"""
        # 使用 input_mapper 转换状态
        subgraph_input = general_qa_input_mapper(sample_global_state)
        
        # 调用子图
        subgraph_output = general_qa_subgraph.invoke(subgraph_input)
        
        # 确保输出是 GeneralQAState 对象
        result = _ensure_general_qa_state(subgraph_output)
        
        assert result is not None
        assert hasattr(result, 'user_input')
        assert result.user_input == sample_global_state.user_input
        print(f"✓ General QA Subgraph 基本调用成功")


class TestGeneralQAAnswerGeneration:
    """General QA 回答生成测试"""
    
    @pytest.mark.parametrize("question", [
        "什么是免疫系统？",
        "请解释一下什么是抗体",
        "生物信息学的主要应用领域有哪些？",
        "什么是蛋白质结构预测？",
    ])
    def test_answer_generation(self, general_qa_subgraph, question):
        """测试回答生成"""
        input_state = GeneralQAState(
            user_input=question,
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 验证回答已生成
        assert result.answer is not None
        assert len(result.answer) > 0
        print(f"\n问题: {question}")
        print(f"回答长度: {len(result.answer)} 字符")
        print(f"回答预览: {result.answer[:100]}...")
    
    def test_answer_with_confidence(self, general_qa_subgraph):
        """测试置信度信息生成"""
        input_state = GeneralQAState(
            user_input="什么是DNA？",
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 验证置信度信息
        assert result.confidence is not None
        assert len(result.confidence) > 0
        print(f"\n置信度: {result.confidence}")
    
    def test_related_topics_generation(self, general_qa_subgraph):
        """测试相关问题生成"""
        input_state = GeneralQAState(
            user_input="什么是免疫系统？",
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 验证相关问题已生成
        assert result.related_topics is not None
        assert isinstance(result.related_topics, list)
        print(f"\n相关问题数量: {len(result.related_topics)}")
        if result.related_topics:
            print(f"相关问题: {result.related_topics}")
    
    def test_sources_suggested_generation(self, general_qa_subgraph):
        """测试参考资料生成"""
        input_state = GeneralQAState(
            user_input="什么是蛋白质折叠？",
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 验证参考资料已生成
        assert result.sources_suggested is not None
        assert isinstance(result.sources_suggested, list)
        print(f"\n参考资料数量: {len(result.sources_suggested)}")
        if result.sources_suggested:
            print(f"参考资料: {result.sources_suggested}")


class TestGeneralQACompleteOutput:
    """General QA 完整输出测试"""
    
    def test_complete_output_structure(self, general_qa_subgraph):
        """测试完整输出结构"""
        input_state = GeneralQAState(
            user_input="什么是CRISPR基因编辑技术？",
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 验证所有字段都存在
        assert result.answer is not None, "回答字段缺失"
        assert result.confidence is not None, "置信度字段缺失"
        assert result.related_topics is not None, "相关问题字段缺失"
        assert result.sources_suggested is not None, "参考资料字段缺失"
        
        # 验证字段类型
        assert isinstance(result.answer, str), "回答应该是字符串"
        assert isinstance(result.confidence, str), "置信度应该是字符串"
        assert isinstance(result.related_topics, list), "相关问题应该是列表"
        assert isinstance(result.sources_suggested, list), "参考资料应该是列表"
        
        print("\n✓ 完整输出结构验证通过")
        print(f"  回答: {len(result.answer)} 字符")
        print(f"  置信度: {len(result.confidence)} 字符")
        print(f"  相关问题: {len(result.related_topics)} 个")
        print(f"  参考资料: {len(result.sources_suggested)} 个")


class TestGeneralQAStateMapping:
    """General QA 状态映射测试"""
    
    def test_input_mapper(self, sample_global_state):
        """测试输入映射函数"""
        subgraph_state = general_qa_input_mapper(sample_global_state)
        
        assert isinstance(subgraph_state, GeneralQAState)
        assert subgraph_state.user_input == sample_global_state.user_input
        assert subgraph_state.answer is None
        assert subgraph_state.confidence is None
        assert subgraph_state.related_topics == []
        assert subgraph_state.sources_suggested == []
        print("✓ 输入映射函数测试通过")
    
    def test_output_mapper(self, sample_global_state):
        """测试输出映射函数"""
        # 创建模拟的子图输出
        subgraph_output = GeneralQAState(
            user_input="什么是免疫系统？",
            answer="免疫系统是...",
            confidence="高置信度",
            related_topics=["免疫细胞", "免疫反应"],
            sources_suggested=["PubMed", "相关期刊"]
        )
        
        # 执行输出映射
        updated_global_state = general_qa_output_mapper(subgraph_output, sample_global_state)
        
        # 验证结果已同步到 merged_result
        assert updated_global_state.merged_result is not None
        assert "general_qa_answer" in updated_global_state.merged_result
        assert "general_qa_confidence" in updated_global_state.merged_result
        assert "general_qa_related_topics" in updated_global_state.merged_result
        assert "general_qa_sources" in updated_global_state.merged_result
        
        assert updated_global_state.merged_result["general_qa_answer"] == "免疫系统是..."
        assert updated_global_state.merged_result["general_qa_confidence"] == "高置信度"
        assert updated_global_state.merged_result["general_qa_related_topics"] == ["免疫细胞", "免疫反应"]
        assert updated_global_state.merged_result["general_qa_sources"] == ["PubMed", "相关期刊"]
        
        print("✓ 输出映射函数测试通过")


class TestGeneralQAEdgeCases:
    """General QA 边界情况测试"""
    
    def test_empty_input(self, general_qa_subgraph):
        """测试空输入"""
        input_state = GeneralQAState(
            user_input="",
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        # 即使输入为空，也应该有降级回答
        assert result.answer is not None
        print("✓ 空输入测试通过")
    
    def test_long_input(self, general_qa_subgraph):
        """测试长输入"""
        long_input = "什么是" + "A" * 500 + "？"
        input_state = GeneralQAState(
            user_input=long_input,
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        assert result.answer is not None
        print(f"✓ 长输入测试通过（输入长度: {len(long_input)} 字符）")
    
    def test_special_characters(self, general_qa_subgraph):
        """测试特殊字符输入"""
        special_input = "什么是@#$%^&*()？"
        input_state = GeneralQAState(
            user_input=special_input,
            answer=None,
            confidence=None,
            related_topics=[],
            sources_suggested=[]
        )
        
        result = general_qa_subgraph.invoke(input_state)
        result = _ensure_general_qa_state(result)
        
        assert result.answer is not None
        print("✓ 特殊字符输入测试通过")


def test_environment_check():
    """检查环境配置"""
    print("\n" + "=" * 50)
    print("检查 General QA Subgraph 环境配置...")
    
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
        print("\n警告：未配置任何LLM API Key，General QA将使用降级方案")
    else:
        print("✓ 至少一个LLM API Key 已配置")
    
    print("=" * 50)
    assert True  # 确保测试通过，即使没有LLM Key

