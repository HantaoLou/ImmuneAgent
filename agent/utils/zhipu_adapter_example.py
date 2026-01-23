"""
ZhipuAI 适配器使用示例

展示如何使用 ZhipuAI 适配器，以 OpenAI 兼容的方式调用智谱 AI 模型。
"""

import os
from langchain_core.messages import HumanMessage, SystemMessage

# 方式1: 直接使用适配器
from agent.utils.zhipu_adapter import ZhipuAIAdapter, create_zhipu_chat_model

# 方式2: 通过 llm_factory 使用（推荐）
from agent.utils.llm_factory import create_reasoning_advanced_llm


def example_1_direct_use():
    """示例1: 直接使用适配器"""
    print("=" * 60)
    print("示例1: 直接使用 ZhipuAIAdapter")
    print("=" * 60)
    
    # 创建适配器实例
    llm = ZhipuAIAdapter(
        model="chatglm3-6b-1001",  # 或使用其他模型，如 "glm-4"
        temperature=0.7,
        api_key=os.getenv("ZHIPU_API_KEY")  # 从环境变量读取
    )
    
    # 准备消息（与 OpenAI 完全相同的格式）
    messages = [
        SystemMessage(content="你是一个名为chatGLM的AI助手。"),
        HumanMessage(content="你好，请介绍一下自己。")
    ]
    
    # 调用（与 OpenAI 完全相同的用法）
    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def example_2_convenience_function():
    """示例2: 使用便捷函数"""
    print("=" * 60)
    print("示例2: 使用便捷函数 create_zhipu_chat_model")
    print("=" * 60)
    
    # 使用便捷函数创建
    llm = create_zhipu_chat_model(
        model="chatglm3-6b-1001",
        temperature=0.7
    )
    
    messages = [
        SystemMessage(content="你是一个专业的生物信息学助手。"),
        HumanMessage(content="什么是抗体？")
    ]
    
    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def example_3_through_factory():
    """示例3: 通过 llm_factory 使用（推荐方式）"""
    print("=" * 60)
    print("示例3: 通过 llm_factory 使用（推荐）")
    print("=" * 60)
    
    # 使用工厂函数创建（会自动根据配置选择模型）
    llm = create_reasoning_advanced_llm(temperature=0.7)
    
    if llm is None:
        print("警告: 无法创建 LLM 实例，请检查 ZHIPU_API_KEY 环境变量")
        return
    
    messages = [
        SystemMessage(content="你是一个AI助手。"),
        HumanMessage(content="请用一句话介绍自己。")
    ]
    
    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def example_4_custom_model():
    """示例4: 使用自定义模型"""
    print("=" * 60)
    print("示例4: 使用自定义模型")
    print("=" * 60)
    
    # 通过 custom_model 参数指定 zhipu 模型
    llm = create_reasoning_advanced_llm(
        temperature=0.7,
        custom_model="zhipu:glm-4"  # 格式: provider:model
    )
    
    if llm is None:
        print("警告: 无法创建 LLM 实例")
        return
    
    messages = [
        HumanMessage(content="解释一下什么是机器学习。")
    ]
    
    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def example_5_with_structured_output():
    """示例5: 使用结构化输出（与 OpenAI 完全兼容）"""
    print("=" * 60)
    print("示例5: 使用结构化输出")
    print("=" * 60)
    
    from pydantic import BaseModel, Field
    
    # 定义输出结构
    class PersonInfo(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(description="年龄")
        city: str = Field(description="城市")
    
    llm = create_zhipu_chat_model(
        model="chatglm3-6b-1001",
        temperature=0.7
    )
    
    # 使用 with_structured_output（与 OpenAI 完全相同的用法）
    structured_llm = llm.with_structured_output(PersonInfo)
    
    messages = [
        HumanMessage(content="请提取以下信息：张三，25岁，来自北京。")
    ]
    
    result = structured_llm.invoke(messages)
    print(f"结构化输出: {result}")
    print(f"类型: {type(result)}")
    print()


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("ZHIPU_API_KEY"):
        print("警告: 请先设置 ZHIPU_API_KEY 环境变量")
        print("例如: export ZHIPU_API_KEY='your_api_key_here'")
        print()
    
    # 运行示例
    try:
        example_1_direct_use()
        example_2_convenience_function()
        example_3_through_factory()
        example_4_custom_model()
        example_5_with_structured_output()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

