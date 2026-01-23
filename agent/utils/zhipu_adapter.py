"""
ZhipuAI OpenAI 兼容适配器

将 ZhipuAI SDK 的调用方式适配为 OpenAI 兼容的接口，使其可以与 LangChain 无缝集成。
"""

from typing import Optional, Any, List, Dict, Iterator, Type, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel
import os
import json
import re


class ZhipuAIAdapter(BaseChatModel):
    """
    ZhipuAI 适配器类
    
    将 ZhipuAI SDK 的调用方式适配为 LangChain 兼容的接口。
    使用方式与 OpenAI 的 ChatOpenAI 完全一致。
    """
    
    model: str = "chatglm3-6b-1001"
    temperature: float = 0.7
    api_key: Optional[str] = None
    zhipu_client: Optional[Any] = None
    
    def __init__(
        self,
        model: str = "chatglm3-6b-1001",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 ZhipuAI 适配器
        
        Args:
            model: 模型名称，默认为 "chatglm3-6b-1001"
            temperature: 温度参数，默认为 0.7
            api_key: API 密钥，如果不提供则从环境变量 ZHIPU_API_KEY 读取
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.temperature = temperature
        
        # 获取 API Key
        if api_key is None:
            api_key = os.getenv("ZHIPU_API_KEY")
        
        if not api_key:
            raise ValueError("ZHIPU_API_KEY 未设置，请设置环境变量或传入 api_key 参数")
        
        self.api_key = api_key
        
        # 初始化 ZhipuAI 客户端
        try:
            from zai import ZhipuAiClient
            self.zhipu_client = ZhipuAiClient(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 zai 库: pip install zai")
        except Exception as e:
            raise RuntimeError(f"初始化 ZhipuAI 客户端失败: {e}")
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        return "zhipu"
    
    def _convert_messages_to_zhipu_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        将 LangChain 的 Message 对象转换为 ZhipuAI 格式
        
        Args:
            messages: LangChain 的 Message 列表
            
        Returns:
            ZhipuAI 格式的消息列表
        """
        zhipu_messages = []
        
        for message in messages:
            # 获取消息内容，确保是字符串类型
            if isinstance(message, SystemMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 system 消息
                if not content.strip():
                    continue
                zhipu_messages.append({
                    "role": "system",
                    "content": content
                })
            elif isinstance(message, HumanMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 user 消息
                if not content.strip():
                    continue
                zhipu_messages.append({
                    "role": "user",
                    "content": content
                })
            elif isinstance(message, AIMessage):
                content = message.content
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空的 assistant 消息
                if not content.strip():
                    continue
                zhipu_messages.append({
                    "role": "assistant",
                    "content": content
                })
            else:
                # 对于其他类型的消息，尝试获取内容
                content = getattr(message, 'content', None)
                if content is None:
                    content = str(message)
                elif not isinstance(content, str):
                    content = str(content)
                # 跳过空消息
                if not content.strip():
                    continue
                zhipu_messages.append({
                    "role": "user",
                    "content": content
                })
        
        # 确保至少有一条消息
        if not zhipu_messages:
            raise ValueError("消息列表为空或所有消息内容为空")
        
        # 确保最后一条消息是 user 消息（ZhipuAI 要求）
        if zhipu_messages and zhipu_messages[-1]["role"] not in ["user", "assistant"]:
            raise ValueError("最后一条消息必须是 user 或 assistant 消息")
        
        return zhipu_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成回复（核心方法）
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult 对象
        """
        # 转换消息格式
        try:
            zhipu_messages = self._convert_messages_to_zhipu_format(messages)
        except Exception as e:
            raise RuntimeError(f"消息格式转换失败: {e}")
        
        # 验证消息格式
        if not zhipu_messages:
            raise ValueError("转换后的消息列表为空")
        
        # 调试：打印消息格式（仅在开发时使用）
        # print(f"调试：ZhipuAI 消息格式: {zhipu_messages}")
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": zhipu_messages,
            "temperature": self.temperature,
        }
        
        # 添加其他参数
        if stop:
            request_params["stop"] = stop
        
        request_params.update(kwargs)
        
        try:
            # 调用 ZhipuAI SDK
            response = self.zhipu_client.chat.completions.create(**request_params)
            
            # 提取回复内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
            elif hasattr(response, 'data') and hasattr(response.data, 'choices'):
                content = response.data.choices[0].message.content
            else:
                # 尝试直接获取 content
                content = getattr(response, 'content', str(response))
            
            # 创建 ChatGeneration 对象
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={
                    "model": self.model,
                    "temperature": self.temperature,
                }
            )
            
            # 创建 ChatResult 对象
            return ChatResult(generations=[generation])
            
        except Exception as e:
            # 提供更详细的错误信息
            error_msg = str(e)
            if "messages" in error_msg.lower() or "参数非法" in error_msg:
                # 打印消息格式以便调试
                print(f"错误：ZhipuAI 消息格式问题")
                print(f"消息数量: {len(zhipu_messages)}")
                for i, msg in enumerate(zhipu_messages):
                    print(f"  消息 {i+1}: role={msg.get('role')}, content长度={len(str(msg.get('content', '')))}")
            raise RuntimeError(f"ZhipuAI 调用失败: {e}")
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """
        流式生成回复
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Yields:
            ChatGeneration 对象
        """
        # 转换消息格式
        zhipu_messages = self._convert_messages_to_zhipu_format(messages)
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": zhipu_messages,
            "temperature": self.temperature,
            "stream": True,  # 启用流式输出
        }
        
        if stop:
            request_params["stop"] = stop
        
        request_params.update(kwargs)
        
        try:
            # 调用 ZhipuAI SDK（流式）
            response = self.zhipu_client.chat.completions.create(**request_params)
            
            # 处理流式响应
            if hasattr(response, '__iter__'):
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield ChatGeneration(
                                message=AIMessage(content=delta.content),
                                generation_info={"chunk": True}
                            )
            else:
                # 如果不是流式响应，回退到普通生成
                result = self._generate(messages, stop, run_manager, **kwargs)
                yield result.generations[0]
                
        except Exception as e:
            raise RuntimeError(f"ZhipuAI 流式调用失败: {e}")
    
    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], Dict[str, Any], Type],
        **kwargs: Any
    ) -> Any:
        """
        创建支持结构化输出的 LLM 包装器
        
        Args:
            schema: Pydantic 模型类、字典或类型
            **kwargs: 其他参数
            
        Returns:
            支持结构化输出的 LLM 包装器
        """
        from langchain_core.runnables import RunnableLambda
        
        # 创建输出解析器
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic 模型
            output_parser = PydanticOutputParser(pydantic_object=schema)
            schema_json = schema.model_json_schema()
        elif isinstance(schema, dict):
            # 字典格式的 schema
            output_parser = JsonOutputParser()
            schema_json = schema
        else:
            # 其他类型，使用 JsonOutputParser
            output_parser = JsonOutputParser()
            schema_json = {}
        
        # 创建包装器函数
        def structured_invoke(messages: List[BaseMessage]) -> Any:
            """结构化调用的包装函数"""
            # 如果消息是字符串，转换为 HumanMessage
            if isinstance(messages, str):
                messages = [HumanMessage(content=messages)]
            elif not isinstance(messages, list):
                messages = [messages]
            
            # 获取 JSON schema 字符串
            schema_str = json.dumps(schema_json, ensure_ascii=False, indent=2) if schema_json else ""
            
            # 修改最后一条消息，添加 JSON 格式要求
            modified_messages = messages.copy()
            if modified_messages:
                last_msg = modified_messages[-1]
                if isinstance(last_msg, HumanMessage):
                    # 在提示词末尾添加 JSON 格式要求
                    original_content = last_msg.content
                    json_instruction = f"""

请严格按照以下 JSON 格式返回结果，不要包含任何其他文本或说明：

{schema_str}

重要：只返回 JSON 对象，不要包含 markdown 代码块标记（如 ```json）或其他文本。"""
                    modified_messages[-1] = HumanMessage(content=original_content + json_instruction)
            
            # 调用原始 LLM
            response = self.invoke(modified_messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON
            try:
                # 尝试直接解析
                parsed = output_parser.parse(response_content)
                return parsed
            except Exception as e:
                # 如果解析失败，尝试提取 JSON
                try:
                    # 尝试提取 JSON 代码块
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
                    if json_match:
                        parsed = output_parser.parse(json_match.group(1))
                        return parsed
                    
                    # 尝试提取第一个 JSON 对象
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        parsed = output_parser.parse(json_match.group(0))
                        return parsed
                    
                    # 如果都失败了，尝试直接解析整个响应
                    parsed = json.loads(response_content)
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        return schema(**parsed)
                    return parsed
                except Exception as e2:
                    raise ValueError(f"无法解析结构化输出: {e2}. 原始响应: {response_content[:200]}")
        
        # 创建可调用对象（兼容 LangChain 接口）
        class StructuredOutputWrapper:
            """结构化输出包装器"""
            def __init__(self, invoke_func, base_llm):
                self._invoke = invoke_func
                self._base_llm = base_llm
                # 保持原始 LLM 的属性
                self.model = base_llm.model
                self.temperature = base_llm.temperature
            
            def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
                return self._invoke(messages)
            
            def __call__(self, messages: List[BaseMessage], **kwargs) -> Any:
                return self._invoke(messages)
            
            def generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
                """生成方法（兼容 LangChain）"""
                result = self._invoke(messages)
                # 将结果包装为 ChatResult
                if isinstance(result, BaseModel):
                    content = result.model_dump_json()
                elif isinstance(result, dict):
                    content = json.dumps(result, ensure_ascii=False)
                else:
                    content = str(result)
                
                generation = ChatGeneration(
                    message=AIMessage(content=content),
                    generation_info={"structured": True}
                )
                return ChatResult(generations=[generation])
        
        return StructuredOutputWrapper(structured_invoke, self)


# ===================== 便捷函数 =====================

def create_zhipu_chat_model(
    model: str = "chatglm3-6b-1001",
    temperature: float = 0.7,
    api_key: Optional[str] = None
) -> ZhipuAIAdapter:
    """
    创建 ZhipuAI 聊天模型实例（兼容 OpenAI 接口）
    
    Args:
        model: 模型名称，默认为 "chatglm3-6b-1001"
        temperature: 温度参数，默认为 0.7
        api_key: API 密钥，如果不提供则从环境变量 ZHIPU_API_KEY 读取
        
    Returns:
        ZhipuAIAdapter 实例，可以像 ChatOpenAI 一样使用
        
    Examples:
        >>> from agent.utils.zhipu_adapter import create_zhipu_chat_model
        >>> from langchain_core.messages import HumanMessage, SystemMessage
        >>> 
        >>> llm = create_zhipu_chat_model(model="chatglm3-6b-1001", temperature=0.7)
        >>> 
        >>> messages = [
        ...     SystemMessage(content="你是一个AI助手。"),
        ...     HumanMessage(content="你好，请介绍一下自己。")
        ... ]
        >>> 
        >>> response = llm.invoke(messages)
        >>> print(response.content)
    """
    return ZhipuAIAdapter(
        model=model,
        temperature=temperature,
        api_key=api_key
    )

