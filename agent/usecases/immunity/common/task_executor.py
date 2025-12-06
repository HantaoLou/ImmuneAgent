from typing import List, Dict, Any, Optional
import json
import time
from huggingface_hub import reject_access_request
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Interrupt
from common.factory import get_mcp_client, get_reasoning_model
from usecases.immunity.config.immunity_config import get_runnable_config
from usecases.immunity.common.hil_wrapper import add_human_in_the_loop
from usecases.execute.graph.generic_executor import get_all_tools
from usecases.immunity.schema.common_schemas import TaskInfo
from usecases.immunity.prompts.prompts import ImmunityPrompts
from usecases.immunity.common.utils import download_geo_dataset

class TaskExecutor:
    """
    基于deepagents架构的任务执行器
    
    功能特性：
    - 使用create_deep_agent创建主代理
    - 使用HIL装饰器包装MCP工具，支持人机交互
    - 支持异步任务执行
    """
    
    def __init__(self, use_deep_agent: bool = False, checkpointer=None, sse_streamer=None):
        """初始化任务执行器
        
        Args:
            use_deep_agent: 是否使用create_deep_agent而不是create_react_agent，默认为False
            checkpointer: 外部传入的checkpointer，如果为None则创建新的MemorySaver
            sse_streamer: SSE流处理器，用于与前端通信
        """
        self.agent = None
        self.tools = []
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.use_deep_agent = use_deep_agent
        self.sse_streamer = sse_streamer
        self.pending_action_response = None
        self.action_response_event = None
        self._geo_download_cache: set[str] = set()
        self.geo_download_registry: Dict[str, Dict[str, Any]] = {}
        self._tool_call_sequence: int = 0
    
    def _get_model_with_config(self, config: RunnableConfig) -> Any:
        """
        统一的模型获取方法，优化配置处理逻辑
        
        Args:
            config: 运行配置
            
        Returns:
            配置好的推理模型
            
        Raises:
            Exception: 当模型获取失败时抛出异常
        """
        try:
            # 检查传入配置是否包含模型配置
            if config.get("configurable", {}).get("model_config"):
                # 如果有模型配置，直接使用
                model = get_reasoning_model(config)
                return model
            else:
                # 如果没有模型配置，使用默认配置
                default_config = get_runnable_config()
                model = get_reasoning_model(default_config)
                return model
                
        except Exception as e:
            # 后备方案：强制使用默认配置
            try:
                default_config = get_runnable_config()
                model = get_reasoning_model(default_config)
                return model
            except Exception as fallback_error:
                raise Exception(f"模型获取完全失败: 主要错误={str(e)}, 后备错误={str(fallback_error)}")

    async def initialize_agent(self, config: RunnableConfig):
        """
        初始化主代理和工具
        
        Args:
            config: 运行配置
        """
        try:
            # 使用统一的模型获取方法
            model = self._get_model_with_config(config)

            # 获取所有MCP工具
            self.tools = await get_all_tools(config)
            
            # 使用HIL装饰器包装工具，支持人机交互
            # from usecases.immunity.common.hil_wrapper import hil
            from usecases.deepagents.tools import hil
            hil_tools = [add_human_in_the_loop(tool) for tool in self.tools] if self.tools else []
            # hil_tools = [hil(tool) for tool in self.tools] if self.tools else []
            
            # 根据use_deep_agent参数选择创建函数
            if self.use_deep_agent:
                from deepagents import create_deep_agent
                
                self.agent = create_deep_agent(
                    model=model,
                    tools=hil_tools,
                    checkpointer=self.checkpointer,
                    # 根据deepagents文档，使用system_prompt参数传递自定义指令
                    system_prompt="你是一个专业的免疫学研究助手。当需要使用工具时，请直接调用相应的工具，不要询问更多信息。遇到工具调用中断时，等待用户确认后继续执行。如果工具执行失败，请提供清晰的错误信息而不是抛出异常。始终尝试完成用户的任务，即使遇到部分失败也要提供有用的结果。"
                )
            else:
                from langgraph.prebuilt import create_react_agent
                self.agent = create_react_agent(
                    model=model,
                    tools=hil_tools,
                    checkpointer=self.checkpointer
                )
        except Exception as e:
            print(f"- 代理初始化失败: {str(e)}")
            raise
    
    async def execute_task(self, task: TaskInfo, original_planning, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行单个任务
        
        Args:
            task: 任务对象，包含task_id、name、description、tools等完整信息
            config: 运行配置
            
        Returns:
            任务执行结果
        """
        if not self.agent:
            await self.initialize_agent(config)
        
        # 确保configurable存在
        if "configurable" not in config:
            config["configurable"] = {}
            
        # 确保有thread_id用于checkpointer
        if "thread_id" not in config["configurable"]:
            import uuid
            config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # 构建输入数据
        from langchain_core.messages import HumanMessage
        
        # 获取完整工具集合信息
        from usecases.immunity.common.constants import get_tools_json
        tools_info = get_tools_json()
        
        # 构建推荐工具信息
        import json
        recommended_tools = []
        if task.tools:
            for tool in task.tools:
                recommended_tools.append({
                    "tool_name": tool.tool_name,
                    "description": tool.description
                })
        
        recommended_tools_json = json.dumps(recommended_tools, ensure_ascii=False, indent=2) if recommended_tools else "无推荐工具"
        
        # Build task description - Direct execution mode
        task_description = ImmunityPrompts.TASK_EXECUTION_PROMPT.format(
            original_planning=original_planning,
            task_description=task.description,
            tools_info=tools_info,
            recommended_tools_json=recommended_tools_json
        )
        
        human_message = HumanMessage(content=task_description)
        input_data = {"messages": [human_message]}
        
        # 使用传入的config，确保thread_id一致（中断恢复需要相同的thread_id）
        # 如果没有thread_id，使用task_id创建固定的thread_id
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = f"task_{task.task_id if hasattr(task, 'task_id') else task.id if hasattr(task, 'id') else 'default'}"
        
        # 使用config中的thread_id构建thread_config
        thread_config = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
        
        # checkpointer已在agent创建时配置，不需要在thread_config中重复配置
        
        try:
            # 使用同步invoke执行，与正常工作的代码保持一致
            print(f"   - 使用thread_id: {thread_config['configurable']['thread_id']} 执行任务")
            result = self.agent.invoke(input_data, thread_config)
                    
            # 使用while循环处理所有中断事件
            # 修复：添加None检查，防止result为None时出现TypeError
            while result is not None and isinstance(result, dict) and "__interrupt__" in result:
                interrupt_info = result["__interrupt__"][0]  # 修复：取第一个中断信息

                # 处理中断并获取用户响应
                user_response = await self._handle_interrupt(interrupt_info)
                
                # 检查用户响应
                if not user_response.get("accept", False):
                    # 用户拒绝执行
                    print(f"   - 用户拒绝执行工具")
                    result = {
                        "status": "rejected",
                        "message": "用户拒绝执行",
                        "result": None,
                        "messages": result.get("messages", []) if result else []
                    }
                    break
                
                # 用户接受执行，恢复LangGraph流程
                # 使用Command(resume=...)恢复执行，resume参数应该是字典，不是JSON字符串
                # 重要：必须使用相同的thread_config，否则无法找到之前的中断状态
                print(f"   - 准备恢复执行，user_response: {user_response}")
                print(f"   - 使用相同的thread_id恢复: {thread_config['configurable']['thread_id']}")
                print(f"   - 调用Command(resume=...)恢复执行...")
                
                try:
                    # 直接传递字典，LangGraph会自动处理序列化
                    result = self.agent.invoke(Command(resume=user_response), thread_config)
                    print(f"   - ✅ Command(resume=...)调用成功")
                    print(f"   - 恢复后的result类型: {type(result)}")
                    if isinstance(result, dict):
                        print(f"   - 恢复后的result键: {list(result.keys())}")
                        print(f"   - 恢复后的result是否有__interrupt__: {'__interrupt__' in result}")
                        # 如果有messages，打印最后一条消息
                        if "messages" in result and result["messages"]:
                            last_msg = result["messages"][-1]
                            print(f"   - 最后一条消息: {type(last_msg).__name__}")
                    # 恢复执行后，while循环会检查是否还有__interrupt__，如果没有则退出循环继续处理
                except Exception as resume_error:
                    print(f"   - ❌ Command(resume=...)调用失败: {str(resume_error)}")
                    import traceback
                    traceback.print_exc()
                    # 重新抛出异常，让上层处理
                    raise
            
            # 检查是否为跳过结果
            if isinstance(result, dict) and result.get("status") == "skipped":
                print(f"   - 任务被用户跳过: {result.get('message', '')}")
                # 直接返回跳过结果，不需要进一步处理
                return {
                    "status": "skipped",
                    "task": task,
                    "result": result.get("message", "用户跳过当前任务"),
                    "tool_calls": [],
                    "tool_results": []
                }
            
            print(f"   - 代理执行完成，分析结果...")
            
            # 处理执行结果
            execution_result = self._process_execution_result(result, task)
                
            return execution_result
            
        except Exception as e:
            # 捕获所有异常，包括deepagents可能抛出的异常
            print(f"   - 任务执行过程中发生异常: {str(e)}")
            print(f"   - 异常类型: {type(e).__name__}")
            

    async def _handle_interrupt(self, interrupt_info) -> Dict[str, Any]:
        """
        处理中断，通过SSE推送action信息到前端
        
        根据LangGraph最佳实践，处理interrupt函数返回的中断信息
        
        Args:
            interrupt_info: 中断信息，可能是单个对象或列表
            
        Returns:
            用户响应数据
        """
        try:
            # 处理中断信息，兼容不同格式
            if hasattr(interrupt_info, 'value'):
                # 如果是带有value属性的对象（如graph版本）
                interrupt_value = interrupt_info.value
            elif isinstance(interrupt_info, list) and len(interrupt_info) > 0:
                # 如果是列表，取第一个元素
                interrupt_data = interrupt_info[0]
                if hasattr(interrupt_data, 'value'):
                    interrupt_value = interrupt_data.value
                else:
                    interrupt_value = interrupt_data.get("value", interrupt_data)
            else:
                # 直接使用interrupt_info
                interrupt_value = interrupt_info
            
            # 如果是工具调用中断
            if isinstance(interrupt_value, dict):
                action_request = interrupt_value.get("action_request", {})
                tool_info = interrupt_value.get("tool_info", {})
                
                if action_request:
                    tool_name = action_request.get("action", "未知工具")
                    tool_args = action_request.get("args", {})
                    
                    # 构建action信息，通过SSE推送到前端
                    action_data = {
                        "type": "tool_action_request",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_info": tool_info,
                        "args_schema": tool_info.get("args_schema"),
                        "description": tool_info.get("description", ""),
                        "timestamp": self._get_timestamp()
                    }
                    
                    # 通过SSE推送action信息
                    action_event_id = str(action_data.get("timestamp") or self._get_timestamp())
                    action_data["timestamp"] = action_event_id
                    self._push_action_to_frontend(action_data)
                    
                    # 等待前端响应（需要await，因为这是异步方法）
                    user_response = await self._wait_for_frontend_response(action_event_id, timeout=600)
                    
                    if user_response:
                        # 将前端响应转换为LangGraph期望的格式
                        return self._convert_frontend_response_to_langgraph_format(user_response, tool_args)
                    else:
                        # 超时或错误，默认拒绝执行
                        return {"accept": False}
            
            # 默认拒绝执行
            return {"accept": False}
            
        except Exception as e:
            print(f"   - 处理中断时发生错误: {str(e)}")
            # 默认拒绝执行
            return {
                "type": "response", 
                "args": f"中断处理错误: {str(e)}"
            }



    def _process_execution_result(self, result: Dict[str, Any], task: TaskInfo) -> Dict[str, Any]:
        """
        处理代理执行结果，提取消息内容和工具调用信息
        
        Args:
            result: 代理执行结果
            task: 任务对象，包含完整的任务信息
            
        Returns:
            处理后的执行结果
        """
        # 修复：添加None检查，防止result为None时出现AttributeError
        if result is None:
            print(f"   - 警告: 执行结果为None，返回默认结果")
            return {
                "status": "error",
                "message": "执行结果为空",
                "result": None,
                "messages": []
            }
        
        # 从结果中提取消息和工具调用信息
        all_messages = result.get("messages", [])
        tool_calls = self._extract_tool_calls(result)
        print(f"   - 消息数量: {len(all_messages)}")
        print(f"   - 工具调用数量: {len(tool_calls)}")
        
        # 获取最终响应内容和工具结果
        ai_contents = []  # 收集所有AI消息内容
        tool_results = []
 
        if all_messages:
            # 遍历所有消息，提取AI回复和工具结果
            for msg in all_messages:
                if hasattr(msg, 'content') and msg.content:
                    # 使用isinstance进行严格的类型检查
                    if isinstance(msg, AIMessage):
                        ai_contents.append(msg.content)
                        # 检查无效工具调用
                        if hasattr(msg, 'invalid_tool_calls') and msg.invalid_tool_calls:
                            for invalid_call in msg.invalid_tool_calls:
                                print(f"        ⚠️ 无效工具调用: {invalid_call.get('name', 'unknown')} - {invalid_call.get('error', 'unknown error')}")
                    elif isinstance(msg, ToolMessage):
                        # 提取工具执行结果
                        tool_results.append({
                            'content': msg.content,
                            'tool_call_id': getattr(msg, 'tool_call_id', 'unknown'),
                            'call_id': getattr(msg, 'tool_call_id', 'unknown'),
                            'status': 'completed',
                            'completed_at': time.time(),
                        })
            
            # 合并AI内容，优先使用最后一个非空内容
            final_content = ai_contents[-1] if ai_contents else ""

            # Align tool names with results for consistent display
            tool_name_map = {
                call.get('call_id') or call.get('id'): call.get('tool_name') or call.get('name')
                for call in tool_calls
            }
            for result_item in tool_results:
                call_id = result_item.get('call_id') or result_item.get('tool_call_id')
                if call_id in tool_name_map:
                    result_item.setdefault('tool_name', tool_name_map[call_id])
 
            # 显示工具调用详情和结果
            self._display_tool_call_details(tool_calls, tool_results)
        else:
            final_content = ""
        
        # 构建执行结果
        execution_result = {
            "status": "success",
            "task": task,
            "result": final_content or "任务完成",
            "tool_calls": tool_calls,
            "tool_results": tool_results
        }

        if tool_calls:
            self._handle_geo_downloads(tool_calls, tool_results)
            
        return execution_result

    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从代理执行结果中提取工具调用信息
        
        Args:
            result: 代理执行结果
            
        Returns:
            工具调用信息列表
        """
        tool_calls = []
        
        # 检查是否有messages字段
        if 'messages' in result:
            for message in result['messages']:
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        call_id = tool_call.get('id') or tool_call.get('call_id') or f"tool_call_{len(tool_calls) + 1}"
                        tool_calls.append({
                            'name': tool_call['name'],
                            'tool_name': tool_call['name'],
                            'args': tool_call['args'],
                            'id': call_id,
                            'call_id': call_id,
                            'status': 'completed',
                            'timestamp': time.time(),
                        })
        
        return tool_calls

    def _display_tool_call_details(self, tool_calls: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> None:
        """
        显示工具调用详情和结果
        
        Args:
            tool_calls: 工具调用列表
            tool_results: 工具结果列表
        """
        if tool_calls:
            print(f"   - 工具调用详情:")
            for i, tool_call in enumerate(tool_calls, 1):
                print(f"     {i}. {tool_call['name']} (ID: {tool_call['id']})")
                
                # 查找对应的工具结果
                matching_result = None
                for result_item in tool_results:
                    if result_item['tool_call_id'] == tool_call['id']:
                        matching_result = result_item
                        break
                
                if matching_result:
                    # 显示工具结果的前200个字符
                    result_content = matching_result['content']
                    if len(result_content) > 500:
                        result_preview = result_content[:500] + "..."
                    else:
                        result_preview = result_content
                    print(f"        结果: {result_preview}")
                else:
                    print(f"        结果: 未找到对应结果")

    def _format_tool_parameter_info(self, args_schema) -> str:
        """
        格式化工具参数信息，用于在中断时展示
        
        Args:
            args_schema: 工具的参数schema（Pydantic模型）
            
        Returns:
            格式化的参数信息字符串
        """
        if not args_schema:
            return "    无参数"
        
        try:
            # 修复：兼容不同版本的Pydantic，处理args_schema可能是dict的情况
            if isinstance(args_schema, dict):
                # 如果args_schema是字典，直接处理
                if not args_schema:
                    return "    无参数"
                
                param_info = []
                for field_name, field_data in args_schema.items():
                    if isinstance(field_data, dict):
                        field_type = field_data.get('type', 'Any')
                        description = field_data.get('description', '')
                        required = field_data.get('required', True)
                        
                        required_text = "必需" if required else "可选"
                        desc_text = f" - {description}" if description else ""
                        param_info.append(f"    • {field_name} ({field_type}, {required_text}){desc_text}")
                    else:
                        param_info.append(f"    • {field_name}: {field_data}")
                
                return "\n".join(param_info) if param_info else "    无参数"
            
            # 处理Pydantic模型
            if hasattr(args_schema, 'model_fields'):
                model_fields = args_schema.model_fields
            elif hasattr(args_schema, '__fields__'):
                # 兼容Pydantic v1
                model_fields = args_schema.__fields__
            else:
                return f"    不支持的schema类型: {type(args_schema)}"
            
            if not model_fields:
                return "    无参数"
            
            param_info = []
            for field_name, field_info in model_fields.items():
                # 获取字段类型
                field_type = getattr(field_info, 'annotation', 'Any')
                if hasattr(field_type, '__name__'):
                    type_name = field_type.__name__
                else:
                    type_name = str(field_type)
                
                # 检查是否必需
                is_required = field_info.is_required() if hasattr(field_info, 'is_required') else True
                required_text = "必需" if is_required else "可选"
                
                # 获取默认值
                default_value = getattr(field_info, 'default', None)
                default_text = f", 默认值: {default_value}" if default_value is not None else ""
                
                # 获取描述
                description = getattr(field_info, 'description', None)
                desc_text = f" - {description}" if description else ""
                
                param_info.append(f"    • {field_name} ({type_name}, {required_text}{default_text}){desc_text}")
            
            return "\n".join(param_info)
        except Exception as e:
            return f"    参数信息解析失败: {str(e)}"

    async def _edit_tool_parameters(self, tool_args: dict, args_schema: dict) -> dict:
        """
        交互式编辑工具参数 - 通过SSE与前端交互
        
        Args:
            tool_args: 原始工具参数
            args_schema: 参数模式定义
            
        Returns:
            dict: 编辑后的参数
        """
        # 如果有SSE流处理器，通过前端UI编辑参数
        if self.sse_streamer:
            return await self._edit_parameters_via_sse(tool_args, args_schema)
        
        # 否则使用控制台编辑（保留原有逻辑作为fallback）
        return self._edit_parameters_via_console(tool_args, args_schema)

    async def _edit_parameters_via_sse(self, tool_args: dict, args_schema: dict) -> dict:
        """
        通过SSE与前端交互编辑参数
        
        Args:
            tool_args: 原始工具参数
            args_schema: 参数模式定义
            
        Returns:
            dict: 编辑后的参数
        """
        try:
            # 构建参数编辑请求
            edit_request = {
                "type": "parameter_edit_request",
                "tool_args": tool_args,
                "args_schema": args_schema,
                "timestamp": self._get_timestamp()
            }
            
            # 通过SSE推送编辑请求（使用plain事件ID）
            session_id = None
            try:
                session_id = self.rc.get("configurable", {}).get("session_id")
            except Exception:
                session_id = None
            event_id = str(edit_request.get("timestamp"))
            self.sse_streamer.push_action_request({**edit_request, "event_name": event_id, "session_id": session_id})
            print(f"   - 已推送参数编辑请求到前端")
            
            # 等待前端响应
            user_response = await self.sse_streamer.wait_for_action_response(timeout=300, event_name=event_id, session_id=session_id)
            
            if user_response and user_response.get("type") == "edit":
                # 获取编辑后的参数
                edit_args = user_response.get("args", {})
                if isinstance(edit_args, dict) and "args" in edit_args:
                    modified_args = edit_args["args"]
                    print(f"   - 收到前端编辑的参数: {modified_args}")
                    return modified_args
                else:
                    print(f"   - 前端编辑参数格式错误，使用原参数")
                    return tool_args
            else:
                print(f"   - 前端编辑超时或取消，使用原参数")
                return tool_args
                
        except Exception as e:
            print(f"   - SSE参数编辑错误: {str(e)}")
            return tool_args

    def _edit_parameters_via_console(self, tool_args: dict, args_schema: dict) -> dict:
        """
        通过控制台交互编辑参数（fallback方法）
        
        Args:
            tool_args: 原始工具参数
            args_schema: 参数模式定义
            
        Returns:
            dict: 编辑后的参数
        """
        edited_args = tool_args.copy()
        
        print(f"当前参数：")
        for key, value in tool_args.items():
            print(f"  {key}: {value}")
        
        print("\n请选择要编辑的参数（输入参数名，或输入 'done' 完成编辑）：")
        
        while True:
            try:
                param_name = input("参数名 > ").strip()
                
                if param_name.lower() in ['done', '完成', 'finish']:
                    break
                    
                if param_name not in tool_args:
                    print(f"参数 '{param_name}' 不存在。可用参数：{list(tool_args.keys())}")
                    continue
                
                current_value = edited_args[param_name]
                print(f"\n当前值：{current_value}")
                
                # 获取参数类型信息
                param_info = args_schema.get('properties', {}).get(param_name, {})
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', '无描述')
                
                print(f"参数类型：{param_type}")
                print(f"参数描述：{param_desc}")
                
                new_value = input(f"新值（留空保持不变）> ").strip()
                
                if new_value:
                    try:
                        # 根据参数类型转换值
                        if param_type == 'integer':
                            edited_args[param_name] = int(new_value)
                        elif param_type == 'number':
                            edited_args[param_name] = float(new_value)
                        elif param_type == 'boolean':
                            edited_args[param_name] = new_value.lower() in ['true', '1', 'yes', '是']
                        elif param_type == 'array':
                            # 简单的数组处理，用逗号分隔
                            edited_args[param_name] = [item.strip() for item in new_value.split(',')]
                        else:
                            # 默认为字符串
                            edited_args[param_name] = new_value
                        
                        print(f"✓ 参数 '{param_name}' 已更新为：{edited_args[param_name]}")
                        
                    except ValueError as e:
                        print(f"✗ 值转换失败：{e}")
                        print("保持原值不变")
                else:
                    print("保持原值不变")
                    
            except KeyboardInterrupt:
                print("\n\n用户取消编辑")
                break
            except Exception as e:
                print(f"❌ 编辑参数时发生错误: {str(e)}")
                continue
        
        print("\n=== 编辑完成 ===")
        print("最终参数：")
        for key, value in edited_args.items():
            if edited_args[key] != tool_args[key]:
                print(f"  {key}: {value} ← 已修改")
            else:
                print(f"  {key}: {value}")
        
        return edited_args

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _push_action_to_frontend(self, action_data: Dict[str, Any]):
        """通过SSE推送action信息到前端"""
        if self.sse_streamer:
            try:
                # 通过SSE推送action信息（仅使用plain事件ID，避免键不一致）
                session_id = None
                try:
                    session_id = self.rc.get("configurable", {}).get("session_id")
                except Exception:
                    session_id = None
                event_id = str(action_data.get("timestamp") or self._get_timestamp())
                self.sse_streamer.push_action_request({**action_data, "event_name": event_id, "session_id": session_id})
                print(f"   - 已推送action信息到前端: {action_data['tool_name']}")
            except Exception as e:
                print(f"   - SSE推送失败: {str(e)}")
        else:
            print(f"   - 警告: 没有SSE流处理器，无法推送action信息")

    async def _wait_for_frontend_response(self, event_id: str, timeout: int = 600) -> Optional[Dict[str, Any]]:
        """等待前端响应"""
        if hasattr(self, 'sse_streamer') and self.sse_streamer:
            try:
                # 等待前端响应，设置超时时间（使用plain事件ID）
                session_id = None
                try:
                    session_id = self.rc.get("configurable", {}).get("session_id")
                except Exception:
                    session_id = None
                response = await self.sse_streamer.wait_for_action_response(timeout=timeout, event_name=event_id, session_id=session_id)
                if response:
                    print(f"   - 收到前端响应: {response.get('type', 'unknown')}")
                    return response
                else:
                    print(f"   - 前端响应超时")
                    return None
            except Exception as e:
                print(f"   - 等待前端响应时发生错误: {str(e)}")
                return None
        else:
            print(f"   - 警告: 没有SSE流处理器，使用默认拒绝")
            return {
                "type": "response",
                "args": "没有SSE流处理器，默认拒绝执行"
            }

    def _convert_frontend_response_to_langgraph_format(self, frontend_response: Dict[str, Any], original_tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        将前端响应转换为LangGraph Command(resume=...)期望的格式
        
        Args:
            frontend_response: 前端发送的响应，格式如 {"type": "accept", "args": {...}}
            original_tool_args: 原始工具参数
            
        Returns:
            LangGraph格式的响应，格式如 {"accept": True, "args": {...}}
        """
        response_type = frontend_response.get("type", "reject")
        
        if response_type == "accept":
            # 用户接受执行
            return {"accept": True}
        elif response_type == "edit":
            # 用户编辑了参数
            edit_args = frontend_response.get("args", {})
            if isinstance(edit_args, dict) and "args" in edit_args:
                modified_args = edit_args["args"]
                print(f"   - 使用编辑后的参数: {modified_args}")
                return {"accept": True, "args": modified_args}
            else:
                # 参数格式错误，使用原始参数
                print(f"   - 编辑参数格式错误，使用原始参数")
                return {"accept": True, "args": original_tool_args}
        else:
            # reject或其他类型，拒绝执行
            return {"accept": False}

    def set_action_response(self, response: Dict[str, Any]):
        """设置action响应（由SSE流处理器调用）"""
        self.pending_action_response = response
        if self.action_response_event:
            self.action_response_event.set()

    def _handle_geo_downloads(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """Automatically download GEO datasets when download_geo_sequences tool is used."""
        for call in tool_calls:
            if call.get("name") != "download_geo_sequences":
                continue
            matching_result = next(
                (item for item in tool_results if item.get("tool_call_id") == call.get("id")),
                None,
            )
            if not matching_result:
                call.setdefault("status", "completed")
                continue
            content = matching_result.get("content")
            if not content:
                call.setdefault("status", "completed")
                continue
            try:
                download_info = download_geo_dataset(
                    content,
                    cache=self._geo_download_cache,
                )
            except Exception as geo_error:  # noqa: BLE001
                print(f"[CommonTaskExecutor] GEO auto-download failed: {geo_error}")
                download_info = None
            if not download_info:
                call.setdefault("status", "completed")
                continue
            geo_key = (
                download_info.get("geo_id")
                or download_info.get("ftp_url")
                or f"geo_{len(self.geo_download_registry) + 1}"
            )
            self.geo_download_registry[geo_key] = download_info
            matching_result["download_info"] = download_info
            matching_result.setdefault("status", "completed")
            call.setdefault("status", "completed")
            print(
                f"[CommonTaskExecutor] GEO dataset downloaded to {download_info.get('destination')}"
            )

