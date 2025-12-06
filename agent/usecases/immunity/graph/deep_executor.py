from typing import List, Dict, Any, Optional, Callable
from langchain_core.runnables.config import RunnableConfig

# deepagents核心导入
from deepagents import async_create_deep_agent
from deepagents.interrupt import ToolInterruptConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.errors import GraphInterrupt  # 添加GraphInterrupt导入
from langgraph.types import Command  # 添加Command导入
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER  # 添加CONFIG_KEY_CHECKPOINTER导入

# 项目内部导入
from common.factory import get_mcp_client, get_reasoning_model
from usecases.immunity.config.immunity_config import get_runnable_config
from langchain_core.output_parsers import JsonOutputParser
from usecases.immunity.schema.common_schemas import PlanStep, TaskExtractionResult
from usecases.immunity.state.state import ImprovedCellState

class TaskExecutor:
    """
    基于deepagents架构的任务执行器
    
    功能特性：
    - 使用create_deep_agent创建主代理
    - 为每个MCP服务器类型创建专门的SubAgent
    - 支持异步任务执行
    - 支持MCP工具的人机交互interrupt机制
    - 支持生产级的interrupt处理和恢复
    """
    
    def __init__(self, interrupt_handler: Optional[Callable] = None, sse_streamer=None):
        """
        初始化任务执行器
        
        Args:
            interrupt_handler: 可选的自定义interrupt处理函数
            sse_streamer: SSE流处理器，用于与前端通信
        """
        self.agent = None
        self.checkpointer = MemorySaver()  # 添加checkpointer支持中断和恢复
        self.sse_streamer = sse_streamer
        self.interrupt_handler = interrupt_handler or self._default_interrupt_handler

    @staticmethod
    def create_web_interrupt_handler(callback_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
        """
        创建Web界面专用的interrupt处理器
        
        Args:
            callback_func: Web界面的回调函数，接收interrupt数据，返回用户响应
            
        Returns:
            配置好的interrupt处理器
        """
        def web_interrupt_handler(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Web界面interrupt处理器
            
            Args:
                interrupt_data: interrupt数据，包含工具调用信息
                
            Returns:
                用户的响应数据
            """
            try:
                # 调用Web界面的回调函数获取用户响应
                return callback_func(interrupt_data)
            except Exception as e:
                print(f"❌ Web interrupt处理器错误: {str(e)}")
                # 默认跳过工具执行
                return {
                    "type": "response",
                    "args": f"Web界面处理错误，跳过工具执行: {str(e)}"
                }
        
        return web_interrupt_handler



    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _push_action_to_frontend(self, action_data: Dict[str, Any]):
        """通过SSE推送action信息到前端"""
        if hasattr(self, 'sse_streamer') and self.sse_streamer:
            try:
                # 通过SSE推送action信息
                # 复合键：session_id + event_name
                session_id = None
                try:
                    session_id = self.rc.get("configurable", {}).get("session_id")
                except Exception:
                    session_id = None
                event_id = str(action_data.get("timestamp") or self._get_timestamp())
                composite_event_name = f"{(session_id or 'no-session')}:{event_id}"
                self.sse_streamer.push_action_request({**action_data, "event_name": composite_event_name, "session_id": session_id})
                print(f"   - 已推送action信息到前端: {action_data['tool_name']}")
            except Exception as e:
                print(f"   - SSE推送失败: {str(e)}")
        else:
            print(f"   - 警告: 没有SSE流处理器，无法推送action信息")

    async def _wait_for_frontend_response(self, event_id: str, timeout: int = 600) -> Optional[Dict[str, Any]]:
        """等待前端响应"""
        if hasattr(self, 'sse_streamer') and self.sse_streamer:
            try:
                # 等待前端响应，设置超时时间
                session_id = None
                try:
                    session_id = self.rc.get("configurable", {}).get("session_id")
                except Exception:
                    session_id = None
                composite_event_name = f"{(session_id or 'no-session')}:{event_id}"
                response = await self.sse_streamer.wait_for_action_response(timeout=timeout, event_name=composite_event_name, session_id=session_id)
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

    async def _default_interrupt_handler(self, interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        默认的interrupt处理器 - 通过SSE推送action请求到前端
        符合deepagents的HumanResponse格式要求
        
        Args:
            interrupt_data: interrupt数据，包含工具调用信息
            
        Returns:
            HumanResponse格式的用户响应数据
        """
        print("\n" + "="*60)
        print("🔔 工具执行需要您的确认")
        print("="*60)
        
        # 提取工具信息
        action_request = interrupt_data.get('action_request', {})
        tool_name = action_request.get('action', 'Unknown')
        tool_args = action_request.get('args', {})
        tool_info = interrupt_data.get('tool_info', {})
        description = interrupt_data.get('description', '')
        
        print(f"📋 工具名称: {tool_name}")
        print(f"📝 工具参数: {tool_args}")
        if description:
            print(f"📄 描述: {description}")
        
        # 构建action信息，通过SSE推送到前端
        timestamp = self._get_timestamp()
        action_data = {
            "type": "tool_action_request",
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_info": tool_info,
            "args_schema": tool_info.get("args_schema"),
            "description": description,
            "timestamp": timestamp
        }
        
        # 通过SSE推送action信息
        event_id = str(timestamp)
        action_data["timestamp"] = event_id
        self._push_action_to_frontend(action_data)
        
        # 等待前端响应
        user_response = await self._wait_for_frontend_response(event_id, timeout=600)
        
        if user_response:
            return user_response
        else:
            # 超时或错误，默认拒绝执行
            return {
                "type": "response",
                "args": "前端响应超时，默认拒绝执行"
            }

    def _create_mcp_interrupt_config(self, tools: List) -> ToolInterruptConfig:
        """
        为所有MCP工具创建interrupt配置，支持人机交互中断
        使用deepagents的ToolInterruptConfig格式
        
        Args:
            tools: MCP工具列表
            
        Returns:
            ToolInterruptConfig: 工具中断配置字典
        """
        interrupt_config = {}
        
        for tool in tools:
            # 为每个工具创建interrupt配置 - 使用HumanInterruptConfig格式
            interrupt_config[tool.name] = {
                "allow_accept": True,    # 允许用户接受工具调用
                "allow_edit": True,      # 允许用户编辑工具参数
                "allow_respond": True,   # 允许用户提供文本回复而不执行工具
                "allow_ignore": False    # deepagents暂不支持ignore
            }
            
        print(f"✅ 为 {len(interrupt_config)} 个MCP工具创建了interrupt配置")
        return interrupt_config

    async def initialize_agent(self, config: RunnableConfig):
        """
        初始化主代理和SubAgent
        
        Args:
            config: 运行配置
        """
        try:
            # 直接获取所有MCP工具，不创建SubAgent
            all_tools = await self._get_all_mcp_tools(config)
            
            # 创建MCP工具的interrupt配置
            interrupt_config = self._create_mcp_interrupt_config(all_tools)
            
            # 获取推理模型
            try:
                # 如果config中没有模型配置，使用immunity_config的默认配置
                if not config.get("configurable", {}).get("model_config"):
                    default_config = get_runnable_config()
                    config["configurable"]["model_config"] = default_config["configurable"]["model_config"]
                
                model = get_reasoning_model(config)
                print("✅ 成功获取推理模型")
            except Exception as e:
                print(f"⚠️ 获取推理模型失败，使用默认配置: {str(e)}")
                # 使用immunity_config的默认配置
                default_config = get_runnable_config()
                enhanced_config = RunnableConfig(
                    configurable={
                        **config.get("configurable", {}),
                        "model_config": default_config["configurable"]["model_config"]
                    }
                )
                model = get_reasoning_model(enhanced_config)
            
            # 创建主代理 - 使用async_create_deep_agent支持MCP工具的异步操作
            self.agent = async_create_deep_agent(
                tools=all_tools,  # 主代理直接使用所有MCP工具
                instructions="""你是一个专业的免疫学研究助手，专门处理抗体发现、B细胞分析和免疫组学研究任务。

重要指导原则：
1. 当用户要求执行特定任务时，你必须立即使用相应的工具
2. 不要询问更多信息，直接根据任务描述选择合适的工具执行
3. 根据任务类型选择合适的工具：
   - MetaBCR工具 - 处理抗体BCR序列预测和分析
   - R Analysis工具 - 处理统计分析和数据可视化  
   - B Cell Analysis工具 - 处理B细胞分析和免疫组学
4. 如果任务明确要求使用特定工具，请直接调用该工具
5. 始终提供具体的执行结果，而不是仅仅解释或询问

请根据任务需求立即执行相应的工具调用。""",
                model=model,
                interrupt_config=interrupt_config,  # 使用正确的interrupt_config参数
                checkpointer=self.checkpointer  # 添加checkpointer支持
            )
            
            print(f"✅ 主代理初始化成功，包含 {len(all_tools)} 个MCP工具")
            print(f"   - Checkpointer: 已启用 (支持中断和恢复)")
            print(f"   - Interrupt Config: 已启用 (支持 {len(interrupt_config)} 个工具的人机交互)")
            
        except Exception as e:
            print(f"❌ 代理初始化失败: {str(e)}")
            raise
 
    async def execute_task_with_interrupt_handling(self, task_message: str, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行单个任务，支持人机交互中断处理
        使用ainvoke而非astream，正确处理GraphInterrupt异常
        
        Args:
            task_message: 任务描述
            config: 运行配置
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if not self.agent:
            await self.initialize_agent(config)
        
        print(f"📋 开始执行任务: {task_message}")
        
        try:
            # 构建任务消息
            enhanced_message = f"""请立即执行以下任务：

{task_message}

执行要求：
1. 立即分析任务需求并选择合适的工具
2. 如果任务要求使用特定工具，直接调用该工具
3. 不要询问更多信息或解释，直接执行并提供结果
4. 必须使用工具来完成任务，不能仅提供文字回答
5. 等待工具执行完成后再提供最终结果

现在立即开始执行。"""

            # 使用thread_id以支持中断和恢复
            thread_id = f"task_{hash(task_message) % 10000}"
            
            # 构建执行配置 - 确保包含thread_id和checkpointer用于中断恢复
            execution_config = RunnableConfig(
                configurable={
                    **config.get("configurable", {}),
                    "thread_id": thread_id,
                    CONFIG_KEY_CHECKPOINTER: self.checkpointer  # 使用正确的CONFIG_KEY_CHECKPOINTER
                }
            )
            
            print(f"🔧 执行配置: thread_id={thread_id}")
            
            # 准备输入数据
            input_data = {"messages": [{"role": "user", "content": enhanced_message}]}
            
            # 使用astream执行，正确处理GraphInterrupt异常
            while True:
                try:
                    print(f"🔧 开始astream调用...")
                    
                    # 使用astream执行代理以正确捕获GraphInterrupt
                    result = None
                    async for chunk in self.agent.astream(input_data, config=execution_config):
                        result = chunk
                    
                    print(f"✅ 代理执行完成，无中断发生")
                    
                    # 提取工具调用信息
                    extracted_tool_calls = self._extract_tool_calls(result)
                    
                    # 获取最终响应内容
                    final_content = ""
                    if "messages" in result and result["messages"]:
                        # 遍历所有消息，提取AI回复
                        for msg in result["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                if hasattr(msg, 'type') and msg.type == 'ai':
                                    final_content = msg.content
                            elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                                final_content = msg.get('content', '')
                    
                    # 提取执行结果
                    execution_result = {
                        "status": "success",
                        "task": task_message,
                        "result": final_content or "任务完成",
                        "tool_calls": extracted_tool_calls,
                        "interrupt_handled": False
                    }
                    
                    print(f"✅ 任务执行完成: {task_message}")
                    if extracted_tool_calls:
                        print(f"   - 成功调用 {len(extracted_tool_calls)} 个工具")
                    
                    return execution_result
                    
                except GraphInterrupt as e:
                    print(f"⚠️  捕获到GraphInterrupt异常，开始人机交互流程")
                    
                    # 提取中断数据 - GraphInterrupt的args[0]可能是tuple，需要正确提取
                    interrupt_obj = e.args[0] if e.args else None
                    if not interrupt_obj:
                        print(f"❌ 无法获取中断对象")
                        raise
                    
                    print(f"🔍 中断对象类型: {type(interrupt_obj)}")
                    print(f"🔍 中断对象: {interrupt_obj}")
                    
                    # 如果interrupt_obj是tuple，获取第一个元素
                    if isinstance(interrupt_obj, tuple) and len(interrupt_obj) > 0:
                        actual_interrupt = interrupt_obj[0]
                    else:
                        actual_interrupt = interrupt_obj
                    
                    print(f"🔍 实际中断对象类型: {type(actual_interrupt)}")
                    print(f"🔍 实际中断对象: {actual_interrupt}")
                    
                    # 从Interrupt对象获取value列表
                    interrupt_value = actual_interrupt.value if hasattr(actual_interrupt, 'value') else []
                    print(f"🔍 中断value: {interrupt_value}")
                    print(f"🔍 中断value类型: {type(interrupt_value)}")
                    print(f"🔍 中断value长度: {len(interrupt_value) if interrupt_value else 0}")
                    
                    if not interrupt_value or len(interrupt_value) == 0:
                        print(f"❌ 中断数据为空")
                        raise
                    
                    # 获取第一个中断请求数据
                    first_interrupt_data = interrupt_value[0]
                    print(f"🔍 处理中断请求数据: {first_interrupt_data}")
                    
                    # 调用interrupt_handler进行人机交互
                    print(f"🔔 调用interrupt_handler进行人机交互...")
                    user_response = await self.interrupt_handler(first_interrupt_data)
                    print(f"✅ 用户响应: {user_response}")
                    
                    # 根据用户响应构建恢复值 - 使用HumanResponse格式
                    # 根据官方文档，Command.resume必须是列表格式，即使只有一个恢复值
                    resume_value = user_response  # HumanResponse格式的用户响应
                    
                    print(f"🔧 构建恢复值: {resume_value}")
                    
                    # 使用正确的LangGraph恢复机制 - Command(resume=...)
                    # 根据官方文档，resume参数应该是单个字典，而不是列表
                    print(f"🔄 使用LangGraph Command(resume=...)机制继续执行...")
                    
                    # 构建恢复命令 - 根据官方文档的正确格式
                    # resume参数应该是单个HumanResponse格式的响应字典
                    resume_command = Command(resume=resume_value)
                    
                    print(f"🔄 构建恢复命令: {resume_command}")
                    print(f"🔄 使用Command恢复执行，继续astream流程...")
                    
                    # 恢复执行 - 使用astream保持异步一致性
                    # 官方文档中的stream示例是同步版本，我们使用异步版本需要astream
                    print(f"🔄 使用astream恢复执行...")
                    result = None
                    
                    # 恢复执行时需要继续使用原始的input_data，而不是resume_command
                    # resume_command只是告诉LangGraph如何处理中断，实际执行仍需原始输入
                    async for chunk in self.agent.astream(resume_command, config=execution_config):
                        result = chunk
                        print(f"🔄 恢复执行chunk: {chunk}")
                        
                        # 检查是否有新的工具调用需要执行
                        if "messages" in chunk:
                            for msg in chunk["messages"]:
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    print(f"🔧 检测到工具调用恢复: {[tc['name'] for tc in msg.tool_calls]}")
                    
                    print(f"✅ 恢复执行完成，无进一步中断发生")
                    
                    # 提取工具调用信息
                    extracted_tool_calls = self._extract_tool_calls(result)
                    
                    # 获取最终响应内容
                    final_content = ""
                    if "messages" in result and result["messages"]:
                        # 遍历所有消息，提取AI回复
                        for msg in result["messages"]:
                            if hasattr(msg, 'content') and msg.content:
                                if hasattr(msg, 'type') and msg.type == 'ai':
                                    final_content = msg.content
                            elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                                final_content = msg.get('content', '')
                    
                    # 提取执行结果
                    execution_result = {
                        "status": "success",
                        "task": task_message,
                        "result": final_content or "任务完成",
                        "tool_calls": extracted_tool_calls,
                        "interrupt_handled": True  # 标记已处理中断
                    }
                    
                    print(f"✅ 任务执行完成(含中断处理): {task_message}")
                    if extracted_tool_calls:
                        print(f"   - 成功调用 {len(extracted_tool_calls)} 个工具")
                    
                    return execution_result
        except Exception as e:
            import traceback
            
            # 只处理非GraphInterrupt异常，GraphInterrupt已在内层处理
            if not isinstance(e, GraphInterrupt):
                error_result = {
                    "status": "error", 
                    "task": task_message,
                    "error": str(e),
                    "tool_calls": [],
                    "interrupt_handled": False
                }
                print(f"❌ 任务执行失败: {task_message}, 错误: {str(e)}")
                print(f"   - 错误详情: {traceback.format_exc()}")
                return error_result
            else:
                # GraphInterrupt异常应该在内层被处理，如果到这里说明有问题
                print(f"⚠️  GraphInterrupt异常未被内层处理，重新抛出")
                raise

    async def execute_task(self, task_message: str, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行单个任务 - 兼容性方法，调用新的interrupt处理版本
        
        Args:
            task_message: 任务描述
            config: 运行配置
            
        Returns:
            任务执行结果
        """
        return await self.execute_task_with_interrupt_handling(task_message, config)

    async def _get_all_mcp_tools(self, config: RunnableConfig) -> List:
        """
        获取所有MCP工具，不创建SubAgent
        
        Args:
            config: 运行配置，包含MCP服务器信息
            
        Returns:
            所有工具列表
        """
        all_tools = []
        
        # MCP服务器ID列表
        service_ids = ["metabcr", "r_analysis", "bcell_analysis"]
        
        # 为每个服务ID获取工具
        for service_id in service_ids:
            try:
                # 创建该服务的MCP配置
                service_config_obj = RunnableConfig(
                    configurable={
                        **config.get("configurable", {}),
                        "mcp_config": {"service_ids": [service_id]}
                    }
                )
                
                # 获取该服务的MCP工具
                mcp_client = await get_mcp_client(service_config_obj)
                tools = await mcp_client.get_tools()
                
                if tools:
                    all_tools.extend(tools)
                    print(f"✅ 获取到 {service_id} 服务的 {len(tools)} 个工具")
                else:
                    print(f"⚠️ {service_id} 服务没有可用工具")
                    
            except Exception as e:
                print(f"❌ 获取 {service_id} 服务工具失败: {str(e)}")
                continue
        
        # 去重all_tools（避免重复工具）
        unique_tools = []
        seen_tool_names = set()
        for tool in all_tools:
            if tool.name not in seen_tool_names:
                unique_tools.append(tool)
                seen_tool_names.add(tool.name)
        
        print(f"📋 总共获取了 {len(unique_tools)} 个唯一MCP工具")
        return unique_tools
    
    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从代理执行结果中提取工具调用信息
        
        Args:
            result: 代理执行结果
            
        Returns:
            工具调用信息列表
        """
        tool_calls = []
        
        print(f"🔍 _extract_tool_calls - result类型: {type(result)}")
        print(f"🔍 _extract_tool_calls - result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # 检查是否有messages字段
        if 'messages' in result:
            print(f"🔍 _extract_tool_calls - messages数量: {len(result['messages'])}")
            for i, message in enumerate(result['messages']):
                print(f"🔍 _extract_tool_calls - message[{i}]类型: {type(message)}")
                print(f"🔍 _extract_tool_calls - message[{i}]内容: {message}")
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"🔍 _extract_tool_calls - message[{i}]有tool_calls: {len(message.tool_calls)}个")
                    for j, tool_call in enumerate(message.tool_calls):
                        print(f"🔍 _extract_tool_calls - tool_call[{j}]: {tool_call}")
                        tool_calls.append({
                            'name': tool_call['name'],
                            'args': tool_call['args'],
                            'id': tool_call.get('id', 'unknown')
                        })
                else:
                    print(f"🔍 _extract_tool_calls - message[{i}]没有tool_calls")
        else:
            print(f"🔍 _extract_tool_calls - result中没有messages字段")
        
        print(f"🔍 _extract_tool_calls - 最终提取到 {len(tool_calls)} 个工具调用")
        return tool_calls
    
    def _extract_tool_result(self, result: Dict[str, Any], tool_name: str) -> str:
        """
        从代理执行结果中提取特定工具的执行结果
        
        Args:
            result: 代理执行结果
            tool_name: 工具名称
            
        Returns:
            工具执行结果
        """
        if 'messages' in result:
            for message in result['messages']:
                if hasattr(message, 'type') and message.type == 'tool':
                    # 检查是否是对应工具的结果
                    if hasattr(message, 'name') and message.name == tool_name:
                        return message.content
                    elif hasattr(message, 'content'):
                        # 如果没有name字段，返回内容
                        return message.content
        
        return "No result found"

    def create_deep_graph(self):
        """
        创建深度执行图 - 简单的开始->execute_task_list->结束的workflow
        
        Returns:
            编译后的StateGraph
        """
        # 创建StateGraph，使用ImprovedCellState作为状态类型
        workflow = StateGraph(ImprovedCellState)
        
        # 添加节点
        workflow.add_node("execute_task_list", execute_task_list)
        
        # 设置入口点和边
        workflow.set_entry_point("execute_task_list")
        workflow.add_edge("execute_task_list", END)
        
        # 编译图
        graph = workflow.compile()
        
        # 打印工作流程图
        try:
            print("\n===== Deep Executor Workflow Diagram =====")
            print(graph.get_graph().draw_mermaid())
        except Exception as e:
            print(f"生成工作流程图时出错: {str(e)}")
        
        return graph

    async def run_deep_graph(self, decomposed_tasks: List[str], config: RunnableConfig):
        """
        运行深度执行图
        
        Args:
            decomposed_tasks: 分解后的任务列表
            config: 运行配置
            
        Returns:
            执行结果
        """
        # 创建图
        graph = self.create_deep_graph()
        
        # 创建初始状态
        initial_state = ImprovedCellState(
            original_question="执行分解后的任务列表",  # 添加必需的字段
            decomposed_tasks=decomposed_tasks
        )
        
        # 使用异步流执行图
        final_state = None
        async for event in graph.astream(initial_state, config):
            print(f"当前节点: {list(event.keys())}")
            final_state = event
        
        # 提取执行结果
        if final_state and "execute_task_list" in final_state:
            # execute_task_list现在返回字典格式的状态更新
            task_results = final_state["execute_task_list"].get("task_results", [])
        else:
            print("警告: execute_task_list结果未找到")
            task_results = []
        
        print(f"\n深度执行工作流完成")
        print(f"任务数量: {len(decomposed_tasks)}")
        print(f"执行结果数量: {len(task_results)}")
        
        return {
            "task_results": task_results,
            "total_tasks": len(decomposed_tasks),
            "completed_tasks": len([r for r in task_results if r.get('status') == 'success']),
            "failed_tasks": len([r for r in task_results if r.get('status') == 'error'])
        }

    async def complete_deep_pipeline(self, decomposed_tasks: List[str], config: RunnableConfig):
        """
        完整的深度执行管道 - 作为主要的执行入口
        
        Args:
            decomposed_tasks: 分解后的任务列表
            config: 运行配置
            
        Returns:
            完整的执行结果
        """
        print("=== 深度执行管道启动 ===")
        
        # 初始化代理（如果尚未初始化）
        if not self.agent:
            await self.initialize_agent(config)
        
        # 运行深度执行图
        result = await self.run_deep_graph(decomposed_tasks, config)
        
        # 添加执行统计信息
        success_rate = (result["completed_tasks"] / result["total_tasks"] * 100) if result["total_tasks"] > 0 else 0
        
        final_result = {
            **result,
            "success_rate": success_rate,
            "execution_summary": {
                "total_tasks": result["total_tasks"],
                "completed_tasks": result["completed_tasks"],
                "failed_tasks": result["failed_tasks"],
                "success_rate": f"{success_rate:.1f}%"
            }
        }
        
        print(f"\n✅ 深度执行管道完成")
        print(f"📊 执行统计: {final_result['execution_summary']}")
        
        return final_result


async def task_decomposition_node(state: ImprovedCellState, config: RunnableConfig)  -> ImprovedCellState:
    """任务分解节点 - 兼容性函数"""
    """
    Task decomposition node - Decompose refine_plan into specific executable tasks
    
    This node receives the refine_plan from planning_graph.py and decomposes it into
    specific, executable task step lists.
    
    Args:
        state: Cell module state object containing refine_plan
        config: Runtime configuration
        
    Returns:
        ExecuteState: Updated cell state containing decomposed task list
    """
    print("[task_decomposition_node] Starting task decomposition node")
    
    try:
        # Get refine_plan
        plan = state.final_enhanced_plan
        
        if not plan or plan.strip() == "":
            print("[task_decomposition_node] No plan available for decomposition")
            return state
        from usecases.immunity.prompts.prompts import ImmunityPrompts
        from usecases.immunity.common.constants import get_tools_json
        
        # Get tools registry information
        tools_info = get_tools_json()
        
        # Create task extraction chain
        model = get_reasoning_model(config)
        output_parser = JsonOutputParser(pydantic_object=TaskExtractionResult)
        
        # Execute task extraction
        decomposed_tasks = (model | output_parser).invoke(
            ImmunityPrompts.TASK_EXTRACTION_PROMPT.format(
                plan=plan,
                tools_info=tools_info
            )
        )
         
        # JsonOutputParser returns a dictionary, not a TaskExtractionResult instance
        # Need to get the tasks list from the dictionary
        structured_tasks: List[Dict[str, Any]] = []
        if isinstance(decomposed_tasks, dict) and "tasks" in decomposed_tasks:
            tasks_list = decomposed_tasks["tasks"]
            print(f"[task_decomposition_node] Task decomposition completed, extracted {len(tasks_list)} tasks")
            
            task_descriptions = []
            for task_dict in tasks_list:
                if isinstance(task_dict, dict) and "description" in task_dict:
                    description = task_dict["description"]
                    if description and description.strip():
                        task_descriptions.append(description.strip())
                if isinstance(task_dict, dict):
                    structured_tasks.append(task_dict)
        else:
            print(f"[task_decomposition_node] Error: Unable to get tasks list from decomposition result")
            task_descriptions = []
            structured_tasks = []
        
        state.decomposed_tasks = task_descriptions
        def _normalize_tool_list(value: Any) -> List[str]:
            if not value:
                return []
            items = value if isinstance(value, list) else [value]
            normalized = []
            for item in items:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    name = (
                        item.get("tool_name")
                        or item.get("name")
                        or item.get("id")
                        or item.get("label")
                    )
                    if name:
                        normalized.append(str(name))
                else:
                    normalized.append(str(item))
            return normalized

        plan_steps: List[PlanStep] = []
        for idx, task_entry in enumerate(structured_tasks, 1):
            try:
                step = PlanStep(
                    step_id=str(task_entry.get("task_id") or idx),
                    title=task_entry.get("name") or task_entry.get("title") or f"Step {idx}",
                    description=task_entry.get("description", ""),
                    objective=task_entry.get("objective", ""),
                    tools=_normalize_tool_list(task_entry.get("tools")),
                    toolchain=_normalize_tool_list(task_entry.get("toolchain")),
                    recommended_tools=_normalize_tool_list(task_entry.get("recommended_tools")),
                    notes=task_entry.get("notes", ""),
                    inputs=task_entry.get("inputs", []) or [],
                    outputs=task_entry.get("outputs", []) or [],
                    metadata={
                        "raw_task": task_entry,
                    },
                    suggested_alternatives=task_entry.get("suggested_alternatives", []) or [],
                )
                plan_steps.append(step)
            except Exception as e:
                print(f"[task_decomposition_node] Warning: failed to parse plan step {idx}: {e}")
        state.plan_step_details = plan_steps
        print(f"[task_decomposition_node] Successfully extracted {len(task_descriptions)} task descriptions")
        
        return state
        
    except Exception as e:
        import traceback
        print(f"[task_decomposition_node] Task decomposition failed: {e}")
        print(f"[task_decomposition_node] Error type: {type(e).__name__}")
        print(f"[task_decomposition_node] Detailed stack trace:")
        print(traceback.format_exc())
        return state

async def execute_task_list(state: ImprovedCellState, config: RunnableConfig) -> Dict[str, Any]:
    """执行任务列表 - 兼容性函数"""
    # 从config中获取sse_streamer（如果存在）
    sse_streamer = None
    plan_steps_payload = []
    plan_id = None
    if config and "configurable" in config:
        configurable = config["configurable"]
        sse_streamer = configurable.get("sse_streamer")
        plan_steps_payload = configurable.get("plan_steps") or []
        plan_id = configurable.get("plan_id")
    
    executor = TaskExecutor(sse_streamer=sse_streamer)
    normalized_steps: List[Dict[str, Any]] = []
    for idx, step_payload in enumerate(plan_steps_payload, 1):
        if isinstance(step_payload, PlanStep):
            normalized_steps.append(step_payload.model_dump())
        elif isinstance(step_payload, dict):
            normalized_steps.append(step_payload)
    executor.plan_steps = normalized_steps
    executor.plan_id = plan_id
    await executor.initialize_agent(config)
    tasks = state.decomposed_tasks
    print(f"📋 开始批量执行 {len(tasks)} 个任务")
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"🔄 执行任务 {i}/{len(tasks)}: {task}")
        # 使用支持中断处理的方法
        result = await executor.execute_task_with_interrupt_handling(task, config)
        results.append(result)
    
    print(f"✅ 批量任务执行完成，成功: {sum(1 for r in results if r['status'] == 'success')}, 失败: {sum(1 for r in results if r['status'] == 'error')}")
    
    # 返回字典格式的状态更新，LangGraph会自动合并到状态中
    return {"task_results": results}