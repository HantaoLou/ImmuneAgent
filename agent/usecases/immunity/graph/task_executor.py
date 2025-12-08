"""
Task Executor

Core Functions:
1. Execute task lists
2. Automatic tool matching
3. Human-in-the-loop confirmation
4. Parameter modification and passing
"""

import asyncio
import json
import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from common.factory import get_reasoning_model
from usecases.deepagents.tools import hil
from usecases.execute.graph.generic_executor import get_all_tools
from usecases.immunity.common.constants import get_intelligent_tool_mapping
from usecases.immunity.common.utils import download_geo_dataset
from usecases.immunity.schema.common_schemas import PlanStep, TaskExtractionResult
from usecases.immunity.state.state import (
    ImprovedCellState,
)


class TaskExecutor:
    """Task Executor - Core Class"""

    _EMBEDDED_ERROR_PATTERNS = [
        re.compile(r"\berror\s*[:\-]", re.IGNORECASE),
        re.compile(r"\bexception\b", re.IGNORECASE),
        re.compile(r"\btraceback\b", re.IGNORECASE),
        re.compile(r"\bfailed to\b", re.IGNORECASE),
        re.compile(r"\bnot found\b", re.IGNORECASE),
        re.compile(r"\bdoes not exist\b", re.IGNORECASE),
        re.compile(r"\bno such file\b", re.IGNORECASE),
        re.compile(r"\bpermission denied\b", re.IGNORECASE),
        re.compile(r"\bmissing (?:required|input)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        config: Optional[RunnableConfig] = None,
        ui_interaction_mode: bool = False,
        ui_callback=None,
        sse_streamer=None,
    ):
        """
        初始化TaskExecutor

        Args:
            config: 运行配置
            ui_interaction_mode: UI交互模式开关，True为UI模式，False为Terminal模式
            ui_callback: UI交互回调函数，仅在ui_interaction_mode=True时使用
            sse_streamer: SSE流处理器，用于与前端通信
        """
        self.config = config or RunnableConfig()
        self.agent = None
        self.tools = []
        self.checkpointer = MemorySaver()

        # UI交互模式相关配置
        self.ui_interaction_mode = ui_interaction_mode
        self.ui_callback = ui_callback
        self.sse_streamer = sse_streamer
        self.plan_steps: List[Dict[str, Any]] = []
        self.plan_id: Optional[str] = None
        self._geo_download_cache: set[str] = set()
        self._tool_call_sequence: int = 0
        
        # CSV结果收集器
        from usecases.immunity.common.csv_result_collector import CSVResultCollector
        self.csv_collector: Optional[CSVResultCollector] = None

    async def initialize_agent(self):
        """Initialize agent and tools"""

        model = get_reasoning_model(self.config)
        if self.sse_streamer:
            try:
                self.sse_streamer.send_execution_progress(
                    {
                        "planId": self.plan_id,
                        "stepId": "initializing",
                        "status": "running",
                        "message": "Preparing task executor",
                    }
                )
            except Exception as progress_err:
                pass

        try:
            tools = await get_all_tools(self.config)
        except Exception as tool_error:
            tools = []
        self.tools = tools

        # 关键修复：必须使用 hil() 包装器来触发 interrupt
        # 但是，我们需要确保 resume payload 只影响当前工具调用，不影响后续工具调用
        hil_tools = [hil(tool) for tool in self.tools] if self.tools else []
        self.agent = create_react_agent(
            model=model, tools=hil_tools, checkpointer=self.checkpointer
        )

        if self.sse_streamer:
            try:
                self.sse_streamer.send_execution_progress(
                    {
                        "planId": self.plan_id,
                        "stepId": "initializing",
                        "status": "completed",
                        "message": f"Loaded {len(hil_tools)} tools",
                    }
                )
            except Exception as progress_err:
                pass

    def _report_tool_progress(self, server_name: str, status: str, tool_count: int | None = None):
        if not self.sse_streamer:
            return
        message = ""
        if status == "connecting":
            message = f"Connecting to {server_name}..."
        elif status == "loaded":
            message = f"Loaded {tool_count or 0} tools from {server_name}"
        elif status == "failed":
            message = f"Failed to load tools from {server_name}"
        try:
            self.sse_streamer.send_execution_progress(
                {
                    "planId": self.plan_id,
                    "stepId": "initializing",
                    "status": "running",
                    "message": message,
                    "server": server_name,
                    "tools": tool_count,
                }
            )
        except Exception as progress_err:
            pass

    async def execute_task(
        self,
        task_description: str,
        progress_hook: Optional[Any] = None,
        step_context: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        initial_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single task with interactive retries for tool failures."""
        if not self.agent:
            await self.initialize_agent()
        
        # 确保CSV收集器已初始化（懒加载）
        if self.csv_collector is None:
            from usecases.immunity.common.csv_result_collector import CSVResultCollector
            self.csv_collector = CSVResultCollector()
            await self.csv_collector.initialize()

        tools_called: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        thread_config = {
            "configurable": {"thread_id": f"task_{asyncio.get_event_loop().time()}"}
        }

        context = step_context or {}
        preferred_tools = context.get("tools") or context.get("toolchain") or []
        preferred_tool_text = (
            "\n\nPreferred tools for this step (execute these if applicable):\n"
            + "\n".join(f"- {tool}" for tool in preferred_tools)
            if preferred_tools
            else ""
        )
        
        # 如果提供了文件路径（初始文件或累积的合并文件），在任务描述中添加提示
        initial_file_hint = ""
        if initial_file_path:
            initial_file_hint = f"""

**IMPORTANT: Cumulative Merged CSV/Excel File Available**
A cumulative merged CSV/Excel file is available at:
  {initial_file_path}

**Smart Parameter Selection Instructions:**
- This file contains data accumulated from all previous tasks in the workflow.
- **ONLY use this file path for CSV/Excel-related parameters**, such as:
  * Parameters explicitly requiring CSV/Excel files (e.g., `input_file`, `csv_file_path`, `data_file`, `sequences` when expecting CSV/Excel format)
  * Parameters with file type constraints indicating CSV/Excel (check tool description and parameter schema)
  * Parameters named with CSV/Excel context (e.g., `sc_rna_csv_path`, `feature_data_path`, `clone_results_path`)
  
- **DO NOT use this file path for other parameter types**, such as:
  * RDS file parameters (e.g., `rds_file_path`) - use RDS files from previous steps or history
  * Directory parameters (e.g., `bulk_raw_data_dir`, `output_dir`) - use appropriate directory paths
  * UMAP coordinates parameters (e.g., `umap_coordinates_path`) - use UMAP output from previous tools
  * FASTQ file parameters - use FASTQ file paths from data sources
  * Other specialized file types (PDB, JSON, etc.) - use appropriate file paths from context
  
- **How to determine parameter type:**
  1. Check the parameter name and description in the tool schema
  2. Look for file type hints in parameter descriptions (e.g., "RDS file", "CSV file", "directory")
  3. Review previous tool outputs in the conversation history to find appropriate file paths
  4. Use the merged CSV file ONLY when the parameter clearly expects CSV/Excel format data
  
- The merged CSV file path is: {initial_file_path}
- After tool execution, any new CSV/Excel outputs will be automatically merged into this file for the next task.
"""

        tool_mapping_guide = get_intelligent_tool_mapping()
        task_message = HumanMessage(
            content=f"""
Task: {task_description}

{tool_mapping_guide}

## Execution Instructions

Please analyze the task description according to the above intelligent tool mapping guide and select the most appropriate MCP tool to complete this task.

**Execution Steps:**
1. Carefully analyze keywords and analysis types in the task description
2. Select the most appropriate MCP tool according to the mapping strategy
3. Ensure correct parameter settings (especially input_file and base_dir)
4. **CRITICAL RULE #1**: Call ONLY ONE tool per response. Do NOT call multiple tools.
5. **CRITICAL RULE #2**: After calling ONE tool, STOP immediately. Do NOT call any additional tools.
6. **CRITICAL RULE #3**: Wait for user confirmation before calling the next tool, even if tools have dependencies.
7. If the task involves multiple steps, call tools one at a time in logical order, but **STOP after each tool call** and wait for user confirmation.
8. Prioritize tools with the best functional match and avoid using generic tool names
{initial_file_hint}

**ABSOLUTE REQUIREMENTS**: 
- **You MUST call only ONE tool per response**
- **After calling a tool, you MUST STOP and wait**
- **Do NOT call multiple tools, even if they have dependencies**
- **Do NOT automatically chain tool calls**
- **Each tool call must be individually confirmed by the user**

Please start by calling the FIRST tool only. STOP after calling it and wait for user confirmation.{preferred_tool_text}
"""
        )

        try:
            result = self.agent.invoke({"messages": [task_message]}, thread_config)
            
            # 重试计数器，防止死循环
            retry_count = 0
            max_retries = 10

            while "__interrupt__" in result:
                # 关键修复：确保每次只处理一个工具调用请求
                # 即使 result 中包含多个 interrupt，也只处理第一个
                interrupt_info = result["__interrupt__"][0]

                tool_payload = self._parse_tool_call(interrupt_info.value)
                if not tool_payload:
                    break

                tool_info = self._get_complete_parameters(tool_payload) or tool_payload
                
                # 关键修复：每次新的工具调用都必须等待用户确认
                # 不要使用任何缓存的响应，确保用户对每个工具调用都明确确认
                print(f"[DEBUG] execute_task: 🔔 检测到新的工具调用请求，工具: {tool_info.get('tool_name')}")
                print(f"[DEBUG] execute_task: 🔔 准备调用 _get_user_confirmation，等待用户明确确认")
                print(f"[DEBUG] execute_task: 🔔 这是第 {len(tools_called) + 1} 个工具调用")

                # 关键修复：在内层循环开始前，确保用户已经确认要调用这个工具
                # 即使工具执行完成后代理恢复并立即调用下一个工具，这里也会再次触发用户确认
                while True:
                    # 关键：每次都必须等待真实的用户响应，不能使用缓存或默认值
                    user_decision = await self._get_user_confirmation(tool_info)
                    action = (user_decision.get("action") or "").lower()
                    
                    print(f"[DEBUG] execute_task: ✅ 收到用户确认结果，action: {action}")
                    print(f"[DEBUG] execute_task: ✅ 用户决策详情: {user_decision}")

                    if action == "accept":
                        if user_decision.get("modified_args") is not None:
                            tool_info["args"] = user_decision["modified_args"]

                        # 关键修复：用户已确认，现在需要 resume 代理以执行工具
                        # resume 的 payload 会被传递给 hil() 中当前等待的 interrupt()
                        # 这个 payload 只会影响当前工具调用，不会影响后续的工具调用
                        resume_payload = {"accept": True}
                        if tool_info.get("args"):
                            # 确保参数格式正确
                            if isinstance(tool_info["args"], dict) and "args" not in tool_info["args"]:
                                # 如果参数已经是字典，但没有 "args" 键，可能需要包装
                                resume_payload["args"] = tool_info["args"]
                            else:
                                resume_payload["args"] = tool_info["args"]
                        
                        print(f"[DEBUG] execute_task: 🔄 准备 resume 代理以执行工具")
                        print(f"[DEBUG] execute_task: 🔄 payload: {json.dumps(resume_payload, ensure_ascii=False)}")
                        print(f"[DEBUG] execute_task: 🔄 工具: {tool_info.get('tool_name')}")
                        print(f"[DEBUG] execute_task: 🔄 这是第 {len(tools_called) + 1} 个工具调用")
                        
                        # 关键：resume 代理，传递用户确认
                        # 这个 resume payload 会被传递给 hil() 中当前等待的 interrupt()
                        # 工具会通过 hil() 包装器执行
                        result = self.agent.invoke(
                            Command(resume=json.dumps(resume_payload, ensure_ascii=False)),
                            thread_config,
                        )
                        
                        print(f"[DEBUG] execute_task: ✅ 代理 resume 完成，检查结果...")
                        print(f"[DEBUG] execute_task: ✅ result 类型: {type(result)}")
                        print(f"[DEBUG] execute_task: ✅ result 键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                        
                        # 关键：提取工具执行结果
                        # 工具执行结果可能在 result["messages"] 中
                        tool_result = None
                        if "messages" in result and result["messages"]:
                            # 查找工具执行的消息
                            for msg in result["messages"]:
                                # 检查是否是工具消息
                                if hasattr(msg, "name") and msg.name == tool_info.get("tool_name"):
                                    tool_result = msg.content if hasattr(msg, "content") else str(msg)
                                    print(f"[DEBUG] execute_task: ✅ 从消息中提取工具结果: {str(tool_result)[:200]}")
                                    break
                                # 检查是否是 ToolMessage
                                elif hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
                                    # 可能是工具执行结果
                                    tool_result = msg.content if hasattr(msg, "content") else str(msg)
                                    print(f"[DEBUG] execute_task: ✅ 从 ToolMessage 中提取工具结果: {str(tool_result)[:200]}")
                        
                        # 记录工具调用结果
                        call_entry = self._create_tool_call_entry(
                            tool_info.get("tool_name"),
                            tool_info.get("args"),
                            status="completed" if tool_result else "running",
                            service_id=tool_info.get("service_id"),
                        )
                        if tool_result:
                            call_entry["result"] = tool_result
                        tools_called.append(call_entry)
                        
                        if tool_result:
                            tool_result_data = {
                                "tool_name": tool_info.get("tool_name"),
                                "args": tool_info.get("args"),
                                "result": tool_result,
                                "status": "completed",
                            }
                            tool_results.append(tool_result_data)
                            
                            # 收集 CSV 结果
                            if self.csv_collector:
                                try:
                                    merged_csv_path = await self.csv_collector.collect_tool_output(tool_result_data)
                                    if merged_csv_path:
                                        tool_result_data["merged_csv_path"] = merged_csv_path
                                except Exception:
                                    pass
                            
                            await self._emit_tool_progress(
                                progress_hook, tools_called, tool_results, call_entry
                            )
                        
                        # 关键：检查是否还有下一个工具调用（__interrupt__）
                        # 如果有，说明代理在工具执行完成后立即决定调用下一个工具
                        # 外层循环会继续，捕获新的 interrupt，并调用 _get_user_confirmation 让用户确认
                        # 
                        # 关键修复：确保 resume payload 不会影响新的 interrupt
                        # 如果 result 中包含 "__interrupt__"，说明有新的工具调用请求
                        # 这个新的 interrupt 应该等待新的用户确认，而不是使用之前的 resume payload
                        
                        if "__interrupt__" not in result:
                            # 没有下一个工具调用，任务执行完成
                            print(f"[DEBUG] execute_task: ✅ 没有下一个工具调用，任务执行完成")
                            break
                        
                        # 如果有下一个工具调用，外层循环会继续处理
                        # 新的 interrupt 会被捕获，并调用 _get_user_confirmation 让用户确认
                        print(f"[DEBUG] execute_task: 🔔 检测到下一个工具调用请求，继续外层循环")
                        print(f"[DEBUG] execute_task: 🔔 重要：新的 interrupt 会等待新的用户确认，不会使用之前的 resume payload")
                        # break 内层循环，让外层循环处理新的工具调用
                        break
                        # 注意：status == "retry"的情况已经在_execute_tool_with_retry内部处理了
                        # 如果返回retry状态，说明有逻辑错误，应该记录并继续
                        if status == "retry":
                            # 这种情况不应该发生，但为了安全，我们继续循环
                            continue
                        if status == "skipped":
                            return {
                                "success": False,
                                "result": execution_outcome.get(
                                    "message", "Tool execution skipped by user"
                                ),
                                "tools_called": tools_called,
                                "tool_results": tool_results,
                            }
                        if status == "abort":
                            raise Exception(
                                execution_outcome.get(
                                    "message", "Tool execution aborted by user"
                                )
                            )
                        continue

                    if action == "skip":
                        call_entry = self._create_tool_call_entry(
                            tool_info.get("tool_name"),
                            tool_info.get("args"),
                            status="skipped",
                            service_id=tool_info.get("service_id"),
                        )
                        tools_called.append(call_entry)
                        await self._emit_tool_progress(
                            progress_hook, tools_called, tool_results, call_entry
                        )
                        return {
                            "success": False,
                            "result": "Tool execution skipped by user",
                            "tools_called": tools_called,
                            "tool_results": tool_results,
                        }

                    if action in {"reject", "cancel"}:
                        call_entry = self._create_tool_call_entry(
                            tool_info.get("tool_name"),
                            tool_info.get("args"),
                            status="rejected",
                            service_id=tool_info.get("service_id"),
                        )
                        tools_called.append(call_entry)
                        await self._emit_tool_progress(
                            progress_hook, tools_called, tool_results, call_entry
                        )
                        result = self.agent.invoke(
                            Command(resume=json.dumps({"accept": False})),
                            thread_config,
                        )
                        break

                    if user_decision.get("modified_args") is not None:
                        tool_info["args"] = user_decision["modified_args"]
                        continue

                    call_entry = self._create_tool_call_entry(
                        tool_info.get("tool_name"),
                        tool_info.get("args"),
                        status="rejected",
                        service_id=tool_info.get("service_id"),
                    )
                    tools_called.append(call_entry)
                    await self._emit_tool_progress(
                        progress_hook, tools_called, tool_results, call_entry
                    )
                    result = self.agent.invoke(
                        Command(resume=json.dumps({"accept": False})),
                        thread_config,
                    )
                    break

            final_result = ""
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    final_result = last_message.content

            if tool_results:
                print(f"\n{'=' * 60}")
                print(f"Tool Execution Results")
                print(f"{'=' * 60}")
                print(f"Task: {task_description}")
                print(f"Tools Executed: {len(tool_results)}")
                print(f"{'=' * 60}")
                for i, tool_result in enumerate(tool_results, 1):
                    tool_name = tool_result.get("tool_name", "Unknown Tool")
                    args = tool_result.get("args", {})
                    result_content = tool_result.get("result", "No result")
                    merged_csv_path = tool_result.get("merged_csv_path")
                    print(f"\nTool {i}: {tool_name}")
                    print(f"Result: {result_content}")
                    if merged_csv_path:
                        print(f"Merged CSV: {merged_csv_path}")
                    print(f"{'─' * 40}")
                print(f"{'=' * 60}\n")

            return {
                "success": True,
                "result": final_result,
                "tools_called": tools_called,
                "tool_results": tool_results,
            }
        except Exception as task_error:  # noqa: BLE001
            error_message = str(task_error)
            print(f"Error: Task execution exception: {error_message}")
            print(f"Exception type: {type(task_error).__name__}")
            for call in tools_called:
                if call.get("status") == "running":
                    call["status"] = "failed"
                    call.setdefault("error", error_message)
                    call["finished_at"] = time.time()
                    call["duration"] = call["finished_at"] - call.get(
                        "started_at", call["finished_at"]
                    )
            await self._emit_tool_progress(
                progress_hook,
                tools_called,
                tool_results,
                message=error_message,
            )
            return {
                "success": False,
                "result": error_message,
                "tools_called": tools_called,
                "tool_results": tool_results,
            }

    async def _execute_tool_with_retry(
        self,
        tool_info: Dict[str, Any],
        tools_called: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        thread_config: Dict[str, Any],
        progress_hook: Optional[Any],
        task_description: Optional[str],
        step_id: Optional[str],
    ) -> Dict[str, Any]:
        """Execute a tool with retry support based on user feedback."""
        attempt_args = tool_info.get("args")

        while True:
            call_entry = self._create_tool_call_entry(
                tool_info.get("tool_name"),
                attempt_args,
                status="pending",  # 改为 pending，等待用户确认
                service_id=tool_info.get("service_id"),
            )
            tools_called.append(call_entry)
            await self._emit_tool_progress(
                progress_hook, tools_called, tool_results, call_entry
            )

            # 注意：不要在这里自动设置 accept: True
            # hil() 装饰器会在工具调用时触发 interrupt，用户需要通过 interrupt 确认
            # 这里不应该直接 resume，而是应该等待 interrupt 处理完成
            # 但是由于这个方法是处理已经中断的调用，我们需要先等待用户确认
            
            # 等待用户确认（通过 interrupt 机制）
            # 这里不应该自动 accept，应该等待用户的明确确认
            # 但实际上，如果代理已经中断，说明 hil() 已经触发了 interrupt
            # 所以这里不应该再次 invoke，而应该等待 interrupt 响应
            
            # 重要：这里不应该自动 accept，需要检查是否有中断等待处理
            # 如果有中断，应该等待用户响应，而不是自动 accept
            
            print(f"[DEBUG] 准备执行工具 {tool_info.get('tool_name')}，等待用户确认...")
            print(f"[DEBUG] 参数: {attempt_args}")
            
            # 注意：如果代理是通过 hil() 包装的，工具调用会自动触发 interrupt
            # 这里我们不应该绕过这个机制，应该让 interrupt 正常处理
            
            # 重要：这里不应该自动 accept！
            # 工具已经被 hil() 包装，hil() 会在调用时触发 interrupt
            # 用户确认已经在 execute_task 的 _get_user_confirmation 中完成
            # 但是，当我们 resume 代理时，我们需要确保 resume 的格式符合 hil() 的期望
            # hil() 期望的格式是 JSON 字符串: {"accept": true, "args": {...}}
            # 
            # 问题：当我们使用 Command(resume=...) 时，我们是在恢复被 hil() interrupt 的工具调用
            # 但是，如果我们在 resume 中传入 accept: True，可能会绕过 hil() 的第二次确认
            # 
            # 解决方案：我们需要确保 resume 的 payload 格式正确，并且不应该自动 accept
            # 实际上，用户已经在 _get_user_confirmation 中确认了，所以这里可以使用确认的参数
            # 但关键是确保格式正确
            
            # 关键修复：这里不应该直接 resume 代理！
            # 用户确认已经在 execute_task 的外层循环（第273行）中通过 _get_user_confirmation 完成
            # 当我们调用 _execute_tool_with_retry 时，代理已经在等待用户确认的状态
            # 但是，由于我们已经在 execute_task 中处理了用户确认，这里不应该再次 resume
            # 
            # 问题：如果在 execute_task 中已经 resume 了代理，那么工具应该已经执行了
            # 但实际上，工具调用是在这里执行的，说明代理还没有 resume
            # 
            # 解决方案：这里应该 resume 代理，传递用户确认的参数
            # 但是，要确保 resume payload 只影响当前的工具调用，不影响后续的工具调用
            
            # ⚠️ 关键问题：这里不应该 resume 代理！
            # 工具已经被 hil() 包装，当代理调用工具时会触发 interrupt()
            # 用户确认已经在外层循环（execute_task）中通过 _get_user_confirmation 完成
            # 
            # 正确的流程应该是：
            # 1. 代理调用工具 → hil() 触发 interrupt() → 代理暂停
            # 2. 外层循环捕获 interrupt → 调用 _get_user_confirmation → 获取用户确认
            # 3. 用户确认后，在外层循环中 resume 代理，传递用户确认
            # 4. 代理恢复，hil() 中的 interrupt() 返回用户确认的值，工具执行
            # 5. 工具执行完成后，如果代理决定调用下一个工具，会触发新的 interrupt()
            # 6. 新的 interrupt() 应该等待新的用户确认，而不是使用之前的 resume payload
            #
            # 问题：如果我们在 _execute_tool_with_retry 中 resume，resume payload 可能被传递给了新的 interrupt()
            #
            # 解决方案：不在 _execute_tool_with_retry 中 resume，而是在外层循环中 resume
            # 但是，这个函数被调用时，说明外层循环需要工具执行结果
            # 所以，我们需要在这里 resume，但要确保 resume payload 只影响当前工具调用
            
            # 构建 resume payload，只包含当前工具调用的确认信息
            resume_payload = {"accept": True}
            if attempt_args is not None:
                if tool_info.get("structured", False) and isinstance(attempt_args, dict) and "args" not in attempt_args:
                    resume_payload["args"] = {"args": attempt_args}
                else:
                    resume_payload["args"] = attempt_args

            call_start = time.time()
            try:
                # ⚠️ 关键：这里 resume 代理，resume payload 会被传递给当前等待的 interrupt()
                # 但是，如果代理在工具执行完成后立即调用下一个工具，新的 interrupt() 应该不会收到这个 payload
                # 问题可能在于：LangGraph 的 resume 机制可能会将 payload 传递给下一个 interrupt()
                #
                # 解决方案：确保每个工具调用都有唯一的标识，或者使用不同的 thread_id
                resume_json = json.dumps(resume_payload, ensure_ascii=False)
                print(f"[DEBUG] _execute_tool_with_retry: ⚠️ 准备 resume 代理，payload: {resume_json}")
                print(f"[DEBUG] _execute_tool_with_retry: ⚠️ 警告：这个 resume 可能影响后续的工具调用！")
                invoke_result = self.agent.invoke(
                    Command(resume=resume_json), thread_config
                )
                print(f"[DEBUG] _execute_tool_with_retry: ✅ 代理 resume 完成")
            except Exception as call_error:  # noqa: BLE001
                error_message = str(call_error)
                call_entry["status"] = "failed"
                call_entry["error"] = error_message
                call_entry["finished_at"] = time.time()
                call_entry["duration"] = call_entry["finished_at"] - call_start
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message=error_message,
                )

                retry_response = await self._prompt_tool_retry(
                    tool_info=tool_info,
                    current_args=attempt_args,
                    error_message=error_message,
                    step_id=step_id,
                    task_description=task_description,
                )

                next_action = (
                    retry_response.get("action")
                    or retry_response.get("type")
                    or "abort"
                ).lower()
                if next_action == "retry":
                    new_args = retry_response.get("args") or retry_response.get("toolArgs")
                    if new_args is not None:
                        attempt_args = deepcopy(new_args)
                    else:
                        attempt_args = deepcopy(tool_info.get("args") or {})
                    tool_info["args"] = attempt_args
                    retry_entry = self._create_tool_call_entry(
                        tool_info.get("tool_name"),
                        attempt_args,
                        status="retrying",
                        service_id=tool_info.get("service_id"),
                    )
                    retry_entry["message"] = "Retrying tool with updated parameters"
                    tools_called.append(retry_entry)
                    await self._emit_tool_progress(
                        progress_hook,
                        tools_called,
                        tool_results,
                        retry_entry,
                        message="Retrying tool with updated parameters",
                    )
                    continue

                if next_action == "skip":
                    call_entry["status"] = "skipped"
                    await self._emit_tool_progress(
                        progress_hook,
                        tools_called,
                        tool_results,
                        call_entry,
                        message=error_message,
                    )
                    return {"status": "skipped", "message": error_message}

                return {"status": "abort", "message": error_message}

            # 工具调用完成（无论成功还是失败），现在进行reasoning分析
            # 工具调用和reasoning分析是一个整体，只有reasoning分析完成且通过，才算真正完成
            
            tool_result = self._extract_tool_result(
                invoke_result, tool_info["tool_name"]
            )

            tool_result_data = {
                "tool_name": tool_info["tool_name"],
                "args": attempt_args,
                "result": tool_result,
                "call_id": call_entry["call_id"],
            }

            post_processing = self._handle_post_tool_execution(tool_result_data)
            if post_processing:
                tool_result_data.update(post_processing)

            # 检测工具执行错误（即使工具出错，也需要reasoning分析来给用户建议）
            error_message = self._detect_tool_error(tool_result)
            if error_message:
                error_message = error_message.strip()
                tool_result_data["detected_error"] = {"message": error_message}
                tool_result_data["error"] = error_message
                call_entry["status"] = "failed"
                call_entry["error"] = error_message

            # 进行reasoning分析（工具调用和reasoning分析是一个整体）
            # 无论工具成功还是失败，都需要reasoning分析
            try:
                # 进入评估
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message="Starting result evaluation",
                )
                # 评估中（可作为心跳提示）
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message="Evaluating...",
                )
                # 进行reasoning分析，传入工具结果和错误信息（如果有）
                reasoning_input = tool_result
                if error_message:
                    # 如果工具出错，将错误信息也包含在reasoning输入中
                    reasoning_input = {
                        "result": tool_result,
                        "error": error_message,
                        "has_error": True,
                    }
                
                reasoning_payload = await self._run_post_tool_reasoning(
                    tool_name=tool_info.get("tool_name"),
                    tool_args=attempt_args,
                    tool_output=reasoning_input,
                    task_goal=task_description or "",
                    step_id=step_id or "",
                )
                if reasoning_payload:
                    tool_result_data["reasoning"] = reasoning_payload
                
                # 评估完成
                reasoning_status = (reasoning_payload or {}).get("status", "uncertain")
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message=f"Evaluation completed: {reasoning_status}",
                )
                
                # 根据reasoning结果判断工具调用是否真正完成
                # 只有reasoning为"valid"时，工具调用才算真正完成
                if reasoning_status == "valid":
                    # reasoning通过，工具调用真正完成
                    call_entry["status"] = "completed"
                    call_entry["finished_at"] = time.time()
                    call_entry["duration"] = call_entry["finished_at"] - call_start
                    tool_result_data["status"] = "completed"
                    call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                    tool_results.append(tool_result_data)
                    
                    # 收集工具产生的CSV/Excel文件（在reasoning通过的情况下也要收集）
                    merged_csv_path = None
                    if self.csv_collector is None:
                        from usecases.immunity.common.csv_result_collector import CSVResultCollector
                        self.csv_collector = CSVResultCollector()
                        await self.csv_collector.initialize()
                    
                    if self.csv_collector:
                        try:
                            merged_csv_path = await self.csv_collector.collect_tool_output(tool_result_data)
                            if merged_csv_path:
                                # 合并后的CSV路径仅用于记录和展示，不会作为工具参数传递给后续工具
                                tool_result_data["merged_csv_path"] = merged_csv_path
                        except Exception as csv_err:
                            pass
                    
                    await self._emit_tool_progress(
                        progress_hook, tools_called, tool_results, call_entry
                    )
                    # 工具调用真正完成，返回结果并退出函数
                    return {
                        "status": "completed",
                        "result": invoke_result,
                        "tool_result": tool_result_data,
                    }
                else:
                    # Reasoning evaluation failed, trigger interrupt for user decision
                    await self._emit_tool_progress(
                        progress_hook,
                        tools_called,
                        tool_results,
                        call_entry,
                        message=f"Evaluation failed ({reasoning_status}), waiting for user decision",
                    )
                    
                    # 触发中断，让用户选择重试或继续
                    reasoning_response = await self._prompt_reasoning_decision(
                        tool_info=tool_info,
                        current_args=attempt_args,
                        reasoning_payload=reasoning_payload,
                        tool_result=tool_result,
                        step_id=step_id,
                        task_description=task_description,
                        call_entry=call_entry,
                        tools_called=tools_called,
                        tool_results=tool_results,
                    )
                    
                    next_action = (
                        reasoning_response.get("action")
                        or reasoning_response.get("type")
                        or "continue"
                    ).lower()
                    
                    if next_action == "retry":
                        # 用户选择重试工具：检查是否已经在前端修改了参数
                        # 如果reasoning_response中已经包含了modified_args，说明用户已经在前端修改了参数并确认
                        # 此时不需要再次弹出参数修改弹窗，直接用新参数重新调用工具
                        new_args = reasoning_response.get("args") or reasoning_response.get("toolArgs") or reasoning_response.get("modified_args")
                        if new_args is not None:
                            # 用户已经在前端修改了参数并确认，直接用新参数重新调用工具
                            attempt_args = new_args
                            tool_info["args"] = attempt_args
                            # 移除当前工具结果，准备重试
                            if tool_results and tool_results[-1].get("tool_name") == tool_info.get("tool_name"):
                                tool_results.pop()
                            # 继续执行工具调用（在while True循环中重新开始）
                            continue
                        
                        # 如果reasoning_response中没有包含参数，说明用户只是点击了"重试工具"按钮，需要弹出参数修改弹窗
                        # 移除当前工具结果，准备重试
                        if tool_results and tool_results[-1].get("tool_name") == tool_info.get("tool_name"):
                            tool_results.pop()
                        
                        # 确保tool_info包含完整信息，用于弹出参数修改弹窗
                        if not tool_info.get("args_schema"):
                            # 尝试从工具对象获取schema
                            tool_info = self._get_complete_parameters(tool_info) or tool_info
                        # 更新tool_info的timestamp，确保前端能识别这是一个新的请求
                        tool_info["timestamp"] = self._get_timestamp()
                        user_decision = await self._get_user_confirmation(tool_info)
                        action = (user_decision.get("action") or "").lower()
                        
                        if action == "accept":
                            # 用户确认重试，更新参数（如果有修改）
                            if user_decision.get("modified_args") is not None:
                                attempt_args = user_decision["modified_args"]
                                tool_info["args"] = attempt_args
                            # 继续执行工具调用（在while True循环中重新开始）
                            continue
                        elif action in {"skip", "reject", "cancel"}:
                            # 用户取消重试，返回跳过状态
                            call_entry["status"] = "skipped"
                            call_entry["finished_at"] = time.time()
                            call_entry["duration"] = call_entry["finished_at"] - call_start
                            tool_result_data["status"] = "skipped"
                            tool_result_data["reasoning"] = reasoning_payload
                            call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                            tool_results.append(tool_result_data)
                            await self._emit_tool_progress(
                                progress_hook, tools_called, tool_results, call_entry
                            )
                            return {
                                "status": "skipped",
                                "message": "用户取消重试工具",
                                "tool_result": tool_result_data,
                            }
                        else:
                            # 其他情况，默认继续执行
                            continue
                    
                    if next_action == "abort":
                        # 用户选择中止
                        call_entry["status"] = "aborted"
                        call_entry["finished_at"] = time.time()
                        call_entry["duration"] = call_entry["finished_at"] - call_start
                        tool_result_data["status"] = "aborted"
                        tool_result_data["reasoning"] = reasoning_payload
                        call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                        tool_results.append(tool_result_data)
                        return {
                            "status": "abort",
                            "message": f"Reasoning评估未通过: {reasoning_payload.get('rationale', 'N/A')}",
                            "reasoning": reasoning_payload,
                        }
                    
                    # 用户选择继续（skip或continue），尽管reasoning未通过
                    call_entry["status"] = "completed"
                    call_entry["finished_at"] = time.time()
                    call_entry["duration"] = call_entry["finished_at"] - call_start
                    tool_result_data["status"] = "completed"
                    tool_result_data["reasoning"] = reasoning_payload
                    call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                    tool_results.append(tool_result_data)
                    
                    # 收集工具产生的CSV/Excel文件
                    merged_csv_path = None
                    # 懒加载：如果收集器还未初始化，现在初始化
                    if self.csv_collector is None:
                        from usecases.immunity.common.csv_result_collector import CSVResultCollector
                        self.csv_collector = CSVResultCollector()
                        await self.csv_collector.initialize()
                    
                    if self.csv_collector:
                        try:
                            merged_csv_path = await self.csv_collector.collect_tool_output(tool_result_data)
                            if merged_csv_path:
                                # 将合并后的CSV路径添加到工具结果中，便于用户查看
                                tool_result_data["merged_csv_path"] = merged_csv_path
                        except Exception as csv_err:
                            pass
                    
                    # 工具调用完成，返回结果（不要continue，否则会重复执行）
                    await self._emit_tool_progress(
                        progress_hook, tools_called, tool_results, call_entry
                    )
                    return {
                        "status": "completed",
                        "result": invoke_result,
                        "tool_result": tool_result_data,
                    }
            except Exception as eval_err:  # noqa: BLE001
                # 评估失败，降级为不确定状态，但仍需要用户决策
                reasoning_payload = {
                    "status": "uncertain",
                    "confidence": 0.0,
                    "rationale": f"reasoning_failed: {eval_err}",
                    "recommended_actions": ["Manually review or retry the tool"],
                }
                tool_result_data["reasoning"] = reasoning_payload
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message="Evaluation failed (downgraded to uncertain), waiting for user decision",
                )
                
                # 即使reasoning失败，也触发中断让用户决策
                reasoning_response = await self._prompt_reasoning_decision(
                    tool_info=tool_info,
                    current_args=attempt_args,
                    reasoning_payload=reasoning_payload,
                    tool_result=tool_result,
                    step_id=step_id,
                    task_description=task_description,
                    call_entry=call_entry,
                    tools_called=tools_called,
                    tool_results=tool_results,
                )
                
                next_action = (
                    reasoning_response.get("action")
                    or reasoning_response.get("type")
                    or "continue"
                ).lower()
                
                if next_action == "retry":
                    new_args = reasoning_response.get("args") or reasoning_response.get("toolArgs")
                    if new_args is not None:
                        attempt_args = new_args
                    tool_info["args"] = attempt_args
                    if tool_results and tool_results[-1].get("tool_name") == tool_info.get("tool_name"):
                        tool_results.pop()
                    return {
                        "status": "retry",
                        "tool_info": tool_info,
                        "message": "用户选择重试工具",
                    }
                
                if next_action == "abort":
                    call_entry["status"] = "aborted"
                    call_entry["finished_at"] = time.time()
                    call_entry["duration"] = call_entry["finished_at"] - call_start
                    tool_result_data["status"] = "aborted"
                    tool_result_data["reasoning"] = reasoning_payload
                    call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                    tool_results.append(tool_result_data)
                    return {
                        "status": "abort",
                        "message": f"Reasoning评估失败: {eval_err}",
                        "reasoning": reasoning_payload,
                    }
                
                # 用户选择继续
                call_entry["status"] = "completed"
                call_entry["finished_at"] = time.time()
                call_entry["duration"] = call_entry["finished_at"] - call_start
                tool_result_data["status"] = "completed"
                tool_result_data["reasoning"] = reasoning_payload
                call_entry.setdefault("download_info", tool_result_data.get("download_info"))
                tool_results.append(tool_result_data)
                # 工具调用完成，返回结果（不要continue，否则会重复执行）
                await self._emit_tool_progress(
                    progress_hook, tools_called, tool_results, call_entry
                )
                return {
                    "status": "completed",
                    "result": invoke_result,
                    "tool_result": tool_result_data,
                }

            # 注意：工具调用和reasoning分析已经作为一个整体处理完成
            # 如果执行到这里，说明reasoning分析已经完成且通过，或者用户选择继续
            # 工具调用真正完成，返回结果
            
            # 收集工具产生的CSV/Excel文件
            merged_csv_path = None
            # 懒加载：如果收集器还未初始化，现在初始化
            if self.csv_collector is None:
                from usecases.immunity.common.csv_result_collector import CSVResultCollector
                self.csv_collector = CSVResultCollector()
                await self.csv_collector.initialize()
            
            if self.csv_collector:
                try:
                    merged_csv_path = await self.csv_collector.collect_tool_output(tool_result_data)
                    if merged_csv_path:
                        # 将合并后的CSV路径添加到工具结果中，便于用户查看
                        tool_result_data["merged_csv_path"] = merged_csv_path
                except Exception as csv_err:
                    pass
            
            await self._emit_tool_progress(
                progress_hook, tools_called, tool_results, call_entry
            )
            return {
                "status": "completed",
                "result": invoke_result,
                "tool_result": tool_result_data,
            }

    async def _run_post_tool_reasoning(
        self,
        *,
        tool_name: str,
        tool_args: Dict[str, Any] | None,
        tool_output: Any,
        task_goal: str,
        step_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        同步调用云端/本地推理模型，对单次工具调用结果进行有效性评估与建议生成。
        返回结构化结果：{status, confidence, rationale, issues?, recommended_actions?}
        """
        try:
            model = get_reasoning_model(self.config)
            # 构造最小上下文，避免超长
            def _summarize_output(x: Any) -> str:
                try:
                    js = json.dumps(x, ensure_ascii=False)[:2000]
                    return js
                except Exception:
                    text = str(x)
                    return text[:2000]

            prompt = (
                "You are a result evaluation assistant. Please evaluate whether the tool invocation result is valid based on the given information and provide structured recommendations.\n"
                "Please strictly output JSON with the following fields:\n"
                'status: "valid" | "uncertain" | "invalid",\n'
                "confidence: a number between 0 and 1,\n"
                "rationale: a brief English explanation,\n"
                "issues: optional array of strings,\n"
                "recommended_actions: an array of strings (recommendations for next steps).\n\n"
                f"Task goal: {task_goal}\n"
                f"Step ID: {step_id}\n"
                f"Tool: {tool_name}\n"
                f"Parameters: {json.dumps(tool_args or {}, ensure_ascii=False)[:1000]}\n"
                f"Output summary: {_summarize_output(tool_output)}\n"
                "Please output only JSON."
            )
            resp = model.invoke(prompt)
            # 解析
            if isinstance(resp, dict):
                data = resp
            else:
                text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
                # 提取首个大括号 JSON
                try:
                    # 简单查找第一个 '{' 到最后一个 '}' 的片段
                    s = text.find("{")
                    e = text.rfind("}")
                    snippet = text[s : e + 1] if s != -1 and e != -1 and e > s else text
                    data = json.loads(snippet)
                except Exception:
                    data = {
                        "status": "uncertain",
                        "confidence": 0.0,
                        "rationale": "模型返回非 JSON，已降级为不确定。",
                        "recommended_actions": ["手动复核或重试工具"],
                        "raw": text,
                    }
            # 兜底字段
            status = str(data.get("status", "uncertain")).lower()
            if status not in {"valid", "invalid", "uncertain"}:
                status = "uncertain"
            confidence = data.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            data["status"] = status
            data["confidence"] = confidence
            data.setdefault("rationale", "")
            data.setdefault("recommended_actions", [])
            return data
        except Exception as e:  # noqa: BLE001
            return {
                "status": "uncertain",
                "confidence": 0.0,
                "rationale": f"reasoning_exception: {e}",
                "recommended_actions": ["手动复核或重试工具"],
            }

    async def _prompt_tool_retry(
        self,
        tool_info: Dict[str, Any],
        current_args: Optional[Any] = None,
        error_message: Optional[str] = None,
        step_id: Optional[str] = None,
        task_description: Optional[str] = None,
        call_entry: Optional[Dict[str, Any]] = None,
        tools_called: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        step_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """通过SSE提示用户调整参数后重试工具。兼容不同调用场景。"""
        if not self.sse_streamer:
            return {"action": "abort"}

        # 聚合基础信息
        args_for_retry = current_args
        if args_for_retry is None and call_entry:
            args_for_retry = call_entry.get("args")
        if args_for_retry is None:
            args_for_retry = tool_info.get("args") or {}
        error_msg = error_message or (call_entry.get("error") if call_entry else "Tool execution failed")
        service_id = (
            tool_info.get("service_id")
            or (call_entry.get("service_id") if call_entry else None)
            or tool_info.get("serviceId")
        )
        timestamp = str(self._get_timestamp())
        session_id = None
        try:
            session_id = self.config.get("configurable", {}).get("session_id")
        except Exception:
            session_id = None
        composite_event_name = ":".join(
            str(part) for part in [session_id or "no-session", step_id or tool_info.get("tool_name"), timestamp]
        )
        serialized_schema = self._serialize_schema(tool_info.get("args_schema"))

        tool_info_payload: Dict[str, Any] = {
            "name": tool_info.get("tool_name"),
            "description": tool_info.get("description"),
            "args_schema": serialized_schema or tool_info.get("args_schema"),
            "service_id": service_id,
            "args": deepcopy(args_for_retry),
        }

        action_data: Dict[str, Any] = {
            "type": "tool_action_request",
            "mode": "retry",
            "error": error_msg,
            "tool_name": tool_info.get("tool_name"),
            "tool_args": deepcopy(args_for_retry),
            "tool_info": tool_info_payload,
            "description": task_description or tool_info.get("description", ""),
            "serviceId": service_id,
            "stepId": step_id,
            "task": task_description,
            "timestamp": timestamp,
            "action_id": timestamp,
        }

        # 广播错误上下文，方便前端展示
        execution_error_payload: Dict[str, Any] = {
            "type": "tool_error_retry",
            "planId": self.plan_id,
            "stepId": step_id,
            "toolName": tool_info.get("tool_name"),
            "error": error_msg,
            "description": task_description,
            "toolArgs": deepcopy(args_for_retry),
            "actionRequest": action_data,
        }
        if tools_called is not None:
            execution_error_payload["toolCalls"] = json.loads(json.dumps(tools_called, default=str))
        if tool_results is not None:
            execution_error_payload["toolResults"] = json.loads(json.dumps(tool_results, default=str))
        if step_context is not None:
            execution_error_payload["stepContext"] = json.loads(json.dumps(step_context, default=str))

        try:
            if hasattr(self.sse_streamer, "send_execution_error"):
                self.sse_streamer.send_execution_error(execution_error_payload)
        except Exception as notify_err:  # noqa: BLE001
            pass

        # Push action request and wait for user feedback
        self.sse_streamer.push_action_request({**action_data, "event_name": composite_event_name, "session_id": session_id})
        response = await self.sse_streamer.wait_for_action_response(
            timeout=600,
            event_name=composite_event_name,
            session_id=session_id,
        )
        return response or {"action": "abort"}

    async def _prompt_reasoning_decision(
        self,
        tool_info: Dict[str, Any],
        current_args: Optional[Any] = None,
        reasoning_payload: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Any] = None,
        step_id: Optional[str] = None,
        task_description: Optional[str] = None,
        call_entry: Optional[Dict[str, Any]] = None,
        tools_called: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Prompt user to choose retry tool or continue process when reasoning evaluation fails."""
        if not self.sse_streamer:
            return {"action": "continue"}

        # 聚合基础信息
        args_for_retry = current_args
        if args_for_retry is None and call_entry:
            args_for_retry = call_entry.get("args")
        if args_for_retry is None:
            args_for_retry = tool_info.get("args") or {}
        
        reasoning_status = (reasoning_payload or {}).get("status", "uncertain")
        reasoning_rationale = (reasoning_payload or {}).get("rationale", "Evaluation failed")
        reasoning_confidence = (reasoning_payload or {}).get("confidence", 0.0)
        recommended_actions = (reasoning_payload or {}).get("recommended_actions", [])
        
        service_id = (
            tool_info.get("service_id")
            or (call_entry.get("service_id") if call_entry else None)
            or tool_info.get("serviceId")
        )
        timestamp = str(self._get_timestamp())
        session_id = None
        try:
            session_id = self.config.get("configurable", {}).get("session_id")
        except Exception:
            session_id = None
        composite_event_name = ":".join(
            str(part) for part in [session_id or "no-session", step_id or tool_info.get("tool_name"), "reasoning", timestamp]
        )
        serialized_schema = self._serialize_schema(tool_info.get("args_schema"))

        tool_info_payload: Dict[str, Any] = {
            "name": tool_info.get("tool_name"),
            "description": tool_info.get("description"),
            "args_schema": serialized_schema or tool_info.get("args_schema"),
            "service_id": service_id,
            "args": deepcopy(args_for_retry),
        }

        action_data: Dict[str, Any] = {
            "type": "reasoning_decision_request",
            "mode": "reasoning_failed",
            "tool_name": tool_info.get("tool_name"),
            "tool_args": deepcopy(args_for_retry),
            "tool_info": tool_info_payload,
            "description": task_description or tool_info.get("description", ""),
            "serviceId": service_id,
            "stepId": step_id,
            "task": task_description,
            "timestamp": timestamp,
            "action_id": timestamp,
            "reasoning": {
                "status": reasoning_status,
                "confidence": reasoning_confidence,
                "rationale": reasoning_rationale,
                "recommended_actions": recommended_actions,
            },
        }

        # 广播reasoning决策请求
        reasoning_decision_payload: Dict[str, Any] = {
            "type": "reasoning_decision",
            "planId": self.plan_id,
            "stepId": step_id,
            "toolName": tool_info.get("tool_name"),
            "reasoning": reasoning_payload or {},
            "description": task_description,
            "toolArgs": deepcopy(args_for_retry),
            "actionRequest": action_data,
        }
        if tools_called is not None:
            reasoning_decision_payload["toolCalls"] = json.loads(json.dumps(tools_called, default=str))
        if tool_results is not None:
            reasoning_decision_payload["toolResults"] = json.loads(json.dumps(tool_results, default=str))

        try:
            if hasattr(self.sse_streamer, "send_execution_progress"):
                self.sse_streamer.send_execution_progress(reasoning_decision_payload)
        except Exception as notify_err:  # noqa: BLE001
            pass

        # Push action request and wait for user feedback
        self.sse_streamer.push_action_request({**action_data, "event_name": composite_event_name, "session_id": session_id})
        response = await self.sse_streamer.wait_for_action_response(
            timeout=600,
            event_name=composite_event_name,
            session_id=session_id,
        )
        return response or {"action": "continue"}

    def _detect_tool_error(self, tool_result: Any) -> Optional[str]:
        """Detect error information embedded inside tool result payloads."""
        if tool_result is None:
            return None

        if isinstance(tool_result, str):
            return self._find_error_line(tool_result)

        if isinstance(tool_result, dict):
            status_value = tool_result.get("status")
            if isinstance(status_value, str) and status_value.lower() in {
                "error",
                "failed",
                "failure",
                "not_found",
                "fail",
            }:
                candidate = (
                    tool_result.get("message")
                    or tool_result.get("error")
                    or tool_result.get("detail")
                )
                if isinstance(candidate, str):
                    detected = self._find_error_line(candidate)
                    return detected or candidate.strip()
                return self._find_error_line(
                    self._stringify_result_payload(tool_result)
                )

            for key in ("error", "errors", "exception", "detail", "message"):
                if key not in tool_result:
                    continue
                value = tool_result.get(key)
                detected = self._detect_tool_error(value)
                if detected:
                    return detected
            return None

        if isinstance(tool_result, (list, tuple, set)):
            for item in tool_result:
                detected = self._detect_tool_error(item)
                if detected:
                    return detected
            return None

        text_blob = self._stringify_result_payload(tool_result)
        return self._find_error_line(text_blob)

    def _stringify_result_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, bytes):
            try:
                return payload.decode("utf-8")
            except Exception:
                return payload.decode("utf-8", errors="ignore")
        if isinstance(payload, (list, tuple, set)):
            try:
                return "\n".join(
                    self._stringify_result_payload(item) for item in payload
                )
            except Exception:
                return str(payload)
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            try:
                return str(payload)
            except Exception:
                return ""

    def _find_error_line(self, text: str) -> Optional[str]:
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = [text.strip()]

        for line in lines:
            for pattern in self._EMBEDDED_ERROR_PATTERNS:
                if pattern.search(line):
                    return line

        combined = " ".join(lines)
        for pattern in self._EMBEDDED_ERROR_PATTERNS:
            if pattern.search(combined):
                return combined
        return None

    def _create_tool_call_entry(
        self,
        tool_name: str,
        args: Any,
        status: str = "running",
        service_id: str | None = None,
    ) -> Dict[str, Any]:
        self._tool_call_sequence += 1
        entry = {
            "call_id": f"tool_call_{self._tool_call_sequence}",
            "tool_name": tool_name,
            "args": args,
            "status": status,
            "started_at": time.time(),
            "service_id": service_id,
        }
        if status != "running":
            entry["finished_at"] = entry["started_at"]
            entry["duration"] = 0.0
        return entry

    def _serialize_schema(self, schema: Any) -> Any:
        if schema is None:
            return None
        try:
            if hasattr(schema, "model_json_schema"):
                return schema.model_json_schema()
            if hasattr(schema, "json"):
                return json.loads(schema.json())
        except Exception as err:  # noqa: BLE001
            pass
        if isinstance(schema, dict):
            return schema
        try:
            return json.loads(json.dumps(schema, default=str))
        except Exception:
            return None

    async def _emit_tool_progress(
        self,
        progress_hook: Optional[Any],
        tools_called: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        latest_call: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> None:
        if not progress_hook:
            return
        payload = {
            "tools_called": deepcopy(tools_called),
            "tool_results": deepcopy(tool_results),
            "status": latest_call.get("status") if latest_call else None,
        }
        if message:
            payload["message"] = message
        try:
            maybe_awaitable = progress_hook(payload)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        except Exception as hook_error:  # noqa: BLE001
            pass

    async def _execute_tool_with_retries(
        self,
        tool_info: Dict[str, Any],
        call_entry: Dict[str, Any],
        tools_called: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        thread_config: Dict[str, Any],
        progress_hook: Optional[Any],
        step_context: Dict[str, Any],
        task_description: str,
        step_info: Dict[str, Any],
        step_id: str,
    ) -> Optional[Dict[str, Any]]:
        current_args = tool_info.get("args")

        while True:
            call_entry["args"] = current_args
            call_entry["started_at"] = time.time()
            resume_payload = {"accept": True}
            if current_args is not None:
                resume_payload["args"] = current_args

            try:
                result = self.agent.invoke(
                    Command(resume=json.dumps(resume_payload)),
                    thread_config,
                )
            except Exception as call_error:  # noqa: BLE001
                call_entry["status"] = "failed"
                call_entry["error"] = str(call_error)
                call_entry["finished_at"] = time.time()
                call_entry["duration"] = call_entry["finished_at"] - call_entry["started_at"]
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                    message=str(call_error),
                )

                retry_response = await self._prompt_tool_retry(
                    tool_info=tool_info,
                    call_entry=call_entry,
                    tools_called=tools_called,
                    tool_results=tool_results,
                    step_context=step_context,
                    task_description=task_description,
                    step_id=step_id,
                )

                if not retry_response:
                    return None

                action = (retry_response.get("action") or retry_response.get("type") or "").lower()

                if action == "retry":
                    override_args = (
                        retry_response.get("args")
                        or retry_response.get("toolArgs")
                        or retry_response.get("parameters")
                    )
                    if override_args is not None:
                        current_args = override_args
                        tool_info["args"] = override_args

                    call_entry.setdefault("attempts", 1)
                    call_entry["attempts"] += 1
                    call_entry.pop("error", None)
                    call_entry["status"] = "retrying"
                    await self._emit_tool_progress(
                        progress_hook,
                        tools_called,
                        tool_results,
                        call_entry,
                        message="Retrying tool with updated parameters",
                    )
                    continue

                if action == "skip":
                    call_entry["status"] = "skipped"
                    await self._emit_tool_progress(
                        progress_hook,
                        tools_called,
                        tool_results,
                        call_entry,
                        message="Tool execution skipped by user",
                    )
                    return None

                if action in {"abort", "cancel", "reject"}:
                    raise RuntimeError("Tool execution aborted by user")

                return None

            else:
                call_entry["finished_at"] = time.time()
                call_entry["duration"] = call_entry["finished_at"] - call_entry["started_at"]
                call_entry["status"] = "completed"
                await self._emit_tool_progress(
                    progress_hook,
                    tools_called,
                    tool_results,
                    call_entry,
                )
                return result

    def _parse_tool_call(self, interrupt_value: str) -> Optional[Dict[str, Any]]:
        """Parse tool call information"""
        try:
            if "call tool:" in interrupt_value and "with" in interrupt_value:
                parts = interrupt_value.split("call tool:")[1].split("with")
                tool_name = parts[0].strip()
                args_str = parts[1].strip()

                # Parse parameters
                import ast

                try:
                    args = ast.literal_eval(args_str)
                except:
                    try:
                        args = json.loads(args_str.replace("'", '"'))
                    except:
                        args = {}

                return {"tool_name": tool_name, "args": args}
        except Exception as e:
            pass
        return None

    def _get_complete_parameters(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get complete tool parameters (including default values)"""
        tool_name = tool_info.get("tool_name", "")
        current_args = tool_info.get("args", {})

        # Find tool object
        tool_obj = None
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                tool_obj = tool
                break

        if not tool_obj:
            return tool_info

        # Get all parameters
        all_args = current_args.copy()
        serialized_schema = None
        if hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
            schema = tool_obj.args_schema
            if hasattr(schema, "model_fields"):
                for field_name, field_info in schema.model_fields.items():
                    if field_name not in all_args:
                        # Get default value
                        default_value = None
                        if (
                            hasattr(field_info, "default")
                            and field_info.default is not None
                            and field_info.default != ...
                        ):
                            default_value = field_info.default
                        elif hasattr(field_info, "annotation"):
                            annotation = field_info.annotation
                            if (
                                hasattr(annotation, "__origin__")
                                and annotation.__origin__ is Union
                            ):
                                args = annotation.__args__
                                if len(args) == 2 and type(None) in args:
                                    default_value = None

                        all_args[field_name] = default_value
            serialized_schema = self._serialize_schema(schema)

        metadata: Dict[str, Any] = {
            "tool_name": tool_name,
            "args": all_args,
        }
        metadata["structured"] = isinstance(tool_obj, StructuredTool)
        if serialized_schema:
            metadata["args_schema"] = serialized_schema
        description = getattr(tool_obj, "description", None)
        if description:
            metadata["description"] = description
        service_id = getattr(tool_obj, "service_id", None)
        if service_id:
            metadata["service_id"] = service_id
        metadata["name"] = getattr(tool_obj, "name", tool_name)

        return metadata

    async def _get_user_confirmation(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get user confirmation - 支持SSE模式、UI模式和Terminal模式"""

        # SSE交互模式（优先级最高）
        if self.sse_streamer:
            return await self._get_sse_confirmation(tool_info)

        # UI交互模式
        if self.ui_interaction_mode and self.ui_callback:
            return self._get_ui_confirmation(tool_info)

        # Terminal交互模式（原有逻辑）
        return self._get_terminal_confirmation(tool_info)

    async def _get_sse_confirmation(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """SSE模式下的用户确认 - 通过SSE推送action请求到前端"""
        try:
            # 构建action信息，通过SSE推送到前端
            action_data = {
                "type": "tool_action_request",
                "tool_name": tool_info["tool_name"],
                "tool_args": tool_info["args"],
                "tool_info": tool_info,
                "args_schema": tool_info.get("args_schema"),
                "description": tool_info.get("description", ""),
                "timestamp": tool_info.get("timestamp") or self._get_timestamp()
            }
            
            # 通过SSE推送action信息
            action_event_id = str(action_data.get("timestamp") or self._get_timestamp())
            action_data["timestamp"] = action_event_id
            # 复合键：session_id + event_id
            session_id = None
            try:
                session_id = self.config.get("configurable", {}).get("session_id")
            except Exception:
                session_id = None
            composite_event_name = f"{(session_id or 'no-session')}:{action_event_id}"
            self.sse_streamer.push_action_request({**action_data, "event_name": composite_event_name, "session_id": session_id})
            
            # 关键修复：等待前端响应，必须等待真实的用户输入
            print(f"[_get_sse_confirmation] ⏳ 等待前端响应，event_name: {composite_event_name}, session_id: {session_id}")
            print(f"[_get_sse_confirmation] ⏳ 工具: {tool_info.get('tool_name')}, 参数: {tool_info.get('args')}")
            user_response = await self.sse_streamer.wait_for_action_response(timeout=600, event_name=composite_event_name, session_id=session_id)
            
            if user_response:
                print(f"[_get_sse_confirmation] ✅ 收到前端响应: {user_response}")
                # 转换前端响应格式为内部格式
                return self._convert_sse_response_to_internal(user_response, tool_info)
            else:
                # 关键修复：如果没有收到响应（超时或错误），返回 reject 而不是 accept
                print(f"[_get_sse_confirmation] ⚠️ 未收到前端响应（超时或错误），返回 reject")
                return {"action": "reject"}
                
        except Exception as e:
            return {"action": "reject"}

    def _convert_sse_response_to_internal(self, sse_response: Dict[str, Any], tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """将SSE响应转换为内部格式"""
        response_type = sse_response.get("type", "response")
        
        if response_type == "accept":
            return {"action": "accept"}
        elif response_type == "edit":
            # 编辑参数
            edit_args = sse_response.get("args", {})
            if isinstance(edit_args, dict) and "args" in edit_args:
                modified_args = edit_args["args"]
                return {"action": "accept", "modified_args": modified_args}
            else:
                return {"action": "accept"}
        elif response_type == "retry":
            # 重试工具（带参数）
            retry_args = sse_response.get("args")
            if retry_args:
                return {"action": "accept", "modified_args": retry_args}
            else:
                return {"action": "accept"}
        elif response_type == "response":
            # 文本回复，视为拒绝
            return {"action": "reject"}
        else:
            # 默认拒绝
            return {"action": "reject"}

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_ui_confirmation(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """UI模式下的用户确认 - 通过回调函数实现"""
        try:
            # 构造交互请求数据
            interaction_data = {
                "type": "tool_confirmation",
                "tool_name": tool_info["tool_name"],
                "parameters": tool_info["args"],
                "options": [
                    {"id": "1", "text": "确认执行", "action": "accept"},
                    {"id": "2", "text": "修改参数", "action": "modify"},
                    {"id": "3", "text": "拒绝执行", "action": "reject"},
                    {"id": "4", "text": "跳过任务", "action": "skip"},
                ],
            }

            # 调用UI回调函数，等待用户响应
            response = self.ui_callback(interaction_data)

            if response.get("action") == "accept":
                modified_args = response.get("modified_args")
                if modified_args:
                    return {"action": "accept", "modified_args": modified_args}
                return {"action": "accept"}
            elif response.get("action") == "modify":
                # 如果是修改参数，返回修改后的参数
                modified_args = response.get("modified_args", tool_info["args"])
                return {"action": "accept", "modified_args": modified_args}
            elif response.get("action") == "reject":
                return {"action": "reject"}
            elif response.get("action") == "skip":
                return {"action": "skip"}
            else:
                # 默认跳过
                return {"action": "skip"}

        except Exception as e:
            # 出错时默认跳过
            return {"action": "skip"}

    def _get_terminal_confirmation(self, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Terminal模式下的用户确认 - 原有逻辑"""
        print(f"\n{'=' * 50}")
        print(f"Tool Call Confirmation")
        print(f"{'=' * 50}")
        print(f"Tool: {tool_info['tool_name']}")
        print(f"Parameters:")
        for key, value in tool_info["args"].items():
            print(f"  {key}: {value}")

        print(f"\nChoose action:")
        print(f"  1. Confirm execution")
        print(f"  2. Modify parameters")
        print(f"  3. Reject execution")
        print(f"  4. Skip task")

        while True:
            try:
                choice = input("Please select (1-4): ").strip()

                if choice == "1":
                    return {"action": "accept"}

                elif choice == "2":
                    print(f"\nCurrent parameters:")
                    for key, value in tool_info["args"].items():
                        print(f"  {key}: {value}")

                    modified_input = input(
                        "\nEnter modified parameters (JSON format): "
                    ).strip()
                    if modified_input:
                        try:
                            modified_args = json.loads(modified_input)
                            final_args = tool_info["args"].copy()
                            final_args.update(modified_args)
                            return {"action": "accept", "modified_args": final_args}
                        except json.JSONDecodeError:
                            print("Error: Invalid JSON format, please re-enter")
                            continue
                    return {"action": "accept"}

                elif choice == "3":
                    return {"action": "reject"}

                elif choice == "4":
                    return {"action": "skip"}

                else:
                    print("Invalid selection, please enter 1-4")

            except KeyboardInterrupt:
                return {"action": "skip"}
            except Exception as e:
                print(f"Error: Input error: {e}")

    def _extract_tool_result(self, agent_result: Dict[str, Any], tool_name: str) -> Any:
        """
        提取工具方法的真实返回结果

        Args:
            agent_result: Agent执行后的结果
            tool_name: 工具名称

        Returns:
            工具的真实返回结果，如果未找到则返回None
        """
        try:
            # 从agent结果中提取消息列表
            if "messages" in agent_result and agent_result["messages"]:
                messages = agent_result["messages"]

                # 遍历消息，查找ToolMessage类型的消息
                for message in reversed(messages):  # 从最新的消息开始查找
                    # 检查是否是工具消息
                    if hasattr(message, "type") and message.type == "tool":
                        # 检查工具名称是否匹配
                        if hasattr(message, "name") and message.name == tool_name:
                            # 返回工具的内容结果
                            if hasattr(message, "content"):
                                return message.content

                    # 也检查是否有tool_calls相关的结果
                    elif hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if tool_call.get("name") == tool_name:
                                # 查找对应的工具响应
                                for response_msg in messages:
                                    if hasattr(
                                        response_msg, "tool_call_id"
                                    ) and response_msg.tool_call_id == tool_call.get(
                                        "id"
                                    ):
                                        return (
                                            response_msg.content
                                            if hasattr(response_msg, "content")
                                            else None
                                        )

            # 如果没有找到特定的工具结果，返回None
            return None

        except Exception as e:
            return None

    def _display_tool_result_via_ui(self, tool_result_data: Dict[str, Any]):
        """
        通过UI回调显示工具执行结果

        Args:
            tool_result_data: 包含工具名称、参数和结果的字典
        """
        try:
            # 构造工具结果显示数据
            result_display_data = {
                "type": "tool_result",
                "tool_name": tool_result_data.get("tool_name", "Unknown Tool"),
                "args": tool_result_data.get("args", {}),
                "result": tool_result_data.get("result", "No result"),
                "status": "completed",
            }

            # 调用UI回调函数显示结果
            self.ui_callback(result_display_data)

        except Exception as e:
            pass

    async def execute_tasks(
        self, 
        task_descriptions: List[str],
        initial_file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute task list"""
        if not self.agent:
            await self.initialize_agent()

        # 初始化CSV结果收集器
        if self.csv_collector is None:
            from usecases.immunity.common.csv_result_collector import CSVResultCollector
            self.csv_collector = CSVResultCollector()
            # 如果提供了初始文件路径，直接使用它；否则创建新的空CSV文件
            if initial_file_path:
                self.csv_collector.merged_csv_path = initial_file_path
                self.csv_collector._initialized = True
            else:
                await self.csv_collector.initialize()

        results = []
        total_steps = len(task_descriptions)
        for i, task_desc in enumerate(task_descriptions, 1):
            print(f"\nExecuting task {i}/{len(task_descriptions)}: {task_desc[:60]}...")
            step_info: Dict[str, Any] = {}
            if 0 <= i - 1 < len(self.plan_steps):
                step_candidate = self.plan_steps[i - 1]
                if isinstance(step_candidate, PlanStep):
                    step_info = step_candidate.model_dump()
                elif isinstance(step_candidate, dict):
                    step_info = step_candidate
            step_id = step_info.get("step_id") or f"step-{i}"
            planned_tools = step_info.get("tools") or step_info.get("toolchain") or []
            if self.sse_streamer:
                try:
                    self.sse_streamer.send_execution_progress(
                        {
                            "planId": self.plan_id,
                            "stepId": step_id,
                            "index": i,
                            "total": total_steps,
                            "status": "running",
                            "description": task_desc,
                            "step": step_info,
                            "plannedTools": planned_tools,
                        }
                    )
                except Exception as progress_err:
                    pass

            async def progress_hook(update: Dict[str, Any]):
                if not self.sse_streamer:
                    return
                payload = {
                    "planId": self.plan_id,
                    "stepId": step_id,
                    "index": i,
                    "total": total_steps,
                    "status": update.get("status") or "running",
                    "description": task_desc,
                    "step": step_info,
                    "plannedTools": planned_tools,
                    "toolsCalled": update.get("tools_called"),
                    "toolResults": update.get("tool_results"),
                }
                if update.get("message"):
                    payload["message"] = update["message"]
                self.sse_streamer.send_execution_progress(payload)
                if hasattr(self.sse_streamer, "persist_plan_state"):
                    await self.sse_streamer.persist_plan_state(
                        execution_state={
                            step_id: {
                                "status": payload["status"],
                                "description": task_desc,
                                "plannedTools": planned_tools,
                                "toolsCalled": payload.get("toolsCalled"),
                                "toolResults": payload.get("toolResults"),
                                "message": payload.get("message"),
                            }
                        }
                    )

            try:
                # 确定当前任务应该使用的文件路径
                # 优先级：1. CSV收集器的merged_csv_path（如果已初始化且有值） 2. initial_file_path（仅第一个任务）
                current_file_path = None
                if self.csv_collector and self.csv_collector.merged_csv_path:
                    # 使用CSV收集器中的合并文件路径（这是累积的结果，包含之前所有任务的输出）
                    current_file_path = self.csv_collector.merged_csv_path
                elif i == 1 and initial_file_path:
                    # 第一个任务且没有合并文件时，使用初始文件路径
                    current_file_path = initial_file_path
                
                result = await self.execute_task(
                    task_desc,
                    progress_hook=progress_hook,
                    step_context=step_info,
                    step_id=step_id,
                    initial_file_path=current_file_path,
                )
                success = result.get("success", False)
                
                # After task execution, check if CSV collector has updated merged file path
                if self.csv_collector and self.csv_collector.merged_csv_path:
                    updated_path = self.csv_collector.merged_csv_path
                    if updated_path != current_file_path:
                        # This updated path will be used in the next task
                        pass
            except Exception as task_error:  # noqa: BLE001
                success = False
                raw_error = str(task_error)
                result = {
                    "success": False,
                    "result": raw_error,
                    "tools_called": [],
                    "tool_results": [],
                }
                if self.sse_streamer:
                    try:
                        self.sse_streamer.send_execution_error(
                            {
                                "type": "execution-error",
                                "planId": self.plan_id,
                                "stepId": step_id,
                                "error": raw_error,
                                "description": task_desc,
                            }
                        )
                        if hasattr(self.sse_streamer, "persist_plan_state"):
                            await self.sse_streamer.persist_plan_state(
                                execution_state=
                                {
                                    step_id:
                                    {
                                        "status": "failed",
                                        "description": task_desc,
                                        "plannedTools": planned_tools,
                                        "message": raw_error,
                                    }
                                }
                            )
                    except Exception as err:  # noqa: BLE001
                        pass

            if self.sse_streamer:
                try:
                    if success:
                        validation_outcome = await self._validate_step_result(
                            step_info, result
                        )
                        if not validation_outcome.get("ok", True):
                            reason = validation_outcome.get(
                                "reason", "Result did not meet downstream requirements"
                            )
                            error_payload = {
                                "planId": self.plan_id,
                                "stepId": step_id,
                                "index": i,
                                "total": total_steps,
                                "status": "needs_review",
                                "description": task_desc,
                                "step": step_info,
                                "plannedTools": planned_tools,
                                "toolsCalled": result.get("tools_called", []),
                                "toolResults": result.get("tool_results", []),
                                "message": reason,
                                "validation": validation_outcome,
                                "options": [
                                    {"action": "approve", "label": "Accept result"},
                                    {"action": "retry", "label": "Retry current tool"},
                                    {"action": "switch_tool", "label": "Switch to another tool"},
                                    {"action": "provide_input", "label": "Provide alternative output"},
                                ],
                            }
                            self.sse_streamer.send_execution_error(error_payload)
                            intervention_request = {
                                "type": "step_intervention_request",
                                "planId": self.plan_id,
                                "stepId": step_id,
                                "description": task_desc,
                                "validation": validation_outcome,
                                "tools": planned_tools,
                                "timestamp": self._get_timestamp(),
                            }
                            intervention_event_id = str(intervention_request.get("timestamp") or self._get_timestamp())
                            intervention_request["timestamp"] = intervention_event_id
                            self.sse_streamer.push_action_request(intervention_request)
                            # 使用复合键
                            session_id = None
                            try:
                                session_id = config.get("configurable", {}).get("session_id")  # type: ignore[attr-defined]
                            except Exception:
                                session_id = None
                            composite_event_name = f"{(session_id or 'no-session')}:{intervention_event_id}"
                            # 推送带 session_id 的请求
                            try:
                                self.sse_streamer.push_action_request({**intervention_request, "event_name": composite_event_name, "session_id": session_id})
                            except Exception:
                                pass
                            intervention_response = await self.sse_streamer.wait_for_action_response(
                                timeout=600,
                                event_name=composite_event_name,
                                session_id=session_id,
                            )
                            result["validation"] = {
                                "outcome": validation_outcome,
                                "user_response": intervention_response,
                            }
                        self.sse_streamer.send_execution_progress(
                            {
                                "planId": self.plan_id,
                                "stepId": step_id,
                                "index": i,
                                "total": total_steps,
                                "status": "completed",
                                "description": task_desc,
                                "step": step_info,
                                "plannedTools": planned_tools,
                                "toolsCalled": result.get("tools_called", []),
                                "toolResults": result.get("tool_results", []),
                                "output": result.get("result"),
                            }
                        )
                    else:
                        error_payload = {
                            "planId": self.plan_id,
                            "stepId": step_id,
                            "index": i,
                            "total": total_steps,
                            "status": "failed",
                            "description": task_desc,
                            "step": step_info,
                            "plannedTools": planned_tools,
                            "toolsCalled": result.get("tools_called", []),
                            "toolResults": result.get("tool_results", []),
                            "message": result.get("result", "Execution failed"),
                            "options": [
                                {"action": "retry", "label": "Retry current tool"},
                                {"action": "switch_tool", "label": "Choose alternative tool"},
                                {"action": "provide_input", "label": "Provide results manually"},
                            ],
                        }
                        self.sse_streamer.send_execution_error(error_payload)
                except Exception as progress_err:
                    pass

            results.append(result)

        # 获取最终合并的CSV文件路径
        merged_csv_path = None
        if self.csv_collector:
            merged_csv_path = self.csv_collector.get_merged_csv_path()
            if merged_csv_path:
                if merged_csv_path:
                    print(f"\nAll tasks completed. Final merged CSV: {merged_csv_path}")
        
        # 将合并的CSV路径添加到结果中
        if merged_csv_path:
            for result in results:
                result["merged_csv_path"] = merged_csv_path

        return results

    async def _validate_step_result(
        self, step_info: Dict[str, Any], execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Placeholder for post-step validation logic.

        Returns:
            dict: {"ok": True} if the result is acceptable.
                  When future LLM validation is added, return {"ok": False, "reason": "..."} to trigger intervention.
        """
        return {"ok": True, "reason": "", "details": {}}

    def _handle_post_tool_execution(self, tool_result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Post-process tool results for side effects such as GEO downloads."""
        if not tool_result_data:
            return None

        tool_name = tool_result_data.get("tool_name")
        result_payload = tool_result_data.get("result")

        if tool_name == "download_geo_sequences" and result_payload:
            try:
                download_info = download_geo_dataset(
                    result_payload,
                    cache=self._geo_download_cache,
                )
                if download_info:
                    println = f"[TaskExecutor] GEO dataset fetched to {download_info.get('destination')}"
                    print(
                        println
                    )
                    try:
                        configurable = self.config.setdefault("configurable", {})
                    except AttributeError:
                        configurable = {}
                        try:
                            self.config["configurable"] = configurable
                        except Exception:  # noqa: BLE001
                            configurable = {}
                    geo_registry = configurable.setdefault("geo_downloads", {})
                    geo_key = (
                        download_info.get("geo_id")
                        or download_info.get("ftp_url")
                        or f"geo_{len(geo_registry) + 1}"
                    )
                    geo_registry[geo_key] = download_info
                    return {"download_info": download_info}
            except Exception as geo_error:  # noqa: BLE001
                pass
        return None


async def task_decomposition_node(
    state: ImprovedCellState, config: RunnableConfig
) -> ImprovedCellState:
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

    try:
        # Get refine_plan
        plan = state.final_enhanced_plan

        if not plan or plan.strip() == "":
            return state
        from usecases.immunity.common.constants import get_tools_json
        from usecases.immunity.prompts.prompts import ImmunityPrompts

        # Get tools registry information
        tools_info = get_tools_json()

        # Create task extraction chain
        model = get_reasoning_model(config)
        output_parser = JsonOutputParser(pydantic_object=TaskExtractionResult)

        # Execute task extraction
        decomposed_tasks = (model | output_parser).invoke(
            ImmunityPrompts.TASK_EXTRACTION_PROMPT.format(
                plan=plan, tools_info=tools_info
            )
        )

        # JsonOutputParser returns a dictionary, not a TaskExtractionResult instance
        # Need to get the tasks list from the dictionary
        structured_tasks: List[Dict[str, Any]] = []
        if isinstance(decomposed_tasks, dict) and "tasks" in decomposed_tasks:
            tasks_list = decomposed_tasks["tasks"]

            # Extract description field from each task in the task list
            # Store in state.decomposed_tasks list
            task_descriptions = []
            for task_dict in tasks_list:
                if isinstance(task_dict, dict) and "description" in task_dict:
                    description = task_dict["description"]
                    if description and description.strip():
                        task_descriptions.append(description.strip())
                if isinstance(task_dict, dict):
                    structured_tasks.append(task_dict)
        else:
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
                pass
        state.plan_step_details = plan_steps

        return state

    except Exception as e:
        import traceback

        print(f"Error: Task decomposition failed: {e}")
        return state


async def execute_task_list(
    task_descriptions: List[str],
    config: Optional[RunnableConfig] = None,
    ui_interaction_mode: bool = False,
    ui_callback=None,
    sse_streamer=None,
    initial_file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for executing task list"""
    executor = TaskExecutor(
        config, ui_interaction_mode=ui_interaction_mode, ui_callback=ui_callback, sse_streamer=sse_streamer
    )
    if config and "configurable" in config:
        configurable = config["configurable"]
        plan_steps_payload = configurable.get("plan_steps") or []
        plan_id = configurable.get("plan_id")
        normalized_steps: List[Dict[str, Any]] = []
        for idx, step_payload in enumerate(plan_steps_payload, 1):
            if isinstance(step_payload, PlanStep):
                normalized_steps.append(step_payload.model_dump())
            elif isinstance(step_payload, dict):
                normalized_steps.append(step_payload)
        executor.plan_steps = normalized_steps
        executor.plan_id = plan_id
    return await executor.execute_tasks(task_descriptions, initial_file_path=initial_file_path)
