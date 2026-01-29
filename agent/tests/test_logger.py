"""
测试执行流程记录器

用于记录测试过程中的详细执行流程，包括：
- 初始状态
- 每个节点的执行
- 状态变化
- HITL 交互
- 代码生成和执行结果

支持将所有测试用例的日志合并到一个文件中，使用易读的格式输出。
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict


def _safe_serialize_for_json(obj: Any) -> Any:
    """
    安全地将对象序列化为可 JSON 序列化的格式
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        可序列化的对象
    """
    if obj is None:
        return None
    
    # 如果是 Interrupt 对象（通常有 value 和 id 属性）
    if hasattr(obj, 'value') and hasattr(obj, 'id'):
        return {
            "type": "Interrupt",
            "id": str(getattr(obj, 'id', '')),
            "value": _safe_serialize_for_json(getattr(obj, 'value', {}))
        }
    
    # 如果是字典，递归处理
    if isinstance(obj, dict):
        return {k: _safe_serialize_for_json(v) for k, v in obj.items()}
    
    # 如果是列表，递归处理
    if isinstance(obj, list):
        return [_safe_serialize_for_json(item) for item in obj]
    
    # 如果是其他可序列化类型，尝试直接返回
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # 如果无法序列化，转换为字符串
        return str(obj)


class TestCaseLogger:
    """单个测试用例的日志记录器"""
    __test__ = False  # 告诉 pytest 这不是测试类
    
    def __init__(self, test_case_name: str):
        self.test_case_name = test_case_name
        self.logs: List[Dict[str, Any]] = []
        self.current_step = 0
        self.start_time = datetime.now()
        self.end_time = None
        
    def log_initial_state(self, state: Any, description: str = "初始状态"):
        """记录初始状态"""
        self._log("initial_state", {
            "description": description,
            "state": self._serialize_state(state)
        })
    
    def log_node_execution(self, node_name: str, input_state: Any, output_state: Any = None, 
                          description: str = None):
        """记录节点执行"""
        self.current_step += 1
        log_entry = {
            "step": self.current_step,
            "node": node_name,
            "description": description or f"执行节点: {node_name}",
            "input_state": self._serialize_state(input_state),
            "timestamp": datetime.now().isoformat()
        }
        
        if output_state is not None:
            log_entry["output_state"] = self._serialize_state(output_state)
            log_entry["state_changes"] = self._extract_state_changes(input_state, output_state)
        
        self._log("node_execution", log_entry)
    
    def log_parameter_inference(self, task_id: str, parameters: Dict[str, Any], 
                                missing_parameters: List[str], llm_used: bool = True):
        """记录参数推断结果"""
        self._log("parameter_inference", {
            "task_id": task_id,
            "inferred_parameters": parameters,
            "missing_parameters": missing_parameters,
            "llm_used": llm_used,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_hitl_request(self, task_id: str, request_type: str, request_data: Dict[str, Any]):
        """记录 HITL 请求"""
        self._log("hitl_request", {
            "task_id": task_id,
            "type": request_type,
            "request": request_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_hitl_response(self, task_id: str, response_type: str, response_data: Dict[str, Any]):
        """记录 HITL 响应"""
        self._log("hitl_response", {
            "task_id": task_id,
            "type": response_type,
            "response": response_data,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_codeact_input(self, task_id: str, execution_mode: str, parameters: Dict[str, Any],
                         task_description: str):
        """记录传入 codeact 的值"""
        self._log("codeact_input", {
            "task_id": task_id,
            "execution_mode": execution_mode,
            "parameters": parameters,
            "task_description": task_description,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_codeact_output(self, task_id: str, generated_code: str, execution_result: Dict[str, Any]):
        """记录 codeact 生成的代码和执行结果"""
        self._log("codeact_output", {
            "task_id": task_id,
            "generated_code": generated_code,
            "execution_result": execution_result,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_task_execution(self, task_id: str, status: str, execution_mode: str,
                          result: Dict[str, Any], error: Optional[str] = None):
        """记录任务执行结果"""
        self._log("task_execution", {
            "task_id": task_id,
            "status": status,
            "execution_mode": execution_mode,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_task_order(self, task_order: List[Dict[str, Any]]):
        """记录任务调用次序"""
        self._log("task_order", {
            "order": task_order,
            "total_tasks": len(task_order),
            "timestamp": datetime.now().isoformat()
        })
    
    def log_summary(self, summary: Dict[str, Any]):
        """记录测试总结"""
        self._log("summary", {
            "summary": summary,
            "total_steps": self.current_step,
            "timestamp": datetime.now().isoformat()
        })
    
    def finish(self):
        """完成测试用例记录"""
        self.end_time = datetime.now()
    
    def _log(self, event_type: str, data: Dict[str, Any]):
        """内部日志记录方法"""
        log_entry = {
            "event_type": event_type,
            "data": data
        }
        self.logs.append(log_entry)
    
    def _serialize_state(self, state: Any) -> Dict[str, Any]:
        """序列化状态对象"""
        if state is None:
            return None
        
        if isinstance(state, dict):
            return state
        
        # 如果是 Pydantic 模型，使用 model_dump
        if hasattr(state, 'model_dump'):
            try:
                # 排除 parent_state 以避免循环引用
                return state.model_dump(exclude={'parent_state'}, mode='json')
            except:
                return state.model_dump(mode='json')
        
        # 如果是普通对象，尝试转换为字典
        if hasattr(state, '__dict__'):
            return {k: self._serialize_value(v) for k, v in state.__dict__.items() 
                   if not k.startswith('_')}
        
        return str(state)
    
    def _serialize_value(self, value: Any) -> Any:
        """递归序列化值"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if hasattr(value, 'value'):  # Enum
            return value.value
        if hasattr(value, 'model_dump'):  # Pydantic model
            try:
                return value.model_dump(exclude={'parent_state'}, mode='json')
            except:
                return value.model_dump(mode='json')
        return str(value)
    
    def _extract_state_changes(self, input_state: Any, output_state: Any) -> Dict[str, Any]:
        """提取状态变化"""
        changes = {}
        
        input_dict = self._serialize_state(input_state)
        output_dict = self._serialize_state(output_state)
        
        if not isinstance(input_dict, dict) or not isinstance(output_dict, dict):
            return changes
        
        # 比较关键字段的变化
        key_fields = [
            'task_status_map', 'task_results', 'running_tasks',
            'completed_count', 'failed_count', 'hitl_requests', 'hitl_responses'
        ]
        
        for field in key_fields:
            if field in output_dict:
                if field not in input_dict or input_dict[field] != output_dict[field]:
                    changes[field] = {
                        "before": input_dict.get(field),
                        "after": output_dict[field]
                    }
        
        return changes


class GlobalTestLogger:
    """全局测试日志记录器 - 收集所有测试用例的日志"""
    
    def __init__(self, test_file_name: str, logs_dir: Path = None):
        """
        初始化全局测试记录器
        
        Args:
            test_file_name: 测试文件名（如 test_executor_subgraph）
            logs_dir: 日志目录，默认为 tests/logs
        """
        if logs_dir is None:
            logs_dir = Path(__file__).parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 生成日志文件名：[测试文件名]_[时间戳].md
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logs_dir / f"{test_file_name}_{timestamp}.md"
        
        self.test_file_name = test_file_name
        self.test_cases: Dict[str, TestCaseLogger] = {}
        self.start_time = datetime.now()
        
    def get_test_case_logger(self, test_case_name: str) -> TestCaseLogger:
        """获取或创建测试用例记录器"""
        if test_case_name not in self.test_cases:
            self.test_cases[test_case_name] = TestCaseLogger(test_case_name)
        return self.test_cases[test_case_name]
    
    def finish_test_case(self, test_case_name: str):
        """完成测试用例记录"""
        if test_case_name in self.test_cases:
            self.test_cases[test_case_name].finish()
    
    def save(self):
        """保存所有测试用例的日志到文件（Markdown 格式）"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write(f"# 测试执行日志\n\n")
            f.write(f"**测试文件**: `{self.test_file_name}`\n\n")
            f.write(f"**开始时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**结束时间**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**总耗时**: {duration:.2f} 秒\n\n")
            f.write(f"**测试用例数**: {len(self.test_cases)}\n\n")
            f.write("---\n\n")
            
            # 写入每个测试用例的日志
            for idx, (test_case_name, test_logger) in enumerate(self.test_cases.items(), 1):
                self._write_test_case(f, test_case_name, test_logger, idx)
        
        print(f"\n{'='*60}")
        print(f"测试日志已保存: {self.log_file}")
        print(f"{'='*60}\n")
    
    def _write_test_case(self, f, test_case_name: str, test_logger: TestCaseLogger, idx: int):
        """写入单个测试用例的日志"""
        duration = None
        if test_logger.end_time:
            duration = (test_logger.end_time - test_logger.start_time).total_seconds()
        
        f.write(f"## {idx}. {test_case_name}\n\n")
        f.write(f"**开始时间**: {test_logger.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if test_logger.end_time:
            f.write(f"**结束时间**: {test_logger.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**耗时**: {duration:.2f} 秒\n\n")
        f.write(f"**总步骤数**: {test_logger.current_step}\n\n")
        f.write("---\n\n")
        
        # 记录初始状态（入参）
        initial_states = [log for log in test_logger.logs if log["event_type"] == "initial_state"]
        if initial_states:
            f.write("### 入参\n\n")
            for state_log in initial_states:
                desc = state_log["data"].get("description", "初始状态")
                f.write(f"#### {desc}\n\n")
                state = state_log["data"].get("state", {})
                f.write("```json\n")
                f.write(json.dumps(state, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
        
        # 记录任务列表
        task_orders = [log for log in test_logger.logs if log["event_type"] == "task_order"]
        if task_orders:
            f.write("### 任务列表\n\n")
            for task_log in task_orders:
                order = task_log["data"].get("order", [])
                total = task_log["data"].get("total_tasks", 0)
                f.write(f"**总任务数**: {total}\n\n")
                if order:
                    f.write("| 任务ID | 任务描述 | 依赖 |\n")
                    f.write("|--------|----------|------|\n")
                    for task in order:
                        task_id = task.get("task_id", "")
                        content = task.get("content", "")[:50]  # 限制长度
                        deps = ", ".join(task.get("dependencies", []))
                        f.write(f"| {task_id} | {content} | {deps} |\n")
                    f.write("\n")
        
        # 记录节点执行过程（简化版，只记录关键变化）
        node_executions = [log for log in test_logger.logs if log["event_type"] == "node_execution"]
        if node_executions:
            f.write("### 节点执行流程\n\n")
            for node_log in node_executions:
                data = node_log["data"]
                step = data.get("step", 0)
                node_name = data.get("node", "")
                description = data.get("description", "")
                
                f.write(f"#### 步骤 {step}: {node_name}\n\n")
                f.write(f"**描述**: {description}\n\n")
                
                # 只记录状态变化，不记录完整输入输出
                state_changes = data.get("state_changes", {})
                if state_changes:
                    f.write("**关键状态变化**:\n\n")
                    for field, change in state_changes.items():
                        before = change.get('before')
                        after = change.get('after')
                        
                        # 简化显示
                        if field == 'task_status_map':
                            f.write(f"- **任务状态变化**:\n")
                            if isinstance(after, dict):
                                for task_id, status in after.items():
                                    if isinstance(before, dict) and before.get(task_id) != status:
                                        f.write(f"  - {task_id}: {before.get(task_id, 'N/A')} → {status}\n")
                        elif field == 'task_results':
                            f.write(f"- **任务结果更新**:\n")
                            if isinstance(after, dict):
                                for task_id in after.keys():
                                    f.write(f"  - {task_id}: 结果已更新\n")
                        elif field == 'completed_count':
                            f.write(f"- **完成任务数**: {before} → {after}\n")
                        elif field == 'failed_count':
                            f.write(f"- **失败任务数**: {before} → {after}\n")
                        else:
                            # 其他字段，简化显示
                            before_str = json.dumps(before, ensure_ascii=False)[:100]
                            after_str = json.dumps(after, ensure_ascii=False)[:100]
                            f.write(f"- **{field}**: {before_str} → {after_str}\n")
                    f.write("\n")
                else:
                    f.write("*无关键状态变化*\n\n")
                
                f.write("---\n\n")
        
        # 记录参数推断
        param_inferences = [log for log in test_logger.logs if log["event_type"] == "parameter_inference"]
        if param_inferences:
            f.write("### 参数推断结果\n\n")
            for param_log in param_inferences:
                data = param_log["data"]
                task_id = data.get("task_id", "")
                inferred = data.get("inferred_parameters", {})
                missing = data.get("missing_parameters", [])
                
                f.write(f"#### 任务 {task_id}\n\n")
                f.write(f"**推断的参数**: {json.dumps(inferred, ensure_ascii=False)}\n\n")
                f.write(f"**缺失的参数**: {json.dumps(missing, ensure_ascii=False)}\n\n")
                f.write("---\n\n")
        
        # 记录 HITL 交互
        hitl_requests = [log for log in test_logger.logs if log["event_type"] == "hitl_request"]
        hitl_responses = [log for log in test_logger.logs if log["event_type"] == "hitl_response"]
        if hitl_requests or hitl_responses:
            f.write("### HITL 交互\n\n")
            
            if hitl_requests:
                f.write("#### HITL 请求\n\n")
                for hitl_log in hitl_requests:
                    data = hitl_log["data"]
                    task_id = data.get("task_id", "")
                    req_type = data.get("type", "")
                    request = data.get("request", {})
                    
                    f.write(f"**任务 {task_id}** - 类型: {req_type}\n\n")
                    f.write("```json\n")
                    # 安全序列化，确保可以 JSON 序列化
                    safe_request = _safe_serialize_for_json(request)
                    f.write(json.dumps(safe_request, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")
            
            if hitl_responses:
                f.write("#### HITL 响应\n\n")
                for hitl_log in hitl_responses:
                    data = hitl_log["data"]
                    task_id = data.get("task_id", "")
                    resp_type = data.get("type", "")
                    response = data.get("response", {})
                    
                    f.write(f"**任务 {task_id}** - 类型: {resp_type}\n\n")
                    f.write("```json\n")
                    # 安全序列化，确保可以 JSON 序列化
                    safe_response = _safe_serialize_for_json(response)
                    f.write(json.dumps(safe_response, ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")
            
            f.write("---\n\n")
        
        # 记录 CodeAct 输入输出
        codeact_inputs = [log for log in test_logger.logs if log["event_type"] == "codeact_input"]
        codeact_outputs = [log for log in test_logger.logs if log["event_type"] == "codeact_output"]
        if codeact_inputs or codeact_outputs:
            f.write("### CodeAct 执行详情\n\n")
            
            for codeact_log in codeact_inputs:
                data = codeact_log["data"]
                task_id = data.get("task_id", "")
                exec_mode = data.get("execution_mode", "")
                params = data.get("parameters", {})
                task_desc = data.get("task_description", "")
                
                f.write(f"#### 任务 {task_id} - CodeAct 输入\n\n")
                f.write(f"**执行模式**: {exec_mode}\n\n")
                f.write(f"**任务描述**: {task_desc}\n\n")
                f.write(f"**参数**: {json.dumps(params, ensure_ascii=False)}\n\n")
            
            for codeact_log in codeact_outputs:
                data = codeact_log["data"]
                task_id = data.get("task_id", "")
                code = data.get("generated_code", "")
                result = data.get("execution_result", {})
                
                f.write(f"#### 任务 {task_id} - CodeAct 输出\n\n")
                f.write("**生成的代码**:\n\n")
                f.write("```python\n")
                f.write(code)
                f.write("\n```\n\n")
                f.write("**执行结果**:\n\n")
                f.write("```json\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
            
            f.write("---\n\n")
        
        # 记录任务执行结果（重点记录）
        task_executions = [log for log in test_logger.logs if log["event_type"] == "task_execution"]
        if task_executions:
            f.write("### 任务执行结果\n\n")
            for exec_log in task_executions:
                data = exec_log["data"]
                task_id = data.get("task_id", "")
                status = data.get("status", "")
                exec_mode = data.get("execution_mode", "")
                result = data.get("result", {})
                error = data.get("error", "")
                
                f.write(f"#### 任务 {task_id}\n\n")
                f.write(f"**状态**: {status}\n\n")
                f.write(f"**执行模式**: {exec_mode}\n\n")
                
                # 提取关键结果信息
                key_result_info = self._extract_key_task_result(result)
                if key_result_info:
                    f.write("**关键结果**:\n\n")
                    
                    # 优先显示final_result（最重要的最终结果）
                    if "final_result" in key_result_info:
                        f.write("**最终结果** (final_result):\n\n")
                        f.write("```json\n")
                        f.write(json.dumps(key_result_info["final_result"], ensure_ascii=False, indent=2))
                        f.write("\n```\n\n")
                        # 移除final_result，避免重复显示
                        key_result_info_display = {k: v for k, v in key_result_info.items() if k != "final_result"}
                    else:
                        key_result_info_display = key_result_info
                    
                    # 显示其他关键信息
                    if key_result_info_display:
                        f.write("**其他关键信息**:\n\n")
                        f.write("```json\n")
                        f.write(json.dumps(key_result_info_display, ensure_ascii=False, indent=2))
                        f.write("\n```\n\n")
                
                # 如果有错误，详细记录
                if error:
                    f.write(f"**错误信息**: {error}\n\n")
                
                # 显示result消息（工具执行结果）
                if "result_messages" in result and result["result_messages"]:
                    f.write("**工具执行结果** (type: result 消息):\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(result["result_messages"], ensure_ascii=False, indent=2))
                    f.write("\n```\n\n")
                
                # 显示完整结果（但messages中只包含result消息）
                f.write("**完整执行结果** (已过滤progress消息，只保留result消息):\n\n")
                f.write("```json\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")

                # 展开每次迭代的代码内容（如果有）
                execution_summary = result.get("execution_summary", {})
                code_iterations = []
                if isinstance(execution_summary, dict):
                    code_iterations = execution_summary.get("code_iterations", []) or []
                if code_iterations:
                    f.write("**代码迭代历史**:\n\n")
                    for idx, entry in enumerate(code_iterations, 1):
                        status = entry.get("status", "unknown")
                        error_type = entry.get("error_type")
                        exec_time = entry.get("execution_time")
                        f.write(f"- 第 {idx} 次迭代: status={status}, error_type={error_type}, execution_time={exec_time}\n\n")
                        code = entry.get("code")
                        if code:
                            f.write("```python\n")
                            f.write(code)
                            f.write("\n```\n\n")
                
                f.write("---\n\n")
        
        # 记录总结
        summaries = [log for log in test_logger.logs if log["event_type"] == "summary"]
        if summaries:
            f.write("### 测试总结\n\n")
            for summary_log in summaries:
                summary = summary_log["data"].get("summary", {})
                f.write("```json\n")
                f.write(json.dumps(summary, ensure_ascii=False, indent=2))
                f.write("\n```\n\n")
        
        f.write("\n\n")
    
    def _simplify_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """简化状态，只保留关键信息"""
        if not isinstance(state, dict):
            return state
        
        simplified = {}
        key_fields = [
            'subtasks', 'task_status_map', 'task_results', 'running_tasks',
            'completed_count', 'failed_count', 'hitl_requests', 'hitl_responses',
            'total_tasks', 'max_parallel_tasks'
        ]
        
        for field in key_fields:
            if field in state:
                value = state[field]
                # Handle SubTask objects and other Pydantic models
                if isinstance(value, list):
                    simplified[field] = [
                        item.model_dump(mode='json') if hasattr(item, 'model_dump') else _safe_serialize_for_json(item)
                        for item in value
                    ]
                elif hasattr(value, 'model_dump'):
                    simplified[field] = value.model_dump(mode='json')
                else:
                    simplified[field] = _safe_serialize_for_json(value)
        
        return simplified
    
    def _extract_key_task_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取任务执行结果中的关键信息"""
        if not isinstance(result, dict):
            return {}
        
        key_info = {}
        
        # 提取状态（从result中）
        if "status" in result:
            key_info["status"] = result["status"]
        
        # 提取错误信息（优先从result中，其次从error字段）
        if "error" in result and result["error"]:
            key_info["error"] = result["error"]
        elif "error_category" in result:
            key_info["error_category"] = result["error_category"]
        
        # 提取其他关键字段
        for field in ["confidence_score", "retry_count"]:
            if field in result and result[field] is not None:
                key_info[field] = result[field]
        
        # 提取输出文件路径和关键信息
        output_files = []
        if "output" in result:
            output = result["output"]
            
            # 如果是字符串，尝试解析
            if isinstance(output, str):
                # 先尝试解析JSON
                try:
                    output = json.loads(output)
                except:
                    # 如果不是JSON，尝试从字符串中提取文件路径
                    import re
                    # 匹配文件路径模式
                    file_patterns = [
                        r'["\']([^"\']*\.(csv|fasta|json|txt|xlsx|tsv|pdb))["\']',
                        r'([/\\][^/\\\s]+\.(csv|fasta|json|txt|xlsx|tsv|pdb))',
                    ]
                    for pattern in file_patterns:
                        matches = re.findall(pattern, output)
                        for match in matches:
                            if isinstance(match, tuple):
                                file_path = match[0] if match[0] else match[1] if len(match) > 1 else None
                            else:
                                file_path = match
                            if file_path and file_path not in output_files:
                                output_files.append(file_path)
            
            # 如果是字典，提取关键信息
            if isinstance(output, dict):
                # 优先提取final_result（最重要的，包含最终结果和文件路径）
                if "final_result" in output and output["final_result"]:
                    final_result = output["final_result"]
                    if isinstance(final_result, dict):
                        # 从final_result中提取文件路径
                        for key in ["output_file", "output_file_path", "result_file", "file_path", "file", "output", "result"]:
                            if key in final_result:
                                value = final_result[key]
                                if isinstance(value, str) and any(ext in value.lower() for ext in ['.csv', '.fasta', '.json', '.txt', '.xlsx', '.tsv', '.pdb']):
                                    if value not in output_files:
                                        output_files.append(value)
                                elif isinstance(value, dict):
                                    # 如果value是字典，递归查找文件路径
                                    for sub_key in ["path", "file", "file_path"]:
                                        if sub_key in value and isinstance(value[sub_key], str):
                                            if any(ext in value[sub_key].lower() for ext in ['.csv', '.fasta', '.json', '.txt', '.xlsx', '.tsv', '.pdb']):
                                                if value[sub_key] not in output_files:
                                                    output_files.append(value[sub_key])
                        
                        # 保存final_result的关键信息
                        key_info["final_result"] = final_result
                
                # 提取任务ID和服务ID
                if "task_id" in output:
                    key_info["task_id"] = output["task_id"]
                if "service_id" in output:
                    key_info["service_id"] = output["service_id"]
                
                # 检查各种可能的输出文件字段（在output的顶层）
                for key in ["output_file", "output_file_path", "result_file", "file_path", "file", "output"]:
                    if key in output:
                        value = output[key]
                        if isinstance(value, str) and any(ext in value.lower() for ext in ['.csv', '.fasta', '.json', '.txt', '.xlsx', '.tsv', '.pdb']):
                            if value not in output_files:
                                output_files.append(value)
                
                # 检查messages中的result消息（工具执行结果）
                if "messages" in output and isinstance(output["messages"], list):
                    # 只提取type为result的消息（工具执行结果）
                    result_messages = []
                    for msg in output["messages"]:
                        if isinstance(msg, dict):
                            msg_type = msg.get("type", "")
                            
                            # 只处理type为result的消息
                            if msg_type == "result":
                                result_messages.append(msg)
                                
                                # 从result消息中提取文件路径信息
                                for field in ["output_file", "file_path", "result_file", "file", "output"]:
                                    if field in msg:
                                        value = msg[field]
                                        if isinstance(value, str) and any(ext in value.lower() for ext in ['.csv', '.fasta', '.json', '.txt', '.xlsx', '.tsv', '.pdb']):
                                            if value not in output_files:
                                                output_files.append(value)
                                
                                # 检查data字段中是否有文件路径
                                if "data" in msg and isinstance(msg["data"], dict):
                                    for field in ["output_file", "file_path", "result_file", "file", "output"]:
                                        if field in msg["data"]:
                                            value = msg["data"][field]
                                            if isinstance(value, str) and any(ext in value.lower() for ext in ['.csv', '.fasta', '.json', '.txt', '.xlsx', '.tsv', '.pdb']):
                                                if value not in output_files:
                                                    output_files.append(value)
                    
                    # 记录result消息（工具执行结果）
                    if result_messages:
                        key_info["result_messages"] = result_messages
                        key_info["result_messages_count"] = len(result_messages)
                    
                    # 也检查result_messages字段（如果已经提取过）
                    if "result_messages" in output:
                        key_info["result_messages"] = output["result_messages"]
                        key_info["result_messages_count"] = len(output["result_messages"])
            
            # 如果找到了输出文件，记录
            if output_files:
                key_info["output_files"] = list(set(output_files))  # 去重
        
        return key_info
    
    def _summarize_result(self, result: Dict[str, Any], max_length: int = None) -> Dict[str, Any]:
        """对结果进行摘要（暂时不限制长度，用于调试）"""
        # 暂时不进行任何截断，完整返回结果
        return result


# 全局日志记录器实例（在测试模块级别初始化）
_global_logger: Optional[GlobalTestLogger] = None


def init_global_logger(test_file_name: str):
    """初始化全局日志记录器"""
    global _global_logger
    _global_logger = GlobalTestLogger(test_file_name)
    return _global_logger


def get_global_logger() -> Optional[GlobalTestLogger]:
    """获取全局日志记录器"""
    return _global_logger


def save_global_logger():
    """保存全局日志记录器"""
    global _global_logger
    if _global_logger:
        _global_logger.save()
        _global_logger = None


# 为了向后兼容，保留 TestLogger 类（作为单个测试用例记录器的别名）
TestLogger = TestCaseLogger
