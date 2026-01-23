"""
HITL 用户交互模块

提供无前端环境下的用户交互功能，支持：
1. 控制台交互（默认）
2. 回调函数（可扩展）
3. 文件交互（可选）
"""
from typing import Dict, Any, Optional, Callable
import json


# ===================== HITL 交互回调函数类型 =====================

HITLCallback = Callable[[Dict[str, Any]], Dict[str, Any]]
"""
HITL 回调函数类型

Args:
    interrupt_data: 中断数据，包含：
        - type: 中断类型（"missing_parameters" 或 "result_confirmation"）
        - requests: 请求列表
        - message: 提示消息

Returns:
    用户响应字典，格式：
    - 对于 "missing_parameters" 类型：
        {
            "type": "response_parameters",
            "responses": {
                "task_id": {
                    "parameters": {
                        "param_name": "param_value",
                        ...
                    }
                },
                ...
            }
        }
    - 对于 "result_confirmation" 类型：
        {
            "type": "response_confirmation",
            "responses": {
                "task_id": {
                    "continue": True/False
                },
                ...
            }
        }
"""


# ===================== 控制台交互实现 =====================

def console_interact_for_parameters(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    控制台交互：处理参数请求
    
    Args:
        interrupt_data: 中断数据
        
    Returns:
        用户响应字典
    """
    requests = interrupt_data.get("requests", [])
    responses = {}
    
    print(f"\n{'='*80}")
    print(f"【HITL 请求：需要提供参数】")
    print(f"{'='*80}")
    
    for request in requests:
        task_id = request.get("task_id", "unknown")
        message = request.get("message", "")
        missing_params = request.get("missing_parameters", [])
        
        print(f"\n任务ID: {task_id}")
        print(f"消息: {message}")
        print(f"缺失参数: {', '.join(missing_params)}")
        print(f"\n请为以下参数提供值（每行一个，格式：参数名=参数值，或参数名:参数值）：")
        print(f"（如果参数名包含点号，请使用完整格式，如 'tool_name.param_name'）")
        print(f"（输入 'skip' 跳过此任务，输入 'quit' 退出）")
        
        task_responses = {}
        for param in missing_params:
            # 提取参数名（可能包含工具名前缀）
            if '.' in param:
                param_name = param.split('.')[-1]  # 取最后一部分作为参数名
            else:
                param_name = param
            
            while True:
                user_input = input(f"  {param} = ").strip()
                
                if user_input.lower() == 'quit':
                    raise KeyboardInterrupt("用户退出")
                elif user_input.lower() == 'skip':
                    print(f"  跳过参数 {param}")
                    # Mark parameter as skipped (use None as value to indicate skipped)
                    task_responses[param] = None
                    break
                elif not user_input:
                    print(f"  参数 {param} 不能为空，请重新输入")
                    continue
                else:
                    # 解析输入（支持 key=value 或 key:value 格式）
                    if '=' in user_input:
                        key, value = user_input.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                    elif ':' in user_input:
                        key, value = user_input.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                    else:
                        # 直接使用输入作为值
                        key = param_name
                        value = user_input
                    
                    # 验证参数名是否匹配
                    if key != param_name and key != param:
                        print(f"  警告：参数名不匹配（期望: {param_name} 或 {param}，实际: {key}）")
                        confirm = input(f"  是否使用 '{key}' 作为参数名？(y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    task_responses[key] = value
                    break
        
        if task_responses:
            responses[task_id] = {"parameters": task_responses}
    
    print(f"{'='*80}\n")
    
    return {
        "type": "response_parameters",
        "responses": responses
    }


def console_interact_for_confirmation(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    控制台交互：处理结果确认请求
    
    Args:
        interrupt_data: 中断数据
        
    Returns:
        用户响应字典
    """
    requests = interrupt_data.get("requests", [])
    responses = {}
    
    print(f"\n{'='*80}")
    print(f"【HITL 请求：需要确认是否继续】")
    print(f"{'='*80}")
    
    for request in requests:
        task_id = request.get("task_id", "unknown")
        message = request.get("message", "")
        result = request.get("result", "")
        reason = request.get("reason", "")
        
        print(f"\n任务ID: {task_id}")
        print(f"消息: {message}")
        if reason:
            print(f"原因: {reason}")
        if result:
            print(f"执行结果: {result[:500]}...")  # 限制显示长度
        
        while True:
            user_input = input(f"\n是否继续执行后续任务？(y/n/quit): ").strip().lower()
            
            if user_input == 'quit':
                raise KeyboardInterrupt("用户退出")
            elif user_input == 'y' or user_input == 'yes':
                responses[task_id] = {"continue": True}
                break
            elif user_input == 'n' or user_input == 'no':
                responses[task_id] = {"continue": False}
                break
            else:
                print("  请输入 'y' (是)、'n' (否) 或 'quit' (退出)")
    
    print(f"{'='*80}\n")
    
    return {
        "type": "response_confirmation",
        "responses": responses
    }


def console_interact(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    控制台交互：根据中断类型调用相应的处理函数
    
    Args:
        interrupt_data: 中断数据
        
    Returns:
        用户响应字典
    """
    interrupt_type = interrupt_data.get("type", "")
    
    if interrupt_type == "missing_parameters":
        return console_interact_for_parameters(interrupt_data)
    elif interrupt_type == "result_confirmation":
        return console_interact_for_confirmation(interrupt_data)
    else:
        raise ValueError(f"未知的中断类型: {interrupt_type}")


# ===================== 文件交互实现（可选） =====================

def save_hitl_request_to_file(interrupt_data: Dict[str, Any], file_path: str = "hitl_request.json") -> str:
    """
    将 HITL 请求保存到文件
    
    Args:
        interrupt_data: 中断数据
        file_path: 文件路径
        
    Returns:
        文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(interrupt_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"【HITL 请求已保存到文件】")
    print(f"文件路径: {file_path}")
    print(f"{'='*80}\n")
    
    return file_path


def load_hitl_response_from_file(file_path: str = "hitl_response.json") -> Optional[Dict[str, Any]]:
    """
    从文件加载 HITL 响应
    
    Args:
        file_path: 文件路径
        
    Returns:
        用户响应字典，如果文件不存在则返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ===================== 通用 HITL 交互接口 =====================

def handle_hitl_interrupt(
    interrupt_data: Dict[str, Any],
    callback: Optional[HITLCallback] = None,
    use_file: bool = False,
    request_file: str = "hitl_request.json",
    response_file: str = "hitl_response.json"
) -> Dict[str, Any]:
    """
    处理 HITL 中断，获取用户响应
    
    优先级：
    1. 如果提供了 callback，使用 callback
    2. 如果 use_file=True，使用文件交互
    3. 否则，使用控制台交互
    
    Args:
        interrupt_data: 中断数据
        callback: 用户提供的回调函数（可选）
        use_file: 是否使用文件交互
        request_file: HITL 请求文件路径
        response_file: HITL 响应文件路径
        
    Returns:
        用户响应字典
    """
    # 优先级1：使用回调函数
    if callback:
        try:
            return callback(interrupt_data)
        except Exception as e:
            print(f"⚠ 回调函数执行失败: {e}，回退到控制台交互")
    
    # 优先级2：使用文件交互
    if use_file:
        # 保存请求到文件
        save_hitl_request_to_file(interrupt_data, request_file)
        
        # 等待用户编辑响应文件
        print(f"请编辑文件 {response_file} 提供响应，然后按 Enter 继续...")
        input()
        
        # 加载响应
        response = load_hitl_response_from_file(response_file)
        if response:
            return response
        else:
            print(f"⚠ 未找到响应文件 {response_file}，回退到控制台交互")
    
    # 优先级3：使用控制台交互（默认）
    return console_interact(interrupt_data)

