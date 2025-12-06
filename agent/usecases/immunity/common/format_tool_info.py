"""
工具信息格式化模块 - 为HIL提供工具参数展示功能

提供工具参数信息的提取和格式化功能，用于在人机交互中断时展示给用户。
"""

import inspect
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool


def extract_tool_info(tool: BaseTool) -> Dict[str, Any]:
    """
    从LangChain工具对象中提取完整的工具信息
    
    Args:
        tool: LangChain工具对象
        
    Returns:
        包含工具详细信息的字典
    """
    try:
        # 获取工具基本信息
        tool_info = {
            "name": tool.name,
            "description": tool.description or "无描述信息",
            "parameters": []
        }
        
        # 获取参数schema信息
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.model_json_schema()
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])
            
            # 提取每个参数的详细信息
            for param_name, param_info in properties.items():
                param_detail = {
                    "name": param_name,
                    "type": param_info.get("type", "unknown"),
                    "required": param_name in required_fields,
                    "default": param_info.get("default", None),
                    "title": param_info.get("title", param_name),
                    "description": param_info.get("description", "")
                }
                tool_info["parameters"].append(param_detail)
        
        # 尝试获取原始函数的默认值信息
        if hasattr(tool, 'func') and tool.func:
            try:
                signature = inspect.signature(tool.func)
                for param_name, param in signature.parameters.items():
                    # 更新默认值信息
                    for param_detail in tool_info["parameters"]:
                        if param_detail["name"] == param_name:
                            if param.default != inspect.Parameter.empty:
                                param_detail["default"] = param.default
                            break
            except Exception:
                pass  # 忽略签名获取失败
        
        return tool_info
        
    except Exception as e:
        # 返回基本信息作为fallback
        return {
            "name": getattr(tool, 'name', 'unknown_tool'),
            "description": getattr(tool, 'description', '工具信息获取失败'),
            "parameters": []
        }


def format_tool_info_for_hil(tool: BaseTool) -> str:
    """
    将工具信息格式化为适合HIL展示的字符串格式
    
    Args:
        tool: LangChain工具对象
        
    Returns:
        格式化后的工具信息字符串
    """
    tool_info = extract_tool_info(tool)
    
    # 构建格式化字符串
    lines = []
    lines.append(f"🔧 工具: {tool_info['name']}")
    lines.append(f"📝 描述: {tool_info['description']}")
    
    if tool_info['parameters']:
        lines.append("📋 参数列表:")
        for param in tool_info['parameters']:
            # 构建参数行
            param_type = param['type']
            required_text = "必需" if param['required'] else "可选"
            
            param_line = f"  • {param['name']} ({param_type}) - {required_text}"
            
            # 添加默认值信息
            if not param['required'] and param['default'] is not None:
                param_line += f", 默认值: {param['default']}"
            
            lines.append(param_line)
        
        # 添加详细说明部分
        has_descriptions = any(param.get('description') for param in tool_info['parameters'])
        if has_descriptions:
            lines.append("📖 详细说明:")
            for param in tool_info['parameters']:
                if param.get('description'):
                    lines.append(f"  • {param['name']}: {param['description']}")
    else:
        lines.append("📋 参数列表: 无参数")
    
    return "\n".join(lines)