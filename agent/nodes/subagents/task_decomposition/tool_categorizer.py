"""
工具分类模块

对MCP工具按service进行分类，支持基于service_id的筛选。
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import sys

# 添加agent目录到路径
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


def load_service_list() -> List[Dict[str, Any]]:
    """
    加载service列表
    
    Returns:
        service列表，每个service包含 service_id 和 description
    """
    service_list_path = agent_dir / "config" / "service_list.json"
    
    try:
        if service_list_path.exists():
            with open(service_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"警告：service_list.json 不存在: {service_list_path}")
            return []
    except Exception as e:
        print(f"警告：加载service_list.json失败: {e}")
        return []


def get_tools_by_service_ids(
    all_tools: List[Dict[str, Any]], 
    required_service_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    根据需要的service_id筛选工具
    
    Args:
        all_tools: 所有工具列表
        required_service_ids: 需要的service_id列表
    
    Returns:
        筛选后的工具列表
    """
    if not required_service_ids:
        return []
    
    # 筛选工具
    selected_tools = []
    for tool in all_tools:
        tool_service = tool.get("service", "")
        if tool_service in required_service_ids:
            selected_tools.append(tool)
    
    return selected_tools


def get_service_summary(service_list: List[Dict[str, Any]]) -> str:
    """
    生成service列表摘要（用于提示词）
    
    Args:
        service_list: service列表
    
    Returns:
        格式化的service摘要文本
    """
    summary_parts = []
    for service in service_list:
        # 处理 service 可能是字符串或字典的情况
        if isinstance(service, str):
            # 如果是字符串，直接使用
            summary_parts.append(f"- {service}")
        elif isinstance(service, dict):
            service_id = service.get("service_id", "")
            description = service.get("description", "")
            summary_parts.append(f"- {service_id}: {description}")
        else:
            # 其他类型，转换为字符串
            summary_parts.append(f"- {str(service)}")
    
    return "\n".join(summary_parts)


def get_service_summary_by_id(service_id: str) -> Optional[str]:
    """
    根据 service_id 获取单个服务的描述
    
    Args:
        service_id: 服务ID
    
    Returns:
        服务描述，如果找不到则返回 None
    """
    service_list = load_service_list()
    for service in service_list:
        if isinstance(service, dict):
            if service.get("service_id") == service_id:
                return service.get("description", "")
        elif isinstance(service, str) and service == service_id:
            return service
    return None

