"""
Tool Categorization Module

Categorize MCP tools by service, supporting filtering based on service_id.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import sys

# Add agent directory to path
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


def load_service_list() -> List[Dict[str, Any]]:
    """
    Load service list
    
    Returns:
        Service list, each service contains service_id and description
    """
    service_list_path = agent_dir / "config" / "service_list.json"
    
    try:
        if service_list_path.exists():
            with open(service_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: service_list.json does not exist: {service_list_path}")
            return []
    except Exception as e:
        print(f"Warning: Failed to load service_list.json: {e}")
        return []


def get_tools_by_service_ids(
    all_tools: List[Dict[str, Any]], 
    required_service_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter tools based on required service_ids
    
    Args:
        all_tools: All tools list
        required_service_ids: List of required service_ids
    
    Returns:
        Filtered tools list
    """
    if not required_service_ids:
        return []
    
    # Filter tools
    selected_tools = []
    for tool in all_tools:
        tool_service = tool.get("service", "")
        if tool_service in required_service_ids:
            selected_tools.append(tool)
    
    return selected_tools


def get_service_summary(service_list: List[Dict[str, Any]]) -> str:
    """
    Generate service list summary (for prompts)
    
    Args:
        service_list: Service list
    
    Returns:
        Formatted service summary text
    """
    summary_parts = []
    for service in service_list:
        # Handle case where service may be string or dict
        if isinstance(service, str):
            # If it's a string, use directly
            summary_parts.append(f"- {service}")
        elif isinstance(service, dict):
            service_id = service.get("service_id", "")
            description = service.get("description", "")
            summary_parts.append(f"- {service_id}: {description}")
        else:
            # Other types, convert to string
            summary_parts.append(f"- {str(service)}")
    
    return "\n".join(summary_parts)


def get_service_summary_by_id(service_id: str) -> Optional[str]:
    """
    Get description of a single service by service_id
    
    Args:
        service_id: Service ID
    
    Returns:
        Service description, returns None if not found
    """
    service_list = load_service_list()
    for service in service_list:
        if isinstance(service, dict):
            if service.get("service_id") == service_id:
                return service.get("description", "")
        elif isinstance(service, str) and service == service_id:
            return service
    return None
