"""
Tool Categorization Module

Categorize MCP tools by service, supporting filtering based on service_id.
Enhanced with skill.yaml loading for comprehensive service descriptions.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import sys

# Add agent directory to path
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# Import skill loader for enhanced service descriptions
from .skill_loader import get_cached_skills


def load_service_list() -> List[Dict[str, Any]]:
    """
    Load service list from config/service_list.json
    
    Enhanced to include descriptions from skill.yaml files when available.
    
    Returns:
        Service list, each service contains service_id and description
    """
    service_list_path = agent_dir / "config" / "service_list.json"
    service_list = []
    
    try:
        if service_list_path.exists():
            with open(service_list_path, 'r', encoding='utf-8') as f:
                service_list = json.load(f)
        else:
            print(f"Warning: service_list.json does not exist: {service_list_path}")
    except Exception as e:
        print(f"Warning: Failed to load service_list.json: {e}")
    
    # Enhance with skill descriptions
    try:
        skills = get_cached_skills()
        if skills:
            service_list = _enhance_service_list_with_skills(service_list, skills)
    except Exception as e:
        print(f"Warning: Failed to enhance service list with skills: {e}")
    
    return service_list


def _enhance_service_list_with_skills(
    service_list: List[Dict[str, Any]], 
    skills: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enhance service list with information from skill.yaml files
    
    Args:
        service_list: Original service list from config
        skills: Dictionary of loaded skills
        
    Returns:
        Enhanced service list with skill descriptions
    """
    enhanced_list = []
    
    for service in service_list:
        service_id = service.get("service_id", "")
        
        # Create enhanced copy
        enhanced_service = dict(service)
        
        # Try to find matching skill
        matching_skill = None
        
        # Try exact match first
        if service_id in skills:
            matching_skill = skills[service_id]
        else:
            # Try partial match
            for skill_name, skill_data in skills.items():
                if service_id.lower() in skill_name.lower() or skill_name.lower() in service_id.lower():
                    matching_skill = skill_data
                    break
        
        # Enhance with skill information
        if matching_skill:
            meta = matching_skill.get("meta", {})
            
            # Use skill summary as enhanced description
            skill_summary = meta.get("summary", "")
            skill_description = meta.get("description", "")
            capabilities = meta.get("capabilities", [])
            tags = meta.get("tags", [])
            
            # Build enhanced description
            enhanced_desc = skill_summary if skill_summary else service.get("description", "")
            if capabilities:
                enhanced_desc += f" Capabilities: {', '.join(capabilities[:3])}"
            
            enhanced_service["description"] = enhanced_desc
            enhanced_service["skill_name"] = meta.get("name", service_id)
            enhanced_service["capabilities"] = capabilities
            enhanced_service["tags"] = tags
            
            # Add workflow information for better task planning
            workflow = matching_skill.get("workflow", {})
            if workflow:
                steps = workflow.get("steps", [])
                workflow_desc = "; ".join([f"{s.get('step', '?')}. {s.get('name', '')}" for s in steps[:5]])
                enhanced_service["workflow"] = workflow_desc
        
        enhanced_list.append(enhanced_service)
    
    return enhanced_list


def get_service_description_from_skill(service_id: str) -> Optional[str]:
    """
    Get enhanced service description from skill.yaml
    
    Args:
        service_id: Service ID to look up
        
    Returns:
        Enhanced description, or None if not found
    """
    try:
        skills = get_cached_skills()
        
        # Try exact match
        if service_id in skills:
            meta = skills[service_id].get("meta", {})
            return meta.get("description", "") or meta.get("summary", "")
        
        # Try partial match
        for skill_name, skill_data in skills.items():
            if service_id.lower() in skill_name.lower() or skill_name.lower() in service_id.lower():
                meta = skill_data.get("meta", {})
                return meta.get("description", "") or meta.get("summary", "")
    except Exception:
        pass
    
    return None


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
