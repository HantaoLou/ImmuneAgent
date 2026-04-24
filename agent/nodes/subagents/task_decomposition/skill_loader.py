"""
Skill Loader Module

Load and parse skill.yaml files from mcp_tools directory.
These skills provide comprehensive tool descriptions for task decomposition enhancement.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import sys

# Add agent directory to path
agent_dir = Path(__file__).parent.parent.parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))


# Path to mcp_tools directory
MCP_TOOLS_DIR = agent_dir / "mcp_tools"

# Path to task generation guide
TASK_GENERATION_GUIDE_PATH = agent_dir / "nodes" / "subagents" / "code_act" / "TASK_GENERATION_GUIDE.md"


def load_skill_yaml(skill_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single skill.yaml file
    
    Args:
        skill_path: Path to skill.yaml file
        
    Returns:
        Parsed skill dictionary, or None if loading fails
    """
    try:
        with open(skill_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load skill.yaml from {skill_path}: {e}")
        return None


def get_all_skill_dirs() -> List[Path]:
    """
    Get all skill directories under mcp_tools
    
    Returns:
        List of skill directory paths
    """
    if not MCP_TOOLS_DIR.exists():
        print(f"Warning: mcp_tools directory does not exist: {MCP_TOOLS_DIR}")
        return []
    
    skill_dirs = []
    for item in MCP_TOOLS_DIR.iterdir():
        if item.is_dir() and (item / "skill.yaml").exists():
            skill_dirs.append(item)
    
    return skill_dirs


def load_all_skills() -> Dict[str, Dict[str, Any]]:
    """
    Load all skill.yaml files from mcp_tools directory
    
    Returns:
        Dictionary mapping service_name -> skill_content
    """
    skills = {}
    skill_dirs = get_all_skill_dirs()
    
    for skill_dir in skill_dirs:
        skill_path = skill_dir / "skill.yaml"
        skill_data = load_skill_yaml(skill_path)
        
        if skill_data:
            # Use the meta.name or directory name as the key
            meta = skill_data.get("meta", {})
            service_name = meta.get("name", skill_dir.name)
            skills[service_name] = skill_data
            print(f"  [OK] Loaded skill: {service_name}")
    
    return skills


def load_task_generation_guide() -> str:
    """
    Load the task generation guide markdown file
    
    Returns:
        Content of the guide file, or empty string if not found
    """
    try:
        if TASK_GENERATION_GUIDE_PATH.exists():
            with open(TASK_GENERATION_GUIDE_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Task generation guide not found: {TASK_GENERATION_GUIDE_PATH}")
            return ""
    except Exception as e:
        print(f"Warning: Failed to load task generation guide: {e}")
        return ""


def extract_skill_summary(skill_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key information from a skill for use in prompts
    
    Args:
        skill_data: Full skill data from skill.yaml
        
    Returns:
        Simplified skill summary with meta, workflow, and key tool info
    """
    meta = skill_data.get("meta", {})
    workflow = skill_data.get("workflow", {})
    tools = skill_data.get("tools", [])
    constraints = skill_data.get("constraints", {})
    file_column_specs = skill_data.get("file_column_specs", {})
    
    # Extract simplified tool info
    simplified_tools = []
    for tool in tools:
        simplified_tool = {
            "name": tool.get("name", ""),
            "summary": tool.get("summary", ""),
            "category": tool.get("category", ""),
            "priority": tool.get("priority", ""),
            "execution_order": tool.get("execution_order", 0),
            "when_to_use": tool.get("when_to_use", []),
            "dependencies": tool.get("dependencies", {}),
        }
        
        # Extract parameter names and required status
        params = tool.get("parameters", [])
        simplified_params = []
        for param in params:
            simplified_params.append({
                "name": param.get("name", ""),
                "type": param.get("type", ""),
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "default": param.get("default"),
                "example": param.get("example", ""),
            })
        simplified_tool["parameters"] = simplified_params
        
        # Extract returns info
        returns = tool.get("returns", {})
        simplified_tool["returns"] = {
            "type": returns.get("type", ""),
            "description": returns.get("description", ""),
        }
        
        simplified_tools.append(simplified_tool)
    
    return {
        "meta": meta,
        "workflow": workflow,
        "tools": simplified_tools,
        "constraints": constraints,
        "file_column_specs": file_column_specs,
    }


def get_skills_for_services(service_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load and filter skills for specified services
    
    Args:
        service_ids: List of service IDs to load skills for
        
    Returns:
        Dictionary of skill summaries for the specified services
    """
    all_skills = load_all_skills()
    filtered_skills = {}
    
    for service_id in service_ids:
        # Try exact match first
        if service_id in all_skills:
            filtered_skills[service_id] = extract_skill_summary(all_skills[service_id])
        else:
            # Try partial match (e.g., "r_bcell" might match "bcell")
            for skill_name, skill_data in all_skills.items():
                if service_id.lower() in skill_name.lower() or skill_name.lower() in service_id.lower():
                    filtered_skills[service_id] = extract_skill_summary(skill_data)
                    break
    
    return filtered_skills


def format_skills_for_prompt(skills: Dict[str, Dict[str, Any]], max_skills: int = 10) -> str:
    """
    Format skills data for inclusion in LLM prompt
    
    Args:
        skills: Dictionary of skill summaries
        max_skills: Maximum number of skills to include
        
    Returns:
        Formatted string for prompt
    """
    if not skills:
        return "No skill information available."
    
    formatted_parts = []
    skill_count = 0
    
    for service_name, skill in skills.items():
        if skill_count >= max_skills:
            break
        
        meta = skill.get("meta", {})
        workflow = skill.get("workflow", {})
        tools = skill.get("tools", [])
        constraints = skill.get("constraints", {})
        
        # Format service header
        part = f"\n## Service: {service_name}\n"
        part += f"**Summary:** {meta.get('summary', 'N/A')}\n"
        part += f"**Description:** {meta.get('description', 'N/A')[:500]}\n"
        
        # Format capabilities
        capabilities = meta.get("capabilities", [])
        if capabilities:
            part += f"**Capabilities:**\n"
            for cap in capabilities[:5]:  # Limit to 5 capabilities
                part += f"  - {cap}\n"
        
        # Format workflow
        if workflow:
            steps = workflow.get("steps", [])
            if steps:
                part += f"**Workflow Steps:**\n"
                for step in steps:
                    step_name = step.get("name", "")
                    step_tools = step.get("tools", [])
                    is_optional = step.get("is_optional", False)
                    optional_mark = " (optional)" if is_optional else ""
                    part += f"  {step.get('step', '?')}. {step_name}{optional_mark}: {', '.join(step_tools)}\n"
        
        # Format tools (simplified)
        if tools:
            part += f"**Available Tools ({len(tools)}):**\n"
            for tool in tools[:8]:  # Limit to 8 tools per service
                tool_name = tool.get("name", "")
                tool_summary = tool.get("summary", "")
                tool_priority = tool.get("priority", "")
                when_to_use = tool.get("when_to_use", [])
                
                part += f"  - **{tool_name}** [{tool_priority}]: {tool_summary}\n"
                if when_to_use:
                    part += f"    When to use: {when_to_use[0]}\n"
                
                # Include key parameters
                params = tool.get("parameters", [])
                required_params = [p for p in params if p.get("required")]
                if required_params:
                    param_names = [p.get("name", "") for p in required_params[:3]]
                    part += f"    Required params: {', '.join(param_names)}\n"
        
        # Format key constraints
        if constraints:
            limitations = constraints.get("limitations", [])
            if limitations:
                part += f"**Limitations:**\n"
                for lim in limitations[:3]:
                    part += f"  - {lim}\n"
        
        formatted_parts.append(part)
        skill_count += 1
    
    return "\n---\n".join(formatted_parts)


def format_task_guide_for_prompt(guide_content: str, max_length: int = 3000) -> str:
    """
    Format task generation guide for inclusion in prompt
    
    Args:
        guide_content: Full guide content
        max_length: Maximum length to include
        
    Returns:
        Formatted guide excerpt
    """
    if not guide_content:
        return "Task generation guide not available."
    
    # Extract key sections
    key_sections = []
    
    # Find task structure section
    structure_start = guide_content.find("## Task Structure")
    if structure_start != -1:
        structure_end = guide_content.find("##", structure_start + 10)
        if structure_end != -1:
            key_sections.append(guide_content[structure_start:structure_end])
    
    # Find best practices section
    practices_start = guide_content.find("## Best Practices")
    if practices_start != -1:
        practices_end = guide_content.find("##", practices_start + 10)
        if practices_end != -1:
            key_sections.append(guide_content[practices_start:practices_end])
    
    # Find output constraints section
    constraints_start = guide_content.find("## Output Constraints")
    if constraints_start != -1:
        constraints_end = guide_content.find("##", constraints_start + 10)
        if constraints_end != -1:
            key_sections.append(guide_content[constraints_start:constraints_end])
    
    combined = "\n\n".join(key_sections)
    
    # Truncate if too long
    if len(combined) > max_length:
        combined = combined[:max_length] + "\n... (truncated)"
    
    return combined


# Cache for loaded skills
_skills_cache: Optional[Dict[str, Dict[str, Any]]] = None
_guide_cache: Optional[str] = None


def get_cached_skills() -> Dict[str, Dict[str, Any]]:
    """
    Get skills from cache or load if not cached
    
    Returns:
        Dictionary of all skills
    """
    global _skills_cache
    if _skills_cache is None:
        _skills_cache = load_all_skills()
    return _skills_cache


def get_cached_task_guide() -> str:
    """
    Get task generation guide from cache or load if not cached
    
    Returns:
        Task generation guide content
    """
    global _guide_cache
    if _guide_cache is None:
        _guide_cache = load_task_generation_guide()
    return _guide_cache


def clear_cache() -> None:
    """Clear the skills and guide cache"""
    global _skills_cache, _guide_cache, _tool_params_cache
    _skills_cache = None
    _guide_cache = None
    _tool_params_cache = None


# Cache for tool parameters extracted from skills
_tool_params_cache: Optional[Dict[str, Dict[str, Any]]] = None


def load_tool_parameters_from_skills() -> Dict[str, Dict[str, Any]]:
    """
    Extract tool parameter definitions from all skill.yaml files
    
    This is the Single Source of Truth for tool parameters
    Replaces tools_params_table.json
    
    Returns:
        Dict: tool_name -> {input_params: [...], output_params: [...]}
        Format compatible with tools_params_table.json
    """
    global _tool_params_cache
    
    if _tool_params_cache is not None:
        return _tool_params_cache
    
    skills = get_cached_skills()
    tools_params_map: Dict[str, Dict[str, Any]] = {}
    
    for service_name, skill_data in skills.items():
        tools = skill_data.get("tools", [])
        
        for tool in tools:
            tool_name = tool.get("name", "")
            if not tool_name:
                continue
            
            # Build full tool name: service_tool_name (e.g., "nettcr_predict_tcr_binding_complete")
            # Also keep pure tool name as alias
            full_tool_name = f"{service_name}_{tool_name}"
            
            # Extract parameter definitions
            parameters = tool.get("parameters", [])
            input_params = []
            
            for param in parameters:
                param_info = {
                    "name": param.get("name", ""),
                    "type": param.get("type", "string"),
                    "required": param.get("required", False),
                    "description": param.get("description", ""),
                    "default": param.get("default"),
                    "example": param.get("example", ""),
                    "options": param.get("options", []),
                }
                input_params.append(param_info)
            
            # Extract return value info
            returns = tool.get("returns", {})
            output_params = []
            
            if returns:
                output_info = {
                    "type": returns.get("type", ""),
                    "description": returns.get("description", ""),
                    "schema": returns.get("schema", []),
                }
                output_params.append(output_info)
            
            # Build tool parameter mapping
            tool_params = {
                "input_params": input_params,
                "output_params": output_params,
                "tool_name": tool_name,
                "service_name": service_name,
                "full_tool_name": full_tool_name,
                "summary": tool.get("summary", ""),
                "category": tool.get("category", ""),
                "priority": tool.get("priority", ""),
            }
            
            # Store keys in two formats
            # 1. Full name service_tool_name
            tools_params_map[full_tool_name] = tool_params
            # 2. Pure tool name (allows fuzzy matching)
            if tool_name not in tools_params_map:
                tools_params_map[tool_name] = tool_params
    
    _tool_params_cache = tools_params_map
    print(f"[OK] Loaded {len(tools_params_map)} tool parameter definitions from skill.yaml")
    return tools_params_map


def get_tool_params(tool_name: str, service_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get parameter definitions for specified tool
    
    Args:
        tool_name: Tool name
        service_name: Optional service name for exact matching
        
    Returns:
        Tool parameter definition, or None if not found
    """
    tools_params_map = load_tool_parameters_from_skills()
    
    # If service name provided, prefer exact match
    if service_name:
        full_name = f"{service_name}_{tool_name}"
        if full_name in tools_params_map:
            return tools_params_map[full_name]
    
    # Try direct tool name match
    if tool_name in tools_params_map:
        return tools_params_map[tool_name]
    
    # Fuzzy matching
    tool_name_lower = tool_name.lower()
    for key in tools_params_map:
        key_lower = key.lower()
        if tool_name_lower in key_lower or key_lower in tool_name_lower:
            return tools_params_map[key]
    
    return None


if __name__ == "__main__":
    # Test skill loading
    print("Testing skill loader...")
    print(f"MCP_TOOLS_DIR: {MCP_TOOLS_DIR}")
    print(f"TASK_GENERATION_GUIDE_PATH: {TASK_GENERATION_GUIDE_PATH}")
    
    print("\n1. Loading all skills...")
    skills = get_cached_skills()
    print(f"Loaded {len(skills)} skills")
    
    print("\n2. Loading task generation guide...")
    guide = get_cached_task_guide()
    print(f"Guide length: {len(guide)} characters")
    
    print("\n3. Formatting skills for prompt...")
    formatted = format_skills_for_prompt(skills)
    print(f"Formatted length: {len(formatted)} characters")
    print("\nFirst 1000 characters:")
    print(formatted[:1000])

