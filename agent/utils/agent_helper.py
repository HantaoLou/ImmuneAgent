"""
Agent Helper Module

提供从 AGENTS.md 加载和解析 agent 信息的公共工具函数。
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

_AGENTS_CACHE: Optional[str] = None


def get_enable_agents() -> str:
    """
    从 AGENTS.md 中提取可用的 agent 能力描述

    Returns:
        格式化的 agent 能力描述字符串
    """
    global _AGENTS_CACHE
    if _AGENTS_CACHE is not None:
        return _AGENTS_CACHE

    agents_md_path = Path(__file__).parent.parent / "AGENTS.md"
    if not agents_md_path.exists():
        return "无可用 agent 信息"

    try:
        with open(agents_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"读取 AGENTS.md 失败: {str(e)}"

    _AGENTS_CACHE = _extract_capabilities(content)
    return _AGENTS_CACHE


def _extract_capabilities(content: str) -> str:
    """
    从 AGENTS.md 内容提取能力描述

    Args:
        content: AGENTS.md 文件内容

    Returns:
        格式化的 agent 能力描述
    """
    if not content:
        return "无可用 agent 信息"

    lines = content.split("\n")
    capabilities = []
    current_agent = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("### "):
            if current_agent:
                capabilities.append(current_agent)
            agent_name = line.replace("### ", "").strip()
            current_agent = {"name": agent_name, "description": ""}
        elif line.startswith("- **职责**"):
            current_agent["responsibility"] = line.replace("- **职责**:", "").strip()
        elif line.startswith("- **能力**"):
            current_agent["capability"] = line.replace("- **能力**:", "").strip()
        elif line.startswith("- **适用场景**"):
            current_agent["scenario"] = line.replace("- **适用场景**:", "").strip()

    if current_agent:
        capabilities.append(current_agent)

    if not capabilities:
        return "无可用 agent 信息"

    formatted = []
    for agent in capabilities:
        name = agent.get("name", "")
        resp = agent.get("responsibility", "")
        cap = agent.get("capability", "")
        scenario = agent.get("scenario", "")
        formatted.append(
            f"### {name}\n- 职责: {resp}\n- 能力: {cap}\n- 适用场景: {scenario}"
        )

    return "\n\n".join(formatted)


def get_agent_list() -> List[Dict[str, str]]:
    """
    获取 agent 列表（结构化格式）

    Returns:
        agent 列表，每项包含 name, responsibility, capability, scenario
    """
    agents_md_path = Path(__file__).parent.parent / "AGENTS.md"
    if not agents_md_path.exists():
        return []

    try:
        with open(agents_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    lines = content.split("\n")
    agents = []
    current_agent = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("### "):
            if current_agent:
                agents.append(current_agent)
            agent_name = line.replace("### ", "").strip()
            current_agent = {
                "name": agent_name,
                "responsibility": "",
                "capability": "",
                "scenario": "",
            }
        elif line.startswith("- **职责**"):
            current_agent["responsibility"] = line.replace("- **职责**:", "").strip()
        elif line.startswith("- **能力**"):
            current_agent["capability"] = line.replace("- **能力**:", "").strip()
        elif line.startswith("- **适用场景**"):
            current_agent["scenario"] = line.replace("- **适用场景**:", "").strip()

    if current_agent:
        agents.append(current_agent)

    return agents


def clear_agents_cache() -> None:
    """清除 agents 缓存"""
    global _AGENTS_CACHE
    _AGENTS_CACHE = None
