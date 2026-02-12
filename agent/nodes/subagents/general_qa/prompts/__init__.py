"""
Prompts Module for General QA
Domain-specific prompt modules organized by domain
"""

from .domain_mapper import (
    get_prompt_module,
    detect_domain_from_state,
    identify_domain,
    detect_cross_domain,
)

__all__ = [
    "get_prompt_module",
    "detect_domain_from_state",
    "identify_domain",
    "detect_cross_domain",
]

