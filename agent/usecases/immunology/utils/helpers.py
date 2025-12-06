"""
Utility functions for ImmuneAgent.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_results_summary(results: Dict[str, Any]) -> str:
    """Format execution results as a summary."""
    lines = []
    lines.append("EXECUTION RESULTS SUMMARY")
    lines.append("=" * 50)

    total_tools = len(results)
    successful = sum(1 for r in results.values() if r.get("status") != "error")

    lines.append(f"Total Tools Executed: {total_tools}")
    lines.append(f"Successful: {successful}/{total_tools}")
    lines.append("")

    for tool_name, result in results.items():
        status = result.get("status", "unknown")
        icon = "✅" if status != "error" else "❌"
        lines.append(f"{icon} {tool_name}: {status}")

        if "result" in result and isinstance(result["result"], dict):
            # Show key metrics
            for key, value in list(result["result"].items())[:3]:
                lines.append(f"    - {key}: {value}")

    return "\n".join(lines)


def calculate_confidence_score(components: Dict[str, float]) -> float:
    """Calculate overall confidence score from components."""
    if not components:
        return 0.0

    weights = {
        "planning": 0.25,
        "feasibility": 0.25,
        "evidence_support": 0.25,
        "execution": 0.25,
    }

    total_weight = 0
    weighted_sum = 0

    for component, score in components.items():
        weight = weights.get(component, 0.1)
        weighted_sum += score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def format_citations(citations: List[str], max_citations: int = 10) -> str:
    """Format citations in a standard format."""
    if not citations:
        return "No citations available."

    lines = ["## References"]
    unique_citations = list(set(citations))[:max_citations]

    for i, citation in enumerate(unique_citations, 1):
        lines.append(f"{i}. {citation}")

    if len(citations) > max_citations:
        lines.append(f"... and {len(citations) - max_citations} more")

    return "\n".join(lines)


def extract_key_terms(text: str) -> List[str]:
    """Extract key immunology terms from text."""
    immunology_terms = {
        "t cell",
        "b cell",
        "antibody",
        "antigen",
        "epitope",
        "mhc",
        "hla",
        "tcr",
        "bcr",
        "car-t",
        "checkpoint",
        "pd-1",
        "ctla-4",
        "cytokine",
        "chemokine",
        "macrophage",
        "dendritic",
        "neutrophil",
        "nk cell",
        "tumor",
        "cancer",
        "autoimmune",
        "infection",
        "vaccine",
    }

    text_lower = text.lower()
    found_terms = []

    for term in immunology_terms:
        if term in text_lower:
            found_terms.append(term)

    return found_terms


def validate_research_question(question: str) -> Dict[str, Any]:
    """Validate and analyze a research question."""
    validation = {
        "is_valid": True,
        "has_immunology_focus": False,
        "complexity_level": "low",
        "suggested_category": "general_immunology",
        "warnings": [],
    }

    # Check length
    if len(question) < 10:
        validation["warnings"].append("Question is very short")
        validation["is_valid"] = False
    elif len(question) > 1000:
        validation["warnings"].append(
            "Question is very long - consider breaking into sub-questions"
        )

    # Check for immunology terms
    key_terms = extract_key_terms(question)
    if key_terms:
        validation["has_immunology_focus"] = True
        validation["key_terms"] = key_terms
    else:
        validation["warnings"].append("No clear immunology focus detected")

    # Estimate complexity
    if len(key_terms) > 5:
        validation["complexity_level"] = "high"
    elif len(key_terms) > 2:
        validation["complexity_level"] = "medium"

    # Suggest category based on terms
    question_lower = question.lower()
    if "antibody" in question_lower or "mab" in question_lower:
        validation["suggested_category"] = "antibody_discovery"
    elif "t cell" in question_lower or "car-t" in question_lower:
        validation["suggested_category"] = "tcr_discovery"
    elif "single-cell" in question_lower or "scrna" in question_lower:
        validation["suggested_category"] = "single_cell_analysis"
    elif "structure" in question_lower or "fold" in question_lower:
        validation["suggested_category"] = "structural_immunology"
    elif "epitope" in question_lower or "mhc" in question_lower:
        validation["suggested_category"] = "epitope_prediction"

    return validation


def create_execution_summary(state: Any) -> Dict[str, Any]:
    """Create a summary of execution state."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "question": getattr(state, "research_question", ""),
        "status": "completed"
        if getattr(state, "analysis_completed", False)
        else "in_progress",
        "tools_used": len(getattr(state, "selected_tools", [])),
        "hypotheses_generated": len(getattr(state, "hypotheses", [])),
        "citations_found": len(getattr(state, "citations", [])),
        "confidence_scores": getattr(state, "confidence_scores", {}),
        "key_findings": getattr(state, "key_findings", [])[:5],
        "recommendations": getattr(state, "recommendations", [])[:3],
    }

    # Calculate overall success
    if hasattr(state, "execution_results"):
        results = state.execution_results
        successful = sum(1 for r in results.values() if r.get("status") != "error")
        summary["execution_success_rate"] = successful / len(results) if results else 0

    return summary


def save_results_to_json(results: Dict[str, Any], filename: str = None) -> str:
    """Save results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"immune_agent_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return filename


def load_results_from_json(filename: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


# Export utility functions
__all__ = [
    "format_results_summary",
    "calculate_confidence_score",
    "format_citations",
    "extract_key_terms",
    "validate_research_question",
    "create_execution_summary",
    "save_results_to_json",
    "load_results_from_json",
]
