"""Shared output constraints for all tool modules.

Every tool in agent/tools/ uses these utilities to prevent unbounded output
that floods the LLM context window.

Design rules:
    - MAX_OUTPUT_CHARS = 6000 (~2000 tokens)
    - All tool functions return truncated strings
    - Pydantic response models strip API boilerplate
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Output constraints
# ---------------------------------------------------------------------------
MAX_OUTPUT_CHARS = 6000  # ~2000 tokens


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate text to max_chars with a notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} chars omitted ...]"


def format_results(
    results: List[Dict[str, Any]],
    tool_name: str,
    max_results: int = 10,
) -> str:
    """Format query results into a readable string for LLM consumption.

    Mirrors the pattern from biomedical_tools._format_results() but applies
    truncation via truncate_output().
    """
    if not results:
        return f"[{tool_name}] No results found."
    if results and results[0].get("error"):
        return f"[{tool_name}] Error: {results[0]['error']}"

    truncated = results[:max_results]
    lines = [f"[{tool_name}] Found {len(results)} results (showing {len(truncated)}):\n"]
    for i, row in enumerate(truncated, 1):
        parts = [f"--- Result {i} ---"]
        for k, v in row.items():
            if v is not None and str(v).strip():
                parts.append(f"  {k}: {v}")
        lines.append("\n".join(parts))
    text = "\n\n".join(lines)
    return truncate_output(text)


# ---------------------------------------------------------------------------
# Pydantic response models — strip API boilerplate, return structured summaries
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: str  # "First A, Second B, et al."
    journal: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None

    def to_short(self) -> str:
        parts = []
        if self.pmid:
            parts.append(f"PMID:{self.pmid}")
        if self.doi:
            parts.append(f"DOI:{self.doi}")
        parts.append(self.title)
        parts.append(self.authors)
        if self.journal:
            parts.append(self.journal)
        if self.year:
            parts.append(str(self.year))
        if self.citation_count is not None:
            parts.append(f"Citations:{self.citation_count}")
        return " | ".join(parts)


class TrialSummary(BaseModel):
    nct_id: str
    title: str
    status: str
    phase: Optional[str] = None

    def to_short(self) -> str:
        parts = [self.nct_id, self.title, f"Status:{self.status}"]
        if self.phase:
            parts.append(f"Phase:{self.phase}")
        return " | ".join(parts)


class TrialDetail(TrialSummary):
    conditions: Optional[str] = None
    interventions: Optional[str] = None
    eligibility: Optional[str] = None
    primary_outcome: Optional[str] = None

    def to_short(self) -> str:
        parts = [super().to_short()]
        if self.conditions:
            parts.append(f"Conditions:{self.conditions}")
        if self.interventions:
            parts.append(f"Interventions:{self.interventions}")
        if self.eligibility:
            parts.append(f"Eligibility:{self.eligibility}")
        if self.primary_outcome:
            parts.append(f"Outcome:{self.primary_outcome}")
        return " | ".join(parts)


class ProteinSummary(BaseModel):
    accession: str
    name: str
    gene: Optional[str] = None
    organism: Optional[str] = None
    function_summary: Optional[str] = None

    def to_short(self) -> str:
        parts = [self.accession, self.name]
        if self.gene:
            parts.append(f"Gene:{self.gene}")
        if self.organism:
            parts.append(self.organism)
        if self.function_summary:
            parts.append(self.function_summary[:200])
        return " | ".join(parts)
