"""
Prompt Optimization Utilities

This module provides utilities for optimizing LLM prompts:
1. State normalization - Convert state data to compact format
2. Prompt compression - Compress data structures for prompt injection
3. Duplicate detection - Prevent redundant content in prompts

Key design principles:
- Each node's output should be normalized to a standard format
- Subsequent nodes use normalized data instead of raw state
- Compressed prompts reduce token usage by ~50%
"""

from typing import Dict, List, Any, Optional
import json
import re


# ==================== Configuration ====================

# Maximum lengths for various content types
MAX_KNOWLEDGE_LENGTH = 2000  # characters per domain
MAX_CONSTRAINTS = 5  # max constraints to include
MAX_ENTITIES = 10  # max entities to include
MAX_INFERENCE_STEPS = 8  # max steps to show
MAX_TOOL_INSTRUCTION_LENGTH = 500  # characters for tool usage

# Domain mapping (moved from base.py for externalization)
DOMAIN_MODULE_MAPPING = {
    "genetics": ["HWE", "Fst", "GWAS", "variant", "allele", "inheritance", "genotype", "phenotype"],
    "immunology": ["T cell", "B cell", "MHC", "antibody", "antigen", "TCR", "BCR", "immune"],
    "clinical_medicine": ["drug", "medication", "treatment", "diagnosis", "patient", "clinical"],
    "molecular_biology": ["gene expression", "transcription", "translation", "pathway", "protein"],
    "biochemistry": ["enzyme", "metabolism", "concentration", "binding", "kinetics", "reaction"],
    "bioinformatics": ["sequence", "alignment", "variant calling", "BLAST", "genome"],
    "biophysics": ["membrane", "lipid", "receptor", "signaling", "structure"],
    "cell_biology": ["cell", "stem", "differentiation", "proliferation", "apoptosis"],
    "microbiology": ["virus", "bacteria", "pathogen", "infection", "antibiotic"],
    "population_genetics": ["diversity", "theta", "pi", "segregating", "frequency", "selection"],
}


# ==================== State Normalization Functions ====================

def normalize_structured_info(
    structured_subject: Optional[Dict],
    structured_condition: Optional[Dict],
    structured_goal: Optional[Dict]
) -> str:
    """
    Normalize structured three-dimensional information to compact format.
    
    Output format: "Subject: type=... attr=... | Condition: type=... features=... | Goal: type=... intent=..."
    """
    parts = []
    
    if structured_subject and isinstance(structured_subject, dict):
        s_type = structured_subject.get("type", "?")
        s_attr = structured_subject.get("attribute", "?")
        parts.append(f"S:{s_type}|{s_attr}")
    
    if structured_condition and isinstance(structured_condition, dict):
        c_type = structured_condition.get("type", "?")
        c_features = str(structured_condition.get("key_features", "?"))[:100]
        parts.append(f"C:{c_type}|{c_features}")
    
    if structured_goal and isinstance(structured_goal, dict):
        g_type = structured_goal.get("type", "?")
        g_intent = structured_goal.get("intent", "neutral")
        parts.append(f"G:{g_type}|{g_intent}")
    
    return " || ".join(parts) if parts else "N/A"


def normalize_constraints(
    negative_constraints: Optional[List[str]],
    exclusive_constraints: Optional[List[str]],
    strong_restrictions: Optional[List[str]],
    key_constraints: Optional[List[str]],
    max_items: int = MAX_CONSTRAINTS
) -> Dict[str, List[str]]:
    """
    Normalize all constraint types to a unified format.
    
    Returns:
        {
            "all": [...],  # Merged constraints (deduplicated)
            "negative": [...],  # Cannot/except constraints
            "exclusive": [...],  # Only/single constraints
        }
    """
    result = {
        "all": [],
        "negative": [],
        "exclusive": []
    }
    
    seen = set()
    
    def add_constraint(constraint: str, ctype: str):
        if constraint and constraint not in seen:
            seen.add(constraint)
            result["all"].append(constraint)
            if ctype == "negative":
                result["negative"].append(constraint)
            elif ctype == "exclusive":
                result["exclusive"].append(constraint)
    
    # Add by priority
    for c in (exclusive_constraints or [])[:max_items]:
        add_constraint(c, "exclusive")
    
    for c in (negative_constraints or [])[:max_items]:
        add_constraint(c, "negative")
    
    for c in (strong_restrictions or [])[:max_items]:
        add_constraint(c, "strong")
    
    for c in (key_constraints or [])[:max_items]:
        add_constraint(c, "key")
    
    # Limit total
    result["all"] = result["all"][:max_items * 2]
    
    return result


def normalize_knowledge_map(
    domain_knowledge_map: Optional[Dict[str, Dict]],
    max_length: int = MAX_KNOWLEDGE_LENGTH
) -> str:
    """
    Compress domain knowledge map to compact string format.
    
    Output format:
    [Domain1]
    - knowledge point 1
    - knowledge point 2
    [Domain2]
    - ...
    """
    if not domain_knowledge_map:
        return "No knowledge retrieved"
    
    lines = []
    total_len = 0
    
    for domain, knowledge in domain_knowledge_map.items():
        if total_len >= max_length:
            lines.append(f"... (truncated, {len(domain_knowledge_map)} domains total)")
            break
        
        lines.append(f"[{domain}]")
        
        # Extract knowledge points
        if isinstance(knowledge, dict):
            for ktype in ["specialized_knowledge", "foundational_knowledge"]:
                kpoints = knowledge.get(ktype, [])
                if isinstance(kpoints, list):
                    for kp in kpoints[:3]:  # Max 3 per type
                        kp_str = str(kp)[:200]
                        lines.append(f"- {kp_str}")
                        total_len += len(kp_str)
                        if total_len >= max_length:
                            break
        
        if total_len >= max_length:
            break
    
    return "\n".join(lines)


def normalize_inference_path(
    closed_inference_path: Optional[List[Dict]],
    max_steps: int = MAX_INFERENCE_STEPS
) -> str:
    """
    Compress inference path to compact format.
    
    Output format:
    1. [type] content...
    2. [type] content...
    """
    if not closed_inference_path:
        return "No inference path"
    
    lines = []
    for i, step in enumerate(closed_inference_path[:max_steps], 1):
        step_type = step.get("step_type", step.get("type", "?"))
        content = str(step.get("step_content", step.get("conclusion", "?")))[:150]
        lines.append(f"{i}. [{step_type}] {content}")
    
    if len(closed_inference_path) > max_steps:
        lines.append(f"... ({len(closed_inference_path) - max_steps} more steps)")
    
    return "\n".join(lines)


def normalize_entities(
    key_entities: Optional[List[str]],
    core_keywords: Optional[List[str]],
    max_items: int = MAX_ENTITIES
) -> str:
    """
    Normalize entities and keywords to compact format.
    """
    entities = list(key_entities or [])[:max_items]
    keywords = list(core_keywords or [])[:max_items]
    
    # Deduplicate
    all_terms = list(dict.fromkeys(entities + keywords))
    
    return ", ".join(all_terms[:max_items]) if all_terms else "N/A"


# ==================== Prompt Building Utilities ====================

def build_compact_context(state: Any) -> str:
    """
    Build a compact context string from state for prompt injection.
    This replaces verbose state dumps with structured, compact format.
    """
    context_parts = []
    
    # Question info
    if hasattr(state, 'cleaned_text') and state.cleaned_text:
        question_preview = state.cleaned_text[:300] + "..." if len(state.cleaned_text) > 300 else state.cleaned_text
        context_parts.append(f"[Question]\n{question_preview}")
    
    # Structured info (normalized)
    if all(hasattr(state, attr) for attr in ['structured_subject', 'structured_condition', 'structured_goal']):
        struct_info = normalize_structured_info(
            state.structured_subject,
            state.structured_condition,
            state.structured_goal
        )
        context_parts.append(f"[Structured]\n{struct_info}")
    
    # Entities (normalized)
    if hasattr(state, 'key_entities'):
        entities = normalize_entities(
            getattr(state, 'key_entities', None),
            getattr(state, 'core_keywords', None)
        )
        context_parts.append(f"[Entities]\n{entities}")
    
    # Constraints (normalized)
    if hasattr(state, 'inference_core_restrictions') and state.inference_core_restrictions:
        constraints = normalize_constraints(
            getattr(state, 'negative_constraints', None),
            getattr(state, 'exclusive_constraints', None),
            getattr(state, 'strong_restrictions', None),
            getattr(state, 'key_constraints', None)
        )
        if constraints["all"]:
            context_parts.append(f"[Constraints]\n" + "\n".join(f"- {c}" for c in constraints["all"]))
    
    # Knowledge (normalized)
    if hasattr(state, 'domain_knowledge_map') and state.domain_knowledge_map:
        knowledge = normalize_knowledge_map(state.domain_knowledge_map)
        context_parts.append(f"[Knowledge]\n{knowledge}")
    
    return "\n\n".join(context_parts)


def build_option_block(
    question_options: Optional[List[str]],
    max_option_length: int = 200
) -> str:
    """
    Build compact option block for prompts.
    """
    if not question_options:
        return ""
    
    lines = []
    for i, opt in enumerate(question_options):
        label = chr(65 + i)  # A, B, C, ...
        opt_text = str(opt)[:max_option_length]
        lines.append(f"{label}. {opt_text}")
    
    return "\n".join(lines)


def build_constraint_block(state: Any) -> str:
    """
    Build constraint instruction block - unified for all nodes.
    This replaces multiple constraint injection points.
    """
    constraints = normalize_constraints(
        getattr(state, 'negative_constraints', None),
        getattr(state, 'exclusive_constraints', None),
        getattr(state, 'strong_restrictions', None),
        getattr(state, 'key_constraints', None)
    )
    
    if not constraints["all"]:
        return ""
    
    lines = ["[Constraints - MUST follow]"]
    
    if constraints["negative"]:
        lines.append("CANNOT: " + " | ".join(constraints["negative"]))
    
    if constraints["exclusive"]:
        lines.append("ONLY: " + " | ".join(constraints["exclusive"]))
    
    other = [c for c in constraints["all"] if c not in constraints["negative"] and c not in constraints["exclusive"]]
    if other:
        lines.append("MUST: " + " | ".join(other[:3]))
    
    return "\n".join(lines)


# ==================== Tool Instruction Templates ====================

TOOL_USAGE_INSTRUCTION_COMPACT = """
[Tools] {count} tools available. Rules:
1. Call 2+ tools for factual queries
2. Map: drugs→drug tools | genes→gene tools | diseases→disease tools
3. For sequence IDs: use igblast/sequence tools
4. Skip tools for pure reasoning questions
"""

MULTI_STATEMENT_INSTRUCTION_COMPACT = """
[Multi-Statement] Statements I,II,III... detected:
1. Extract each statement
2. Verify against knowledge: Statement I → TRUE/FALSE/UNCERTAIN
3. Match to options based on goal (TRUE→select / FALSE→exclude)
"""

LOGICAL_FALLACY_WARNING_COMPACT = """
[Logic Rules]
- Observation ≠ Causation: "X did well on Y" ≠ "X adapted to Y"
- Correlation ≠ Causation: A happened → B happened ≠ A caused B
- Data First: Don't override experimental data with general knowledge
"""


# ==================== Duplicate Detection ====================

def detect_duplicate_content(prompt: str, content: str) -> bool:
    """
    Check if content already exists in prompt (case insensitive).
    """
    return content.lower() in prompt.lower()


def remove_duplicate_sections(prompt: str) -> str:
    """
    Remove duplicate sections from prompt.
    """
    # Remove duplicate CRITICAL blocks
    critical_pattern = r'\*\*CRITICAL:[^*]+\*\*'
    matches = re.findall(critical_pattern, prompt, re.IGNORECASE)
    
    seen = set()
    for match in matches:
        match_lower = match.lower()
        if match_lower in seen:
            prompt = prompt.replace(match, "", 1)
        else:
            seen.add(match_lower)
    
    return prompt


# ==================== Prompt Length Management ====================

def truncate_prompt(prompt: str, max_length: int = 15000) -> str:
    """
    Truncate prompt to maximum length while preserving structure.
    """
    if len(prompt) <= max_length:
        return prompt
    
    # Try to truncate at section boundary
    lines = prompt.split("\n")
    result = []
    current_len = 0
    
    for line in lines:
        if current_len + len(line) + 1 > max_length:
            result.append(f"\n... [truncated, original length: {len(prompt)}]")
            break
        result.append(line)
        current_len += len(line) + 1
    
    return "\n".join(result)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough: 1 token ≈ 4 characters for English).
    """
    return len(text) // 4


# ==================== Domain Detection ====================

def detect_prompt_module(text: str) -> List[str]:
    """
    Detect which prompt modules should be used based on text content.
    """
    text_lower = text.lower()
    modules = []
    
    for module, keywords in DOMAIN_MODULE_MAPPING.items():
        if any(kw in text_lower for kw in keywords):
            modules.append(module)
    
    return modules[:3] if modules else ["general"]


# ==================== Export Functions ====================

__all__ = [
    # Normalization
    'normalize_structured_info',
    'normalize_constraints',
    'normalize_knowledge_map',
    'normalize_inference_path',
    'normalize_entities',
    # Building
    'build_compact_context',
    'build_option_block',
    'build_constraint_block',
    # Templates
    'TOOL_USAGE_INSTRUCTION_COMPACT',
    'MULTI_STATEMENT_INSTRUCTION_COMPACT',
    'LOGICAL_FALLACY_WARNING_COMPACT',
    # Utilities
    'detect_duplicate_content',
    'remove_duplicate_sections',
    'truncate_prompt',
    'estimate_tokens',
    'detect_prompt_module',
    # Config
    'DOMAIN_MODULE_MAPPING',
    'MAX_KNOWLEDGE_LENGTH',
    'MAX_CONSTRAINTS',
    'MAX_ENTITIES',
]



