"""
GeneralQA agent subgraph

Implements a complete biomedical question-answering system with 12 nodes:
N0: Input Preprocessing & Question Classification
N1: Question Decomposition & Domain Localization
N2: Calculation/Algorithm Requirement Recognition
N3: Cross-Domain Knowledge Retrieval
N4: Calculation Step Decomposition & Formula Matching
N5: Algorithm Parameter Extraction & Applicability Validation
N6: Initial Association Inference
N7: Complete Logical Inference
N8: Multi-Type Answer Generation
N9: Result Validation & Consistency Judgment
N10: Knowledge/Calculation Exception Handling
N11: Manual Intervention Trigger
"""

from typing import Dict, List, Any, Optional, Tuple
from langgraph.graph import StateGraph, START, END
import json
import re
import os

from agent.utils.llm_factory import create_bioinformatics_llm, create_reasoning_llm
from agent.nodes.subagents.general_qa.state import GeneralQAState
from agent.nodes.subagents.general_qa.prompt import (
    get_input_preprocessing_prompt,
    get_question_decomposition_prompt,
    get_calculation_algorithm_recognition_prompt,
    get_knowledge_retrieval_prompt,
    get_calculation_decomposition_prompt,
    get_algorithm_validation_prompt,
    get_initial_inference_prompt,
    get_complete_inference_prompt,
    get_answer_generation_prompt,
    get_result_validation_prompt,
    get_exception_handling_prompt,
    get_manual_intervention_prompt
)
from agent.nodes.subagents.general_qa.prompts.domain_mapper import detect_domain_from_state

try:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    StructuredTool = None
    print("Warning: langchain libraries not installed, general QA functionality will be unavailable")

# Import tool loader
try:
    from agent.nodes.subagents.general_qa.tools.tool_loader import get_tools_for_node, load_all_tools
    from agent.nodes.subagents.general_qa.tools.tool_trigger import (
        get_tools_by_keywords,
        should_force_tool_usage
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    get_tools_for_node = None
    load_all_tools = None
    get_tools_by_keywords = None
    should_force_tool_usage = None
    print("Warning: Tool loader not available, tools will not be bound to nodes")


# ===================== Helper Functions =====================

def _call_llm(llm: Any, prompt: str, tools: Optional[List[StructuredTool]] = None, max_iterations: int = 5, state: Optional[GeneralQAState] = None, node_name: Optional[str] = None) -> Optional[str]:
    """
    Call LLM with prompt and optional tools, handling tool calls iteratively
    
    Args:
        llm: LLM instance
        prompt: Prompt text
        tools: Optional list of tools to bind to LLM
        max_iterations: Maximum number of tool call iterations
        state: Optional state object to record tool calls
        node_name: Optional node name for context in tool call history
    
    Returns:
        Final LLM response text after all tool calls, None if failed
    """
    if not LLM_AVAILABLE or llm is None:
        return None
    
    # Initialize tool calls history in state if provided
    if state and state.tool_calls_history is None:
        state.tool_calls_history = []
    
    try:
        # Bind tools if provided
        llm_with_tools = llm
        tool_map = {}
        available_tool_names = []
        if tools and TOOLS_AVAILABLE and len(tools) > 0:
            try:
                llm_with_tools = llm.bind_tools(tools)
                tool_map = {tool.name: tool for tool in tools}
                available_tool_names = [tool.name for tool in tools]
                print(f"  🔧 Bound {len(tools)} tool(s) to LLM: {', '.join(available_tool_names)}")
            except Exception as e:
                print(f"  ⚠ Failed to bind tools: {e}")
                llm_with_tools = llm
        else:
            print(f"  ℹ No tools provided for this LLM call")
        
        messages = [HumanMessage(content=prompt)]
        iteration = 0
        total_tool_calls = 0
        
        while iteration < max_iterations:
            response = llm_with_tools.invoke(messages)
            
            # Check for tool calls
            tool_calls = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls
            elif hasattr(response, 'response_metadata') and 'tool_calls' in response.response_metadata:
                tool_calls = response.response_metadata['tool_calls']
            
            # Add response to messages
            messages.append(response)
            
            # If no tool calls, return final response
            if not tool_calls:
                response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                if state and total_tool_calls == 0:
                    # Record that no tools were used
                    state.tool_calls_history.append({
                        "node": node_name or "unknown",
                        "iteration": iteration + 1,
                        "tools_available": available_tool_names,
                        "tools_used": [],
                        "note": "LLM did not use any tools"
                    })
                return response_text
            
            # Execute tool calls
            print(f"  🔧 Executing {len(tool_calls)} tool call(s) (iteration {iteration + 1}/{max_iterations})")
            iteration_tool_calls = []
            
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('args', {})
                tool_call_id = tool_call.get('id', '')
                
                tool_call_record = {
                    "node": node_name or "unknown",
                    "iteration": iteration + 1,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_call_id": tool_call_id,
                    "status": "unknown",
                    "result": None,
                    "error": None
                }
                
                if tool_name in tool_map:
                    try:
                        tool = tool_map[tool_name]
                        result = tool.invoke(tool_args)
                        total_tool_calls += 1
                        
                        # Convert result to string if needed
                        if isinstance(result, (list, dict)):
                            result_str = json.dumps(result, ensure_ascii=False, indent=2)
                            # Store full result for logging
                            tool_call_record["result"] = result
                        else:
                            result_str = str(result)
                            tool_call_record["result"] = result_str
                        
                        tool_call_record["status"] = "success"
                        
                        messages.append(ToolMessage(
                            content=result_str,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        ))
                        print(f"    ✓ {tool_name} executed successfully")
                        print(f"      - Args: {json.dumps(tool_args, ensure_ascii=False)[:200]}...")
                        if isinstance(result, (list, dict)):
                            result_preview = json.dumps(result, ensure_ascii=False)[:300]
                            print(f"      - Result preview: {result_preview}...")
                        else:
                            result_preview = str(result)[:300]
                            print(f"      - Result preview: {result_preview}...")
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        tool_call_record["status"] = "error"
                        tool_call_record["error"] = str(e)
                        
                        messages.append(ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        ))
                        print(f"    ✗ {tool_name} failed: {e}")
                else:
                    error_msg = f"Tool {tool_name} not found in available tools"
                    tool_call_record["status"] = "not_found"
                    tool_call_record["error"] = error_msg
                    
                    messages.append(ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call_id,
                        name=tool_name
                    ))
                    print(f"    ✗ {error_msg}")
                
                iteration_tool_calls.append(tool_call_record)
            
            # Record all tool calls for this iteration
            if state and iteration_tool_calls:
                state.tool_calls_history.extend(iteration_tool_calls)
            
            iteration += 1
        
        # If we've exhausted iterations, return the last response
        final_response = messages[-1]
        if hasattr(final_response, 'content'):
            return final_response.content.strip()
        return str(final_response).strip()
        
    except Exception as e:
        print(f"⚠ LLM call exception: {type(e).__name__}: {str(e)[:100]}")
        if state:
            state.tool_calls_history.append({
                "node": node_name or "unknown",
                "status": "llm_exception",
                "error": str(e)
            })
        return None


def _parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response
    
    Args:
        response_text: LLM response text
    
    Returns:
        Parsed JSON dictionary, None if failed
    """
    if not response_text:
        return None
    
    # Try direct JSON parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object in text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def _normalize_question_options(raw_options: Any) -> List[str]:
    """Normalize question options into a clean list of strings."""
    if not raw_options:
        return []
    if isinstance(raw_options, list):
        return [str(item).strip() for item in raw_options if str(item).strip()]
    if isinstance(raw_options, str):
        raw_text = raw_options.strip()
        if not raw_text:
            return []
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        parts = [part.strip() for part in re.split(r'[\n;]+', raw_text) if part.strip()]
        return parts
    return []


def _extract_question_options(text: str) -> List[str]:
    """Extract multiple-choice options from the raw question text."""
    if not text:
        return []
    options: List[str] = []
    current_option: Optional[str] = None
    for line in text.splitlines():
        match = re.match(r'^\s*([A-Z])[\).:]\s+(.*)$', line)
        if match:
            if current_option is not None:
                options.append(current_option.strip())
            current_option = match.group(2).strip()
            continue
        if current_option is not None:
            if not line.strip():
                continue
            current_option = f"{current_option} {line.strip()}"
    if current_option is not None:
        options.append(current_option.strip())
    return options


def _infer_answer_format(question_type_label: Optional[str], user_input: str, question_options: List[str]) -> Optional[str]:
    """Infer answer format when the LLM does not provide one."""
    text_lower = (user_input or "").lower()
    if question_type_label == "Multiple Choice":
        if "select all" in text_lower or "choose all" in text_lower or "select all that apply" in text_lower:
            return "Multi-Select"
        return "Single Choice"
    if question_type_label == "Numerical Calculation":
        return "Numeric"
    if question_type_label == "Text Matching":
        return "Short Text"
    if question_type_label == "Professional Algorithm":
        return "Procedure"
    if question_type_label == "Mechanism Explanation":
        return "Long Text"
    if question_options:
        return "Single Choice"
    return None


def _infer_answer_constraints_from_text(text: str) -> List[str]:
    """Infer lightweight answer constraints from raw question text."""
    if not text:
        return []
    constraints: List[str] = []
    lower = text.lower()
    round_match = re.search(r'round(?:ed)? to ([^.\n]+)', lower)
    if round_match:
        constraints.append(f"round to {round_match.group(1).strip()}")
    if "5'" in text or "3'" in text:
        constraints.append("include 5' and 3' orientation")
    if re.search(r'\b(inches|inch|cm|mm|%|mol/l|mM|uM|nm|kb|bp)\b', text, re.IGNORECASE):
        constraints.append("include specified units")
    count_match = re.search(r'\b(recommend|list|provide|name|identify|select)\s+(\d+)\b', lower)
    if count_match:
        constraints.append(f"answer count: {count_match.group(2)}")
    if "select all" in lower or "choose all" in lower or "all that apply" in lower:
        constraints.append("multi-select")
    seen = set()
    ordered = []
    for item in constraints:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _extract_option_labels(text: str) -> List[str]:
    """Extract standalone option labels from text (e.g., A, B, C)."""
    if not text:
        return []
    labels = re.findall(r'(?<![A-Za-z])([A-Z])(?![A-Za-z])', text)
    seen = set()
    ordered = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _tokenize_text(text: str) -> set:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return set()
    return set(re.findall(r'[A-Za-z0-9]+', text.lower()))


def _map_conclusion_to_option(core_conclusion: Optional[str], options: List[str]) -> List[str]:
    """Map a core conclusion to the best matching option label."""
    if not core_conclusion or not options:
        return []
    conclusion_lower = core_conclusion.lower()
    for idx, option in enumerate(options):
        option_lower = option.lower()
        label = chr(65 + idx)
        if "none of the above" in conclusion_lower or "none of the options" in conclusion_lower:
            if "none" in option_lower:
                return [label]
        if "all of the above" in conclusion_lower:
            if "all" in option_lower and "above" in option_lower:
                return [label]
    for idx, option in enumerate(options):
        label = chr(65 + idx)
        option_lower = option.lower()
        if option_lower in conclusion_lower or conclusion_lower in option_lower:
            return [label]
    conclusion_tokens = _tokenize_text(core_conclusion)
    if not conclusion_tokens:
        return []
    best_idx = None
    best_score = 0.0
    for idx, option in enumerate(options):
        option_tokens = _tokenize_text(option)
        if not option_tokens:
            continue
        score = len(conclusion_tokens & option_tokens) / max(len(conclusion_tokens), 1)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None and best_score >= 0.25:
        return [chr(65 + best_idx)]
    return []


def _normalize_choice_answer(
    final_answer: Optional[str],
    option_matching_table: Optional[Dict[str, Any]],
    question_options: List[str],
    answer_format_label: Optional[str],
    core_conclusion: Optional[str]
) -> Optional[str]:
    """
    Normalize multiple-choice answers to option label(s).
    Enhanced with semantic matching using tools.
    """
    labels: List[str] = []
    if isinstance(option_matching_table, dict):
        for label, status in option_matching_table.items():
            if str(status).strip().lower().startswith("match"):
                labels.append(label)
    if not labels and final_answer:
        labels = _extract_option_labels(final_answer)
    if not labels and core_conclusion and question_options:
        labels = _map_conclusion_to_option(core_conclusion, question_options)
    
    # Enhanced: If no direct match but we have core_conclusion, try semantic matching
    if not labels and core_conclusion and question_options and TOOLS_AVAILABLE:
        # Try to find semantic relationships using tools
        # For example, if conclusion is "PRS" and option is "Ventral foregut budding defect"
        # We should query the relationship
        try:
            if load_all_tools:
                all_tools = load_all_tools()
                # Use disease/gene tools to find relationships
                disease_tools = [t for t in all_tools if any(name in t.name for name in ["disgenet", "omim", "hpo"])]
                # This is a hint for the LLM to use tools, actual matching happens in N8
                print(f"  🔍 Attempting semantic matching for conclusion: {core_conclusion[:50]}...")
        except:
            pass
    
    if not labels:
        return final_answer
    if answer_format_label == "Single Choice":
        return labels[0]
    return ", ".join(labels)


def _validate_answer_format(
    final_answer: Optional[str],
    answer_format_label: Optional[str],
    question_options: List[str],
    answer_constraints: Optional[List[str]]
) -> (str, List[str]):
    """Validate answer format using lightweight deterministic checks."""
    issues: List[str] = []
    if not answer_format_label:
        return "Valid", []
    if not final_answer or not str(final_answer).strip():
        return "Invalid", ["empty answer"]
    answer_text = str(final_answer).strip()
    if answer_format_label in ["Single Choice", "Multi-Select"]:
        labels = _extract_option_labels(answer_text)
        if not labels:
            issues.append("missing option label")
        else:
            if answer_format_label == "Single Choice" and len(labels) != 1:
                issues.append("expected a single option label")
            if question_options:
                allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(question_options)])
                invalid = [label for label in labels if label not in allowed]
                if invalid:
                    issues.append(f"invalid option label(s): {', '.join(invalid)}")
    elif answer_format_label == "Numeric":
        if not re.search(r'\d', answer_text):
            issues.append("numeric answer expected")
    elif answer_format_label == "Formula":
        if not re.search(r'[=+\-*/^]', answer_text):
            issues.append("formula expression expected")
    elif answer_format_label == "Sequence":
        if answer_constraints:
            requires_orientation = any("5'" in item or "3'" in item for item in answer_constraints)
            if requires_orientation and ("5'" not in answer_text and "3'" not in answer_text):
                issues.append("sequence orientation missing")
    return ("Invalid", issues) if issues else ("Valid", [])


# LLM instance cache (key: (provider, model, temperature), value: LLM instance)
_llm_cache: Dict[Tuple[Optional[str], Optional[str], float], Any] = {}

def _get_llm() -> Optional[Any]:
    """Get LLM instance for general QA
    
    Supports dynamic configuration via environment variables:
    - GENERAL_QA_LLM_PROVIDER: LLM provider (dashscope, zhipu, etc.)
    - GENERAL_QA_LLM_MODEL: LLM model name
    - GENERAL_QA_LLM_TEMPERATURE: Temperature parameter (default: 0.3)
    
    Uses caching to avoid recreating LLM instances for the same configuration.
    This improves performance while maintaining stateless behavior (context is in state, not LLM).
    """
    if not LLM_AVAILABLE:
        return None
    
    # Check for dynamic configuration via environment variables
    provider = os.getenv("GENERAL_QA_LLM_PROVIDER")
    model = os.getenv("GENERAL_QA_LLM_MODEL")
    temperature_str = os.getenv("GENERAL_QA_LLM_TEMPERATURE", "0.3")
    
    try:
        temperature = float(temperature_str)
    except (ValueError, TypeError):
        temperature = 0.3
    
    # Create cache key
    cache_key = (provider, model, temperature)
    
    # Check cache
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    
    # Create new LLM instance
    if provider and model:
        custom_model = f"{provider}:{model}"
        llm = create_bioinformatics_llm(temperature=temperature, custom_model=custom_model)
    else:
        # Default: Use bioinformatics LLM for biomedical questions
        llm = create_bioinformatics_llm(temperature=temperature)
    
    # Cache the instance
    if llm is not None:
        _llm_cache[cache_key] = llm
    
    return llm


# ===================== Node Implementations =====================

def n0_input_preprocessing_node(state: GeneralQAState) -> GeneralQAState:
    """
    N0: Input Preprocessing & Question Classification
    
    Structure: Input validation => Data preparation => Execution => Result organization
    """
    # Input validation
    if not state.user_input or not state.user_input.strip():
        state.error_message = "user_input cannot be empty"
        return state
    
    print("=" * 60)
    print("N0: Input Preprocessing & Question Classification")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable, cannot preprocess input"
        return state
    
    # Detect domain from state (if available) or from user input
    domain = detect_domain_from_state(state) if hasattr(state, 'question_type_label') and state.question_type_label else None
    question_type = getattr(state, 'question_type_label', None)
    core_domains = getattr(state, 'core_domains', None)
    
    prompt = get_input_preprocessing_prompt(
        state.user_input,
        domain=domain,
        question_type=question_type,
        core_domains=core_domains
    )
    
    # Execution
    response = _call_llm(llm, prompt, state=state, node_name="n0_input_preprocessing")
    if not response:
        state.error_message = "LLM call failed for input preprocessing"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for input preprocessing"
        return state
    
    state.cleaned_text = result.get("cleaned_text", state.user_input)
    state.question_type_label = result.get("question_type_label")
    state.question_category_standard = result.get("question_category_standard")
    state.category_specific_constraints = result.get("category_specific_constraints", [])
    state.data_completeness_label = result.get("data_completeness_label")
    raw_options = result.get("question_options")
    options = _normalize_question_options(raw_options)
    if not options:
        options = _extract_question_options(state.user_input)
    state.question_options = options if options else None
    state.answer_format_label = result.get("answer_format_label") or _infer_answer_format(
        state.question_type_label,
        state.user_input,
        options
    )
    
    # OPTIMIZATION: Extract core keywords and option features
    state.core_keywords = result.get("core_keywords", [])
    if state.core_keywords:
        print(f"  ✓ Core keywords extracted: {state.core_keywords}")
    
    state.option_features = result.get("option_features", {})
    if state.option_features:
        print(f"  ✓ Option features extracted: {len(state.option_features)} options")
    
    # Extract synonyms and tool intent
    state.synonyms = result.get("synonyms", [])
    if state.synonyms:
        print(f"  ✓ Retrieval keywords normalized: {state.synonyms}")
    
    state.tool_intent = result.get("tool_intent", {})
    if state.tool_intent:
        print(f"  ✓ Tool intent marked: {state.tool_intent}")
    
    # Extract structured three-dimensional information (结构化三维度信息)
    state.structured_subject = result.get("structured_subject")
    state.structured_condition = result.get("structured_condition")
    state.structured_goal = result.get("structured_goal")
    
    # Rule 1: Extract key_constraints (关键约束单独标记)
    state.key_constraints = result.get("key_constraints", [])
    if state.key_constraints:
        print(f"  ✓ Key constraints extracted: {state.key_constraints}")
    
    # OPTIMIZATION: Extract and mark critical constraints for downstream nodes
    # Extract negative constraints (cannot/except/not occur)
    state.negative_constraints = result.get("negative_constraints", [])
    if not state.negative_constraints and state.cleaned_text:
        # Auto-detect negative constraints from cleaned_text
        import re
        text_lower = state.cleaned_text.lower()
        negative_keywords = ["cannot", "can not", "except", "not occur", "exclude", "never", "must not", "not happen"]
        sentences = re.split(r'[.!?]\s+', state.cleaned_text)
        for sent in sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in negative_keywords):
                state.negative_constraints.append(sent.strip())
    
    # Extract exclusive constraints (category 1/only 1/single)
    state.exclusive_constraints = result.get("exclusive_constraints", [])
    if not state.exclusive_constraints and state.cleaned_text:
        import re
        text_lower = state.cleaned_text.lower()
        exclusive_keywords = ["category 1", "only 1", "single", "unique", "exclusively", "solely", "merely", "only one"]
        sentences = re.split(r'[.!?]\s+', state.cleaned_text)
        for sent in sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in exclusive_keywords):
                state.exclusive_constraints.append(sent.strip())
    
    # Extract strong restrictions (necessarily true/must be)
    state.strong_restrictions = result.get("strong_restrictions", [])
    if not state.strong_restrictions and state.cleaned_text:
        import re
        text_lower = state.cleaned_text.lower()
        restriction_keywords = ["necessarily true", "must be", "only when", "strictly required", "must", "necessarily"]
        sentences = re.split(r'[.!?]\s+', state.cleaned_text)
        for sent in sentences:
            sent_lower = sent.lower()
            if any(keyword in sent_lower for keyword in restriction_keywords):
                state.strong_restrictions.append(sent.strip())
    
    if state.negative_constraints:
        print(f"  ✓ Negative constraints extracted: {len(state.negative_constraints)} constraint(s)")
    if state.exclusive_constraints:
        print(f"  ✓ Exclusive constraints extracted: {len(state.exclusive_constraints)} constraint(s)")
    if state.strong_restrictions:
        print(f"  ✓ Strong restrictions extracted: {len(state.strong_restrictions)} restriction(s)")
    
    # Validate structured information completeness
    if not state.structured_subject or not state.structured_condition or not state.structured_goal:
        # Check if any dimension is missing
        missing_dims = []
        if not state.structured_subject:
            missing_dims.append("subject")
        if not state.structured_condition:
            missing_dims.append("condition")
        if not state.structured_goal:
            missing_dims.append("goal")
        
        if missing_dims:
            print(f"  ⚠ Missing structured dimensions: {', '.join(missing_dims)}")
            # If any dimension is missing, mark as Severe Missing
            if state.data_completeness_label != "Severe Missing":
                state.data_completeness_label = "Severe Missing"
                print(f"  ⚠ Data completeness updated to Severe Missing due to missing structured dimensions")
    
    # Validate sub-fields completeness
    if state.structured_subject:
        if not state.structured_subject.get("type") or not state.structured_subject.get("attribute"):
            print(f"  ⚠ Subject missing sub-fields (type or attribute)")
            state.data_completeness_label = "Severe Missing"
    if state.structured_condition:
        if not state.structured_condition.get("type") or not state.structured_condition.get("key_features"):
            print(f"  ⚠ Condition missing sub-fields (type or key_features)")
            state.data_completeness_label = "Severe Missing"
    if state.structured_goal:
        if not state.structured_goal.get("type") or not state.structured_goal.get("constraint") or not state.structured_goal.get("intent"):
            print(f"  ⚠ Goal missing sub-fields (type, constraint, or intent)")
            state.data_completeness_label = "Severe Missing"
    
    print(f"✓ Cleaned text: {state.cleaned_text[:100]}...")
    print(f"✓ Question type: {state.question_type_label}")
    print(f"✓ Data completeness: {state.data_completeness_label}")
    print(f"✓ Answer format: {state.answer_format_label}")
    if state.question_options:
        print(f"✓ Options extracted: {len(state.question_options)}")
    
    # Print structured three-dimensional information (结构化三维度信息)
    print(f"\n  📊 Structured Three-Dimensional Information (结构化三维度信息):")
    if state.structured_subject:
        subject_type = state.structured_subject.get('type', 'N/A')
        subject_attr = state.structured_subject.get('attribute', 'N/A')
        print(f"    ✓ Subject: type={subject_type}, attribute={subject_attr[:80]}...")
    else:
        print(f"    ❌ Subject: MISSING")
    
    if state.structured_condition:
        condition_type = state.structured_condition.get('type', 'N/A')
        condition_features = state.structured_condition.get('key_features', 'N/A')
        print(f"    ✓ Condition: type={condition_type}, key_features={condition_features[:80]}...")
    else:
        print(f"    ❌ Condition: MISSING")
    
    if state.structured_goal:
        goal_type = state.structured_goal.get('type', 'N/A')
        goal_constraint = state.structured_goal.get('constraint', 'N/A')
        goal_intent = state.structured_goal.get('intent', 'N/A')
        print(f"    ✓ Goal: type={goal_type}, constraint={goal_constraint[:80]}..., intent={goal_intent}")
    else:
        print(f"    ❌ Goal: MISSING")
    
    # Check if all dimensions are present
    if state.structured_subject and state.structured_condition and state.structured_goal:
        print(f"  ✅ All three dimensions extracted successfully")
    else:
        print(f"  ⚠ WARNING: Missing structured dimensions - this may affect downstream processing")
    
    return state


def n1_question_decomposition_node(state: GeneralQAState) -> GeneralQAState:
    """
    N1: Question Decomposition & Domain Localization
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses basic entity lookup tools to help identify key entities and domains.
    """
    # Input validation
    if not state.cleaned_text:
        state.error_message = "cleaned_text is required for question decomposition"
        return state
    
    print("=" * 60)
    print("N1: Question Decomposition & Domain Localization")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for question decomposition"
        return state
    
    # Detect domain for tool allocation
    domain = detect_domain_from_state(state) if hasattr(state, 'question_type_label') and state.question_type_label else None
    question_type = getattr(state, 'question_type_label', None)
    
    # Load tools for entity lookup
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n1_question_decomposition", domain=domain, question_type=question_type)
            print(f"  📚 Loaded {len(tools)} tool(s) for entity lookup")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    prompt = get_question_decomposition_prompt(
        state.cleaned_text,
        state.question_type_label or "Unknown",
        structured_subject=state.structured_subject,
        structured_condition=state.structured_condition,
        structured_goal=state.structured_goal,
        question_category_standard=state.question_category_standard,
        category_specific_constraints=state.category_specific_constraints or []
    )
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=3, state=state, node_name="n1_question_decomposition")
    if not response:
        state.error_message = "LLM call failed for question decomposition"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for question decomposition"
        return state
    
    state.structured_conditions = result.get("structured_conditions")
    state.core_domains = result.get("core_domains", [])
    state.research_objective = result.get("research_objective")
    state.key_entities = result.get("key_entities", [])
    state.answer_constraints = result.get("answer_constraints", [])
    if not state.answer_constraints:
        state.answer_constraints = _infer_answer_constraints_from_text(state.user_input)
    
    # Enhanced: Extract critical constraints from LLM response or infer from constraints
    critical_constraints = result.get("critical_constraints", [])
    if not critical_constraints and state.structured_conditions and isinstance(state.structured_conditions, dict):
        constraints = state.structured_conditions.get("constraints", [])
        # Extract critical constraints (e.g., "k4影响极大", "理想条件下")
        critical_keywords = ["极大", "extremely large", "ideal", "理想", "关键", "critical", "重要", "important", "much larger", "much smaller"]
        for constraint in constraints:
            constraint_str = str(constraint).lower()
            if any(keyword in constraint_str for keyword in critical_keywords):
                critical_constraints.append(constraint)
    
    if critical_constraints:
        state.critical_constraints = critical_constraints
        print(f"  ⚠ Critical constraints detected: {critical_constraints}")
    else:
        state.critical_constraints = None
    
    # OPTIMIZATION: Extract inference core restrictions (推理必需的核心限制)
    inference_core_restrictions = result.get("inference_core_restrictions", [])
    # Merge with constraints from n0
    if state.negative_constraints:
        inference_core_restrictions.extend(state.negative_constraints)
    if state.exclusive_constraints:
        inference_core_restrictions.extend(state.exclusive_constraints)
    if state.strong_restrictions:
        inference_core_restrictions.extend(state.strong_restrictions)
    if state.key_constraints:
        inference_core_restrictions.extend(state.key_constraints)
    
    # Store in state for downstream nodes
    state.inference_core_restrictions = list(set(inference_core_restrictions)) if inference_core_restrictions else []
    if state.inference_core_restrictions:
        print(f"  ✓ Inference core restrictions extracted: {len(state.inference_core_restrictions)} restriction(s)")
        print(f"    - Restrictions: {state.inference_core_restrictions[:3]}..." if len(state.inference_core_restrictions) > 3 else f"    - Restrictions: {state.inference_core_restrictions}")
    
    # Extract retrieval sub-questions and tool intent
    state.retrieval_sub_questions = result.get("retrieval_sub_questions", [])
    if state.retrieval_sub_questions:
        print(f"  ✓ Retrieval sub-questions generated: {len(state.retrieval_sub_questions)} question(s)")
    
    # Update tool_intent from n1 if provided, otherwise keep from n0
    n1_tool_intent = result.get("tool_intent")
    if n1_tool_intent:
        state.tool_intent = n1_tool_intent
        print(f"  ✓ Tool intent updated: {state.tool_intent}")
    
    print(f"✓ Core domains: {state.core_domains}")
    print(f"✓ Research objective: {state.research_objective}")
    if state.key_entities:
        print(f"✓ Key entities: {state.key_entities}")
    
    return state


def n2_calculation_algorithm_recognition_node(state: GeneralQAState) -> GeneralQAState:
    """
    N2: Calculation/Algorithm Requirement Recognition
    
    Structure: Input validation => Data preparation => Execution => Result organization
    """
    # Input validation
    if not state.cleaned_text:
        state.error_message = "cleaned_text is required for calculation/algorithm recognition"
        return state
    
    print("=" * 60)
    print("N2: Calculation/Algorithm Requirement Recognition")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for calculation/algorithm recognition"
        return state
    
    prompt = get_calculation_algorithm_recognition_prompt(
        state.cleaned_text,
        state.question_type_label or "Unknown"
    )
    
    # Execution
    response = _call_llm(llm, prompt, state=state, node_name="n2_calculation_algorithm_recognition")
    if not response:
        state.error_message = "LLM call failed for calculation/algorithm recognition"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for calculation/algorithm recognition"
        return state
    
    state.calculation_type_label = result.get("calculation_type_label")
    state.key_parameters = result.get("key_parameters")
    
    print(f"✓ Calculation type: {state.calculation_type_label}")
    print(f"✓ Key parameters: {state.key_parameters}")
    
    return state


def n3_knowledge_retrieval_node(state: GeneralQAState) -> GeneralQAState:
    """
    N3: Cross-Domain Knowledge Retrieval
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses ALL available biomedical tools for comprehensive knowledge retrieval.
    Enhanced with:
    1. Keyword-triggered tool selection for forced tool usage
    2. Deep research subgraph for comprehensive research
    3. PaperQA for scientific literature retrieval and evidence gathering
    """
    # Input validation
    if not state.core_domains and not state.calculation_type_label:
        state.error_message = "core_domains or calculation_type_label is required for knowledge retrieval"
        return state
    
    print("=" * 60)
    print("N3: Cross-Domain Knowledge Retrieval")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for knowledge retrieval"
        return state
    
    # ========== Step 1: PaperQA Literature Retrieval (辅助功能，不影响主流程) ==========
    paper_evidence = ""
    paper_confidence = 0.0
    state.paperqa_result = None  # Initialize
    # PaperQA是辅助功能，即使失败也不应该影响知识激活流程
    try:
        from agent.nodes.subagents.paper_qa import safe_paper_pipeline
        import asyncio
        import concurrent.futures
        import traceback
        
        question_text = state.research_objective or state.cleaned_text or state.user_input or ""
        if question_text:
            print(f"  📄 Starting PaperQA literature retrieval...")
            print(f"    - Question: {question_text[:100]}...")
            try:
                # Run paper pipeline asynchronously in a new event loop
                def run_paper_pipeline():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(safe_paper_pipeline(question_text, max_papers=8, timeout=120.0))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_paper_pipeline)
                    try:
                        # PaperQA timeout set to 30 minutes (1800 seconds) to allow for comprehensive literature search
                        paper_timeout = 1800.0
                        paper_result = future.result(timeout=paper_timeout)
                    except concurrent.futures.TimeoutError:
                        print(f"  ❌ PaperQA retrieval failed: TIMEOUT (exceeded {paper_timeout/60:.1f} minutes)")
                        print(f"    - Possible causes:")
                        print(f"      * Network issues connecting to Tavily/Qdrant")
                        print(f"      * Too many papers to process")
                        print(f"      * paper-qa indexing taking too long")
                        print(f"    - Action: Continuing without paper evidence")
                        paper_result = None
                    except Exception as timeout_e:
                        print(f"  ❌ PaperQA retrieval failed: EXECUTION ERROR in thread")
                        print(f"    - Error type: {type(timeout_e).__name__}")
                        print(f"    - Error message: {str(timeout_e)}")
                        print(f"    - Possible causes:")
                        if "timeout" in str(timeout_e).lower():
                            print(f"      * Operation timed out")
                        elif "tavily" in str(timeout_e).lower() or "api" in str(timeout_e).lower():
                            print(f"      * Tavily API issue (check TAVILY_API_KEY)")
                        elif "qdrant" in str(timeout_e).lower() or "connection" in str(timeout_e).lower():
                            print(f"      * Qdrant connection issue (check QDRANT_HOST, QDRANT_PORT)")
                        elif "embedding" in str(timeout_e).lower():
                            print(f"      * Embedding model issue (check EMBEDDING_PROVIDER, EMBEDDING_API_KEY)")
                        elif "paperqa" in str(timeout_e).lower() or "import" in str(timeout_e).lower():
                            print(f"      * paper-qa module issue (may need: pip install paperqa)")
                        else:
                            print(f"      * Unexpected error during execution")
                        print(f"    - Stack trace (last 3 frames):")
                        tb_lines = traceback.format_exc().split('\n')
                        for line in tb_lines[-10:]:
                            if line.strip():
                                print(f"      {line}")
                        paper_result = None
                
                if paper_result:
                    paper_evidence = paper_result.get("evidence_text_block", "")
                    paper_confidence = paper_result.get("confidence", 0.0)
                    # Store PaperQA result in state for logging
                    state.paperqa_result = {
                        "evidence_text_block": paper_evidence,
                        "confidence": paper_confidence,
                        "papers_discovered": paper_result.get('papers_discovered', 0),
                        "papers_indexed": paper_result.get('papers_indexed', 0),
                        "sources": paper_result.get('sources', []),
                        "evidence_items_count": len(paper_result.get('evidence_items', [])),
                        "answer": paper_result.get('answer', ''),
                        "references": paper_result.get('references', ''),
                        "cost": paper_result.get('cost', 0.0)
                    }
                    papers_discovered = paper_result.get('papers_discovered', 0)
                    papers_indexed = paper_result.get('papers_indexed', 0)
                    print(f"  ✓ PaperQA retrieved {papers_discovered} papers")
                    print(f"    - Confidence: {paper_confidence:.2f}")
                    print(f"    - Sources: {', '.join(paper_result.get('sources', []))}")
                    print(f"    - Papers indexed: {papers_indexed}")
                    if papers_indexed == 0 and papers_discovered > 0:
                        print(f"    ⚠ Warning: No papers were indexed into paper-qa (using raw formatting)")
                        print(f"      This may indicate: paper-qa not installed, indexing timeout, or indexing errors")
                elif paper_result is None:
                    # Already handled above
                    state.paperqa_result = {"status": "failed", "reason": "timeout_or_error"}
                else:
                    print(f"  ⚠ PaperQA returned empty result")
                    state.paperqa_result = {"status": "empty"}
            except Exception as e:
                print(f"  ❌ PaperQA retrieval failed: EXCEPTION during execution")
                print(f"    - Error type: {type(e).__name__}")
                print(f"    - Error message: {str(e)}")
                print(f"    - Possible causes:")
                if isinstance(e, ImportError):
                    print(f"      * Missing required module: {e.name if hasattr(e, 'name') else 'unknown'}")
                elif isinstance(e, AttributeError):
                    print(f"      * Missing attribute or method: {str(e)}")
                elif isinstance(e, ValueError):
                    print(f"      * Invalid parameter or configuration")
                elif isinstance(e, RuntimeError):
                    print(f"      * Runtime error in paper pipeline execution")
                else:
                    print(f"      * Unexpected error: {type(e).__name__}")
                print(f"    - Stack trace (last 5 frames):")
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines[-15:]:
                    if line.strip():
                        print(f"      {line}")
                state.paperqa_result = {"status": "failed", "reason": "execution_exception", "error_type": type(e).__name__, "error": str(e)}
    except ImportError as import_e:
        print(f"  ⚠ PaperQA module not available, skipping literature retrieval")
        print(f"    - Import error: {str(import_e)}")
        print(f"    - Missing module: {import_e.name if hasattr(import_e, 'name') else 'unknown'}")
        print(f"    - Action: Install paper_qa module or check import path")
        state.paperqa_result = {"status": "not_available", "reason": "import_error", "error": str(import_e)}
    except Exception as e:
        print(f"  ❌ PaperQA initialization failed")
        print(f"    - Error type: {type(e).__name__}")
        print(f"    - Error message: {str(e)}")
        import traceback
        print(f"    - Stack trace:")
        for line in traceback.format_exc().split('\n')[-10:]:
            if line.strip():
                print(f"      {line}")
        state.paperqa_result = {"status": "failed", "reason": "initialization_error", "error_type": type(e).__name__, "error": str(e)}
    
    # ========== Step 2: Deep Research (辅助功能，不影响主流程) ==========
    deep_research_result = ""
    state.deep_research_result = None  # Initialize
    # Deep Research是辅助功能，即使失败也不应该影响知识激活流程
    try:
        from agent.nodes.subagents.deep_research.deep_researcher import run_deep_research
        import asyncio
        import concurrent.futures
        import traceback
        
        # Only run deep research for complex questions or when paper evidence is insufficient
        should_run_deep_research = (
            paper_confidence < 0.5 or 
            len(state.core_domains or []) > 2 or
            state.question_type_label in ["Mechanism Explanation", "Professional Algorithm"]
        )
        
        if should_run_deep_research:
            question_text = state.research_objective or state.cleaned_text or state.user_input or ""
            if question_text:
                print(f"  🔬 Starting Deep Research analysis...")
                print(f"    - Question: {question_text[:100]}...")
                print(f"    - Trigger reason: ", end="")
                if paper_confidence < 0.5:
                    print(f"PaperQA confidence too low ({paper_confidence:.2f})")
                elif len(state.core_domains or []) > 2:
                    print(f"Multiple domains ({len(state.core_domains)})")
                else:
                    print(f"Question type: {state.question_type_label}")
                
                try:
                    # Run deep research asynchronously in a new event loop
                    def run_deep_research_sync():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(run_deep_research(question_text, return_full_state=False))
                        finally:
                            new_loop.close()
                    
                    # Deep Research typically takes 20-30 minutes, set timeout to 2000 seconds (~33 minutes)
                    # Can be overridden via environment variable DEEP_RESEARCH_TIMEOUT (in seconds)
                    import os
                    import time
                    deep_research_timeout = float(os.getenv("DEEP_RESEARCH_TIMEOUT", "2000.0"))  # Default: 33 minutes
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_deep_research_sync)
                        try:
                            print(f"    - Timeout set to {deep_research_timeout/60:.1f} minutes")
                            print(f"    - Deep Research is running in background...")
                            print(f"    - This may take 20-30 minutes, please wait...")
                            
                            # Add progress monitoring in a separate thread
                            start_time = time.time()
                            progress_interval = 60.0  # Print progress every 60 seconds
                            
                            def progress_monitor():
                                """Monitor and print progress periodically"""
                                last_progress_time = start_time
                                while not future.done():
                                    elapsed = time.time() - start_time
                                    if elapsed >= deep_research_timeout:
                                        break
                                    
                                    # Print progress every 60 seconds
                                    if time.time() - last_progress_time >= progress_interval:
                                        elapsed_minutes = elapsed / 60.0
                                        remaining_minutes = (deep_research_timeout - elapsed) / 60.0
                                        print(f"    ⏳ Deep Research in progress... ({elapsed_minutes:.1f} min elapsed, ~{remaining_minutes:.1f} min remaining)")
                                        last_progress_time = time.time()
                                    
                                    time.sleep(10)  # Check every 10 seconds
                            
                            # Start progress monitor in background
                            import threading
                            progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                            progress_thread.start()
                            
                            # Wait for result
                            deep_research_result = future.result(timeout=deep_research_timeout)
                            
                            elapsed_minutes = (time.time() - start_time) / 60.0
                            print(f"    ✓ Deep Research completed in {elapsed_minutes:.1f} minutes")
                        except concurrent.futures.TimeoutError:
                            print(f"  ❌ Deep Research failed: TIMEOUT (exceeded {deep_research_timeout/60:.1f} minutes)")
                            print(f"    - Possible causes:")
                            print(f"      * Research question too complex")
                            print(f"      * Network issues or API rate limits")
                            print(f"      * Too many research iterations")
                            print(f"    - Note: Deep Research typically takes 20-30 minutes to complete")
                            print(f"    - Action: Continuing without deep research results")
                            deep_research_result = None
                            state.deep_research_result = {"status": "failed", "reason": "timeout", "timeout_seconds": deep_research_timeout}
                        except Exception as timeout_e:
                            print(f"  ❌ Deep Research failed: EXECUTION ERROR in thread")
                            print(f"    - Error type: {type(timeout_e).__name__}")
                            print(f"    - Error message: {str(timeout_e)}")
                            print(f"    - Possible causes:")
                            if "timeout" in str(timeout_e).lower():
                                print(f"      * Operation timed out")
                            elif "connection" in str(timeout_e).lower() or "network" in str(timeout_e).lower():
                                print(f"      * Network connectivity issue")
                            elif "api" in str(timeout_e).lower() or "key" in str(timeout_e).lower():
                                print(f"      * API configuration issue (check API keys)")
                            elif "import" in str(timeout_e).lower() or "module" in str(timeout_e).lower():
                                print(f"      * Missing dependency or import error")
                            else:
                                print(f"      * Unexpected error during execution")
                            print(f"    - Stack trace (last 3 frames):")
                            tb_lines = traceback.format_exc().split('\n')
                            for line in tb_lines[-10:]:  # Show last 10 lines of traceback
                                if line.strip():
                                    print(f"      {line}")
                            deep_research_result = None
                            state.deep_research_result = {"status": "failed", "reason": "execution_error", "error_type": type(timeout_e).__name__, "error": str(timeout_e)}
                    
                    # Store original result before processing
                    deep_research_raw_result = deep_research_result
                    
                    if deep_research_result and isinstance(deep_research_result, dict):
                        research_report = deep_research_result.get("final_report", "")
                        research_brief = deep_research_result.get("research_brief", "")
                        
                        # Store Deep Research result in state for logging
                        state.deep_research_result = {
                            "final_report": research_report,
                            "research_brief": research_brief,
                            "report_length": len(research_report) if research_report else 0,
                            "brief_length": len(research_brief) if research_brief else 0,
                            "message_count": deep_research_result.get("message_count", 0),
                            "thread_id": deep_research_result.get("thread_id", ""),
                            "status": "success"
                        }
                        
                        if research_report:
                            deep_research_result = f"### Deep Research Report\n\n{research_report}\n\n"
                            if research_brief:
                                deep_research_result += f"### Research Brief\n\n{research_brief}\n\n"
                            print(f"  ✓ Deep Research completed successfully")
                            print(f"    - Report length: {len(research_report)} characters")
                        else:
                            print(f"  ⚠ Deep Research completed but no report generated")
                            print(f"    - Result keys: {list(deep_research_result.keys())}")
                            deep_research_result = ""
                    elif deep_research_result is None:
                        # Already handled above
                        state.deep_research_result = {"status": "failed", "reason": "timeout_or_error"}
                    else:
                        print(f"  ⚠ Deep Research returned unexpected result type: {type(deep_research_result)}")
                        state.deep_research_result = {"status": "failed", "reason": "unexpected_result_type", "result_type": str(type(deep_research_result))}
                        deep_research_result = ""
                except Exception as e:
                    print(f"  ❌ Deep Research failed: EXCEPTION during execution")
                    print(f"    - Error type: {type(e).__name__}")
                    print(f"    - Error message: {str(e)}")
                    print(f"    - Possible causes:")
                    if isinstance(e, ImportError):
                        print(f"      * Missing required module: {e.name if hasattr(e, 'name') else 'unknown'}")
                    elif isinstance(e, AttributeError):
                        print(f"      * Missing attribute or method: {str(e)}")
                    elif isinstance(e, ValueError):
                        print(f"      * Invalid parameter or configuration")
                    elif isinstance(e, RuntimeError):
                        print(f"      * Runtime error in deep research execution")
                    else:
                        print(f"      * Unexpected error: {type(e).__name__}")
                    print(f"    - Stack trace (last 5 frames):")
                    tb_lines = traceback.format_exc().split('\n')
                    for line in tb_lines[-15:]:  # Show last 15 lines of traceback
                        if line.strip():
                            print(f"      {line}")
                    state.deep_research_result = {"status": "failed", "reason": "execution_exception", "error_type": type(e).__name__, "error": str(e)}
    except ImportError as import_e:
        print(f"  ⚠ Deep Research module not available, skipping deep research")
        print(f"    - Import error: {str(import_e)}")
        print(f"    - Missing module: {import_e.name if hasattr(import_e, 'name') else 'unknown'}")
        print(f"    - Action: Install deep_research module or check import path")
        state.deep_research_result = {"status": "not_available", "reason": "import_error", "error": str(import_e)}
    except Exception as e:
        print(f"  ❌ Deep Research initialization failed")
        print(f"    - Error type: {type(e).__name__}")
        print(f"    - Error message: {str(e)}")
        import traceback
        print(f"    - Stack trace:")
        for line in traceback.format_exc().split('\n')[-10:]:
            if line.strip():
                print(f"      {line}")
        state.deep_research_result = {"status": "failed", "reason": "initialization_error", "error_type": type(e).__name__, "error": str(e)}
    
    # ========== Step 3: Load and select tools ==========
    # Load all tools first
    all_tools = []
    if TOOLS_AVAILABLE and load_all_tools:
        try:
            all_tools = load_all_tools()
            print(f"  📦 Loaded {len(all_tools)} total tool(s) from tool loader")
        except Exception as e:
            print(f"  ⚠ Failed to load all tools: {e}")
    
    # Smart tool selection based on keywords and domains
    tools = []
    
    # Detect domain for tool allocation
    domain = detect_domain_from_state(state) if hasattr(state, 'question_type_label') and state.question_type_label else None
    question_type = getattr(state, 'question_type_label', None)
    core_domains = getattr(state, 'core_domains', None)
    
    # CRITICAL: N3 should use ALL tools by default, not just keyword-selected ones
    # First, try to get default tools for n3 (which should be all_tools)
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n3_knowledge_retrieval", domain=domain, question_type=question_type)
            print(f"  📚 Loaded {len(tools)} tool(s) from get_tools_for_node('n3_knowledge_retrieval')")
        except Exception as e:
            print(f"  ⚠ Failed to load default tools for n3: {e}")
    
    # If we have all_tools but tools is still empty, use all_tools directly
    if not tools and all_tools:
        tools = all_tools
        print(f"  📚 Using all {len(tools)} tool(s) directly (fallback to all_tools)")
    
    # If we still don't have tools, try keyword-based selection as last resort
    if not tools and TOOLS_AVAILABLE and get_tools_by_keywords and all_tools:
        try:
            # Get tools based on keywords, domains, and entities
            keyword_tools = get_tools_by_keywords(
                text=state.cleaned_text or state.user_input or "",
                domains=state.core_domains,
                key_entities=state.key_entities,
                all_tools=all_tools
            )
            
            if keyword_tools:
                tools = keyword_tools
                print(f"  📚 Loaded {len(tools)} tool(s) via keyword selection (last resort)")
        except Exception as e:
            print(f"  ⚠ Failed to select tools by keywords: {e}")
    
    # Final check: if still no tools, log warning
    if not tools:
        print(f"  ⚠ WARNING: No tools available for n3_knowledge_retrieval!")
        print(f"    - TOOLS_AVAILABLE: {TOOLS_AVAILABLE}")
        print(f"    - load_all_tools available: {load_all_tools is not None}")
        print(f"    - get_tools_for_node available: {get_tools_for_node is not None}")
        print(f"    - all_tools count: {len(all_tools)}")
    else:
        # Check if we should force tool usage
        if should_force_tool_usage and should_force_tool_usage(state.cleaned_text or state.user_input or "", state.core_domains):
            print(f"  🔧 Force tool usage enabled - tools will be actively used")
    
    algorithm_domain = None
    if isinstance(state.key_parameters, dict):
        algorithm_domain = state.key_parameters.get("algorithm_name")
    
    # ========== Step 4: Build enhanced prompt ==========
    # Enhanced prompt with tool usage instructions
    # Detect domain for prompt and tool allocation
    domain = detect_domain_from_state(state) if hasattr(state, 'question_type_label') and state.question_type_label else None
    question_type = getattr(state, 'question_type_label', None)
    
    base_prompt = get_knowledge_retrieval_prompt(
        state.core_domains or [],
        state.calculation_type_label,
        algorithm_domain,
        state.research_objective,
        state.structured_conditions,
        state.key_entities,
        state.answer_format_label,
        state.question_type_label,
        structured_subject=state.structured_subject,
        structured_condition=state.structured_condition,
        domain=domain,
        question_type=question_type,
        structured_goal=state.structured_goal
    )
    
    # OPTIMIZATION: Add constraint information for knowledge filtering
    constraint_filtering_instruction = ""
    if state.inference_core_restrictions:
        constraint_filtering_instruction = "\n\n**CRITICAL: CONSTRAINT-BASED KNOWLEDGE FILTERING (基于约束的知识过滤) - MANDATORY:**\n"
        constraint_filtering_instruction += "You MUST filter retrieved knowledge based on these constraints:\n"
        for i, restriction in enumerate(state.inference_core_restrictions, 1):
            constraint_filtering_instruction += f"{i}. {restriction}\n"
        constraint_filtering_instruction += "\n- Remove ALL knowledge that violates negative constraints (cannot/except/not occur)\n"
        constraint_filtering_instruction += "- Remove ALL knowledge that contradicts exclusive constraints (only 1/category 1)\n"
        constraint_filtering_instruction += "- Only retain knowledge that aligns with these constraints\n"
        base_prompt = base_prompt + constraint_filtering_instruction
        print(f"  ⚠ Added constraint-based filtering instruction: {len(state.inference_core_restrictions)} restriction(s)")
    
    # Add external knowledge sources to prompt
    external_knowledge = ""
    if paper_evidence:
        external_knowledge += f"\n\n{paper_evidence}\n"
    if deep_research_result:
        external_knowledge += f"\n\n{deep_research_result}\n"
    
    # Add tool usage instructions if tools are available
    if tools and len(tools) > 0:
        # Get tool names for display
        tool_names = [tool.name for tool in tools]
        tool_names_str = ", ".join(tool_names[:10])  # Show first 10 tools
        if len(tool_names) > 10:
            tool_names_str += f", ... (and {len(tool_names) - 10} more tools)"
        
        tool_instruction = f"""

**CRITICAL: MANDATORY DATABASE TOOL USAGE (强制使用数据库查询工具) - HIGHEST PRIORITY**

You have access to {len(tools)} biomedical database query tools. These tools provide REAL-TIME, ACCURATE data from authoritative databases.

**MANDATORY RULES:**
1. **YOU MUST USE TOOLS** - Do NOT rely solely on your training data or general knowledge
2. **For ANY question involving:**
   - Drugs/medications → Use drug interaction/query tools (e.g., query_drug_interaction, query_drug_for_disease)
   - Diseases/syndromes → Use disease-gene tools (e.g., query_disgenet, query_omim)
   - Genes/proteins → Use gene/protein tools (e.g., query_gene_info, query_proteinatlas, query_ppi)
   - Clinical guidelines → Use clinical decision tools
   - Expression data → Use expression tools (e.g., query_gtex_expression)
   - Variants/mutations → Use variant tools (e.g., query_variant)
   - Pathways → Use pathway tools
   - Interactions → Use interaction tools (e.g., query_ppi, query_genetic_interaction)
   - Binding/affinity → Use binding tools (e.g., query_bindingdb)
   - Knowledge graphs → Use knowledge graph tools (e.g., query_knowledge_graph)

3. **Tool Usage Pattern:**
   - Step 1: Identify key entities from the question (drug names, gene names, disease names, protein names, etc.)
   - Step 2: For EACH key entity, call the appropriate tool to retrieve real data
   - Step 3: Use the retrieved data to build your knowledge map
   - Step 4: If tool returns no results, try alternative tools or broader queries

4. **Available Tools (sample):** {tool_names_str}

5. **FORBIDDEN:**
   - ❌ Answering based only on training data without tool queries
   - ❌ Skipping tool usage for factual questions
   - ❌ Using general knowledge when specific database queries are available

**ACTION REQUIRED:** Before building your knowledge map, you MUST call at least 2-3 relevant tools to retrieve real data from databases. Your response will be considered INCOMPLETE if you do not use tools for factual queries.

"""
        prompt = base_prompt + external_knowledge + tool_instruction
    else:
        prompt = base_prompt + external_knowledge
    
    # ========== Step 5: Execute with tools ==========
    # OPTIMIZATION: Add structured output requirement to prompt
    structured_output_instruction = "\n\n**CRITICAL: You MUST output ONLY valid JSON. No markdown, no code blocks, no explanations. The response must be parseable JSON starting with { and ending with }.**"
    prompt = prompt + structured_output_instruction
    
    # First attempt
    response = _call_llm(llm, prompt, tools=tools, max_iterations=5, state=state, node_name="n3_knowledge_retrieval")
    if not response:
        state.error_message = "LLM call failed for knowledge retrieval"
        state.exception_type_label = "Knowledge Retrieval Failed"
        return state
    
    # Result organization with retry mechanism
    result = _parse_json_response(response)
    
    # OPTIMIZATION: Parse failure automatic retry (max 1 retry)
    if not result:
        print(f"  ⚠ Failed to parse LLM response, attempting retry...")
        retry_prompt = prompt + "\n\n**RETRY: Your previous response was not valid JSON. Please output ONLY a valid JSON object, no other text.**"
        retry_response = _call_llm(llm, retry_prompt, tools=tools, max_iterations=3, state=state, node_name="n3_knowledge_retrieval_retry")
        if retry_response:
            result = _parse_json_response(retry_response)
            if result:
                print(f"  ✓ Retry succeeded: successfully parsed JSON response")
            else:
                print(f"  ✗ Retry failed: still unable to parse JSON response")
        else:
            print(f"  ✗ Retry failed: LLM call returned no response")
    
    # ========== Step 6: Build domain_knowledge_map from multiple sources (互补而非依赖) ==========
    # 三者（PaperQA、DeepResearch、LLM工具调用）是互补的，任何一个成功都能产生领域知识
    # 如果三者都成功，知识更全面；如果只有部分成功，也能继续执行
    
    # Initialize domain_knowledge_map from LLM tool usage (if successful)
    # 格式标准化：确保 LLM 返回的格式统一为字符串列表
    llm_domain_knowledge_map = {}
    if result:
        raw_llm_map = result.get("domain_knowledge_map", {})
        
        # 标准化格式：确保 foundational_knowledge 和 specialized_knowledge 都是字符串列表
        if isinstance(raw_llm_map, dict):
            for domain, knowledge_dict in raw_llm_map.items():
                if not isinstance(knowledge_dict, dict):
                    continue
                
                llm_domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
                
                # 标准化 foundational_knowledge（确保是字符串列表）
                foundational = knowledge_dict.get("foundational_knowledge", [])
                if isinstance(foundational, list):
                    for item in foundational:
                        if isinstance(item, str):
                            llm_domain_knowledge_map[domain]["foundational_knowledge"].append(item)
                        elif isinstance(item, dict):
                            # 如果是字典格式，提取 knowledge_point 或转换为字符串
                            knowledge_text = item.get("knowledge_point", str(item))
                            llm_domain_knowledge_map[domain]["foundational_knowledge"].append(knowledge_text)
                        else:
                            llm_domain_knowledge_map[domain]["foundational_knowledge"].append(str(item))
                
                # 标准化 specialized_knowledge（确保是字符串列表）
                specialized = knowledge_dict.get("specialized_knowledge", [])
                if isinstance(specialized, list):
                    for item in specialized:
                        if isinstance(item, str):
                            llm_domain_knowledge_map[domain]["specialized_knowledge"].append(item)
                        elif isinstance(item, dict):
                            # 如果是字典格式，提取 knowledge_point 或转换为字符串
                            knowledge_text = item.get("knowledge_point", str(item))
                            llm_domain_knowledge_map[domain]["specialized_knowledge"].append(knowledge_text)
                        else:
                            llm_domain_knowledge_map[domain]["specialized_knowledge"].append(str(item))
        
        # Extract and validate key_facts - ensure all values are strings
        raw_key_facts = result.get("key_facts", {})
        state.key_facts = {}
        if raw_key_facts and isinstance(raw_key_facts, dict):
            for key, value in raw_key_facts.items():
                if isinstance(value, str):
                    state.key_facts[key] = value
                elif isinstance(value, (list, dict)):
                    # Convert list or dict to JSON string
                    try:
                        state.key_facts[key] = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError):
                        state.key_facts[key] = str(value)
                else:
                    # Convert other types to string
                    state.key_facts[key] = str(value)
        
        state.knowledge_validity_label = result.get("knowledge_validity_label", "Missing")
        state.knowledge_unreliable = result.get("knowledge_unreliable", False)
        
        if state.key_facts:
            print(f"  ✓ Key facts extracted: {len(state.key_facts)} fact(s)")
            for key, value in list(state.key_facts.items())[:3]:
                print(f"    - {key}: {value[:80]}..." if len(str(value)) > 80 else f"    - {key}: {value}")
        
        if state.knowledge_unreliable:
            print(f"  ⚠ Knowledge marked as unreliable (tool calls failed)")
        
        # Extract knowledge confidence from LLM result
        knowledge_confidence = result.get("knowledge_confidence")
        if knowledge_confidence is None:
            knowledge_confidence = 0.8 if state.knowledge_validity_label == "Valid" else 0.3
        else:
            knowledge_confidence = float(knowledge_confidence)
            knowledge_confidence = max(0.0, min(1.0, knowledge_confidence))
        
        # Store confidence in parameter_constraints as metadata
        if state.parameter_constraints is None:
            state.parameter_constraints = {}
        if "_knowledge_metadata" not in state.parameter_constraints:
            state.parameter_constraints["_knowledge_metadata"] = {}
        state.parameter_constraints["_knowledge_metadata"]["confidence"] = knowledge_confidence
        print(f"  📊 Knowledge confidence (from LLM): {knowledge_confidence:.2f}")
    else:
        # LLM tool usage failed, but we can still use PaperQA and DeepResearch results
        print(f"  ⚠ LLM tool usage failed, but continuing with PaperQA/DeepResearch results if available")
        state.knowledge_validity_label = "Missing"  # Will be updated if PaperQA/DeepResearch succeed
        state.key_facts = {}
        state.knowledge_unreliable = True
    
    # Build domain_knowledge_map from PaperQA (if successful)
    # 格式统一：foundational_knowledge 和 specialized_knowledge 都是字符串列表
    # 与 LLM 工具调用返回的格式保持一致
    paperqa_domain_knowledge_map = {}
    if paper_evidence and state.paperqa_result and state.paperqa_result.get("status") != "failed":
        print(f"  📄 Converting PaperQA results to domain_knowledge_map format...")
        # Extract domains from PaperQA result or use core_domains
        paperqa_domains = state.core_domains or ["general"]
        for domain in paperqa_domains:
            if domain not in paperqa_domain_knowledge_map:
                paperqa_domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
            # Add PaperQA evidence as specialized knowledge (字符串格式，与 LLM 格式一致)
            if paper_evidence:
                # 格式：字符串，包含知识内容和来源信息
                paperqa_knowledge_str = f"[PaperQA] {paper_evidence[:800]}"  # 限制长度
                if state.paperqa_result.get("papers_discovered", 0) > 0:
                    paperqa_knowledge_str += f" (from {state.paperqa_result.get('papers_discovered', 0)} papers, confidence: {paper_confidence:.2f})"
                paperqa_domain_knowledge_map[domain]["specialized_knowledge"].append(paperqa_knowledge_str)
        print(f"  ✓ PaperQA contributed knowledge to {len(paperqa_domain_knowledge_map)} domain(s)")
    
    # Build domain_knowledge_map from DeepResearch (if successful)
    # 格式统一：foundational_knowledge 和 specialized_knowledge 都是字符串列表
    deepresearch_domain_knowledge_map = {}
    if deep_research_result and state.deep_research_result and state.deep_research_result.get("status") == "success":
        print(f"  🔬 Converting DeepResearch results to domain_knowledge_map format...")
        # Extract domains from DeepResearch result or use core_domains
        deepresearch_domains = state.core_domains or ["general"]
        for domain in deepresearch_domains:
            if domain not in deepresearch_domain_knowledge_map:
                deepresearch_domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
            # Add DeepResearch report as specialized knowledge (字符串格式)
            research_report = state.deep_research_result.get("final_report", "")
            research_brief = state.deep_research_result.get("research_brief", "")
            if research_report:
                # 格式：字符串，包含知识内容和来源信息
                deepresearch_knowledge_str = f"[DeepResearch] {research_report[:1000]}"  # 限制长度
                deepresearch_domain_knowledge_map[domain]["specialized_knowledge"].append(deepresearch_knowledge_str)
            if research_brief:
                # 格式：字符串，包含知识内容和来源信息
                deepresearch_brief_str = f"[DeepResearch-Brief] {research_brief[:500]}"
                deepresearch_domain_knowledge_map[domain]["foundational_knowledge"].append(deepresearch_brief_str)
        print(f"  ✓ DeepResearch contributed knowledge to {len(deepresearch_domain_knowledge_map)} domain(s)")
    
    # Merge all three sources (互补合并)
    state.domain_knowledge_map = {}
    sources_used = []
    
    # Start with LLM tool usage results (if any)
    if llm_domain_knowledge_map:
        state.domain_knowledge_map = llm_domain_knowledge_map.copy()
        sources_used.append("LLM工具调用")
    
    # Merge PaperQA results
    if paperqa_domain_knowledge_map:
        for domain, knowledge in paperqa_domain_knowledge_map.items():
            if domain not in state.domain_knowledge_map:
                state.domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
            # Merge specialized knowledge
            state.domain_knowledge_map[domain]["specialized_knowledge"].extend(
                knowledge.get("specialized_knowledge", [])
            )
            # Merge foundational knowledge
            state.domain_knowledge_map[domain]["foundational_knowledge"].extend(
                knowledge.get("foundational_knowledge", [])
            )
        sources_used.append("PaperQA")
    
    # Merge DeepResearch results
    if deepresearch_domain_knowledge_map:
        for domain, knowledge in deepresearch_domain_knowledge_map.items():
            if domain not in state.domain_knowledge_map:
                state.domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
            # Merge specialized knowledge
            state.domain_knowledge_map[domain]["specialized_knowledge"].extend(
                knowledge.get("specialized_knowledge", [])
            )
            # Merge foundational knowledge
            state.domain_knowledge_map[domain]["foundational_knowledge"].extend(
                knowledge.get("foundational_knowledge", [])
            )
        sources_used.append("DeepResearch")
    
    # Update knowledge validity based on available sources
    if state.domain_knowledge_map and len(state.domain_knowledge_map) > 0:
        if len(sources_used) >= 2:
            state.knowledge_validity_label = "Valid"
            print(f"  ✓ Knowledge merged from {len(sources_used)} source(s): {', '.join(sources_used)}")
        elif len(sources_used) == 1:
            state.knowledge_validity_label = "Valid"
            print(f"  ✓ Knowledge from {sources_used[0]} (single source, but sufficient)")
        else:
            # This shouldn't happen, but handle it
            state.knowledge_validity_label = "Missing"
    else:
        # No knowledge from any source
        print(f"  ❌ No knowledge available from any source (LLM tools, PaperQA, or DeepResearch)")
        state.knowledge_validity_label = "Missing"
    
    # Final validation: if we have domain_knowledge_map from any source, mark as valid
    if state.domain_knowledge_map and len(state.domain_knowledge_map) > 0:
        # Check if we have actual knowledge content (not just empty structures)
        has_actual_knowledge = False
        for domain, knowledge_dict in state.domain_knowledge_map.items():
            if isinstance(knowledge_dict, dict):
                foundational = knowledge_dict.get("foundational_knowledge", [])
                specialized = knowledge_dict.get("specialized_knowledge", [])
                if foundational or specialized:
                    has_actual_knowledge = True
                    break
        
        if has_actual_knowledge:
            if state.knowledge_validity_label == "Missing":
                state.knowledge_validity_label = "Valid"
                print(f"  ✓ Knowledge from {len(sources_used)} source(s) is sufficient, marking as Valid")
        else:
            # Empty knowledge structures
            state.knowledge_validity_label = "Missing"
            print(f"  ⚠ domain_knowledge_map exists but contains no actual knowledge")
    
    # Knowledge confidence calculation (considering all sources)
    if state.parameter_constraints and "_knowledge_metadata" in state.parameter_constraints:
        knowledge_confidence = state.parameter_constraints["_knowledge_metadata"].get("confidence", 0.5)
    else:
        # Calculate confidence based on sources
        if len(sources_used) >= 2:
            knowledge_confidence = 0.8  # Multiple sources = high confidence
        elif len(sources_used) == 1:
            knowledge_confidence = 0.6  # Single source = medium confidence
        else:
            knowledge_confidence = 0.3  # No sources = low confidence
        
        # Store confidence
        if state.parameter_constraints is None:
            state.parameter_constraints = {}
        if "_knowledge_metadata" not in state.parameter_constraints:
            state.parameter_constraints["_knowledge_metadata"] = {}
        state.parameter_constraints["_knowledge_metadata"]["confidence"] = knowledge_confidence
    
    print(f"  📊 Knowledge confidence: {knowledge_confidence:.2f} (from {len(sources_used)} source(s))")
    
    # OPTIMIZATION: Knowledge confidence threshold check (熔断机制)
    CONFIDENCE_THRESHOLD = 0.5  # Lowered threshold since single source is acceptable
    if knowledge_confidence < CONFIDENCE_THRESHOLD:
        print(f"  ⚠ Knowledge confidence ({knowledge_confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD})")
        print(f"    - This knowledge may be unreliable, but will proceed with caution")
        # Mark as low confidence but don't block - let n6/n7 decide
    
    # OPTIMIZATION: Domain-specific rule validation (领域规则校验)
    domain_validation_errors = _validate_biomedical_domain_rules(state.domain_knowledge_map, state.core_domains, state.key_entities)
    if domain_validation_errors:
        print(f"  ⚠ Domain rule validation found {len(domain_validation_errors)} issue(s):")
        for error in domain_validation_errors[:3]:  # Show first 3
            print(f"    - {error}")
        # Mark knowledge as potentially invalid if critical errors
        critical_errors = [e for e in domain_validation_errors if "CRITICAL" in e.upper()]
        if critical_errors:
            print(f"  ❌ Critical domain rule violations detected, marking knowledge as Invalid")
            state.knowledge_validity_label = "Invalid"
            state.exception_type_label = "Knowledge Domain Rule Violation"
    
    # Fix 5: Code-level filtering of knowledge based on Goal
    if state.domain_knowledge_map and state.structured_goal:
        goal_type = state.structured_goal.get("type", "").lower()
        goal_constraint = state.structured_goal.get("constraint", "").lower()
        goal_intent = state.structured_goal.get("intent", "").lower()
        
        # Filter knowledge that is not related to the goal
        filtered_domain_knowledge_map = {}
        removed_count = 0
        
        for domain, knowledge_dict in state.domain_knowledge_map.items():
            if not isinstance(knowledge_dict, dict):
                continue
            
            filtered_knowledge = {
                "foundational_knowledge": [],
                "specialized_knowledge": []
            }
            
            # Check foundational_knowledge
            # 统一格式处理：支持字符串和字典两种格式（向后兼容）
            for item in knowledge_dict.get("foundational_knowledge", []):
                # 如果是字典格式（旧格式），提取 knowledge_point 或转换为字符串
                if isinstance(item, dict):
                    knowledge_text = item.get("knowledge_point", str(item))
                else:
                    knowledge_text = str(item)
                
                if _is_knowledge_relevant_to_goal(knowledge_text, goal_type, goal_constraint, goal_intent):
                    # 统一为字符串格式（与 LLM 格式一致）
                    filtered_knowledge["foundational_knowledge"].append(knowledge_text)
                else:
                    removed_count += 1
            
            # Check specialized_knowledge
            # 统一格式处理：支持字符串和字典两种格式（向后兼容）
            for item in knowledge_dict.get("specialized_knowledge", []):
                # 如果是字典格式（旧格式），提取 knowledge_point 或转换为字符串
                if isinstance(item, dict):
                    knowledge_text = item.get("knowledge_point", str(item))
                else:
                    knowledge_text = str(item)
                
                if _is_knowledge_relevant_to_goal(knowledge_text, goal_type, goal_constraint, goal_intent):
                    # 统一为字符串格式（与 LLM 格式一致）
                    filtered_knowledge["specialized_knowledge"].append(knowledge_text)
                else:
                    removed_count += 1
            
            # Only keep domain if it has relevant knowledge
            if filtered_knowledge["foundational_knowledge"] or filtered_knowledge["specialized_knowledge"]:
                filtered_domain_knowledge_map[domain] = filtered_knowledge
        
        if removed_count > 0:
            print(f"  ✓ Filtered out {removed_count} irrelevant knowledge items based on goal")
            print(f"    - Goal: {goal_type} / {goal_constraint} / {goal_intent}")
            state.domain_knowledge_map = filtered_domain_knowledge_map
    
    # Enhanced: Consider external knowledge sources in validity assessment
    if paper_evidence or deep_research_result:
        # If we have external knowledge, knowledge should be valid
        if state.knowledge_validity_label == "Missing":
            state.knowledge_validity_label = "Valid"
            print(f"  ✓ External knowledge sources improved knowledge validity")
    
    if state.knowledge_validity_label == "Missing":
        state.exception_type_label = "Knowledge Missing"
    
    print(f"✓ Knowledge validity: {state.knowledge_validity_label}")
    print(f"✓ Domains retrieved: {list(state.domain_knowledge_map.keys()) if state.domain_knowledge_map else []}")
    
    # ========== Step 6: Extract parameter constraints from knowledge base ==========
    # Extract parameter constraints for calculation problems
    if state.calculation_type_label == "Numerical" and state.domain_knowledge_map:
        parameter_constraints = _extract_parameter_constraints(state.domain_knowledge_map, state.key_parameters)
        if parameter_constraints:
            state.parameter_constraints = parameter_constraints
            print(f"✓ Parameter constraints extracted: {len(parameter_constraints)} parameter(s)")
            for param_name, constraints in parameter_constraints.items():
                if "range" in constraints:
                    print(f"    - {param_name}: range {constraints['range']}")
                if "sign" in constraints:
                    print(f"    - {param_name}: sign constraint = {constraints['sign']}")
    
    return state


def _extract_parameter_constraints(domain_knowledge_map: Dict[str, Dict[str, Any]], key_parameters: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter constraints from knowledge base.
    
    Args:
        domain_knowledge_map: Domain-knowledge mapping table
        key_parameters: Key parameters extracted from question
    
    Returns:
        Dictionary mapping parameter names to their constraints
        Format: {param_name: {range: {min, max}, sign: "positive"/"negative", physical_constraints: [...]}}
    """
    import re
    constraints = {}
    
    if not domain_knowledge_map:
        return constraints
    
    # Extract parameter names from key_parameters
    param_names = []
    if isinstance(key_parameters, dict):
        # Look for parameters in formula_clues and parameters list
        formula_clues = key_parameters.get("formula_clues", [])
        parameters = key_parameters.get("parameters", [])
        
        # Extract parameter names from formula clues (e.g., "B22(steric)" from "B22(steric) = ...")
        for clue in formula_clues:
            if isinstance(clue, str):
                # Match patterns like "B22(steric)", "B22(electrostatic)", etc.
                matches = re.findall(r'([A-Za-z0-9_]+)\([^)]+\)', clue)
                param_names.extend(matches)
                # Also match simple parameter names
                matches = re.findall(r'\b([A-Z][a-z]*\d*)\b', clue)
                param_names.extend(matches)
        
        # Extract from parameters list
        for param in parameters:
            if isinstance(param, str):
                # Try to extract parameter name
                matches = re.findall(r'([A-Za-z0-9_]+)\([^)]+\)', param)
                param_names.extend(matches)
                matches = re.findall(r'\b([A-Z][a-z]*\d*)\b', param)
                param_names.extend(matches)
    
    # Search knowledge base for constraints
    for domain, knowledge_dict in domain_knowledge_map.items():
        if not isinstance(knowledge_dict, dict):
            continue
        
        # Check both foundational and specialized knowledge
        for knowledge_type in ["foundational_knowledge", "specialized_knowledge"]:
            knowledge_list = knowledge_dict.get(knowledge_type, [])
            for item in knowledge_list:
                if not isinstance(item, str):
                    continue
                
                item_lower = item.lower()
                
                # Extract range constraints (e.g., "range of +6 to +12 mL/g", "between 6 and 12")
                range_patterns = [
                    r'range\s+of\s+([+-]?[\d.]+)\s+to\s+([+-]?[\d.]+)',
                    r'between\s+([+-]?[\d.]+)\s+and\s+([+-]?[\d.]+)',
                    r'([+-]?[\d.]+)\s+to\s+([+-]?[\d.]+)',
                    r'approximately\s+([+-]?[\d.]+)\s+to\s+([+-]?[\d.]+)',
                ]
                
                for pattern in range_patterns:
                    matches = re.finditer(pattern, item_lower)
                    for match in matches:
                        try:
                            min_val = float(match.group(1))
                            max_val = float(match.group(2))
                            
                            # Try to find which parameter this applies to
                            for param_name in param_names:
                                if param_name.lower() in item_lower[:match.start()] or param_name.lower() in item_lower[match.end():]:
                                    if param_name not in constraints:
                                        constraints[param_name] = {}
                                    constraints[param_name]["range"] = {"min": min_val, "max": max_val}
                                    break
                        except (ValueError, IndexError):
                            continue
                
                # Extract sign constraints (e.g., "always positive", "始终为正", "negative value")
                sign_patterns = [
                    (r'always\s+positive|始终为正|始终为正的|always\s+positive\s+value', "positive"),
                    (r'always\s+negative|始终为负|始终为负的|always\s+negative\s+value', "negative"),
                    (r'must\s+be\s+positive|必须为正|必须为正的', "positive"),
                    (r'must\s+be\s+negative|必须为负|必须为负的', "negative"),
                ]
                
                for pattern, sign in sign_patterns:
                    if re.search(pattern, item_lower):
                        # Try to find which parameter this applies to
                        for param_name in param_names:
                            if param_name.lower() in item_lower:
                                if param_name not in constraints:
                                    constraints[param_name] = {}
                                constraints[param_name]["sign"] = sign
                                break
                
                # Extract physical constraints (e.g., "depends on pH", "varies with ionic strength")
                physical_keywords = ["depends on", "varies with", "function of", "related to", "influenced by"]
                for keyword in physical_keywords:
                    if keyword in item_lower:
                        # Try to find which parameter this applies to
                        for param_name in param_names:
                            if param_name.lower() in item_lower:
                                if param_name not in constraints:
                                    constraints[param_name] = {}
                                if "physical_constraints" not in constraints[param_name]:
                                    constraints[param_name]["physical_constraints"] = []
                                # Extract the constraint text
                                constraint_text = item[max(0, item_lower.find(keyword)-50):min(len(item), item_lower.find(keyword)+100)]
                                if constraint_text not in constraints[param_name]["physical_constraints"]:
                                    constraints[param_name]["physical_constraints"].append(constraint_text.strip())
                                break
    
    return constraints


def _validate_biomedical_domain_rules(domain_knowledge_map: Optional[Dict[str, Dict[str, Any]]], core_domains: Optional[List[str]], key_entities: Optional[List[str]]) -> List[str]:
    """
    Validate biomedical domain-specific rules (领域规则校验)
    
    Args:
        domain_knowledge_map: Domain-knowledge mapping
        core_domains: Core domains identified
        key_entities: Key entities from question
    
    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    if not domain_knowledge_map:
        return errors
    
    # Extract domain keywords for rule matching
    domains_lower = [d.lower() for d in (core_domains or [])]
    entities_lower = [e.lower() for e in (key_entities or [])]
    
    # Rule 1: Fluorescence wavelength matching (荧光波长匹配规则)
    # If question mentions fluorescence/probe/excitation, check wavelength consistency
    has_fluorescence_keywords = any(kw in str(entities_lower) for kw in ['fluorescence', 'fluorescent', 'probe', 'excitation', 'wavelength', 'nm', '488', '559', '630'])
    if has_fluorescence_keywords:
        # Check for wavelength-knowledge consistency
        for domain, knowledge_dict in domain_knowledge_map.items():
            if not isinstance(knowledge_dict, dict):
                continue
            for knowledge_type in ["foundational_knowledge", "specialized_knowledge"]:
                for item in knowledge_dict.get(knowledge_type, []):
                    if not isinstance(item, str):
                        continue
                    item_lower = item.lower()
                    # Check for wavelength mismatches (e.g., eGFP should be ~488nm, DsRed should be ~559nm)
                    if 'egfp' in item_lower or 'green' in item_lower:
                        if '630' in item or '559' in item:
                            errors.append(f"CRITICAL: Wavelength mismatch - eGFP typically uses 488nm excitation, not 630nm or 559nm")
                    if 'dsred' in item_lower or 'red' in item_lower:
                        if '488' in item and '630' not in item:
                            errors.append(f"CRITICAL: Wavelength mismatch - DsRed typically uses 559nm excitation, not 488nm")
    
    # Rule 2: DNA/RNA sequence direction (序列方向规则)
    # If question mentions DNA/RNA sequence, check for 5'/3' orientation
    has_sequence_keywords = any(kw in str(entities_lower) for kw in ['dna', 'rna', 'sequence', 'oligo', '5\'', '3\'', 'codon'])
    if has_sequence_keywords:
        sequence_found = False
        orientation_found = False
        for domain, knowledge_dict in domain_knowledge_map.items():
            if not isinstance(knowledge_dict, dict):
                continue
            for knowledge_type in ["foundational_knowledge", "specialized_knowledge"]:
                for item in knowledge_dict.get(knowledge_type, []):
                    if not isinstance(item, str):
                        continue
                    if any(kw in item.lower() for kw in ['sequence', 'dna', 'rna', 'codon']):
                        sequence_found = True
                    if '5\'' in item or '3\'' in item or 'five prime' in item.lower() or 'three prime' in item.lower():
                        orientation_found = True
        if sequence_found and not orientation_found:
            errors.append(f"WARNING: DNA/RNA sequence knowledge found but missing 5'/3' orientation information")
    
    # Rule 3: Drug/BUD time constraints (药剂时间约束规则)
    # If question mentions BUD (beyond use date), check for time unit consistency
    has_bud_keywords = any(kw in str(entities_lower) for kw in ['bud', 'beyond use date', 'sterile', 'puncture', 'hour', 'ampule'])
    if has_bud_keywords:
        time_found = False
        for domain, knowledge_dict in domain_knowledge_map.items():
            if not isinstance(knowledge_dict, dict):
                continue
            for knowledge_type in ["foundational_knowledge", "specialized_knowledge"]:
                for item in knowledge_dict.get(knowledge_type, []):
                    if not isinstance(item, str):
                        continue
                    if any(kw in item.lower() for kw in ['hour', 'minute', 'day', 'time']):
                        time_found = True
        if has_bud_keywords and not time_found:
            errors.append(f"WARNING: BUD question but no time-related knowledge retrieved")
    
    return errors


def _is_knowledge_relevant_to_goal(knowledge_item: str, goal_type: str, goal_constraint: str, goal_intent: str) -> bool:
    """
    Fix 5: Check if knowledge item is relevant to the goal
    
    Args:
        knowledge_item: Knowledge text to check
        goal_type: Goal type (e.g., "conclusion judgment", "calculation result")
        goal_constraint: Goal constraint (e.g., "recommend 3 medications", "determine ordering")
        goal_intent: Goal intent (e.g., "ask_defect", "ask_cause", "neutral")
    
    Returns:
        True if knowledge is relevant, False otherwise
    """
    if not knowledge_item:
        return False
    
    knowledge_lower = str(knowledge_item).lower()
    
    # Extract key terms from goal
    goal_terms = []
    if goal_type:
        goal_terms.extend(goal_type.split())
    if goal_constraint:
        # Extract key terms from constraint (e.g., "recommend 3 medications" -> ["recommend", "medications"])
        goal_terms.extend([term for term in goal_constraint.split() if len(term) > 3])
    
    # Check if knowledge contains goal-related terms
    if goal_terms:
        # Check if any goal term appears in knowledge
        if any(term in knowledge_lower for term in goal_terms if len(term) > 3):
            return True
    
    # Special handling for intent-based filtering
    if goal_intent == "ask_defect":
        # For "ask_defect", only keep knowledge about limitations, drawbacks, errors
        defect_keywords = ["limitation", "drawback", "disadvantage", "error", "assumption", "simplification", "weakness", "defect", "flaw"]
        advantage_keywords = ["advantage", "benefit", "strength", "useful", "effective", "robust", "comprehensive"]
        
        # If knowledge contains advantage keywords but no defect keywords, it's not relevant
        if any(kw in knowledge_lower for kw in advantage_keywords) and not any(kw in knowledge_lower for kw in defect_keywords):
            return False
    
    # For "recommend medications" type goals, filter out knowledge about other diseases/problems
    if "recommend" in goal_constraint and "medication" in goal_constraint:
        # Extract the target condition (e.g., "HTN" from "recommend 3 medications for HTN")
        # This is a simple heuristic - in practice, you might want more sophisticated extraction
        if "htn" in goal_constraint or "hypertension" in goal_constraint:
            # Filter out knowledge about other conditions (e.g., hypercholesterolemia, hypothyroidism)
            irrelevant_conditions = ["hypercholesterolemia", "hypothyroidism", "diabetes", "hyperlipidemia"]
            if any(condition in knowledge_lower for condition in irrelevant_conditions):
                # Only keep if it's also about HTN
                if "htn" not in knowledge_lower and "hypertension" not in knowledge_lower:
                    return False
    
    # Default: keep the knowledge if we can't determine irrelevance
    return True


def n4_calculation_decomposition_node(state: GeneralQAState) -> GeneralQAState:
    """
    N4: Calculation Step Decomposition & Formula Matching
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses expression and variant tools for calculation support.
    Enhanced with formula validation using binding database tools.
    """
    # Input validation
    if not state.cleaned_text or not state.key_parameters or state.calculation_type_label != "Numerical":
        state.error_message = "Invalid state for calculation decomposition (requires Numerical calculation type)"
        return state
    
    print("=" * 60)
    print("N4: Calculation Step Decomposition & Formula Matching")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for calculation decomposition"
        return state
    
    # Load tools for calculation support, including binding affinity tools
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n4_calculation_decomposition")
            # Add binding affinity tools if the question involves binding
            if load_all_tools:
                all_tools = load_all_tools()
                binding_tools = [t for t in all_tools if "binding" in t.name.lower()]
                tools.extend(binding_tools)
                # Deduplicate
                tool_names = {t.name for t in tools}
                tools = [t for t in tools if t.name in tool_names or tool_names.add(t.name)]
            print(f"  📚 Loaded {len(tools)} tool(s) for calculation support")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    # Include critical constraints in prompt if available
    enhanced_key_parameters = state.key_parameters.copy() if isinstance(state.key_parameters, dict) else {}
    if state.critical_constraints:
        enhanced_key_parameters["critical_constraints"] = state.critical_constraints
        print(f"  ⚠ Including critical constraints: {state.critical_constraints}")
    
    prompt = get_calculation_decomposition_prompt(
        state.cleaned_text,
        enhanced_key_parameters,
        state.domain_knowledge_map or {}
    )
    
    # Add formula validation instruction
    if tools:
        validation_instruction = "\n\nIMPORTANT: For binding affinity calculations, you MUST verify the formula using the binding database tools. "
        validation_instruction += "Do not assume simple linear relationships. Consider cooperative binding models and complex binding kinetics."
        prompt = prompt + validation_instruction
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=3, state=state, node_name="n4_calculation_decomposition")
    if not response:
        state.error_message = "LLM call failed for calculation decomposition"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        # 记录更详细的错误信息
        print(f"  ❌ Failed to parse LLM response for calculation decomposition")
        print(f"    - Response length: {len(response) if response else 0} characters")
        print(f"    - Response preview: {response[:200] if response else 'N/A'}...")
        print(f"    - Attempting to create fallback calculation steps from key_parameters")
        
        # 创建基本的fallback calculation_steps，允许流程继续
        if state.key_parameters and isinstance(state.key_parameters, dict):
            formula_clues = state.key_parameters.get("formula_clues", [])
            parameters = state.key_parameters.get("parameters", [])
            
            # 构建基本的计算步骤（通用fallback，不限制步骤数量）
            fallback_steps = []
            
            # 优先使用formula_clues，如果没有则使用parameters
            source_items = formula_clues if formula_clues else (parameters if isinstance(parameters, list) else [])
            
            if source_items:
                # 使用所有可用的线索/参数，不限制数量
                for i, item in enumerate(source_items, 1):
                    item_str = str(item)
                    # 动态截断，保持合理的描述长度（避免过长）
                    max_desc_length = 200
                    description = item_str[:max_desc_length] + ("..." if len(item_str) > max_desc_length else "")
                    fallback_steps.append({
                        "step_number": i,
                        "step_description": f"Analysis: {description}",
                        "step_type": "objective"
                    })
            
            if fallback_steps:
                state.calculation_steps = fallback_steps
                state.formula_match_result = "Match Failed"  # 标记为失败，但允许继续
                state.matched_formula = None
                state.unit_conversion_rules = []
                print(f"  ⚠ Created {len(fallback_steps)} fallback calculation steps from key_parameters")
                print(f"  ⚠ Formula match marked as 'Match Failed' but allowing process to continue")
            else:
                state.error_message = "Failed to parse LLM response for calculation decomposition and no fallback available"
                return state
        else:
            state.error_message = "Failed to parse LLM response for calculation decomposition and no key_parameters available"
            return state
    else:
        # 正常解析成功
        state.calculation_steps = result.get("calculation_steps", [])
        state.matched_formula = result.get("matched_formula")
        state.unit_conversion_rules = result.get("unit_conversion_rules", [])
        state.formula_match_result = result.get("formula_match_result")
    
    # Enhanced: Validate formula if it involves binding affinity
    if state.matched_formula and isinstance(state.matched_formula, dict):
        formula_name = state.matched_formula.get("formula_name", "").lower()
        if "binding" in formula_name or "affinity" in formula_name or "kd" in formula_name.lower():
            # Check if formula is too simple (linear relationship)
            formula_expr = state.matched_formula.get("formula_expression", "")
            if "n-1" in formula_expr or "n*" in formula_expr:
                print(f"  ⚠ Warning: Formula appears to be a simple linear model. Consider cooperative binding.")
                # Don't fail, but add a warning
    
    if state.formula_match_result == "Match Failed":
        state.exception_type_label = "Formula Match Failed"
    
    print(f"✓ Formula match result: {state.formula_match_result}")
    print(f"✓ Calculation steps: {len(state.calculation_steps)} steps")
    
    return state


def n5_algorithm_validation_node(state: GeneralQAState) -> GeneralQAState:
    """
    N5: Algorithm Parameter Extraction & Applicability Validation
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses pathway and interaction tools for algorithm validation.
    """
    # Input validation - 增强容错处理
    if not state.cleaned_text:
        state.error_message = "cleaned_text is required for algorithm validation"
        return state
    
    if not state.key_parameters:
        state.error_message = "key_parameters is required for algorithm validation"
        return state
    
    # 如果calculation_type_label不是"Algorithm"，尝试容错处理
    if state.calculation_type_label != "Algorithm":
        # 检查是否有algorithm_name，如果有则可能是误分类，允许继续
        if isinstance(state.key_parameters, dict) and state.key_parameters.get("algorithm_name"):
            print(f"  ⚠ calculation_type_label='{state.calculation_type_label}' is not 'Algorithm', but algorithm_name found, allowing to continue")
            # 更新calculation_type_label以匹配
            state.calculation_type_label = "Algorithm"
        else:
            # 没有algorithm_name，创建基本的algorithm_parameters作为fallback，允许流程继续
            print(f"  ⚠ calculation_type_label='{state.calculation_type_label}' is not 'Algorithm' and no algorithm_name found")
            print(f"  ⚠ Creating fallback algorithm_parameters to allow process to continue")
            
            # 创建基本的algorithm_parameters
            if isinstance(state.key_parameters, dict):
                state.algorithm_parameters = {
                    "required_parameters": state.key_parameters.get("parameters", {}),
                    "optional_parameters": {}
                }
                state.applicability_result = "Applicable"  # 标记为适用，允许继续
                state.alternative_algorithms = []
                print(f"  ✓ Created fallback algorithm_parameters from key_parameters")
                print(f"  ⚠ Note: This is a fallback, actual algorithm validation was skipped")
                # 直接返回，跳过实际的LLM调用
                return state
            else:
                state.error_message = f"Invalid state for algorithm validation: calculation_type_label='{state.calculation_type_label}' is not 'Algorithm' and key_parameters is not a dict"
                return state
    
    print("=" * 60)
    print("N5: Algorithm Parameter Extraction & Applicability Validation")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for algorithm validation"
        return state
    
    # Load tools for algorithm validation
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n5_algorithm_validation")
            print(f"  📚 Loaded {len(tools)} tool(s) for algorithm validation")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    algorithm_name = state.key_parameters.get("algorithm_name", "Unknown") if isinstance(state.key_parameters, dict) else "Unknown"
    
    prompt = get_algorithm_validation_prompt(
        state.cleaned_text,
        algorithm_name,
        state.domain_knowledge_map or {}
    )
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=3, state=state, node_name="n5_algorithm_validation")
    if not response:
        state.error_message = "LLM call failed for algorithm validation"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for algorithm validation"
        return state
    
    state.algorithm_parameters = result.get("algorithm_parameters")
    state.applicability_result = result.get("applicability_result")
    state.alternative_algorithms = result.get("alternative_algorithms", [])
    if state.applicability_result == "Not Applicable":
        state.exception_type_label = "Algorithm Not Applicable"
    
    print(f"✓ Applicability result: {state.applicability_result}")
    print(f"✓ Algorithm parameters: {state.algorithm_parameters}")
    
    return state


def n6_initial_inference_node(state: GeneralQAState) -> GeneralQAState:
    """
    N6: Initial Association Inference
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses knowledge retrieval tools for association inference.
    """
    # OPTIMIZATION: Knowledge confidence threshold check (推理前置校验)
    knowledge_confidence = 0.5  # Default
    if state.parameter_constraints and isinstance(state.parameter_constraints, dict):
        metadata = state.parameter_constraints.get("_knowledge_metadata", {})
        if isinstance(metadata, dict):
            knowledge_confidence = metadata.get("confidence", 0.5)
    
    CONFIDENCE_THRESHOLD = 0.7
    if knowledge_confidence < CONFIDENCE_THRESHOLD:
        print(f"  ⚠ Knowledge confidence ({knowledge_confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD})")
        print(f"    - Knowledge may be unreliable, but proceeding with caution")
        # Don't block, but mark as low confidence
    
    # OPTIMIZATION: Check knowledge validity before inference (知识合法性校验)
    if state.knowledge_validity_label == "Invalid":
        print(f"  ❌ Knowledge marked as Invalid, cannot proceed with inference")
        state.exception_type_label = "Knowledge Invalid - Cannot Infer"
        state.error_message = "Knowledge retrieval returned invalid knowledge, cannot perform inference"
        return state
    
    # Input validation - 放宽条件，允许在缺少部分信息时继续执行
    if not state.structured_conditions:
        # 如果没有structured_conditions，尝试从其他字段构建
        state.structured_conditions = {
            "objective_conditions": [],
            "experimental_settings": [],
            "constraints": state.answer_constraints or []
        }
        if state.research_objective:
            state.structured_conditions["objective_conditions"] = [state.research_objective]
        print(f"  ⚠ structured_conditions not available, using fallback")
    
    # Try fallback for domain_knowledge_map BEFORE checking if it's empty
    if not state.domain_knowledge_map or (isinstance(state.domain_knowledge_map, dict) and len(state.domain_knowledge_map) == 0):
        # 如果没有domain_knowledge_map，尝试从core_domains构建基本映射
        if state.core_domains:
            state.domain_knowledge_map = {}
            for domain in state.core_domains:
                state.domain_knowledge_map[domain] = {
                    "foundational_knowledge": [],
                    "specialized_knowledge": []
                }
            print(f"  ⚠ domain_knowledge_map not available, using fallback from core_domains")
        elif state.cleaned_text or state.user_input:
            # 如果连core_domains都没有，尝试从问题文本中提取基本知识
            question_text = state.cleaned_text or state.user_input
            state.domain_knowledge_map = {
                "general": {
                    "foundational_knowledge": [f"Question context: {question_text[:200]}"],
                    "specialized_knowledge": []
                }
            }
            print(f"  ⚠ domain_knowledge_map not available, using minimal fallback from question text")
        else:
            # 所有fallback都失败，返回错误
            print(f"  ❌ No valid knowledge available, cannot perform inference")
            state.exception_type_label = "Knowledge Missing - Cannot Infer"
            state.error_message = "No domain knowledge available for inference"
            return state
    
    print("=" * 60)
    print("N6: Initial Association Inference")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for initial inference"
        return state
    
    # Load tools for knowledge retrieval
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n6_initial_inference")
            print(f"  📚 Loaded {len(tools)} tool(s) for association inference")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    prompt = get_initial_inference_prompt(
        state.structured_conditions,
        state.domain_knowledge_map,
        state.research_objective,
        state.key_entities,
        state.answer_constraints,
        state.answer_format_label,
        state.question_type_label
    )
    
    # OPTIMIZATION: Pass constraint information to n6 for logical validation
    if state.inference_core_restrictions:
        constraint_info = f"\n\n**CRITICAL CONSTRAINTS FOR VALIDATION:**\n"
        for i, restriction in enumerate(state.inference_core_restrictions, 1):
            constraint_info += f"{i}. {restriction}\n"
        constraint_info += "\nYou MUST verify each knowledge match against these constraints. Remove matches that violate constraints."
        prompt = prompt + constraint_info
        print(f"  ⚠ Added constraint information for logical validation: {len(state.inference_core_restrictions)} restriction(s)")
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=4, state=state, node_name="n6_initial_inference")
    if not response:
        state.error_message = "LLM call failed for initial inference"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        print(f"  ❌ Failed to parse LLM response for initial inference")
        print(f"    - Response preview: {response[:300] if response else 'N/A'}...")
        state.error_message = "Failed to parse LLM response for initial inference"
        return state
    
    state.phenomenon_knowledge_match_table = result.get("phenomenon_knowledge_match_table")
    state.core_molecular_function = result.get("core_molecular_function")
    state.match_confidence_label = result.get("match_confidence_label")
    state.need_recheck = result.get("need_recheck", False)
    
    # Diagnostic: Check if required fields are missing
    if state.phenomenon_knowledge_match_table is None:
        print(f"  ⚠ phenomenon_knowledge_match_table is missing from LLM response")
        print(f"    - Available keys in result: {list(result.keys()) if result else 'N/A'}")
        print(f"    - This may indicate the prompt format needs adjustment or LLM did not follow instructions")
        # Try to extract from alternative field names (backward compatibility)
        if "initial_associations" in result:
            print(f"    - Found 'initial_associations' field (old format), attempting conversion...")
            # Convert old format to new format if possible
            initial_associations = result.get("initial_associations", [])
            if initial_associations and isinstance(initial_associations, list):
                # Create a basic match table from initial_associations
                state.phenomenon_knowledge_match_table = {}
                for assoc in initial_associations[:5]:  # Limit to first 5
                    if isinstance(assoc, dict):
                        entity1 = assoc.get("entity1", "")
                        entity2 = assoc.get("entity2", "")
                        if entity1 or entity2:
                            domain = "general"  # Default domain
                            if domain not in state.phenomenon_knowledge_match_table:
                                state.phenomenon_knowledge_match_table[domain] = {
                                    "phenomena": [],
                                    "matched_knowledge": [],
                                    "confidence": "Medium"
                                }
                            if entity1:
                                state.phenomenon_knowledge_match_table[domain]["phenomena"].append(entity1)
                            if entity2:
                                state.phenomenon_knowledge_match_table[domain]["phenomena"].append(entity2)
                            evidence = assoc.get("evidence", "")
                            if evidence:
                                state.phenomenon_knowledge_match_table[domain]["matched_knowledge"].append(evidence)
                if state.phenomenon_knowledge_match_table:
                    print(f"    ✓ Converted from old format: {len(state.phenomenon_knowledge_match_table)} domain(s)")
                    state.match_confidence_label = state.match_confidence_label or "Medium"
    
    if state.core_molecular_function:
        print(f"  ✓ Core molecular function: {state.core_molecular_function}")
    
    if state.need_recheck:
        print(f"  ⚠ Inference needs recheck (knowledge unreliable)")
    
    if state.phenomenon_knowledge_match_table:
        match_count = sum(len(v.get("phenomena", [])) if isinstance(v, dict) else 0 for v in state.phenomenon_knowledge_match_table.values())
        print(f"  ✓ Match table created: {len(state.phenomenon_knowledge_match_table)} domain(s), {match_count} total matches")
    
    print(f"✓ Match confidence: {state.match_confidence_label}")
    
    return state


def n7_complete_inference_node(state: GeneralQAState) -> GeneralQAState:
    """
    N7: Complete Logical Inference (with dynamic/calculation derivation)
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses ALL available tools for comprehensive reasoning and inference.
    Enhanced to consider critical constraints in inference.
    """
    # OPTIMIZATION: Knowledge confidence threshold check (推理前置校验)
    knowledge_confidence = 0.5  # Default
    if state.parameter_constraints and isinstance(state.parameter_constraints, dict):
        metadata = state.parameter_constraints.get("_knowledge_metadata", {})
        if isinstance(metadata, dict):
            knowledge_confidence = metadata.get("confidence", 0.5)
    
    CONFIDENCE_THRESHOLD = 0.7
    if knowledge_confidence < CONFIDENCE_THRESHOLD:
        print(f"  ⚠ Knowledge confidence ({knowledge_confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD})")
        print(f"    - Proceeding with caution, inference may be unreliable")
    
    # OPTIMIZATION: Check knowledge validity before inference (知识合法性校验)
    if state.knowledge_validity_label == "Invalid":
        print(f"  ❌ Knowledge marked as Invalid, cannot proceed with complete inference")
        state.exception_type_label = "Knowledge Invalid - Cannot Infer"
        state.error_message = "Knowledge retrieval returned invalid knowledge, cannot perform complete inference"
        return state
    
    # Input validation - 放宽条件，允许在缺少部分信息时继续执行
    has_initial_inference = state.phenomenon_knowledge_match_table is not None
    has_calculation = state.calculation_steps is not None and len(state.calculation_steps) > 0
    has_algorithm = state.algorithm_parameters is not None
    
    # 如果都没有，尝试使用domain_knowledge_map进行基本推理
    if not (has_initial_inference or has_calculation or has_algorithm):
        if state.domain_knowledge_map:
            # 使用domain_knowledge_map构建基本的phenomenon_knowledge_match_table
            print(f"  ⚠ No initial inference/calculation/algorithm result, using domain_knowledge_map for basic inference")
            state.phenomenon_knowledge_match_table = {}
            for domain, knowledge in state.domain_knowledge_map.items():
                state.phenomenon_knowledge_match_table[domain] = {
                    "matched_phenomena": [],
                    "knowledge_points": knowledge.get("specialized_knowledge", []) + knowledge.get("foundational_knowledge", [])
                }
            state.match_confidence_label = "Low"
            print(f"  ✓ Created basic inference from domain_knowledge_map")
        elif state.key_parameters and isinstance(state.key_parameters, dict):
            # 如果连domain_knowledge_map都没有，尝试从key_parameters创建基本的calculation_steps
            print(f"  ⚠ No domain_knowledge_map, attempting to create basic calculation steps from key_parameters")
            formula_clues = state.key_parameters.get("formula_clues", [])
            parameters = state.key_parameters.get("parameters", [])
            
            # 通用fallback：优先使用formula_clues，如果没有则使用parameters
            source_items = formula_clues if formula_clues else (parameters if isinstance(parameters, list) else [])
            
            if source_items:
                # 使用所有可用的线索/参数，不限制数量
                max_desc_length = 200
                state.calculation_steps = [
                    {
                        "step_number": i + 1,
                        "step_description": f"Analysis: {str(item)[:max_desc_length]}{('...' if len(str(item)) > max_desc_length else '')}",
                        "step_type": "objective"
                    }
                    for i, item in enumerate(source_items)
                ]
                print(f"  ✓ Created {len(state.calculation_steps)} basic calculation steps from key_parameters")
            else:
                state.error_message = "At least one of initial inference, calculation steps, or algorithm parameters is required, and no fallback available"
                return state
        else:
            state.error_message = "At least one of initial inference, calculation steps, or algorithm parameters is required, and no fallback available"
            return state
    
    print("=" * 60)
    print("N7: Complete Logical Inference")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for complete inference"
        return state
    
    # Load tools for complete inference (all tools available)
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n7_complete_inference")
            print(f"  📚 Loaded {len(tools)} tool(s) for complete inference")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    # Enhanced: Include critical constraints in prompt
    enhanced_answer_constraints = state.answer_constraints or []
    if state.critical_constraints:
        enhanced_answer_constraints = enhanced_answer_constraints + state.critical_constraints
        print(f"  ⚠ Including critical constraints in inference: {state.critical_constraints}")
    
    # Prepare structured_condition with hard_constraints if needed
    structured_condition_with_constraints = state.structured_condition
    if structured_condition_with_constraints and isinstance(structured_condition_with_constraints, dict):
        hard_constraints = structured_condition_with_constraints.get("hard_constraints", [])
        if hard_constraints:
            print(f"  ⚠ Hard constraints detected: {hard_constraints}")
    else:
        structured_condition_with_constraints = {}
    
    # Prepare initial_associations from phenomenon_knowledge_match_table
    initial_associations = []
    if state.phenomenon_knowledge_match_table:
        for domain, match_data in state.phenomenon_knowledge_match_table.items():
            if isinstance(match_data, dict):
                matched_phenomena = match_data.get("matched_phenomena", [])
                knowledge_points = match_data.get("knowledge_points", [])
                initial_associations.append({
                    "domain": domain,
                    "phenomena": matched_phenomena,
                    "knowledge": knowledge_points
                })
    
    # Prepare retrieved_knowledge from domain_knowledge_map
    retrieved_knowledge = state.domain_knowledge_map or {}
    
    # Prepare calculation_result from calculation_steps or algorithm_parameters
    calculation_result = None
    if state.calculation_steps:
        calculation_result = {"type": "calculation_steps", "data": state.calculation_steps}
    elif state.algorithm_parameters:
        calculation_result = {"type": "algorithm_parameters", "data": state.algorithm_parameters}
    
    prompt = get_complete_inference_prompt(
        cleaned_text=state.cleaned_text or state.user_input or "",
        research_objective=state.research_objective or "",
        initial_associations=initial_associations,
        retrieved_knowledge=retrieved_knowledge,
        question_options=state.question_options,
        structured_subject=state.structured_subject,
        structured_condition=structured_condition_with_constraints,
        structured_goal=state.structured_goal,
        calculation_result=calculation_result,
        domain=None,  # Will be auto-detected by _get_prompt_func
        question_type=state.question_type_label,
        core_domains=state.core_domains
    )
    
    # Rule 2: Add original question text to prompt for strict anchoring
    if state.cleaned_text:
        question_anchor_instruction = f"\n\n**CRITICAL: Original Question Text (题干原文) - MUST ANCHOR TO THIS:**\n{state.cleaned_text}\n\nAll inference steps MUST be based on this original text. Do NOT add external knowledge that contradicts or exceeds this question."
        prompt = prompt + question_anchor_instruction
    
    # Fix 3: Add instruction to use key_constraints in inference
    if state.key_constraints:
        key_constraint_instruction = "\n\n**CRITICAL: Key Constraints (关键逻辑约束) - MUST USE IN INFERENCE:**\n"
        for i, constraint in enumerate(state.key_constraints, 1):
            key_constraint_instruction += f"{i}. {constraint}\n"
        key_constraint_instruction += "\nAll inference steps MUST use these key constraints. No conclusion without constraint support."
        prompt = prompt + key_constraint_instruction
        print(f"  ⚠ Key constraints added to inference prompt: {state.key_constraints}")
    
    # Add instruction to consider critical constraints
    if state.critical_constraints:
        constraint_instruction = "\n\nCRITICAL: Pay special attention to these constraints that significantly affect the answer: "
        constraint_instruction += ", ".join(state.critical_constraints)
        constraint_instruction += ". These constraints may invalidate simple relationships or require special consideration."
        prompt = prompt + constraint_instruction
    
    # Add parameter constraints to prompt for calculation problems
    if state.calculation_type_label == "Numerical" and state.parameter_constraints:
        param_constraint_instruction = "\n\n**CRITICAL: PARAMETER CONSTRAINTS (参数约束) - MUST VALIDATE ASSUMPTIONS:**\n"
        param_constraint_instruction += "The following parameter constraints have been extracted from the knowledge base. "
        param_constraint_instruction += "If you need to assume values for missing parameters, you MUST ensure they satisfy these constraints:\n\n"
        for param_name, constraints in state.parameter_constraints.items():
            param_constraint_instruction += f"**{param_name}:**\n"
            if "range" in constraints:
                min_val = constraints["range"].get("min")
                max_val = constraints["range"].get("max")
                param_constraint_instruction += f"  - Expected range: [{min_val}, {max_val}]\n"
            if "sign" in constraints:
                param_constraint_instruction += f"  - Sign constraint: must be {constraints['sign']}\n"
            if "physical_constraints" in constraints:
                for pc in constraints["physical_constraints"]:
                    param_constraint_instruction += f"  - Physical constraint: {pc}\n"
            param_constraint_instruction += "\n"
        param_constraint_instruction += "**IMPORTANT:**\n"
        param_constraint_instruction += "1. Before assuming any parameter value, check if it satisfies the constraints above.\n"
        param_constraint_instruction += "2. If you must assume a value, clearly state the assumption and justify it based on the constraints.\n"
        param_constraint_instruction += "3. After calculation, verify that the final result also satisfies the constraints for the target parameter.\n"
        param_constraint_instruction += "4. If the result violates constraints, re-examine your assumptions and calculation steps.\n"
        prompt = prompt + param_constraint_instruction
        print(f"  ⚠ Parameter constraints added to inference prompt: {len(state.parameter_constraints)} parameter(s)")
    
    # OPTIMIZATION 1: Highlight and strictly follow NEGATIVE constraints (cannot/except) and EXCLUSIVE constraints (category 1/only 1)
    negative_constraints = []
    exclusive_constraints = []
    if state.cleaned_text:
        import re  # Import at function level to avoid UnboundLocalError
        text_lower = state.cleaned_text.lower()
        # Detect negative constraints (cannot, except, not, exclude, etc.)
        if any(keyword in text_lower for keyword in ["cannot", "can not", "except", "not", "exclude", "never", "must not"]):
            # Extract sentences containing negative keywords
            sentences = re.split(r'[.!?]\s+', state.cleaned_text)
            for sent in sentences:
                sent_lower = sent.lower()
                if any(keyword in sent_lower for keyword in ["cannot", "can not", "except", "not occur", "exclude", "never", "must not"]):
                    negative_constraints.append(sent.strip())
        # Detect exclusive constraints (category 1, only 1, single, unique, etc.)
        if any(keyword in text_lower for keyword in ["category 1", "only 1", "single", "unique", "exclusively", "solely", "merely"]):
            sentences = re.split(r'[.!?]\s+', state.cleaned_text)
            for sent in sentences:
                sent_lower = sent.lower()
                if any(keyword in sent_lower for keyword in ["category 1", "only 1", "single", "unique", "exclusively", "solely", "merely"]):
                    exclusive_constraints.append(sent.strip())
    
    if negative_constraints:
        negative_instruction = "\n\n**CRITICAL: NEGATIVE CONSTRAINTS (否定约束) - MUST STRICTLY FOLLOW:**\n"
        for i, constraint in enumerate(negative_constraints, 1):
            negative_instruction += f"{i}. {constraint}\n"
        negative_instruction += "\nYour inference MUST explicitly show how you EXCLUDE options/conclusions that violate these negative constraints. Any conclusion that violates these constraints is WRONG."
        prompt = prompt + negative_instruction
        print(f"  ⚠ Negative constraints detected and added: {len(negative_constraints)} constraint(s)")
    
    if exclusive_constraints:
        exclusive_instruction = "\n\n**CRITICAL: EXCLUSIVE CONSTRAINTS (专属约束) - MUST STRICTLY FOLLOW:**\n"
        for i, constraint in enumerate(exclusive_constraints, 1):
            exclusive_instruction += f"{i}. {constraint}\n"
        exclusive_instruction += "\nYour inference MUST identify the ONE AND ONLY option/conclusion that satisfies this exclusive constraint. Multiple matches are NOT allowed."
        prompt = prompt + exclusive_instruction
        print(f"  ⚠ Exclusive constraints detected and added: {len(exclusive_constraints)} constraint(s)")
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=5, state=state, node_name="n7_complete_inference")
    if not response:
        state.error_message = "LLM call failed for complete inference"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for complete inference"
        return state
    
    state.closed_inference_path = result.get("closed_inference_path", [])
    state.core_conclusion = result.get("core_conclusion")
    if not state.core_conclusion or not state.closed_inference_path:
        state.exception_type_label = state.exception_type_label or "Inference Path Incomplete"
    
    # ========== Step 6: Validate parameters and calculation results ==========
    # Validate assumed parameters against constraints
    if state.calculation_type_label == "Numerical" and state.parameter_constraints and state.closed_inference_path:
        validation_issues = []
        for step in state.closed_inference_path:
            step_content = step.get("step_content", "")
            step_type = step.get("step_type", "")
            
            # Check for parameter assumptions in calculation steps
            if step_type == "calculation" and ("assume" in step_content.lower() or "estimated" in step_content.lower()):
                # Extract assumed parameter values
                import re
                # Match patterns like "B22(electrostatic) = -10.0 mL/g"
                param_matches = re.finditer(r'([A-Za-z0-9_]+)\([^)]+\)\s*=\s*([+-]?[\d.]+)', step_content)
                for match in param_matches:
                    param_name = match.group(1)
                    param_value = float(match.group(2))
                    
                    # Check against constraints
                    if param_name in state.parameter_constraints:
                        constraints = state.parameter_constraints[param_name]
                        
                        # Check range
                        if "range" in constraints:
                            min_val = constraints["range"].get("min")
                            max_val = constraints["range"].get("max")
                            if min_val is not None and max_val is not None:
                                if param_value < min_val or param_value > max_val:
                                    validation_issues.append(
                                        f"Assumed parameter {param_name}={param_value} is outside expected range [{min_val}, {max_val}]"
                                    )
                        
                        # Check sign
                        if "sign" in constraints:
                            expected_sign = constraints["sign"]
                            if expected_sign == "positive" and param_value < 0:
                                validation_issues.append(
                                    f"Assumed parameter {param_name}={param_value} violates sign constraint (expected positive)"
                                )
                            elif expected_sign == "negative" and param_value > 0:
                                validation_issues.append(
                                    f"Assumed parameter {param_name}={param_value} violates sign constraint (expected negative)"
                                )
        
        if validation_issues:
            print(f"  ⚠ Parameter validation issues detected:")
            for issue in validation_issues:
                print(f"    - {issue}")
            # Don't fail, but add warning to exception type
            if not state.exception_type_label:
                state.exception_type_label = "Parameter Assumption Warning"
    
    # Validate final calculation result against constraints
    if state.calculation_type_label == "Numerical" and state.parameter_constraints and state.core_conclusion:
        # Extract numerical result from core_conclusion
        import re
        # Try to extract the final numerical value
        result_matches = re.findall(r'([+-]?[\d.]+)\s*(?:mL/g|mg/L|μM|mM|M|g/mol|kDa|Gy|%|units?)', state.core_conclusion)
        if not result_matches:
            # Try simpler pattern
            result_matches = re.findall(r'([+-]?[\d.]+)', state.core_conclusion)
        
        if result_matches:
            try:
                final_result = float(result_matches[-1])  # Use last match as final result
                
                # Check against constraints for the target parameter (usually the one being calculated)
                # Try to identify the target parameter from key_parameters
                target_param = None
                if isinstance(state.key_parameters, dict):
                    formula_clues = state.key_parameters.get("formula_clues", [])
                    for clue in formula_clues:
                        if isinstance(clue, str) and "steric" in clue.lower():
                            # Extract parameter name
                            match = re.search(r'([A-Za-z0-9_]+)\([^)]+\)', clue)
                            if match:
                                target_param = match.group(1)
                                break
                
                # If we can't find target param, check all constraints
                params_to_check = [target_param] if target_param else list(state.parameter_constraints.keys())
                
                for param_name in params_to_check:
                    if param_name in state.parameter_constraints:
                        constraints = state.parameter_constraints[param_name]
                        
                        # Check range
                        if "range" in constraints:
                            min_val = constraints["range"].get("min")
                            max_val = constraints["range"].get("max")
                            if min_val is not None and max_val is not None:
                                if final_result < min_val or final_result > max_val:
                                    print(f"  ⚠ WARNING: Final result {final_result} is outside expected range [{min_val}, {max_val}] for {param_name}")
                                    print(f"    - This may indicate a calculation error or incorrect parameter assumptions")
                                    # Mark as exception for n9 to handle
                                    if not state.exception_type_label or "Warning" not in state.exception_type_label:
                                        state.exception_type_label = (state.exception_type_label or "") + " / Result Out of Range"
                        
                        # Check sign
                        if "sign" in constraints:
                            expected_sign = constraints["sign"]
                            if expected_sign == "positive" and final_result < 0:
                                print(f"  ⚠ WARNING: Final result {final_result} violates sign constraint (expected positive for {param_name})")
                                if not state.exception_type_label or "Warning" not in state.exception_type_label:
                                    state.exception_type_label = (state.exception_type_label or "") + " / Result Sign Violation"
                            elif expected_sign == "negative" and final_result > 0:
                                print(f"  ⚠ WARNING: Final result {final_result} violates sign constraint (expected negative for {param_name})")
                                if not state.exception_type_label or "Warning" not in state.exception_type_label:
                                    state.exception_type_label = (state.exception_type_label or "") + " / Result Sign Violation"
            except (ValueError, IndexError):
                pass  # Could not extract numerical result
    
    # OPTIMIZATION 4: Check numerical precision and unit/magnitude consistency for calculation results
    if state.closed_inference_path and state.calculation_steps:
        import re
        for step in state.closed_inference_path:
            if step.get("step_type") == "calculation" and step.get("intermediate_result"):
                result_str = str(step.get("intermediate_result", ""))
                # Extract numerical value
                num_match = re.search(r'([\d.]+)\s*\*\s*10\^?([+-]?\d+)', result_str)
                if num_match:
                    base = float(num_match.group(1))
                    exp = int(num_match.group(2))
                    value = base * (10 ** exp)
                    
                    # Check unit/magnitude consistency with question text
                    if state.cleaned_text:
                        text_lower = state.cleaned_text.lower()
                        # Check for dose questions (should be in Gy, typically 10^-6 to 10^-3 range)
                        if "dose" in text_lower or "gy" in text_lower:
                            if exp > -3 or exp < -8:
                                print(f"  ⚠ Warning: Calculated dose value magnitude ({exp}) may be outside expected range (10^-6 to 10^-3 Gy)")
                                # Don't fail, but add warning
                        # Check for concentration questions (should be in reasonable range)
                        if "concentration" in text_lower or "mol/l" in text_lower or "μm" in text_lower:
                            if abs(exp) > 3:
                                print(f"  ⚠ Warning: Calculated concentration value magnitude ({exp}) may be unreasonable")
                    
                    # Check precision (±10% error range would be validated in n9, but we can check for obvious errors here)
                    # This is a basic sanity check - full precision validation happens in n9
                    if abs(exp) > 15:
                        print(f"  ⚠ Warning: Calculated value has extreme exponent ({exp}), likely calculation error")
    
    print(f"✓ Core conclusion: {state.core_conclusion[:100] if state.core_conclusion else 'N/A'}...")
    print(f"✓ Inference path steps: {len(state.closed_inference_path)}")
    
    return state


def n8_answer_generation_node(state: GeneralQAState) -> GeneralQAState:
    """
    N8: Multi-Type Answer Generation
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses focused tools for answer refinement and validation.
    """
    # OPTIMIZATION: Generate pre-validation (生成前校验)
    # Check if we have valid inference results
    if not state.core_conclusion:
        print(f"  ❌ No core_conclusion available, cannot generate answer")
        state.error_message = "core_conclusion is required for answer generation"
        state.exception_type_label = "Answer Generation Failed - No Inference Result"
        return state
    
    # Check if inference path is complete
    if not state.closed_inference_path or len(state.closed_inference_path) == 0:
        print(f"  ❌ No inference path available, cannot generate answer")
        state.error_message = "closed_inference_path is required for answer generation"
        state.exception_type_label = "Answer Generation Failed - Incomplete Inference Path"
        return state
    
    # Check knowledge validity
    if state.knowledge_validity_label == "Invalid":
        print(f"  ❌ Knowledge marked as Invalid, cannot generate reliable answer")
        state.error_message = "Invalid knowledge, cannot generate answer"
        state.exception_type_label = "Answer Generation Failed - Invalid Knowledge"
        return state
    
    print("=" * 60)
    print("N8: Multi-Type Answer Generation")
    print("=" * 60)
    
    # Initialize node visit tracking
    if state.node_visit_count is None:
        state.node_visit_count = {}
    
    # Increment N8 visit count
    n8_visits = state.node_visit_count.get("n8_answer_generation", 0)
    state.node_visit_count["n8_answer_generation"] = n8_visits + 1
    
    # Initialize N8 internal retry counter (stored in node_visit_count to avoid adding new state field)
    n8_internal_retry_count = state.node_visit_count.get("n8_internal_retry", 0)
    
    # CRITICAL: Prevent infinite loops - if N8 has been visited too many times, skip processing
    if n8_visits >= 3:
        print(f"  ⚠ Infinite loop detected: N8 visited {n8_visits + 1} times, skipping to prevent infinite loop")
        state.exception_type_label = state.exception_type_label or "Answer Generation Failed - Infinite Loop"
        return state
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for answer generation"
        return state
    
    # Load tools for answer refinement
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n8_answer_generation")
            print(f"  📚 Loaded {len(tools)} tool(s) for answer refinement")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    # Extract calculation result from inference path if available
    calculation_result = None
    if state.closed_inference_path:
        for step in state.closed_inference_path:
            if step.get("step_type") == "calculation" and step.get("intermediate_result"):
                calculation_result = step.get("intermediate_result")
                break
    
    base_prompt = get_answer_generation_prompt(
        state.core_conclusion,
        state.question_type_label or "Unknown",
        state.question_options or [],
        calculation_result,
        state.answer_format_label,
        state.answer_constraints,
        structured_goal=state.structured_goal
    )
    
    # Rule 3: Add original question text for option checking
    if state.cleaned_text and state.question_options:
        question_check_instruction = f"\n\n**CRITICAL: Original Question Text (题干原文) - CHECK OPTIONS AGAINST THIS:**\n{state.cleaned_text}\n\nWhen judging options: FIRST exclude any option that directly CONTRADICTS this question text, THEN match with core conclusion."
        prompt = base_prompt + question_check_instruction
    else:
        prompt = base_prompt
    
    # Enhanced: Add answer format control instructions
    format_instruction = ""
    if state.answer_format_label == "Short Text" or state.question_type_label == "Professional Algorithm":
        format_instruction = "\n\nIMPORTANT: The answer format is 'Short Text' or 'Professional Algorithm'. "
        format_instruction += "You MUST provide a CONCRETE, SPECIFIC answer, NOT a general method or procedure. "
        format_instruction += "For example:\n"
        format_instruction += "- If asked for amino acid replacement: Provide the specific sequence (e.g., 'Gly-Ser-Gly-Gly'), NOT 'use neutral amino acids'\n"
        format_instruction += "- If asked for filtering strategy: Provide the specific threshold (e.g., 'LFC > 4'), NOT 'use a filter function'\n"
        format_instruction += "- If asked for drug recommendation: Provide specific drug names, NOT 'consult guidelines'"
    elif state.answer_format_label == "List":
        format_instruction = "\n\nIMPORTANT: The answer format is 'List'. Provide a specific list of items, not general recommendations."
    elif state.answer_format_label in ["Single Choice", "Multi-Select"]:
        format_instruction = "\n\nIMPORTANT: For multiple choice questions, you MUST select from the provided options. "
        format_instruction += "If your conclusion doesn't match any option exactly, use tools to find semantic relationships. "
        format_instruction += "For example, if you conclude 'Pierre Robin sequence' but the options include 'Ventral foregut budding defect', "
        format_instruction += "you should know that the latter is the anatomical defect causing PRS."
    
    prompt = base_prompt + format_instruction
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=3, state=state, node_name="n8_answer_generation")
    if not response:
        state.error_message = "LLM call failed for answer generation"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        # OPTIMIZATION 3: For factual number questions: retry retrieval if no info
        if state.core_conclusion and ("No Specific Information Available" in str(state.core_conclusion) or "Cannot generate" in str(state.core_conclusion)):
            # This is a factual question that needs specific information
            # Try to retry knowledge retrieval with more specific queries
            print(f"  ⚠ Answer generation failed and core_conclusion indicates no specific info available")
            print(f"    - Attempting to retry knowledge retrieval with more specific queries")
            # Mark for retry - this will be handled by routing logic
            state.exception_type_label = "Answer Generation Failed - No Specific Information"
            return state
        state.error_message = "Failed to parse LLM response for answer generation"
        return state
    
    state.structured_answer = result.get("structured_answer")
    
    # OPTIMIZATION: Ensure final_answer is always a string, not a list (fixes ValidationError)
    if state.structured_answer and isinstance(state.structured_answer, dict):
        raw_final = state.structured_answer.get("final_answer")
        if isinstance(raw_final, list):
            # Convert list to comma-separated string for List format
            if state.answer_format_label == "List":
                state.structured_answer["final_answer"] = ", ".join(str(item) for item in raw_final)
            else:
                # For other formats, convert to string representation
                state.structured_answer["final_answer"] = str(raw_final)
    
    # OPTIMIZATION: For Single Choice questions, FORBID "Cannot generate" - must select one option
    if state.answer_format_label == "Single Choice" and state.structured_answer:
        if isinstance(state.structured_answer, dict):
            answer_content = state.structured_answer.get("answer_content") or {}
            final_answer = state.structured_answer.get("final_answer", "")
            
            # Check if all options are excluded or answer is "Cannot generate"
            if "Cannot generate" in str(final_answer).lower() or final_answer == "":
                option_matching_table = answer_content.get("option_matching_table", {})
                all_excluded = all(
                    status == "exclude" or status == "excluded" 
                    for status in option_matching_table.values()
                )
                
                if all_excluded or "Cannot generate" in str(final_answer).lower():
                    print(f"  ⚠ Single Choice question: All options excluded or 'Cannot generate' detected")
                    print(f"    - FORCING selection of best matching option")
                    
                    # Force selection: find the option with least exclusion reason or best semantic match
                    # Use core_keywords and option_features if available
                    if state.core_keywords and state.option_features:
                        # Try to match core keywords with option features
                        best_match = None
                        best_score = 0
                        for opt_label, opt_feature in state.option_features.items():
                            if opt_label in option_matching_table:
                                # Count keyword matches
                                score = sum(1 for kw in state.core_keywords if kw.lower() in opt_feature.lower())
                                if score > best_score:
                                    best_score = score
                                    best_match = opt_label
                        
                        if best_match:
                            print(f"    - Selected option {best_match} based on keyword matching")
                            if isinstance(state.structured_answer, dict):
                                if "answer_content" not in state.structured_answer:
                                    state.structured_answer["answer_content"] = {}
                                if "option_matching_table" not in state.structured_answer["answer_content"]:
                                    state.structured_answer["answer_content"]["option_matching_table"] = {}
                                state.structured_answer["answer_content"]["option_matching_table"][best_match] = "match"
                                state.structured_answer["final_answer"] = best_match
                                state.final_answer = best_match
                                # Mark other options as excluded
                                for opt in state.question_options or []:
                                    opt_label = opt[0] if opt and len(opt) > 0 else None
                                    if opt_label and opt_label != best_match:
                                        if "option_matching_table" in state.structured_answer["answer_content"]:
                                            state.structured_answer["answer_content"]["option_matching_table"][opt_label] = "exclude"
                    else:
                        # Fallback: select first option if no better match
                        if state.question_options:
                            first_option_label = state.question_options[0][0] if state.question_options[0] else None
                            if first_option_label:
                                print(f"    - Fallback: Selected first option {first_option_label}")
                                if isinstance(state.structured_answer, dict):
                                    if "answer_content" not in state.structured_answer:
                                        state.structured_answer["answer_content"] = {}
                                    if "option_matching_table" not in state.structured_answer["answer_content"]:
                                        state.structured_answer["answer_content"]["option_matching_table"] = {}
                                    state.structured_answer["answer_content"]["option_matching_table"][first_option_label] = "match"
                                    state.structured_answer["final_answer"] = first_option_label
                                    state.final_answer = first_option_label
    
    # Extract and normalize final answer
    if state.structured_answer and isinstance(state.structured_answer, dict):
        answer_content = state.structured_answer.get("answer_content") or {}
        option_matching_table = None
        if isinstance(answer_content, dict):
            option_matching_table = answer_content.get("option_matching_table")
        
        # Fix 1: Code-level validation for Single Choice option matching
        if option_matching_table and isinstance(option_matching_table, dict):
            if state.answer_format_label == "Single Choice":
                # Count how many options are marked as "match"
                match_count = sum(1 for status in option_matching_table.values() 
                                if str(status).strip().lower().startswith("match"))
                
                if match_count > 1:
                    # Multiple matches for Single Choice - this is an error
                    print(f"  ⚠ WARNING: Single Choice question has {match_count} matches, expected 1")
                    print(f"    - Option matching table: {option_matching_table}")
                    
                    # Force correction: keep only the first match
                    match_labels = [label for label, status in option_matching_table.items() 
                                  if str(status).strip().lower().startswith("match")]
                    if match_labels:
                        # Keep only the first match, mark others as exclude
                        first_match = match_labels[0]
                        for label in match_labels[1:]:
                            option_matching_table[label] = "exclude"
                        print(f"    - Corrected: keeping only '{first_match}' as match")
                        
                        # Update the structured_answer
                        if isinstance(answer_content, dict):
                            answer_content["option_matching_table"] = option_matching_table
                            state.structured_answer["answer_content"] = answer_content
                    
                    # Mark as potential issue
                    if not state.exception_type_label:
                        state.exception_type_label = "Option Matching Error"
                
                elif match_count == 0:
                    # OPTIMIZATION: No matches for Single Choice - force semantic matching using tools
                    print(f"  ⚠ WARNING: Single Choice question has 0 matches, forcing semantic matching with tools")
                    # CRITICAL: Prevent infinite retry loops within N8
                    # Use node_visit_count to track internal retry (avoid adding new state field)
                    if state.node_visit_count is None:
                        state.node_visit_count = {}
                    n8_internal_retry = state.node_visit_count.get("n8_internal_retry", 0)
                    
                    if n8_internal_retry >= 1:
                        print(f"  ⚠ N8 internal retry limit reached ({n8_internal_retry}/1), skipping semantic matching retry")
                        print(f"    - This will be handled by N10 exception handling")
                        state.exception_type_label = state.exception_type_label or "Option Matching Error"
                    elif state.core_conclusion and state.question_options and tools:
                        # Force LLM to use tools for semantic matching
                        state.node_visit_count["n8_internal_retry"] = n8_internal_retry + 1
                        print(f"  🔄 Retrying answer generation with forced tool-based semantic matching (attempt {n8_internal_retry + 1}/1)")
                        semantic_matching_instruction = f"\n\n**CRITICAL: ALL OPTIONS WERE EXCLUDED - FORCE SEMANTIC MATCHING:**\n"
                        semantic_matching_instruction += f"Your core conclusion is: {state.core_conclusion}\n"
                        semantic_matching_instruction += f"You MUST use available tools to find semantic relationships between your conclusion and the options.\n"
                        semantic_matching_instruction += f"DO NOT return 'Cannot generate' - you MUST find at least one option that semantically relates to your conclusion.\n"
                        semantic_matching_instruction += f"Pattern: If conclusion mentions '[specific entity/group]', find which option contains/produces this entity.\n"
                        semantic_matching_instruction += f"Pattern: If conclusion is about '[aspect A]' but question asks '[aspect B]', re-analyze to find the correct aspect.\n"
                        
                        # Retry with semantic matching instruction
                        retry_prompt = base_prompt + format_instruction + semantic_matching_instruction
                        if state.cleaned_text and state.question_options:
                            retry_prompt = retry_prompt + question_check_instruction
                        
                        retry_response = _call_llm(llm, retry_prompt, tools=tools, max_iterations=5, state=state, node_name="n8_answer_generation_retry")
                        if retry_response:
                            retry_result = _parse_json_response(retry_response)
                            if retry_result and retry_result.get("structured_answer"):
                                state.structured_answer = retry_result.get("structured_answer")
                                print(f"  ✓ Semantic matching retry succeeded")
                                # Re-extract option_matching_table from retry result
                                if state.structured_answer and isinstance(state.structured_answer, dict):
                                    answer_content = state.structured_answer.get("answer_content") or {}
                                    if isinstance(answer_content, dict):
                                        option_matching_table = answer_content.get("option_matching_table")
                                        if option_matching_table:
                                            # Update the option_matching_table
                                            if isinstance(answer_content, dict):
                                                answer_content["option_matching_table"] = option_matching_table
                                                state.structured_answer["answer_content"] = answer_content
                    elif state.core_conclusion and state.question_options:
                        # This will be handled by _normalize_choice_answer
                        pass
        
        expected_choice = (
            state.answer_format_label in ["Single Choice", "Multi-Select"]
            or state.question_type_label == "Multiple Choice"
        )
        normalized_final = state.structured_answer.get("final_answer")
        if expected_choice:
            normalized_final = _normalize_choice_answer(
                normalized_final,
                option_matching_table,
                state.question_options or [],
                state.answer_format_label or "Single Choice",
                state.core_conclusion
            )
            state.structured_answer["final_answer"] = normalized_final
        
        # OPTIMIZATION: Format conversion for Sequence answers (e.g., rank sequences)
        if state.answer_format_label == "Sequence" and normalized_final and state.core_conclusion:
            import re
            # Check if question asks for rank sequence (e.g., "sequence of aFC ranks", "rank order")
            question_lower = (state.cleaned_text or "").lower()
            if "rank" in question_lower and ("sequence" in question_lower or "order" in question_lower):
                # Check if answer contains aFC values that need to be converted to ranks
                # Pattern: "1/3, 1/2, 3/2, 2, 3" or similar
                afc_pattern = r'(\d+/\d+|\d+\.\d+|\d+)'
                answer_str = str(normalized_final)
                afc_matches = re.findall(afc_pattern, answer_str)
                
                if len(afc_matches) >= 3:  # Likely aFC values
                    # Try to extract rank mapping from question text
                    # Example: "1/3 is rank 1, 3 is rank 5"
                    rank_map = {}
                    rank_pattern = r'(\d+/\d+|\d+\.\d+|\d+)\s+is\s+rank\s+(\d+)'
                    rank_matches = re.findall(rank_pattern, question_lower)
                    for afc_val, rank_val in rank_matches:
                        rank_map[afc_val] = int(rank_val)
                    
                    # If we have rank mapping, convert aFC sequence to rank sequence
                    if rank_map:
                        rank_sequence = []
                        for afc_val in afc_matches:
                            # Try exact match first
                            if afc_val in rank_map:
                                rank_sequence.append(str(rank_map[afc_val]))
                            else:
                                # Try to find closest match (e.g., "1/3" vs "1/3")
                                matched = False
                                for mapped_afc, rank in rank_map.items():
                                    try:
                                        # Compare as fractions or decimals
                                        if "/" in afc_val and "/" in mapped_afc:
                                            val1 = eval(afc_val)
                                            val2 = eval(mapped_afc)
                                            if abs(val1 - val2) < 0.001:
                                                rank_sequence.append(str(rank))
                                                matched = True
                                                break
                                        elif abs(float(afc_val) - float(mapped_afc)) < 0.001:
                                            rank_sequence.append(str(rank))
                                            matched = True
                                            break
                                    except:
                                        pass
                                if not matched:
                                    # If no match found, keep original
                                    rank_sequence.append(afc_val)
                        
                        if len(rank_sequence) == len(afc_matches):
                            # Successfully converted to rank sequence
                            rank_str = "".join(rank_sequence)
                            print(f"  ✓ Converted aFC sequence to rank sequence: {rank_str}")
                            normalized_final = rank_str
                            if state.structured_answer:
                                if isinstance(state.structured_answer, dict):
                                    state.structured_answer["final_answer"] = rank_str
                                    if "answer_content" in state.structured_answer and isinstance(state.structured_answer["answer_content"], dict):
                                        state.structured_answer["answer_content"]["sequence_result"] = rank_str
        
        # OPTIMIZATION: Convert list to string for List format answers
        if isinstance(normalized_final, list):
            # Convert list to comma-separated string
            if state.answer_format_label == "List":
                state.final_answer = ", ".join(str(item) for item in normalized_final)
            else:
                # For other formats, convert to string representation
                state.final_answer = str(normalized_final)
        else:
            state.final_answer = normalized_final if normalized_final is not None else None
    else:
        state.final_answer = None
    
    # OPTIMIZATION 3: Check logical rationality for all answers
    if state.final_answer:
        answer_str = str(state.final_answer)
        rationality_issues = []
        
        # Check for numerical answers: magnitude/scale validation
        import re
        num_match = re.search(r'([\d.]+)\s*\*\s*10\^?([+-]?\d+)', answer_str)
        if num_match:
            base = float(num_match.group(1))
            exp = int(num_match.group(2))
            value = base * (10 ** exp)
            # Check for absurd magnitudes
            if abs(exp) > 10:
                rationality_issues.append(f"Numerical value has extreme exponent ({exp}), likely incorrect")
            # Check for specific question types (e.g., dose in Gy should be 10^-6 to 10^-3 range)
            if "dose" in state.cleaned_text.lower() or "gy" in answer_str.lower():
                if exp > -3 or exp < -8:
                    rationality_issues.append(f"Dose value magnitude ({exp}) outside expected range (10^-6 to 10^-3 Gy)")
        
        # Check for sequence answers: length validation
        if state.answer_format_label == "Sequence":
            if len(answer_str) == 0:
                rationality_issues.append("Sequence answer is empty")
            elif len(answer_str) > 10000:
                rationality_issues.append(f"Sequence answer is extremely long ({len(answer_str)} chars), likely incorrect")
            elif len(answer_str) < 3:
                rationality_issues.append(f"Sequence answer is too short ({len(answer_str)} chars), likely incomplete")
        
        # Check for option consistency
        if state.answer_format_label in ["Single Choice", "Multi-Select"]:
            if state.question_options:
                # Check if selected option exists
                if answer_str not in [chr(65+i) for i in range(len(state.question_options))] and answer_str.upper() not in [chr(65+i) for i in range(len(state.question_options))]:
                    # Check if it's a comma-separated list
                    if "," in answer_str:
                        options = [opt.strip().upper() for opt in answer_str.split(",")]
                        valid_options = [chr(65+i) for i in range(len(state.question_options))]
                        if not all(opt in valid_options for opt in options):
                            rationality_issues.append(f"Selected option(s) '{answer_str}' not in valid options")
                    else:
                        rationality_issues.append(f"Selected option '{answer_str}' not in valid options")
        
        if rationality_issues:
            print(f"  ⚠ Rationality issues detected: {rationality_issues}")
            # Don't fail immediately, but mark for validation
            if not state.exception_type_label:
                state.exception_type_label = "Answer Rationality Check Failed"
    
    if not state.structured_answer or not state.final_answer:
        # OPTIMIZATION 3: For factual questions, try to retry knowledge retrieval
        if state.core_conclusion and ("No Specific Information Available" in str(state.core_conclusion) or "Cannot generate" in str(state.core_conclusion)):
            print(f"  ⚠ No answer generated and core_conclusion indicates no specific info")
            print(f"    - Marking for knowledge retrieval retry")
            state.exception_type_label = "Answer Generation Failed - No Specific Information"
        else:
            state.exception_type_label = state.exception_type_label or "Answer Generation Failed"
    
    print(f"✓ Final answer: {state.final_answer[:100] if state.final_answer else 'N/A'}...")
    
    # ========== X-Masters Enhancement: Generate Multiple Candidate Answers ==========
    # Generate 3-5 candidate answers with different reasoning paths
    num_candidates = state.num_candidates or 3
    print(f"\n{'='*60}")
    print(f"N8: Generating {num_candidates} Candidate Answers (X-Masters Enhancement)")
    print(f"{'='*60}")
    
    candidate_answers = []
    
    # First candidate: use the original answer
    if state.structured_answer and state.final_answer:
        candidate_answers.append({
            "candidate_id": 0,
            "structured_answer": state.structured_answer,
            "final_answer": state.final_answer,
            "reasoning_path": "original",
            "success": True
        })
        print(f"  ✓ Candidate 0: Original answer generated")
    
    # Generate additional candidates with different temperatures/approaches
    for i in range(1, num_candidates):
        try:
            # Use slightly different temperature for diversity
            temp_llm = create_bioinformatics_llm(temperature=0.5 + (i * 0.1))
            
            # Modify prompt slightly for diversity
            diversity_prompt = base_prompt + format_instruction
            if i == 1:
                diversity_prompt += "\n\n**Alternative Approach**: Consider a different reasoning path or interpretation."
            elif i == 2:
                diversity_prompt += "\n\n**Alternative Approach**: Focus on edge cases or alternative explanations."
            else:
                diversity_prompt += "\n\n**Alternative Approach**: Explore complementary perspectives or additional constraints."
            
            if state.cleaned_text and state.question_options:
                diversity_prompt = diversity_prompt + question_check_instruction
            
            # Generate alternative answer
            alt_response = _call_llm(temp_llm, diversity_prompt, tools=tools, max_iterations=3, state=state, node_name=f"n8_candidate_{i}")
            if alt_response:
                alt_result = _parse_json_response(alt_response)
                if alt_result and alt_result.get("structured_answer"):
                    alt_structured = alt_result.get("structured_answer")
                    alt_final = alt_structured.get("final_answer") if isinstance(alt_structured, dict) else None
                    
                    if alt_final:
                        candidate_answers.append({
                            "candidate_id": i,
                            "structured_answer": alt_structured,
                            "final_answer": alt_final,
                            "reasoning_path": f"alternative_{i}",
                            "success": True
                        })
                        print(f"  ✓ Candidate {i}: Alternative answer generated")
                    else:
                        print(f"  ⚠ Candidate {i}: Failed to extract final answer")
                else:
                    print(f"  ⚠ Candidate {i}: Failed to parse response")
            else:
                print(f"  ⚠ Candidate {i}: LLM call failed")
        except Exception as e:
            print(f"  ⚠ Candidate {i}: Error - {e}")
            # Continue with other candidates
    
    # If we have at least one candidate, store them
    if candidate_answers:
        state.candidate_answers = candidate_answers
        print(f"\n✓ Generated {len(candidate_answers)} candidate answer(s)")
    else:
        # Fallback: if no candidates generated, use original answer as single candidate
        if state.structured_answer and state.final_answer:
            state.candidate_answers = [{
                "candidate_id": 0,
                "structured_answer": state.structured_answer,
                "final_answer": state.final_answer,
                "reasoning_path": "original",
                "success": True
            }]
            print(f"  ⚠ Using original answer as single candidate")
    
    return state


def n8_5_critic_review_node(state: GeneralQAState) -> GeneralQAState:
    """
    N8.5: Critic Review (X-Masters Enhancement)
    
    Reviews and corrects each candidate answer independently using CodeActAgent.
    """
    print("=" * 60)
    print("N8.5: Critic Review (X-Masters Enhancement)")
    print("=" * 60)
    
    if not state.candidate_answers or len(state.candidate_answers) == 0:
        print(f"  ⚠ No candidate answers to review, skipping critic stage")
        state.critiqued_answers = []
        return state
    
    # Import X-Masters critic function
    try:
        from agent.nodes.subagents.x_masters.critic import run_single_critic
        from agent.nodes.subagents.x_masters.tools import inject_lightweight_tools_to_namespace
    except ImportError as e:
        print(f"  ⚠ X-Masters critic not available (optional dependency): {e}")
        print(f"  → Continuing without X-Masters enhancement, using original candidate answers")
        state.critiqued_answers = state.candidate_answers  # Fallback: use original candidates
        return state
    
    # Build problem context for critic
    problem_context = state.cleaned_text or state.user_input
    if state.core_conclusion:
        problem_context += f"\n\nCore Conclusion: {state.core_conclusion}"
    if state.closed_inference_path:
        problem_context += f"\n\nInference Path: {str(state.closed_inference_path)[:500]}"
    
    # Build retrieved context from knowledge
    retrieved_context = ""
    if state.domain_knowledge_map:
        for domain, knowledge in state.domain_knowledge_map.items():
            if isinstance(knowledge, dict):
                found_knowledge = knowledge.get("foundational_knowledge", []) + knowledge.get("specialized_knowledge", [])
                if found_knowledge:
                    retrieved_context += f"\n{domain}: {str(found_knowledge)[:500]}\n"
    
    critiqued_answers = []
    
    # Review each candidate answer
    for candidate in state.candidate_answers:
        candidate_id = candidate.get("candidate_id", 0)
        candidate_answer = candidate.get("final_answer", "")
        candidate_structured = candidate.get("structured_answer", {})
        
        if not candidate_answer:
            print(f"  ⚠ Candidate {candidate_id}: No answer to review, skipping")
            critiqued_answers.append({
                "candidate_id": candidate_id,
                "original_answer": candidate_answer,
                "critiqued_answer": candidate_answer,
                "success": False
            })
            continue
        
        print(f"\n  📝 Reviewing Candidate {candidate_id}...")
        
        try:
            # Run critic on this candidate
            critic_result = run_single_critic(
                problem=problem_context,
                solution=candidate_answer,
                solver_id=candidate_id,
                retrieved_context=retrieved_context,
                temperature=0.6,
                llm=None,  # Use default LLM
                source=None,
                base_url=None,
                api_key=None,
                timeout_seconds=120,
            )
            
            critiqued_answer = critic_result.get("solution", candidate_answer)
            success = critic_result.get("success", False)
            
            critiqued_answers.append({
                "candidate_id": candidate_id,
                "original_answer": candidate_answer,
                "original_structured": candidate_structured,
                "critiqued_answer": critiqued_answer,
                "success": success
            })
            
            if success:
                print(f"    ✓ Candidate {candidate_id} reviewed successfully")
            else:
                print(f"    ⚠ Candidate {candidate_id} review failed, using original")
                
        except Exception as e:
            print(f"    ❌ Candidate {candidate_id} review error: {e}")
            critiqued_answers.append({
                "candidate_id": candidate_id,
                "original_answer": candidate_answer,
                "original_structured": candidate_structured,
                "critiqued_answer": candidate_answer,  # Fallback to original
                "success": False
            })
    
    state.critiqued_answers = critiqued_answers
    print(f"\n✓ Reviewed {len(critiqued_answers)} candidate answer(s)")
    
    return state


def n8_6_rewriter_synthesis_node(state: GeneralQAState) -> GeneralQAState:
    """
    N8.6: Rewriter Synthesis (X-Masters Enhancement)
    
    Synthesizes all critiqued answers into new improved answers.
    """
    print("=" * 60)
    print("N8.6: Rewriter Synthesis (X-Masters Enhancement)")
    print("=" * 60)
    
    if not state.critiqued_answers or len(state.critiqued_answers) == 0:
        print(f"  ⚠ No critiqued answers to synthesize, skipping rewriter stage")
        state.rewritten_answers = []
        return state
    
    # Import X-Masters rewriter function
    try:
        from agent.nodes.subagents.x_masters.rewriter import run_single_rewriter
    except ImportError as e:
        print(f"  ⚠ X-Masters rewriter not available (optional dependency): {e}")
        print(f"  → Continuing without X-Masters enhancement, using critiqued answers")
        state.rewritten_answers = state.critiqued_answers  # Fallback
        return state
    
    # Build problem context
    problem_context = state.cleaned_text or state.user_input
    if state.core_conclusion:
        problem_context += f"\n\nCore Conclusion: {state.core_conclusion}"
    
    # Build retrieved context
    retrieved_context = ""
    if state.domain_knowledge_map:
        for domain, knowledge in state.domain_knowledge_map.items():
            if isinstance(knowledge, dict):
                found_knowledge = knowledge.get("foundational_knowledge", []) + knowledge.get("specialized_knowledge", [])
                if found_knowledge:
                    retrieved_context += f"\n{domain}: {str(found_knowledge)[:500]}\n"
    
    # Extract all critiqued solution strings
    all_solutions = [c.get("critiqued_answer", "") for c in state.critiqued_answers if c.get("critiqued_answer")]
    
    if not all_solutions:
        print(f"  ⚠ No valid solutions to synthesize")
        state.rewritten_answers = []
        return state
    
    # Generate 2-3 rewritten solutions
    num_rewriters = min(3, len(all_solutions))
    rewritten_answers = []
    
    for i in range(num_rewriters):
        print(f"\n  🔄 Synthesizing Rewritten Answer {i}...")
        
        try:
            rewriter_result = run_single_rewriter(
                problem=problem_context,
                all_solutions=all_solutions,
                rewriter_id=i,
                retrieved_context=retrieved_context,
                temperature=0.7,
                llm=None,
                source=None,
                base_url=None,
                api_key=None,
                timeout_seconds=120,
            )
            
            rewritten_answer = rewriter_result.get("solution", "")
            success = rewriter_result.get("success", False)
            
            if rewritten_answer:
                rewritten_answers.append({
                    "rewriter_id": i,
                    "rewritten_answer": rewritten_answer,
                    "success": success
                })
                print(f"    ✓ Rewriter {i} completed successfully")
            else:
                print(f"    ⚠ Rewriter {i} produced empty answer")
                
        except Exception as e:
            print(f"    ❌ Rewriter {i} error: {e}")
    
    # If no rewritten answers, use best critiqued answer as fallback
    if not rewritten_answers:
        print(f"  ⚠ No rewritten answers generated, using best critiqued answer")
        best_critiqued = max(state.critiqued_answers, key=lambda x: x.get("success", False))
        rewritten_answers = [{
            "rewriter_id": 0,
            "rewritten_answer": best_critiqued.get("critiqued_answer", ""),
            "success": True
        }]
    
    state.rewritten_answers = rewritten_answers
    print(f"\n✓ Generated {len(rewritten_answers)} rewritten answer(s)")
    
    return state


def _validate_answer_against_biomedical_rules(
    final_answer: Optional[str],
    core_domains: Optional[List[str]],
    key_entities: Optional[List[str]],
    question_options: Optional[List[str]],
    answer_format_label: Optional[str]
) -> List[str]:
    """
    Validate answer against biomedical domain hard rules (领域硬规则校验)
    
    Args:
        final_answer: Final answer to validate
        core_domains: Core domains
        key_entities: Key entities
        question_options: Question options (for multiple choice)
        answer_format_label: Answer format label
    
    Returns:
        List of validation error messages (empty if valid)
    """
    import re
    errors = []
    if not final_answer:
        return errors
    
    answer_str = str(final_answer).lower()
    domains_lower = [d.lower() for d in (core_domains or [])]
    entities_lower = [e.lower() for e in (key_entities or [])]
    entities_str = str(entities_lower)
    
    # Rule 1: Fluorescence wavelength matching (荧光波长匹配规则)
    if any(kw in entities_str for kw in ['fluorescence', 'fluorescent', 'probe', 'excitation', 'wavelength']):
        # eGFP typically uses 488nm, DsRed uses 559nm, HaloTag uses 630nm
        if 'egfp' in answer_str or 'green' in answer_str:
            # Check if answer mentions correct wavelength
            if '488' not in answer_str and '488nm' not in answer_str:
                if any(opt and '488' in str(opt).lower() for opt in (question_options or [])):
                    errors.append("CRITICAL: eGFP excitation wavelength should be 488nm, not matching answer")
        if 'dsred' in answer_str or 'red' in answer_str:
            if '559' not in answer_str and '559nm' not in answer_str:
                if any(opt and '559' in str(opt).lower() for opt in (question_options or [])):
                    errors.append("CRITICAL: DsRed excitation wavelength should be 559nm, not matching answer")
        if 'halo' in answer_str or 'halotag' in answer_str:
            if '630' not in answer_str and '630nm' not in answer_str:
                if any(opt and '630' in str(opt).lower() for opt in (question_options or [])):
                    errors.append("CRITICAL: HaloTag excitation wavelength should be 630nm, not matching answer")
    
    # Rule 2: DNA sequence direction (序列方向规则)
    if answer_format_label == "Sequence" and any(kw in entities_str for kw in ['dna', 'rna', 'sequence', 'oligo']):
        if not ("5'" in str(final_answer) and "3'" in str(final_answer)):
            errors.append("CRITICAL: DNA/RNA sequence answer must include 5' and 3' orientation")
        # Check direction consistency
        if "5'" in str(final_answer) and "3'" in str(final_answer):
            # Extract sequence part
            seq_match = re.search(r"5'\s*([ATCGUatcgu\s]+)\s*3'", str(final_answer))
            if seq_match:
                seq = seq_match.group(1).replace(" ", "").upper()
                # Basic validation: should contain only valid nucleotides
                if not all(c in 'ATCGU' for c in seq):
                    errors.append("CRITICAL: DNA/RNA sequence contains invalid nucleotides")
    
    # Rule 3: BUD time constraints (BUD时间约束规则)
    if any(kw in entities_str for kw in ['bud', 'beyond use date', 'sterile', 'puncture', 'ampule']):
        # BUD for single dose container after puncture in sterile environment is typically 1 hour
        if 'hour' in answer_str or 'h' in answer_str:
            # Extract time value
            time_match = re.search(r'(\d+)\s*(?:hour|h)', answer_str)
            if time_match:
                hours = int(time_match.group(1))
                if hours > 24:
                    errors.append("CRITICAL: BUD time exceeds reasonable limit (typically 1-24 hours)")
        elif 'minute' in answer_str or 'min' in answer_str:
            time_match = re.search(r'(\d+)\s*(?:minute|min)', answer_str)
            if time_match:
                minutes = int(time_match.group(1))
                if minutes > 1440:  # 24 hours
                    errors.append("CRITICAL: BUD time exceeds reasonable limit")
    
    # Rule 4: Codon translation rules (密码子翻译规则)
    if any(kw in entities_str for kw in ['codon', 'translation', 'amino acid', 'dna sequence']):
        # Check if answer mentions valid amino acids or codons
        valid_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        if answer_format_label == "Sequence":
            # Extract amino acid sequence
            seq_match = re.search(r"([A-Z]+)", str(final_answer))
            if seq_match:
                seq = seq_match.group(1)
                invalid_chars = [c for c in seq if c not in valid_aa and c not in ['-', ' ']]
                if invalid_chars:
                    errors.append(f"CRITICAL: Amino acid sequence contains invalid characters: {set(invalid_chars)}")
    
    return errors


def _validate_answer_against_constraints(answer: str, question_type: str, parameter_constraints: Optional[Dict[str, Dict[str, Any]]], domain_knowledge_map: Optional[Dict[str, Dict[str, Any]]]) -> tuple[bool, List[str]]:
    """
    Validate answer against parameter constraints and knowledge base.
    
    Args:
        answer: Final answer to validate
        question_type: Question type label
        parameter_constraints: Parameter constraints extracted from knowledge base
        domain_knowledge_map: Domain knowledge map
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if question_type == "Numerical Calculation" and parameter_constraints:
        # Extract numerical value from answer
        import re
        result_matches = re.findall(r'([+-]?[\d.]+)\s*(?:mL/g|mg/L|μM|mM|M|g/mol|kDa|Gy|%|units?)', answer)
        if not result_matches:
            result_matches = re.findall(r'([+-]?[\d.]+)', answer)
        
        if result_matches:
            try:
                final_result = float(result_matches[-1])
                
                # Check against all parameter constraints
                for param_name, constraints in parameter_constraints.items():
                    # Check range
                    if "range" in constraints:
                        min_val = constraints["range"].get("min")
                        max_val = constraints["range"].get("max")
                        if min_val is not None and max_val is not None:
                            if final_result < min_val or final_result > max_val:
                                issues.append(
                                    f"Answer {final_result} is outside expected range [{min_val}, {max_val}] for {param_name}"
                                )
                    
                    # Check sign
                    if "sign" in constraints:
                        expected_sign = constraints["sign"]
                        if expected_sign == "positive" and final_result < 0:
                            issues.append(
                                f"Answer {final_result} violates sign constraint (expected positive for {param_name})"
                            )
                        elif expected_sign == "negative" and final_result > 0:
                            issues.append(
                                f"Answer {final_result} violates sign constraint (expected negative for {param_name})"
                            )
            except (ValueError, IndexError):
                pass
    
    return len(issues) == 0, issues


def n9_result_validation_node(state: GeneralQAState) -> GeneralQAState:
    """
    N9: Result Validation & Consistency Judgment (Enhanced with X-Masters Selector)
    
    Structure: Input validation => Data preparation => Execution => Result organization
    Enhanced with improved consistency checking and option semantic matching.
    If rewritten_answers exist, uses X-Masters Selector to choose the best answer.
    """
    # ========== X-Masters Selector: Choose Best Answer ==========
    if state.rewritten_answers and len(state.rewritten_answers) > 0:
        print("=" * 60)
        print("N9: X-Masters Selector - Choosing Best Answer")
        print("=" * 60)
        
        try:
            from agent.nodes.subagents.x_masters.selector import run_selector
            
            # Build problem context
            problem_context = state.cleaned_text or state.user_input
            if state.core_conclusion:
                problem_context += f"\n\nCore Conclusion: {state.core_conclusion}"
            
            # Build retrieved context
            retrieved_context = ""
            if state.domain_knowledge_map:
                for domain, knowledge in state.domain_knowledge_map.items():
                    if isinstance(knowledge, dict):
                        found_knowledge = knowledge.get("foundational_knowledge", []) + knowledge.get("specialized_knowledge", [])
                        if found_knowledge:
                            retrieved_context += f"\n{domain}: {str(found_knowledge)[:500]}\n"
            
            # Extract all rewritten solution strings
            all_solutions = [r.get("rewritten_answer", "") for r in state.rewritten_answers if r.get("rewritten_answer")]
            
            if all_solutions:
                print(f"  🔍 Selecting best answer from {len(all_solutions)} rewritten solutions...")
                
                selector_result = run_selector(
                    problem=problem_context,
                    all_solutions=all_solutions,
                    retrieved_context=retrieved_context,
                    temperature=0.7,
                    llm=None,
                    source=None,
                    base_url=None,
                    api_key=None,
                    timeout_seconds=120,
                )
                
                selected_answer = selector_result.get("solution", "")
                selected_index = selector_result.get("selected_index", 0)
                success = selector_result.get("success", False)
                
                if selected_answer and success:
                    # Update state with selected answer
                    state.final_answer = selected_answer
                    
                    # Try to reconstruct structured_answer from selected answer
                    if state.rewritten_answers and selected_index < len(state.rewritten_answers):
                        # Use the structured format from the selected rewritten answer if available
                        selected_rewritten = state.rewritten_answers[selected_index]
                        # Try to find corresponding structured answer
                        if state.critiqued_answers and selected_index < len(state.critiqued_answers):
                            critiqued = state.critiqued_answers[selected_index]
                            if critiqued.get("original_structured"):
                                state.structured_answer = critiqued.get("original_structured")
                    
                    print(f"  ✓ Selected answer {selected_index + 1} as final answer")
                else:
                    print(f"  ⚠ Selector failed, using first rewritten answer")
                    if all_solutions:
                        state.final_answer = all_solutions[0]
            else:
                print(f"  ⚠ No valid rewritten solutions for selection")
                
        except ImportError as e:
            print(f"  ⚠ X-Masters Selector not available: {e}, using original answer")
        except Exception as e:
            print(f"  ⚠ Selector error: {e}, using original answer")
    
    # OPTIMIZATION: Three-layer automated validation (三层自动化校验)
    validation_errors = []
    
    # Layer 1: Format validation (格式校验)
    if not state.structured_answer:
        validation_errors.append("Format Error: structured_answer is missing")
    elif not state.final_answer:
        validation_errors.append("Format Error: final_answer is missing")
    else:
        # Check format matches answer_format_label
        answer_str = str(state.final_answer)
        if state.answer_format_label == "Single Choice":
            if answer_str not in [chr(65+i) for i in range(len(state.question_options or []))]:
                validation_errors.append(f"Format Error: Single Choice answer '{answer_str}' not in valid options")
        elif state.answer_format_label == "Sequence":
            if not ("5'" in answer_str and "3'" in answer_str):
                validation_errors.append("Format Error: Sequence answer missing 5'/3' orientation")
        elif state.answer_format_label == "Numeric":
            import re
            if not re.search(r'[\d.]+', answer_str):
                validation_errors.append("Format Error: Numeric answer contains no numbers")
    
    # Layer 2: Logic validation (逻辑校验)
    if state.closed_inference_path and state.core_conclusion:
        # Check if conclusion follows from inference path
        path_text = " ".join([str(s.get("step_content", "")) for s in state.closed_inference_path]).lower()
        conclusion_lower = str(state.core_conclusion).lower()
        # Simple semantic overlap check
        path_words = set(path_text.split())
        conclusion_words = set(conclusion_lower.split())
        overlap = len(path_words & conclusion_words)
        if overlap < 2 and len(path_words) > 5:
            validation_errors.append("Logic Error: Conclusion does not logically follow from inference path")
    
    # Layer 3: Domain hard rule validation (领域硬规则校验)
    domain_rule_errors = _validate_answer_against_biomedical_rules(
        state.final_answer,
        state.core_domains,
        state.key_entities,
        state.question_options,
        state.answer_format_label
    )
    validation_errors.extend(domain_rule_errors)
    
    # If validation errors found, mark as invalid
    if validation_errors:
        print(f"  ❌ Validation failed with {len(validation_errors)} error(s):")
        for error in validation_errors[:5]:  # Show first 5
            print(f"    - {error}")
        state.consistency_label = "Inconsistent"
        state.reliability_score = 0
        state.format_valid_label = "Invalid"
        state.exception_type_label = "Answer Validation Failed"
        state.format_issues = validation_errors
        return state
    
    # Input validation
    if not state.structured_answer or not state.closed_inference_path:
        state.error_message = "structured_answer and closed_inference_path are required for result validation"
        return state
    
    print("=" * 60)
    print("N9: Result Validation & Consistency Judgment")
    print("=" * 60)
    
    # Initialize node visit tracking
    if state.node_visit_count is None:
        state.node_visit_count = {}
    
    # Increment N9 visit count
    n9_visits = state.node_visit_count.get("n9_result_validation", 0)
    state.node_visit_count["n9_result_validation"] = n9_visits + 1
    
    # CRITICAL: Prevent infinite loops - if N9 has been visited too many times, skip processing
    if n9_visits >= 3:
        print(f"  ⚠ Infinite loop detected: N9 visited {n9_visits + 1} times, skipping to prevent infinite loop")
        state.consistency_label = "Inconsistent"
        state.reliability_score = 0
        state.exception_type_label = state.exception_type_label or "Result Validation Failed - Infinite Loop"
        return state
    
    # OPTIMIZATION: Three-layer automated validation (三层自动化校验)
    validation_errors = []
    
    # Layer 1: Format validation (格式校验)
    if not state.structured_answer:
        validation_errors.append("Format Error: structured_answer is missing")
    elif not state.final_answer:
        validation_errors.append("Format Error: final_answer is missing")
    else:
        # Check format matches answer_format_label
        answer_str = str(state.final_answer)
        if state.answer_format_label == "Single Choice":
            if answer_str not in [chr(65+i) for i in range(len(state.question_options or []))]:
                validation_errors.append(f"Format Error: Single Choice answer '{answer_str}' not in valid options")
        elif state.answer_format_label == "Sequence":
            if not ("5'" in answer_str and "3'" in answer_str):
                validation_errors.append("Format Error: Sequence answer missing 5'/3' orientation")
        elif state.answer_format_label == "Numeric":
            import re
            if not re.search(r'[\d.]+', answer_str):
                validation_errors.append("Format Error: Numeric answer contains no numbers")
    
    # Layer 2: Logic validation (逻辑校验)
    if state.closed_inference_path and state.core_conclusion:
        # Check if conclusion follows from inference path
        path_text = " ".join([str(s.get("step_content", "")) for s in state.closed_inference_path]).lower()
        conclusion_lower = str(state.core_conclusion).lower()
        # Simple semantic overlap check
        path_words = set(path_text.split())
        conclusion_words = set(conclusion_lower.split())
        overlap = len(path_words & conclusion_words)
        if overlap < 2 and len(path_words) > 5:
            validation_errors.append("Logic Error: Conclusion does not logically follow from inference path")
    
    # Layer 3: Domain hard rule validation (领域硬规则校验)
    domain_rule_errors = _validate_answer_against_biomedical_rules(
        state.final_answer,
        state.core_domains,
        state.key_entities,
        state.question_options,
        state.answer_format_label
    )
    validation_errors.extend(domain_rule_errors)
    
    # If validation errors found, mark as invalid and trigger exception
    if validation_errors:
        print(f"  ❌ Validation failed with {len(validation_errors)} error(s):")
        for error in validation_errors[:5]:  # Show first 5
            print(f"    - {error}")
        state.consistency_label = "Inconsistent"
        state.reliability_score = 0
        state.format_valid_label = "Invalid"
        state.exception_type_label = "Answer Validation Failed"
        state.format_issues = validation_errors
        # Don't return here - continue to LLM validation for additional checks
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for result validation"
        return state
    
    # Load tools for option semantic matching
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n9_result_validation")
            print(f"  📚 Loaded {len(tools)} tool(s) for validation")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    # Enhanced: Check inference path consistency
    inference_consistency_issues = []
    if state.closed_inference_path:
        # Check if critical constraints were considered
        if state.critical_constraints:
            path_text = " ".join([
                str(step.get("step_content", "")) for step in state.closed_inference_path
            ]).lower()
            for constraint in state.critical_constraints:
                constraint_lower = str(constraint).lower()
                # Check if constraint keywords appear in path
                constraint_keywords = constraint_lower.split()
                if not any(kw in path_text for kw in constraint_keywords if len(kw) > 3):
                    inference_consistency_issues.append(f"Critical constraint '{constraint}' not considered in inference path")
        
        # Check if conclusion logically follows from path
        if state.core_conclusion and state.closed_inference_path:
            last_step = state.closed_inference_path[-1] if state.closed_inference_path else {}
            last_result = str(last_step.get("intermediate_result", "")).lower()
            conclusion_lower = str(state.core_conclusion).lower()
            # Simple check: conclusion should relate to last step
            if last_result and conclusion_lower:
                # Check for semantic overlap
                last_words = set(last_result.split())
                conclusion_words = set(conclusion_lower.split())
                overlap = len(last_words & conclusion_words)
                if overlap < 2 and len(last_words) > 5:  # If last step has content but no overlap
                    inference_consistency_issues.append("Conclusion may not logically follow from inference path")
    
    # Extract hard_constraints from structured_condition
    hard_constraints = []
    if state.structured_condition and isinstance(state.structured_condition, dict):
        hard_constraints = state.structured_condition.get("hard_constraints", [])
        if hard_constraints:
            print(f"  ⚠ Validating against hard constraints: {hard_constraints}")
    
    # OPTIMIZATION: Add core keywords and option features to validation prompt
    core_keywords_str = ", ".join(state.core_keywords) if state.core_keywords else "N/A"
    option_features_str = str(state.option_features) if state.option_features else "N/A"
    
    base_prompt = get_result_validation_prompt(
        state.structured_answer,
        state.closed_inference_path,
        state.answer_format_label,
        state.question_options or [],
        state.answer_constraints,
        state.question_type_label,
        hard_constraints=hard_constraints,
        structured_goal=state.structured_goal,
        core_keywords=core_keywords_str,
        option_features=option_features_str
    )
    
    # Rule 4: Add original question text for factual correctness check
    if state.cleaned_text:
        factual_check_instruction = f"\n\n**CRITICAL: Original Question Text (题干原文) - FACTUAL CORRECTNESS CHECK:**\n{state.cleaned_text}\n\nCheck if final answer conflicts with this question text. If answer contradicts question text → mark INCONSISTENT, score ≤ 2.0. Only when BOTH factually correct AND logically consistent → score ≥ 4.0."
        base_prompt = base_prompt + factual_check_instruction
    
    # OPTIMIZATION 2: Add 2 core validation checks
    # Check 1: Answer matches question's core/negative constraints
    constraint_check_instruction = "\n\n**CRITICAL VALIDATION CHECK 1: Answer-Constraint Matching (答案与约束匹配性校验) - MANDATORY:**\n"
    constraint_check_instruction += "You MUST verify:\n"
    constraint_check_instruction += "1. If question contains NEGATIVE constraints (cannot/except/not occur), verify that the answer does NOT violate them\n"
    constraint_check_instruction += "2. If question contains EXCLUSIVE constraints (category 1/only 1/single), verify that the answer is the ONLY one that satisfies them\n"
    constraint_check_instruction += "3. If question contains HARD constraints (禁用/排除), verify that the answer does NOT include prohibited items\n"
    constraint_check_instruction += "If answer violates ANY constraint → mark INCONSISTENT, reliability_score ≤ 2.0\n"
    
    # Check 2: Answer logical rationality
    rationality_check_instruction = "\n\n**CRITICAL VALIDATION CHECK 2: Answer Logical Rationality (答案合理性校验) - MANDATORY:**\n"
    rationality_check_instruction += "You MUST verify:\n"
    rationality_check_instruction += "1. For numerical answers: Check if value is within reasonable range (no absurd magnitude, e.g., 10^10 for biological measurements)\n"
    rationality_check_instruction += "2. For sequence answers: Check if sequence length is reasonable (not empty, not extremely long)\n"
    rationality_check_instruction += "3. For option answers: Check if selected option is logically consistent with question context\n"
    rationality_check_instruction += "4. For calculation answers: Check if result magnitude matches expected scale (e.g., dose in Gy should be 10^-6 to 10^-3 range, not 10^3)\n"
    rationality_check_instruction += "If answer is logically unreasonable → mark INCONSISTENT, reliability_score ≤ 2.0\n"
    
    base_prompt = base_prompt + constraint_check_instruction + rationality_check_instruction
    
    # Enhanced: Add consistency checking instructions
    if inference_consistency_issues:
        consistency_instruction = "\n\nCONSISTENCY ISSUES DETECTED:\n"
        for issue in inference_consistency_issues:
            consistency_instruction += f"- {issue}\n"
        consistency_instruction += "\nPlease carefully review the inference path and conclusion for logical consistency."
        prompt = base_prompt + consistency_instruction
    else:
        prompt = base_prompt
    
    # Execution with tools for option matching
    response = _call_llm(llm, prompt, tools=tools, max_iterations=2, state=state, node_name="n9_result_validation")
    if not response:
        state.error_message = "LLM call failed for result validation"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for result validation"
        return state
    
    state.consistency_label = result.get("consistency_label")
    state.reliability_score = result.get("reliability_score")
    state.fact_check_result = result.get("fact_check_result", {})
    llm_format_label = result.get("format_valid_label")
    llm_format_issues = result.get("format_issues", [])
    
    if state.fact_check_result:
        print(f"  ✓ Fact check result: {state.fact_check_result.get('answer_matches_fact', 'N/A')}")
        if state.fact_check_result.get("correct_function"):
            print(f"    - Correct function: {state.fact_check_result['correct_function']}")
    
    # OPTIMIZATION 2: Code-level validation checks
    # Check 1: Answer matches question's core/negative constraints
    constraint_violation = False
    if state.final_answer and state.cleaned_text:
        answer_lower = str(state.final_answer).lower()
        text_lower = state.cleaned_text.lower()
        # Check for negative constraint violations
        if any(keyword in text_lower for keyword in ["cannot", "can not", "except", "not occur", "exclude"]):
            # If question says "cannot X" but answer contains X, it's a violation
            import re
            cannot_patterns = re.findall(r'cannot\s+([^.!?]+)|can\s+not\s+([^.!?]+)|except\s+([^.!?]+)', text_lower)
            for pattern_group in cannot_patterns:
                for pattern in pattern_group:
                    if pattern and pattern.strip():
                        # Check if answer contains the prohibited term
                        prohibited_terms = pattern.strip().split()
                        if any(term in answer_lower for term in prohibited_terms if len(term) > 3):
                            constraint_violation = True
                            print(f"  ⚠ Constraint violation detected: answer contains prohibited term from negative constraint")
                            break
                if constraint_violation:
                    break
        
        # Check for exclusive constraint violations (e.g., "category 1" but answer is not category 1)
        if any(keyword in text_lower for keyword in ["category 1", "only 1", "single"]):
            # This is more complex, rely on LLM validation, but flag if answer seems to violate
            if "category 1" in text_lower and "category 1" not in answer_lower:
                # Check if answer might be wrong category
                if any(keyword in answer_lower for keyword in ["category 2", "category 3", "second", "third"]):
                    constraint_violation = True
                    print(f"  ⚠ Constraint violation detected: exclusive constraint 'category 1' but answer suggests other category")
    
    # Check 2: Answer logical rationality
    rationality_issue = False
    if state.final_answer:
        answer_str = str(state.final_answer)
        # Check for numerical answers
        import re
        # Try to extract numerical value
        num_match = re.search(r'([\d.]+)\s*\*\s*10\^?([+-]?\d+)', answer_str)
        if num_match:
            base = float(num_match.group(1))
            exp = int(num_match.group(2))
            value = base * (10 ** exp)
            # Check for absurd magnitudes (e.g., > 10^6 for biological measurements, < 10^-10 for doses)
            if abs(exp) > 10:
                rationality_issue = True
                print(f"  ⚠ Rationality issue: numerical value has extreme exponent ({exp})")
        # Check for empty or extremely long sequences
        if state.answer_format_label == "Sequence":
            if len(answer_str) == 0:
                rationality_issue = True
                print(f"  ⚠ Rationality issue: sequence answer is empty")
            elif len(answer_str) > 10000:
                rationality_issue = True
                print(f"  ⚠ Rationality issue: sequence answer is extremely long ({len(answer_str)} chars)")
    
    # Apply validation results
    if constraint_violation or rationality_issue:
        if state.consistency_label == "Consistent":
            state.consistency_label = "Inconsistent"
            print(f"  ⚠ Overriding consistency to Inconsistent due to constraint violation or rationality issue")
        if not state.reliability_score or state.reliability_score > 2.0:
            state.reliability_score = 2.0
            print(f"  ⚠ Setting reliability_score to 2.0 due to validation issues")
    
    # Enhanced: Consider inference consistency issues
    if inference_consistency_issues:
        if state.consistency_label == "Consistent":
            # Override if we detected issues
            state.consistency_label = "Inconsistent"
            if not state.reliability_score or state.reliability_score > 3:
                state.reliability_score = 3.0
            print(f"  ⚠ Overriding consistency to Inconsistent due to detected issues")
    
    # ========== Step 7: Validate answer against parameter constraints ==========
    # Additional validation using parameter constraints for calculation problems
    if state.calculation_type_label == "Numerical" and state.parameter_constraints and state.final_answer:
        is_valid, constraint_issues = _validate_answer_against_constraints(
            state.final_answer,
            state.question_type_label or "",
            state.parameter_constraints,
            state.domain_knowledge_map
        )
        
        if not is_valid and constraint_issues:
            print(f"  ⚠ Parameter constraint validation failed:")
            for issue in constraint_issues:
                print(f"    - {issue}")
            
            # Override consistency if constraints are violated
            if state.consistency_label == "Consistent":
                state.consistency_label = "Inconsistent"
                print(f"  ⚠ Overriding consistency to Inconsistent due to parameter constraint violations")
            
            # Reduce reliability score
            if not state.reliability_score or state.reliability_score > 2.0:
                state.reliability_score = 2.0
                print(f"  ⚠ Setting reliability_score to 2.0 due to parameter constraint violations")
            
            # Mark exception
            if not state.exception_type_label or "Result Out of Range" not in state.exception_type_label:
                state.exception_type_label = (state.exception_type_label or "") + " / Result Out of Range"
    
    local_format_label, local_format_issues = _validate_answer_format(
        state.final_answer,
        state.answer_format_label,
        state.question_options or [],
        state.answer_constraints or []
    )
    if local_format_label == "Invalid":
        state.format_valid_label = "Invalid"
        state.format_issues = local_format_issues
    else:
        state.format_valid_label = llm_format_label or "Valid"
        state.format_issues = llm_format_issues
    if llm_format_label == "Invalid" and state.format_valid_label != "Invalid":
        state.format_valid_label = "Invalid"
        state.format_issues = llm_format_issues or ["format invalid"]
    if state.format_valid_label == "Invalid":
        state.consistency_label = "Inconsistent"
        if not state.reliability_score or state.reliability_score > 3:
            state.reliability_score = 3.0
        state.exception_type_label = "Answer Format Invalid"
    elif state.consistency_label == "Inconsistent" and not state.exception_type_label:
        state.exception_type_label = "Inference Path Inconsistent"
    
    print(f"✓ Consistency: {state.consistency_label}")
    print(f"✓ Reliability score: {state.reliability_score}")
    if state.format_valid_label:
        print(f"✓ Format validity: {state.format_valid_label}")
    if inference_consistency_issues:
        print(f"  ⚠ Inference consistency issues: {len(inference_consistency_issues)}")
    
    return state


def n10_exception_handling_node(state: GeneralQAState) -> GeneralQAState:
    """
    N10: Knowledge/Calculation Exception Handling
    
    Structure: Input validation => Data preparation => Execution => Result organization
    This node uses ALL available tools to find alternative solutions when exceptions occur.
    """
    # Input validation - This node is triggered by exceptions, so we need exception context
    print("=" * 60)
    print("N10: Knowledge/Calculation Exception Handling")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for exception handling"
        return state
    
    # Load tools for finding alternative solutions
    tools = []
    if TOOLS_AVAILABLE and get_tools_for_node:
        try:
            tools = get_tools_for_node("n10_exception_handling")
            print(f"  📚 Loaded {len(tools)} tool(s) for exception handling")
        except Exception as e:
            print(f"  ⚠ Failed to load tools: {e}")
    
    # OPTIMIZATION: Comprehensive exception type detection (全覆盖异常类型)
    # Define all possible exception types
    exception_types = {
        "Knowledge Retrieval Parsing Failed": "知识检索解析失败",
        "Knowledge Invalid - Cannot Infer": "知识无效-无法推理",
        "Knowledge Missing - Cannot Infer": "知识缺失-无法推理",
        "Knowledge Domain Rule Violation": "知识领域规则违反",
        "Answer Generation Failed - No Inference Result": "答案生成失败-无推理结果",
        "Answer Generation Failed - Incomplete Inference Path": "答案生成失败-推理路径不完整",
        "Answer Generation Failed - Invalid Knowledge": "答案生成失败-无效知识",
        "Answer Validation Failed": "答案验证失败",
        "Formula Match Failed": "公式匹配失败",
        "Inference Path Incomplete": "推理路径不完整",
        "Answer Rationality Check Failed": "答案合理性检查失败",
        "Result Validation Failed - Infinite Loop": "结果验证失败-无限循环",
        "Knowledge Missing": "知识缺失",
        "Unknown": "未知异常"
    }
    
    # Determine exception type from state with priority
    exception_type = "Unknown"
    if state.exception_type_label:
        exception_type = state.exception_type_label
    elif state.error_message:
        # Try to infer from error message
        error_lower = state.error_message.lower()
        if "parse" in error_lower and "knowledge" in error_lower:
            exception_type = "Knowledge Retrieval Parsing Failed"
        elif "knowledge" in error_lower and ("invalid" in error_lower or "missing" in error_lower):
            exception_type = "Knowledge Missing - Cannot Infer"
        elif "answer" in error_lower and "generation" in error_lower:
            exception_type = "Answer Generation Failed - No Inference Result"
        elif "validation" in error_lower:
            exception_type = "Answer Validation Failed"
    
    exception_context = {
        "exception_type": exception_type,
        "exception_type_cn": exception_types.get(exception_type, "未知异常"),
        "knowledge_validity": state.knowledge_validity_label,
        "knowledge_confidence": None,  # Will be extracted from metadata
        "formula_match_result": state.formula_match_result,
        "applicability_result": state.applicability_result,
        "consistency_label": state.consistency_label,
        "format_validity": state.format_valid_label,
        "format_issues": state.format_issues,
        "answer_format": state.answer_format_label,
        "error_message": state.error_message,
        "has_core_conclusion": bool(state.core_conclusion),
        "has_inference_path": bool(state.closed_inference_path),
        "has_domain_knowledge": bool(state.domain_knowledge_map)
    }
    
    # Extract knowledge confidence if available
    if state.parameter_constraints and isinstance(state.parameter_constraints, dict):
        metadata = state.parameter_constraints.get("_knowledge_metadata", {})
        if isinstance(metadata, dict):
            exception_context["knowledge_confidence"] = metadata.get("confidence")
    
    print(f"  📋 Exception Type: {exception_type} ({exception_types.get(exception_type, '未知')})")
    print(f"    - Knowledge Validity: {state.knowledge_validity_label}")
    print(f"    - Has Core Conclusion: {bool(state.core_conclusion)}")
    print(f"    - Has Inference Path: {bool(state.closed_inference_path)}")
    
    prompt = get_exception_handling_prompt(exception_type, exception_context)
    
    # Execution with tools
    response = _call_llm(llm, prompt, tools=tools, max_iterations=4, state=state, node_name="n10_exception_handling")
    if not response:
        state.error_message = "LLM call failed for exception handling"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for exception handling"
        return state
    
    state.exception_type_label = result.get("exception_type_label", exception_type)
    state.solution_suggestion = result.get("solution_suggestion")
    
    print(f"✓ Exception type: {state.exception_type_label}")
    print(f"✓ Solution suggestion: {state.solution_suggestion}")
    
    return state


def n11_manual_intervention_node(state: GeneralQAState) -> GeneralQAState:
    """
    N11: Manual Intervention Trigger
    
    Structure: Input validation => Data preparation => Execution => Result organization
    """
    # Input validation
    if not state.exception_type_label:
        state.error_message = "exception_type_label is required for manual intervention"
        return state
    
    print("=" * 60)
    print("N11: Manual Intervention Trigger")
    print("=" * 60)
    
    # Data preparation
    llm = _get_llm()
    if llm is None:
        state.error_message = "LLM unavailable for manual intervention"
        return state
    
    intermediate_results = {
        "cleaned_text": state.cleaned_text,
        "question_type": state.question_type_label,
        "core_domains": state.core_domains,
        "key_entities": state.key_entities,
        "answer_format": state.answer_format_label,
        "answer_constraints": state.answer_constraints,
        "question_options": state.question_options,
        "core_conclusion": state.core_conclusion,
        "structured_answer": state.structured_answer,
        "exception_type": state.exception_type_label
    }
    
    prompt = get_manual_intervention_prompt(
        state.exception_type_label,
        intermediate_results
    )
    
    # Execution
    response = _call_llm(llm, prompt, state=state, node_name="n11_manual_intervention")
    if not response:
        state.error_message = "LLM call failed for manual intervention"
        return state
    
    # Result organization
    result = _parse_json_response(response)
    if not result:
        state.error_message = "Failed to parse LLM response for manual intervention"
        return state
    
    state.manual_intervention_guide = result.get("manual_intervention_guide")
    state.intermediate_result_snapshot = result.get("intermediate_result_snapshot")
    
    print(f"✓ Manual intervention guide generated")
    
    return state


# ===================== Routing Functions =====================

def route_after_n0(state: GeneralQAState) -> str:
    """Route after N0: Determine if question is calculation/algorithm or reasoning type"""
    if state.data_completeness_label == "Severe Missing":
        return "n10_exception_handling"
    
    # 修复路由规则：只有真正的计算题才进入计算分支
    # Professional Algorithm 和 Multiple Choice 应该跳过计算分支，直接进入推理路径
    if state.question_type_label == "Numerical Calculation":
        return "n2_calculation_algorithm_recognition"
    else:
        # Multiple Choice, Text Matching, Mechanism Explanation, Professional Algorithm 都走推理路径
        return "n1_question_decomposition"


def route_after_n3(state: GeneralQAState) -> str:
    """Route after N3: Check knowledge validity"""
    if state.knowledge_validity_label == "Missing":
        return "n10_exception_handling"
    
    # OPTIMIZATION: Ensure n6 executes for ALL question types (including calculation)
    # n6 provides knowledge matching that is useful for both reasoning and calculation questions
    # For calculation questions, n6 matches calculation-related knowledge (e.g., grouping logic, minimal set rules)
    # After n6, route based on calculation type
    return "n6_initial_inference"


def route_after_n4(state: GeneralQAState) -> str:
    """Route after N4: Check formula match result"""
    # Rule 5: Formula match failure should NOT immediately route to exception handling
    # Instead, allow fallback to common-sense reasoning in N7
    # Only route to exception if there are no calculation_steps at all
    if state.formula_match_result == "Match Failed" and (not state.calculation_steps or len(state.calculation_steps) == 0):
        # OPTIMIZATION 5: Auto-retry mechanism - retry n2/n4 once before triggering n10/n11
        if not hasattr(state, 'auto_retry_count') or state.auto_retry_count is None:
            state.auto_retry_count = 0
        
        if state.auto_retry_count < 1:
            state.auto_retry_count = (state.auto_retry_count or 0) + 1
            print(f"  🔄 Auto-retry mechanism triggered: retrying n2_calculation_algorithm_recognition (attempt {state.auto_retry_count}/1)")
            # Reset relevant state to allow retry
            state.calculation_steps = None
            state.formula_match_result = None
            state.matched_formula = None
            return "n2_calculation_algorithm_recognition"
        else:
            return "n10_exception_handling"
    # Even if formula match failed, proceed to N7 for fallback reasoning
    return "n7_complete_inference"


def route_after_n5(state: GeneralQAState) -> str:
    """Route after N5: Check applicability result"""
    if state.applicability_result == "Not Applicable":
        return "n10_exception_handling"
    return "n7_complete_inference"


def route_after_n6(state: GeneralQAState) -> str:
    """Route after N6: Check if inference outputs are present and route based on question type"""
    # OPTIMIZATION: Route based on calculation type after n6
    # n6 now executes for ALL question types, providing knowledge matching
    if state.calculation_type_label:
        # We're on calculation/algorithm path - route to n4 or n5 after n6
        if state.calculation_type_label == "Numerical":
            return "n4_calculation_decomposition"
        elif state.calculation_type_label == "Logical Calculation":
            # Logical Calculation also goes through n4 for grouping logic decomposition
            return "n4_calculation_decomposition"
        elif state.calculation_type_label == "Algorithm":
            return "n5_algorithm_validation"
        else:
            # Fallback: treat as Numerical if key_parameters exist
            if state.key_parameters:
                print(f"  ⚠ calculation_type_label='{state.calculation_type_label}' is not recognized, treating as Numerical")
                return "n4_calculation_decomposition"
            else:
                # No key_parameters, proceed to n7
                return "n7_complete_inference"
    
    # Check if phenomenon_knowledge_match_table exists (for reasoning path)
    if not state.phenomenon_knowledge_match_table:
        # Try to construct fallback from domain_knowledge_map
        if state.domain_knowledge_map:
            print(f"  ⚠ phenomenon_knowledge_match_table not available, constructing fallback from domain_knowledge_map")
            state.phenomenon_knowledge_match_table = {}
            for domain, knowledge in state.domain_knowledge_map.items():
                state.phenomenon_knowledge_match_table[domain] = {
                    "matched_phenomena": [],
                    "knowledge_points": knowledge.get("specialized_knowledge", []) + knowledge.get("foundational_knowledge", [])
                }
            state.match_confidence_label = "Low"
            print(f"  ✓ Created fallback inference from domain_knowledge_map")
            return "n7_complete_inference"
        else:
            # Initialize node visit tracking
            if state.node_visit_count is None:
                state.node_visit_count = {}
            
            # Check for infinite loop: if n1 has been visited too many times, stop retrying
            n1_visits = state.node_visit_count.get("n1_question_decomposition", 0)
            if n1_visits >= 2:
                print(f"  ⚠ Infinite loop detected: n1_question_decomposition visited {n1_visits} times, stopping retry")
                state.exception_type_label = state.exception_type_label or "Inference Match Failed"
                return "n10_exception_handling"
            
            # OPTIMIZATION 5: Auto-retry mechanism - retry previous 2 core nodes (n1/n3) once before triggering n10/n11
            if not hasattr(state, 'auto_retry_count') or state.auto_retry_count is None:
                state.auto_retry_count = 0
            
            if state.auto_retry_count < 1:
                state.auto_retry_count = (state.auto_retry_count or 0) + 1
                state.node_visit_count["n1_question_decomposition"] = n1_visits + 1
                print(f"  🔄 Auto-retry mechanism triggered: retrying n1_question_decomposition and n3_knowledge_retrieval (attempt {state.auto_retry_count}/1, total visits: {state.node_visit_count['n1_question_decomposition']})")
                # Reset relevant state to allow retry
                state.phenomenon_knowledge_match_table = None
                state.match_confidence_label = None
                # Instead of routing to n1 (which would cause KeyError), route to exception handling
                # The retry mechanism should be handled differently - for now, route to exception handling
                print(f"  ⚠ Auto-retry would route to n1, but that's not in route_after_n6 edges. Routing to exception handling instead.")
                state.exception_type_label = state.exception_type_label or "Inference Match Failed - Retry Exhausted"
                return "n10_exception_handling"
            else:
                # No fallback available and retry exhausted, route to exception handling
                print(f"  ⚠ No phenomenon_knowledge_match_table and no domain_knowledge_map, auto-retry exhausted, routing to exception handling")
                state.exception_type_label = state.exception_type_label or "Inference Match Failed"
                return "n10_exception_handling"
    
    # Normal flow: proceed to N7
    return "n7_complete_inference"


def route_after_n7(state: GeneralQAState) -> str:
    """Route after N7: Ensure inference outputs are present"""
    if not state.closed_inference_path or not state.core_conclusion:
        # Initialize node visit tracking
        if state.node_visit_count is None:
            state.node_visit_count = {}
        
        # Check for infinite loop: if n6 has been visited more than 2 times, stop retrying
        n6_visits = state.node_visit_count.get("n6_initial_inference", 0)
        if n6_visits >= 2:
            print(f"  ⚠ Infinite loop detected: n6_initial_inference visited {n6_visits} times, stopping retry")
            state.exception_type_label = state.exception_type_label or "Inference Path Incomplete"
            return "n10_exception_handling"
        
        # OPTIMIZATION 5: Auto-retry mechanism - retry previous 2 core nodes (n1/n3 or n6) once before triggering n10/n11
        if not hasattr(state, 'auto_retry_count') or state.auto_retry_count is None:
            state.auto_retry_count = 0
        
        if state.auto_retry_count < 1:
            state.auto_retry_count = (state.auto_retry_count or 0) + 1
            state.node_visit_count["n6_initial_inference"] = n6_visits + 1
            print(f"  🔄 Auto-retry mechanism triggered: retrying n6_initial_inference (attempt {state.auto_retry_count}/1, total visits: {state.node_visit_count['n6_initial_inference']})")
            # Reset relevant state to allow retry
            state.closed_inference_path = None
            state.core_conclusion = None
            return "n6_initial_inference"
        else:
            # Check if we came from n10 retry - if so, mark that retry failed
            if state.retry_count and state.retry_count > 0:
                print(f"  ⚠ Retry from n10 failed: inference path still incomplete after retry")
            state.exception_type_label = state.exception_type_label or "Inference Path Incomplete"
            return "n10_exception_handling"
    # Success: reset auto_retry_count and retry_count for this path
    state.auto_retry_count = 0
    if state.retry_count and state.retry_count > 0:
        print(f"  ✓ Retry successful: inference path completed successfully")
        # Reset retry_count on success
        state.retry_count = 0
        state.retry_target_node = None
    return "n8_answer_generation"


def route_after_n8(state: GeneralQAState) -> str:
    """Route after N8: Go to N8.5 (Critic) if candidates generated, else to N10"""
    if state.exception_type_label:
        return "n10_exception_handling"
    
    # If we have candidate answers, proceed to critic review
    if state.candidate_answers and len(state.candidate_answers) > 0:
        return "n8_5_critic_review"
    
    # Otherwise, skip to validation (fallback)
    return "n10_exception_handling"


def route_after_n8_6(state: GeneralQAState) -> str:
    """Route after N8.6: Go to N9 if rewritten answers exist, else to N10"""
    if state.exception_type_label:
        return "n10_exception_handling"
    
    # If we have rewritten answers, proceed to validation
    if state.rewritten_answers and len(state.rewritten_answers) > 0:
        return "n9_result_validation"
    
    # Fallback: if no rewritten answers but we have critiqued answers, use best one
    if state.critiqued_answers and len(state.critiqued_answers) > 0:
        # Use best critiqued answer
        best = max(state.critiqued_answers, key=lambda x: x.get("success", False))
        if best.get("critiqued_answer"):
            state.final_answer = best.get("critiqued_answer")
            if best.get("original_structured"):
                state.structured_answer = best.get("original_structured")
            return "n9_result_validation"
    
    # If nothing works, go to exception handling
    return "n10_exception_handling"


def route_after_n8_6(state: GeneralQAState) -> str:
    """Route after N8.6: Go to N9 if rewritten answers exist, else to N10"""
    if state.exception_type_label:
        return "n10_exception_handling"
    
    # If we have rewritten answers, proceed to validation
    if state.rewritten_answers and len(state.rewritten_answers) > 0:
        return "n9_result_validation"
    
    # Fallback: if no rewritten answers but we have critiqued answers, use best one
    if state.critiqued_answers and len(state.critiqued_answers) > 0:
        # Use best critiqued answer
        best = max(state.critiqued_answers, key=lambda x: x.get("success", False))
        if best.get("critiqued_answer"):
            state.final_answer = best.get("critiqued_answer")
            if best.get("original_structured"):
                state.structured_answer = best.get("original_structured")
            return "n9_result_validation"
    
    # If nothing works, go to exception handling
    return "n10_exception_handling"


def route_after_n8(state: GeneralQAState) -> str:
    """Route after N8: Go to N8.5 (Critic) if candidates generated, else handle exceptions"""
    # Check for exceptions first
    if state.exception_type_label:
        return "n10_exception_handling"
    
    # If we have candidate answers, proceed to critic review (X-Masters enhancement)
    if state.candidate_answers and len(state.candidate_answers) > 0:
        return "n8_5_critic_review"
    
    # Original logic: check if answer generation failed
    if not state.structured_answer or not state.final_answer:
        # Initialize node visit tracking
        if state.node_visit_count is None:
            state.node_visit_count = {}
        
        # Check for infinite loop: if n7 or n3 has been visited too many times, stop retrying
        n7_visits = state.node_visit_count.get("n7_complete_inference", 0)
        n3_visits = state.node_visit_count.get("n3_knowledge_retrieval", 0)
        
        if n7_visits >= 2 or n3_visits >= 2:
            print(f"  ⚠ Infinite loop detected: n7 visited {n7_visits} times, n3 visited {n3_visits} times, stopping retry")
            state.exception_type_label = state.exception_type_label or "Answer Generation Failed"
            return "n10_exception_handling"
        
        # OPTIMIZATION 5: Auto-retry mechanism - retry n7 once before triggering n10/n11
        if not hasattr(state, 'auto_retry_count') or state.auto_retry_count is None:
            state.auto_retry_count = 0
        
        # Check if this is a factual question that needs knowledge retrieval retry
        if state.exception_type_label == "Answer Generation Failed - No Specific Information":
            if state.auto_retry_count < 1 and n3_visits < 2:
                state.auto_retry_count = (state.auto_retry_count or 0) + 1
                state.node_visit_count["n3_knowledge_retrieval"] = n3_visits + 1
                print(f"  🔄 Auto-retry mechanism triggered: retrying n3_knowledge_retrieval for factual question (attempt {state.auto_retry_count}/1, total visits: {state.node_visit_count['n3_knowledge_retrieval']})")
                # Reset relevant state to allow retry
                state.structured_answer = None
                state.final_answer = None
                state.core_conclusion = None
                state.closed_inference_path = None
                return "n3_knowledge_retrieval"
        
        if state.auto_retry_count < 1 and n7_visits < 2:
            state.auto_retry_count = (state.auto_retry_count or 0) + 1
            state.node_visit_count["n7_complete_inference"] = n7_visits + 1
            print(f"  🔄 Auto-retry mechanism triggered: retrying n7_complete_inference (attempt {state.auto_retry_count}/1, total visits: {state.node_visit_count['n7_complete_inference']})")
            # Reset relevant state to allow retry
            state.structured_answer = None
            state.final_answer = None
            return "n7_complete_inference"
        else:
            # Check if we came from n10 retry - if so, mark that retry failed
            if state.retry_count and state.retry_count > 0:
                print(f"  ⚠ Retry from n10 failed: answer generation still unsuccessful after retry")
                print(f"    - This indicates the issue cannot be resolved by retrying n8")
            state.exception_type_label = state.exception_type_label or "Answer Generation Failed"
            return "n10_exception_handling"
    
    # Success: reset auto_retry_count and retry_count for this path
    state.auto_retry_count = 0
    
    # If we have candidate answers, proceed to critic review (X-Masters enhancement)
    if state.candidate_answers and len(state.candidate_answers) > 0:
        return "n8_5_critic_review"
    
    # Otherwise, proceed to validation (fallback for single answer)
    return "n9_result_validation"
    if state.retry_count and state.retry_count > 0:
        print(f"  ✓ Retry successful: answer generated successfully")
        # Reset retry_count on success
        state.retry_count = 0
        state.retry_target_node = None
    return "n9_result_validation"


def route_after_n9(state: GeneralQAState) -> str:
    """Route after N9: Check consistency and reliability"""
    # Initialize node visit tracking
    if state.node_visit_count is None:
        state.node_visit_count = {}
    
    # Check for infinite loop: if N9 has been visited too many times, stop retrying
    n9_visits = state.node_visit_count.get("n9_result_validation", 0)
    n8_visits = state.node_visit_count.get("n8_answer_generation", 0)
    
    # CRITICAL: Prevent N8->N9->N10->N8 infinite loop
    # If N9 has been visited 3+ times or N8 has been visited 3+ times, stop retrying
    if n9_visits >= 3 or n8_visits >= 3:
        print(f"  ⚠ Infinite loop detected: N9 visited {n9_visits} times, N8 visited {n8_visits} times")
        print(f"    - Stopping retry to prevent infinite loop, routing to manual intervention")
        state.exception_type_label = state.exception_type_label or "Answer Validation Failed - Infinite Loop"
        return "n11_manual_intervention"
    
    if state.consistency_label == "Inconsistent" or (state.reliability_score and state.reliability_score <= 3):
        return "n10_exception_handling"
    return "end"


def route_after_n10(state: GeneralQAState) -> str:
    """Route after N10: Check solution suggestion and implement retry logic"""
    if state.solution_suggestion == "Manual Intervention":
        return "n11_manual_intervention"
    
    # Initialize node visit tracking
    if state.node_visit_count is None:
        state.node_visit_count = {}
    
    # Fix 2: Implement automatic retry mechanism
    if state.solution_suggestion == "Retry" or state.solution_suggestion == "Retry-N7" or state.solution_suggestion == "Retry-N8":
        # Check retry count (max 2 retries)
        if state.retry_count is None:
            state.retry_count = 0
        
        # Determine retry target based on exception type
        if not state.retry_target_node:
            if state.exception_type_label == "Inference Path Inconsistent" or state.exception_type_label == "Inference Path Incomplete":
                state.retry_target_node = "n7_complete_inference"
            elif state.exception_type_label == "Answer Format Invalid" or state.exception_type_label == "Answer Generation Failed" or state.exception_type_label == "Option Matching Error":
                state.retry_target_node = "n8_answer_generation"
            elif "Retry-N7" in str(state.solution_suggestion):
                state.retry_target_node = "n7_complete_inference"
            elif "Retry-N8" in str(state.solution_suggestion):
                state.retry_target_node = "n8_answer_generation"
            else:
                # Default: retry from N7 (inference)
                state.retry_target_node = "n7_complete_inference"
        
        # Check for infinite loop: if target node has been visited too many times, stop retrying
        target_visits = state.node_visit_count.get(state.retry_target_node, 0)
        
        # CRITICAL: If target node has been visited 2+ times (meaning we've already retried it), stop retrying
        # Also check N8 and N9 visits to prevent N8->N9->N10->N8 loops
        n8_visits = state.node_visit_count.get("n8_answer_generation", 0)
        n9_visits = state.node_visit_count.get("n9_result_validation", 0)
        
        if target_visits >= 2 or n8_visits >= 3 or n9_visits >= 3:
            print(f"  ⚠ Infinite loop detected: {state.retry_target_node} visited {target_visits} times, N8 visited {n8_visits} times, N9 visited {n9_visits} times")
            print(f"    - Stopping retry to prevent infinite loop, routing to manual intervention")
            return "n11_manual_intervention"
        
        # Check retry count (max 1 retry per exception type to prevent loops)
        if state.retry_count >= 1:
            print(f"  ⚠ Maximum retry count (1) reached for this exception type, routing to manual intervention")
            print(f"    - Exception type: {state.exception_type_label}")
            print(f"    - Retry target: {state.retry_target_node}")
            return "n11_manual_intervention"
        
        # Increment retry count and node visit count
        state.retry_count = state.retry_count + 1
        state.node_visit_count[state.retry_target_node] = target_visits + 1
        print(f"  🔄 Retry attempt {state.retry_count}/1, routing to {state.retry_target_node} (total visits: {state.node_visit_count[state.retry_target_node]})")
        print(f"    - If this retry fails, will route to manual intervention")
        return state.retry_target_node
    
    # If solution_suggestion is not "Retry", end the flow
    return "end"


# ===================== Graph Construction =====================

def build_general_qa_graph():
    """Build the General QA subgraph"""
    graph = StateGraph(GeneralQAState)
    
    # Add nodes
    graph.add_node("n0_input_preprocessing", n0_input_preprocessing_node)
    graph.add_node("n1_question_decomposition", n1_question_decomposition_node)
    graph.add_node("n2_calculation_algorithm_recognition", n2_calculation_algorithm_recognition_node)
    graph.add_node("n3_knowledge_retrieval", n3_knowledge_retrieval_node)
    graph.add_node("n4_calculation_decomposition", n4_calculation_decomposition_node)
    graph.add_node("n5_algorithm_validation", n5_algorithm_validation_node)
    graph.add_node("n6_initial_inference", n6_initial_inference_node)
    graph.add_node("n7_complete_inference", n7_complete_inference_node)
    graph.add_node("n8_answer_generation", n8_answer_generation_node)
    graph.add_node("n8_5_critic_review", n8_5_critic_review_node)
    graph.add_node("n8_6_rewriter_synthesis", n8_6_rewriter_synthesis_node)
    graph.add_node("n9_result_validation", n9_result_validation_node)
    graph.add_node("n10_exception_handling", n10_exception_handling_node)
    graph.add_node("n11_manual_intervention", n11_manual_intervention_node)
    
    # Set entry point
    graph.set_entry_point("n0_input_preprocessing")
    
    # Add edges
    graph.add_conditional_edges(
        "n0_input_preprocessing",
        route_after_n0,
        {
            "n1_question_decomposition": "n1_question_decomposition",
            "n2_calculation_algorithm_recognition": "n2_calculation_algorithm_recognition",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # Both N1 and N2 lead to N3
    graph.add_edge("n1_question_decomposition", "n3_knowledge_retrieval")
    graph.add_edge("n2_calculation_algorithm_recognition", "n3_knowledge_retrieval")
    
    # N3 routes to N4, N5, or N6
    # OPTIMIZATION: n3 always routes to n6 (n6 now executes for ALL question types)
    graph.add_conditional_edges(
        "n3_knowledge_retrieval",
        route_after_n3,
        {
            "n6_initial_inference": "n6_initial_inference",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # n6 routes to n4/n5/n7 based on calculation type
    graph.add_conditional_edges(
        "n6_initial_inference",
        route_after_n6,
        {
            "n4_calculation_decomposition": "n4_calculation_decomposition",
            "n5_algorithm_validation": "n5_algorithm_validation",
            "n7_complete_inference": "n7_complete_inference",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N4 and N5 route to N7 or N10
    graph.add_conditional_edges(
        "n4_calculation_decomposition",
        route_after_n4,
        {
            "n7_complete_inference": "n7_complete_inference",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    graph.add_conditional_edges(
        "n5_algorithm_validation",
        route_after_n5,
        {
            "n7_complete_inference": "n7_complete_inference",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N6 routing is already handled above (after n3) - removed duplicate
    
    # N7 routes to N8, N6 (retry), or N10
    graph.add_conditional_edges(
        "n7_complete_inference",
        route_after_n7,
        {
            "n8_answer_generation": "n8_answer_generation",
            "n6_initial_inference": "n6_initial_inference",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N8 routes to N8.5 (Critic) or N10
    graph.add_conditional_edges(
        "n8_answer_generation",
        route_after_n8,
        {
            "n8_5_critic_review": "n8_5_critic_review",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N8.5 (Critic) routes to N8.6 (Rewriter)
    graph.add_edge("n8_5_critic_review", "n8_6_rewriter_synthesis")
    
    # N8.6 (Rewriter) routes to N9 or N10
    graph.add_conditional_edges(
        "n8_6_rewriter_synthesis",
        route_after_n8_6,
        {
            "n9_result_validation": "n9_result_validation",
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N9 routes to END or N10
    graph.add_conditional_edges(
        "n9_result_validation",
        route_after_n9,
        {
            "end": END,
            "n10_exception_handling": "n10_exception_handling"
        }
    )
    
    # N10 routes to N11, N7, N8, or END
    graph.add_conditional_edges(
        "n10_exception_handling",
        route_after_n10,
        {
            "n11_manual_intervention": "n11_manual_intervention",
            "n7_complete_inference": "n7_complete_inference",
            "n8_answer_generation": "n8_answer_generation",
            "end": END
        }
    )
    
    # N11 leads to END
    graph.add_edge("n11_manual_intervention", END)
    
    return graph.compile()


# Export the compiled graph
general_qa_graph = build_general_qa_graph()


# ===================== Adapter Functions for Main Graph =====================

def build_general_qa_subgraph():
    """
    Build General QA subgraph (adapter for main graph compatibility)
    
    Returns:
        Compiled General QA subgraph
    """
    return general_qa_graph


def general_qa_input_mapper(global_state: Any) -> GeneralQAState:
    """
    Map main graph state to General QA subgraph state
    
    Args:
        global_state: Main graph global state
    
    Returns:
        General QA subgraph state
    """
    from state import GlobalState
    
    return GeneralQAState(
        user_input=global_state.user_input
    )


def general_qa_output_mapper(general_qa_state: GeneralQAState, global_state: Any) -> Any:
    """
    Map General QA subgraph state back to main graph state
    
    Args:
        general_qa_state: General QA subgraph state
        global_state: Main graph global state
    
    Returns:
        Updated main graph state
    """
    from state import GlobalState
    
    # Store answer to merged_result
    if not global_state.merged_result:
        global_state.merged_result = {}
    
    global_state.merged_result["general_qa_answer"] = general_qa_state.final_answer
    global_state.merged_result["general_qa_error"] = general_qa_state.error_message
    
    # Store detailed results if available
    if general_qa_state.structured_answer:
        global_state.merged_result["general_qa_structured_answer"] = general_qa_state.structured_answer
    
    if general_qa_state.core_conclusion:
        global_state.merged_result["general_qa_conclusion"] = general_qa_state.core_conclusion
    
    print(f"✅ General QA subgraph completed")
    if general_qa_state.final_answer:
        print(f"  - Final answer: {general_qa_state.final_answer[:200]}...")
    if general_qa_state.error_message:
        print(f"  - Error: {general_qa_state.error_message}")
    
    return global_state

