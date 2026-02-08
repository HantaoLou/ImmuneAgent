# General QA Subgraph Optimization Plan

Based on error analysis of the Watterson's θ and π bias judgment question, this document outlines optimization directions to fix logical contradictions and improve answer accuracy.

## Problem Summary

**Core Issue**: Node 4 (Reasoning Engine) correctly derived "θ remains unbiased, while π is likely underestimated", but Node 5 (Conclusion Validation) selected option A ("Only θ is biased"), creating a complete logical contradiction.

**Root Causes**:
1. **Lack of consistency enforcement**: No mandatory check ensuring Final Answer aligns with Preliminary Conclusion
2. **Weak option matching logic**: Option Matching Priority in Node 4 may be incorrect, and Node 5 blindly follows it
3. **Insufficient validation**: Validation checks pass even when conclusion contradicts preliminary analysis
4. **Missing contradiction detection**: Current contradiction detection doesn't catch conclusion-reversal errors

## Optimization Directions

### 1. **Enforce Preliminary Conclusion - Final Answer Consistency** (High Priority)

**Problem**: Node 5 can select an option that contradicts Node 4's Preliminary Conclusion.

**Solution**:
- **Add explicit consistency check in Node 5 prompt**: Require LLM to explicitly verify that Final Answer matches Preliminary Conclusion
- **Add rule-based validation**: Before accepting Final Answer, programmatically check if it contradicts Preliminary Conclusion
- **Add consistency field**: Add "Consistency Check" field in final validation result, explicitly stating whether Final Answer aligns with Preliminary Conclusion

**Implementation**:
```python
# In conclusion_validation_node, after getting final_data:
preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "")
final_answer = final_data.get("Final Answer", "")

# Extract key claims from preliminary conclusion
# e.g., "θ unbiased, π biased" → {"theta": "unbiased", "pi": "biased"}
# Extract key claims from final answer
# e.g., "Only θ is biased" → {"theta": "biased", "pi": "unbiased"}
# Compare and raise error if contradictory
```

**Prompt Enhancement**:
Add to `FINAL_VALIDATION_PROMPT_TEMPLATE`:
```
CRITICAL CONSISTENCY REQUIREMENT:
- The Final Answer MUST be logically consistent with Node 4's "Preliminary Conclusion"
- If Preliminary Conclusion states "θ unbiased, π biased", then Final Answer MUST select an option that matches this (e.g., "Only π is biased")
- If you find Preliminary Conclusion contradicts all options, you MUST explicitly state this in Validation Results and reconsider the Preliminary Conclusion
- DO NOT select an option that contradicts the Preliminary Conclusion, even if Option Matching Priority suggests otherwise
```

### 2. **Improve Option Matching Priority Generation in Node 4** (High Priority)

**Problem**: Node 4's "Option Matching Priority" may be incorrect, leading Node 5 to select wrong option.

**Solution**:
- **Strengthen prompt**: Require Node 4 to explicitly map Preliminary Conclusion to each option
- **Add validation**: Node 4 must verify that its Option Matching Priority aligns with Preliminary Conclusion
- **Add detailed mapping**: Instead of just "A>B>C", require "A matches because X, B contradicts because Y"

**Implementation**:
Enhance `get_logical_derivation_prompt` for Judgment type:
```python
# Add requirement:
"Option Matching Priority": For each option, explicitly state:
- Whether it matches or contradicts Preliminary Conclusion
- Specific reason (e.g., "Option A contradicts because it claims θ is biased, but Preliminary Conclusion states θ is unbiased")
- Final priority ranking with justification
```

### 3. **Enhance Contradiction Detection** (Medium Priority)

**Problem**: Current `detect_logical_contradictions` doesn't catch conclusion-reversal errors.

**Solution**:
- **Add Preliminary Conclusion - Final Answer contradiction check**: Specifically check if Final Answer contradicts Preliminary Conclusion
- **Add semantic comparison**: Use keyword extraction and semantic matching to detect contradictions
- **Add bias direction check**: For bias-related questions, explicitly check if bias direction matches

**Implementation**:
```python
def _check_preliminary_final_consistency(
    preliminary_conclusion: str,
    final_answer: str,
    question_type: str
) -> Tuple[bool, List[str]]:
    """
    Check if Final Answer is consistent with Preliminary Conclusion
    
    Returns:
        (has_contradiction, contradiction_reports)
    """
    contradictions = []
    
    # Extract key claims from preliminary conclusion
    # For bias questions: extract which items are biased/unbiased
    # For judgment questions: extract yes/no, higher/lower, etc.
    
    # Extract key claims from final answer
    
    # Compare and detect contradictions
    
    return len(contradictions) > 0, contradictions
```

### 4. **Add Explicit Bias Direction Validation** (Medium Priority)

**Problem**: For bias-related questions, system may confuse "biased" vs "unbiased" or misidentify which item is biased.

**Solution**:
- **Add bias direction extraction**: Extract from Preliminary Conclusion which items are biased/unbiased
- **Add option bias mapping**: Map each option to its bias claims
- **Add consistency check**: Verify Final Answer's bias claims match Preliminary Conclusion

**Implementation**:
```python
def _extract_bias_claims(text: str) -> Dict[str, str]:
    """
    Extract bias claims from text
    e.g., "θ unbiased, π biased" → {"theta": "unbiased", "pi": "biased"}
    """
    # Use regex or LLM to extract bias claims
    pass

def _map_option_bias_claims(option: str) -> Dict[str, str]:
    """
    Map option to its bias claims
    e.g., "Only θ is biased" → {"theta": "biased", "pi": "unbiased"}
    """
    pass
```

### 5. **Improve Validation Results Reporting** (Low Priority)

**Problem**: Validation Results claim "Previous logic consistent" even when there's a contradiction.

**Solution**:
- **Add explicit consistency statement**: Require LLM to explicitly state whether Final Answer matches Preliminary Conclusion
- **Add detailed comparison**: Show side-by-side comparison of Preliminary Conclusion vs Final Answer
- **Add contradiction flag**: If contradiction detected, mark it clearly in Validation Results

### 6. **Add Post-Validation Consistency Check** (High Priority)

**Problem**: Even with improved prompts, LLM may still make mistakes.

**Solution**:
- **Add programmatic check after LLM response**: After Node 5 gets final_data, programmatically verify consistency
- **Add retry mechanism**: If contradiction detected, regenerate with explicit error message
- **Add fallback**: If retry fails, use Preliminary Conclusion directly to select option

**Implementation**:
```python
# In conclusion_validation_node, after parsing final_data:
preliminary_conclusion = logical_derivation.get("Preliminary Conclusion", "")
final_answer = final_data.get("Final Answer", "")

has_contradiction, reports = _check_preliminary_final_consistency(
    preliminary_conclusion, final_answer, str(question_type)
)

if has_contradiction:
    print(f"⚠ CRITICAL: Final Answer contradicts Preliminary Conclusion!")
    print(f"  Preliminary: {preliminary_conclusion}")
    print(f"  Final Answer: {final_answer}")
    print(f"  Contradictions: {reports}")
    
    # Option 1: Retry with explicit error
    retry_prompt = generate_consistency_fix_prompt(
        original_prompt, preliminary_conclusion, final_answer, reports
    )
    # Retry LLM call...
    
    # Option 2: Use Preliminary Conclusion to directly select option
    # Extract correct option from Preliminary Conclusion
    corrected_answer = _select_option_from_preliminary_conclusion(
        preliminary_conclusion, question_options
    )
    final_data["Final Answer"] = corrected_answer
    final_data["Validation Results"] += " | WARNING: Original answer contradicted Preliminary Conclusion, corrected based on Preliminary Conclusion"
```

## Implementation Priority

1. **Priority 1 (Critical)**: 
   - Enforce Preliminary Conclusion - Final Answer Consistency (#1)
   - Add Post-Validation Consistency Check (#6)

2. **Priority 2 (High)**:
   - Improve Option Matching Priority Generation (#2)
   - Enhance Contradiction Detection (#3)

3. **Priority 3 (Medium)**:
   - Add Explicit Bias Direction Validation (#4)
   - Improve Validation Results Reporting (#5)

## Testing Strategy

After implementing optimizations:
1. Test with the Watterson's θ/π question to verify fix
2. Test with other bias-related questions
3. Test with judgment-type questions to ensure no regression
4. Add unit tests for consistency checking functions

## Expected Outcomes

- **Eliminate logical contradictions**: Final Answer will always align with Preliminary Conclusion
- **Improve accuracy**: Correct option selection rate should increase significantly
- **Better debugging**: Clear error messages when contradictions are detected
- **More reliable**: System will catch and correct its own mistakes

