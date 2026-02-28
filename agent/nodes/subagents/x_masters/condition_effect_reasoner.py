"""Condition-Effect Reasoner for X-Masters Critic

This module provides formalized reasoning about how problem conditions affect 
statistical estimators and calculations. It maps conditions to their effects
on specific statistics, enabling the Critic to verify whether simulations
correctly implement the stated conditions.

Key Use Case:
    When a question mentions conditions like "random per-sample filtering" or
    "reference genome imputation", this module provides formal rules about
    how those conditions affect statistics like theta (Watterson's estimator) and pi.

Example:
    >>> from condition_effect_reasoner import reason_condition_effects
    >>> conditions = {"randomness": {"type": "independent_per_sample"}}
    >>> effects = reason_condition_effects(conditions, ["theta", "pi"])
    >>> print(effects["pi"]["effect"])  # "potential_bias"
"""

from typing import Dict, List, Any, Optional, Tuple


# ============================================================================
# CONDITION-EFFECT RULES: Formal mapping from conditions to statistical effects
# ============================================================================

CONDITION_EFFECT_RULES = {
    # Rule 1: Random per-sample filtering
    # Key insight: Each sample has INDEPENDENTLY random missing patterns
    # Effect on theta: No systematic bias (segregating sites still detectable)
    # Effect on pi: Potential bias (pairwise differences may be masked)
    "random_per_sample_filtering": {
        "theta": {
            "effect": "no_bias",
            "reasoning": (
                "Segregating sites S requires at least one sample to have a variant. "
                "Random per-sample filtering doesn't systematically eliminate all variants "
                "at any site, so S remains unchanged. theta = S/a1 is unbiased."
            ),
            "formula": "theta = S / a1, where a1 = sum(1/i) for i=1 to n-1",
            "key_condition": "Each sample has INDEPENDENTLY random missing patterns",
            "verification": "Check that simulation generates different random patterns for each sample"
        },
        "pi": {
            "effect": "potential_bias",
            "reasoning": (
                "Pairwise differences are calculated across all sites. "
                "Missing sites filled with reference may mask true differences "
                "if reference differs from actual genotypes. "
                "pi may be underestimated if reference = ancestral."
            ),
            "formula": "pi = sum(d_ij) / (n(n-1)/2), where d_ij = pairwise differences",
            "key_condition": "Missing sites are imputed, affecting observed pairwise differences",
            "verification": "Check that imputation affects pairwise comparison correctly"
        },
        "tajima_d": {
            "effect": "biased",
            "reasoning": (
                "Tajima's D = (pi - theta) / sqrt(V). "
                "If theta is unbiased but pi is biased, D will be biased."
            ),
            "formula": "D = (pi - theta) / sqrt(V(pi-theta))",
            "verification": "Check that both pi and theta are calculated correctly"
        }
    },

    # Rule 2: Reference genome imputation
    # Key insight: Missing sites filled with reference genotypes
    # Effect depends on what reference represents (ancestral vs derived)
    "reference_imputation": {
        "pi": {
            "effect": "downward_bias",
            "condition": "reference_is_ancestral",
            "reasoning": (
                "If reference allele is ancestral (most common), all imputed sites "
                "show same genotype, eliminating true pairwise differences at those sites. "
                "pi is underestimated because imputed sites contribute 0 differences."
            ),
            "formula": "pi_observed <= pi_true when reference = ancestral",
            "key_condition": "Reference genome represents ancestral state",
            "verification": "Check that imputed sites are filled with reference genotypes"
        },
        "theta": {
            "effect": "no_bias",
            "reasoning": (
                "theta depends on segregating sites S (sites where at least one sample has variant). "
                "Imputation doesn't change whether a site is segregating - if any sample "
                "has a variant, the site is still segregating regardless of imputation."
            ),
            "formula": "theta = S / a1, S = count of segregating sites (unchanged by imputation)",
            "verification": "S counts sites where ANY sample has variant (not imputed)"
        }
    },

    # Rule 3: Uniform filtering (same pattern for all samples)
    # Key insight: All samples have the SAME missing sites
    # This is DIFFERENT from independent per-sample randomness
    "uniform_filtering": {
        "theta": {
            "effect": "potential_bias",
            "reasoning": (
                "If all samples are missing the SAME variants, some segregating sites "
                "may be completely lost (no sample has the variant), reducing S. "
                "theta would be underestimated."
            ),
            "formula": "theta_observed <= theta_true when variants are lost uniformly",
            "verification": "Check that missing pattern is the same for all samples"
        },
        "pi": {
            "effect": "potential_bias",
            "reasoning": (
                "Same as per-sample filtering: imputed sites may mask differences. "
                "But the set of affected sites is the same across all samples."
            ),
            "verification": "Check that all samples have same missing sites"
        }
    },

    # Rule 4: Ascertainment bias (variants pre-selected)
    # Key insight: Not all variants are observed; selection criteria exist
    "ascertainment_bias": {
        "theta": {
            "effect": "biased",
            "reasoning": (
                "If variants are pre-selected based on certain criteria (e.g., MAF > 0.05), "
                "S no longer represents true segregating sites count. "
                "theta is biased depending on ascertainment scheme."
            ),
            "verification": "Check if variant selection criteria are applied"
        },
        "pi": {
            "effect": "biased",
            "reasoning": (
                "Pairwise differences are only calculated at ascertained variants, "
                "missing contribution from unobserved variants."
            ),
            "verification": "Check if calculation is restricted to ascertained variants"
        }
    }
}


# ============================================================================
# CONDITION DETECTION: Map semantic conditions to rule keys
# ============================================================================

def detect_condition_type(semantic_conditions: Dict[str, Any]) -> List[str]:
    """
    Detect which CONDITION_EFFECT_RULES apply based on semantic conditions.
    
    Args:
        semantic_conditions: Output from extract_structured_conditions()
        
    Returns:
        List of applicable rule keys
    """
    applicable_rules = []
    
    if not semantic_conditions:
        return applicable_rules
    
    # Check randomness condition
    randomness = semantic_conditions.get("randomness")
    if randomness:
        rand_type = randomness.get("type", "")
        if rand_type == "independent_per_sample":
            applicable_rules.append("random_per_sample_filtering")
        elif rand_type == "uniform_across_samples":
            applicable_rules.append("uniform_filtering")
    
    # Check imputation condition
    imputation = semantic_conditions.get("imputation")
    if imputation:
        method = imputation.get("method", "")
        if method == "reference_genome":
            applicable_rules.append("reference_imputation")
    
    # Check for ascertainment bias (detected from data constraints)
    data_constraints = semantic_conditions.get("data_constraints", [])
    for constraint in data_constraints:
        if "maf" in constraint.lower() or "frequency" in constraint.lower():
            applicable_rules.append("ascertainment_bias")
    
    return applicable_rules


# ============================================================================
# MAIN REASONING FUNCTION
# ============================================================================

def reason_condition_effects(
    semantic_conditions: Dict[str, Any],
    statistics: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Reason about how conditions affect specified statistics.
    
    This is the main entry point for condition-effect reasoning.
    
    Args:
        semantic_conditions: Output from extract_structured_conditions()
        statistics: List of statistics to analyze (e.g., ["theta", "pi"])
        
    Returns:
        Dict mapping each statistic to its effect analysis:
        {
            "theta": {
                "effect": "no_bias" | "biased" | "potential_bias" | "downward_bias",
                "reasoning": "...",
                "formula": "...",
                "verification": "..."
            },
            "pi": {...}
        }
    """
    effects = {}
    
    if not semantic_conditions or not statistics:
        return effects
    
    # Detect applicable rules
    applicable_rules = detect_condition_type(semantic_conditions)
    
    if not applicable_rules:
        return effects
    
    # Aggregate effects from all applicable rules
    for stat in statistics:
        stat_lower = stat.lower().replace("theta", "theta").replace("pi", "pi")
        
        # Map common names
        stat_normalized = stat_lower
        if "watterson" in stat_lower or "theta" in stat_lower:
            stat_normalized = "theta"
        elif "pi" in stat_lower or "nucleotide diversity" in stat_lower:
            stat_normalized = "pi"
        elif "tajima" in stat_lower:
            stat_normalized = "tajima_d"
        
        aggregated_effect = {
            "effect": "no_bias",  # Default
            "reasoning": [],
            "formulas": [],
            "verifications": [],
            "applicable_rules": []
        }
        
        for rule_key in applicable_rules:
            rule = CONDITION_EFFECT_RULES.get(rule_key, {})
            stat_effect = rule.get(stat_normalized)
            
            if stat_effect:
                aggregated_effect["applicable_rules"].append(rule_key)
                aggregated_effect["reasoning"].append(stat_effect.get("reasoning", ""))
                if stat_effect.get("formula"):
                    aggregated_effect["formulas"].append(stat_effect["formula"])
                if stat_effect.get("verification"):
                    aggregated_effect["verifications"].append(stat_effect["verification"])
                
                # Aggregate effect (most severe wins)
                current_effect = aggregated_effect["effect"]
                new_effect = stat_effect.get("effect", "no_bias")
                
                # Priority: biased > downward_bias > potential_bias > no_bias
                effect_priority = {
                    "biased": 4,
                    "downward_bias": 3,
                    "potential_bias": 2,
                    "no_bias": 1
                }
                if effect_priority.get(new_effect, 0) > effect_priority.get(current_effect, 0):
                    aggregated_effect["effect"] = new_effect
        
        # Only include if we found relevant rules
        if aggregated_effect["applicable_rules"]:
            effects[stat] = {
                "effect": aggregated_effect["effect"],
                "reasoning": " ".join(aggregated_effect["reasoning"]),
                "formulas": aggregated_effect["formulas"],
                "verifications": aggregated_effect["verifications"],
                "applicable_rules": aggregated_effect["applicable_rules"]
            }
    
    return effects


# ============================================================================
# BIAS COMPARISON HELPER
# ============================================================================

def compare_bias_effects(effects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare bias effects across statistics to answer questions like
    "Which estimator is more biased?"
    
    Args:
        effects: Output from reason_condition_effects()
        
    Returns:
        Dict with comparison results:
        {
            "most_biased": "pi" | "theta" | "both" | "neither",
            "comparison": "pi is biased, theta is not" | ...,
            "explanation": "..."
        }
    """
    if not effects:
        return {
            "most_biased": "unknown",
            "comparison": "No effects analyzed",
            "explanation": "Insufficient condition information"
        }
    
    bias_levels = {
        "no_bias": 0,
        "potential_bias": 1,
        "downward_bias": 2,
        "biased": 3
    }
    
    stat_bias_scores = {}
    for stat, effect_info in effects.items():
        effect = effect_info.get("effect", "no_bias")
        stat_bias_scores[stat] = bias_levels.get(effect, 0)
    
    if not stat_bias_scores:
        return {
            "most_biased": "unknown",
            "comparison": "No statistics analyzed",
            "explanation": ""
        }
    
    # Find most biased
    max_bias = max(stat_bias_scores.values())
    most_biased_stats = [s for s, b in stat_bias_scores.items() if b == max_bias]
    
    if max_bias == 0:
        most_biased = "neither"
        comparison = "Neither estimator is biased"
    elif len(most_biased_stats) == len(stat_bias_scores):
        most_biased = "both"
        comparison = "Both estimators are biased"
    elif len(most_biased_stats) == 1:
        most_biased = most_biased_stats[0]
        other_stats = [s for s in stat_bias_scores if s != most_biased]
        if other_stats:
            other_effect = effects.get(other_stats[0], {}).get("effect", "no_bias")
            comparison = f"{most_biased} is biased ({effects[most_biased]['effect']}), {other_stats[0]} is {other_effect}"
        else:
            comparison = f"{most_biased} is biased"
    else:
        most_biased = "multiple"
        comparison = f"Multiple estimators are biased: {most_biased_stats}"
    
    # Generate explanation
    explanations = []
    for stat, effect_info in effects.items():
        if effect_info.get("reasoning"):
            explanations.append(f"{stat}: {effect_info['reasoning'][:200]}")
    
    return {
        "most_biased": most_biased,
        "comparison": comparison,
        "explanation": " | ".join(explanations),
        "bias_scores": stat_bias_scores
    }


# ============================================================================
# VERIFICATION CHECKLIST GENERATOR
# ============================================================================

def generate_verification_checklist(
    semantic_conditions: Dict[str, Any],
    statistics: List[str]
) -> List[Dict[str, str]]:
    """
    Generate a verification checklist for Critic to use before concluding.
    
    Args:
        semantic_conditions: Output from extract_structured_conditions()
        statistics: Statistics mentioned in the problem
        
    Returns:
        List of verification items with id, description, and check
    """
    checklist = []
    
    if not semantic_conditions:
        return checklist
    
    # Randomness verification
    randomness = semantic_conditions.get("randomness")
    if randomness:
        rand_type = randomness.get("type", "")
        if rand_type == "independent_per_sample":
            checklist.append({
                "id": "random_independent",
                "description": "Verify INDEPENDENT per-sample randomness",
                "check": "Does the simulation generate DIFFERENT random missing patterns for each sample?",
                "common_mistake": "Using same random seed for all samples, causing all samples to miss the same variants",
                "critical": True
            })
        elif rand_type == "uniform_across_samples":
            checklist.append({
                "id": "random_uniform",
                "description": "Verify UNIFORM randomness across samples",
                "check": "Does the simulation apply the SAME random pattern to all samples?",
                "common_mistake": "Generating different patterns when the same pattern should be used",
                "critical": True
            })
    
    # Imputation verification
    imputation = semantic_conditions.get("imputation")
    if imputation:
        checklist.append({
            "id": "imputation_method",
            "description": "Verify imputation method",
            "check": f"Are missing sites filled with {imputation.get('method', 'reference genotypes')}?",
            "common_mistake": "Dropping missing sites or using incorrect imputation method",
            "critical": True
        })
    
    # Statistics-specific verification
    effects = reason_condition_effects(semantic_conditions, statistics)
    for stat, effect_info in effects.items():
        for verification in effect_info.get("verifications", []):
            checklist.append({
                "id": f"verify_{stat}",
                "description": f"Verify {stat} calculation",
                "check": verification,
                "critical": effect_info.get("effect") != "no_bias"
            })
    
    return checklist


# ============================================================================
# CONVENIENCE FUNCTION FOR CRITIC
# ============================================================================

def get_condition_analysis_for_critic(
    semantic_conditions: Dict[str, Any],
    problem_text: str = ""
) -> str:
    """
    Generate a formatted analysis string for injection into Critic prompt.
    
    This provides a human-readable summary of condition effects.
    
    Args:
        semantic_conditions: Output from extract_structured_conditions()
        problem_text: Original problem text (for context)
        
    Returns:
        Formatted string for Critic prompt
    """
    if not semantic_conditions:
        return ""
    
    # Extract statistics from problem text
    stats = semantic_conditions.get("statistics_affected", [])
    if not stats and problem_text:
        # Try to detect from text
        text_lower = problem_text.lower()
        if "theta" in text_lower or "theta" in problem_text or "watterson" in text_lower:
            stats.append("theta")
        if "pi" in text_lower or "pi" in problem_text or "nucleotide diversity" in text_lower:
            stats.append("pi")
    
    effects = reason_condition_effects(semantic_conditions, stats)
    comparison = compare_bias_effects(effects)
    
    output_lines = ["## Condition-Effect Analysis\n"]
    
    if effects:
        output_lines.append("Based on the problem conditions, here is the expected effect on each statistic:\n")
        
        for stat, effect_info in effects.items():
            effect = effect_info.get("effect", "unknown")
            effect_emoji = {
                "no_bias": "[OK]",
                "potential_bias": "[WARN]",
                "downward_bias": "[DOWN]",
                "biased": "[BIASED]"
            }.get(effect, "[?]")
            
            output_lines.append(f"### {stat.upper()}: {effect_emoji} {effect.replace('_', ' ').title()}")
            output_lines.append(f"- **Reasoning**: {effect_info.get('reasoning', 'N/A')[:300]}")
            if effect_info.get('formulas'):
                output_lines.append(f"- **Formula**: {effect_info['formulas'][0]}")
            output_lines.append("")
        
        if comparison.get("comparison"):
            output_lines.append(f"### Comparison: {comparison['comparison']}")
    
    return "\n".join(output_lines)


# ============================================================================
# TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    # Demo: Test with the Watterson vs pi bias question
    demo_conditions = {
        "randomness": {
            "type": "independent_per_sample",
            "description": "Each sample has independently random missing variants"
        },
        "imputation": {
            "method": "reference_genome",
            "assumption": "ancestral_allele"
        },
        "statistics_affected": ["theta", "pi"]
    }
    
    print("=" * 60)
    print("Condition-Effect Reasoner Demo")
    print("=" * 60)
    
    print("\nInput conditions:")
    print(f"  - Randomness: {demo_conditions['randomness']['type']}")
    print(f"  - Imputation: {demo_conditions['imputation']['method']}")
    print(f"  - Statistics: {demo_conditions['statistics_affected']}")
    
    print("\nReasoning about effects:")
    effects = reason_condition_effects(demo_conditions, ["theta", "pi"])
    
    for stat, effect_info in effects.items():
        print(f"\n{stat.upper()}:")
        print(f"  Effect: {effect_info['effect']}")
        print(f"  Reasoning: {effect_info['reasoning'][:150]}...")
    
    print("\nComparison:")
    comparison = compare_bias_effects(effects)
    print(f"  Most biased: {comparison['most_biased']}")
    print(f"  Comparison: {comparison['comparison']}")
    
    print("\n" + "=" * 60)
    print("Generated Analysis for Critic:")
    print("=" * 60)
    print(get_condition_analysis_for_critic(demo_conditions))

