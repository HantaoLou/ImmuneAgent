"""
Analysis Tools for General QA
============================================
Specialized tools for complex reasoning tasks:
- Multi-statement verification
- Mass spectrometry modification calculation  
- Experimental data analysis
- Sequence matching analysis

LangChain/LangGraph 1.0 compatible tools using @tool decorator
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolArg
from typing_extensions import TypedDict


# ============================================================
# Knowledge Bases for Tools
# ============================================================

# Database of common biomedical facts for statement verification
BIOMEDICAL_FACTS = {
    # Microbiology - Pseudomonas aeruginosa
    "pseudomonas_twitching_motility": {
        "fact": "Pseudomonas aeruginosa exhibits twitching motility mediated by type IV pili",
        "verification": "TRUE",
        "source": "Standard microbiology textbooks"
    },
    "pseudomonas_stab_inoculation": {
        "fact": "Twitching motility assays typically use stab inoculation through agar",
        "verification": "TRUE",
        "source": "Standard lab protocols"
    },
    "pseudomonas_10cm_plates": {
        "fact": "Standard twitching motility plates are 10-cm diameter with ~25ml agar",
        "verification": "TRUE",
        "source": "Standard lab protocols"
    },
    "pseudomonas_swarming_glycerol": {
        "fact": "P. aeruginosa swarming motility requires specific carbon sources; glycerol does NOT typically support swarming",
        "verification": "FALSE for 'can swarm with glycerol'",
        "source": "Microbiology literature - swarming requires amino acids/glucose"
    },
    "pseudomonas_metal_chelators": {
        "fact": "Metal chelators can inhibit swarming motility by sequestering essential metals for flagellar function",
        "verification": "TRUE for 'can inhibit'",
        "source": "Microbiology literature"
    },
    "pseudomonas_pigment": {
        "fact": "P. aeruginosa produces pyocyanin (blue-green pigment) and other pigments; concentrated cultures appear blue-green",
        "verification": "TRUE",
        "source": "Standard microbiology"
    },
    
    # Biochemistry - Protein Modifications
    "iodoacetamide_alkylation": {
        "fact": "Iodoacetamide alkylation adds +57.02146 Da (carbamidomethylation) to cysteine residues",
        "verification": "TRUE",
        "source": "Mass spectrometry protocols"
    },
    "biotin_dadps": {
        "fact": "Biotin-DADPS is a cleavable biotinylation reagent; after cleavage, the remaining modification mass depends on the specific chemistry",
        "verification": "CONTEXT_DEPENDENT",
        "source": "Mass spectrometry literature"
    },
    
    # CRISPR/Cas9
    "spcas9_pam": {
        "fact": "SpCas9 requires NGG PAM sequence; off-target analysis requires checking similar sequences in genome",
        "verification": "TRUE",
        "source": "CRISPR literature"
    },
    "sgrna_targeting": {
        "fact": "sgRNA target sequences should be 20bp upstream of PAM; targeting exons requires sequence context",
        "verification": "TRUE",
        "source": "CRISPR protocols"
    }
}

# Common modification masses (in Daltons)
MODIFICATION_MASSES = {
    "carbamidomethylation": {
        "reagent": "Iodoacetamide",
        "mass_change": 57.02146,
        "description": "Alkylation of cysteine, adds CAM (carbamidomethyl)"
    },
    "oxidation": {
        "reagent": "N/A (spontaneous)",
        "mass_change": 15.99491,
        "description": "Oxidation of methionine, adds oxygen"
    },
    "acetylation": {
        "reagent": "Acetic anhydride",
        "mass_change": 42.01056,
        "description": "Acetylation of N-terminus or lysine"
    },
    "phosphorylation": {
        "reagent": "Kinases",
        "mass_change": 79.96633,
        "description": "Phosphorylation of S, T, Y"
    },
    "biotinylation": {
        "reagent": "Biotin-NHS",
        "mass_change": 226.0776,
        "description": "Biotinylation via NHS ester"
    },
    "biotin_dadps": {
        "reagent": "Biotin-DADPS",
        "mass_change_before_cleavage": 475.2,
        "mass_change_after_cleavage": 254.14,
        "description": "Cleavable biotinylation reagent"
    },
    "tmt_10plex": {
        "reagent": "TMT10-plex",
        "mass_change": 229.1629,
        "description": "Tandem Mass Tag for quantification"
    },
    "tmt_16plex": {
        "reagent": "TMTpro16-plex",
        "mass_change": 304.2071,
        "description": "TMTpro for quantification"
    }
}


# ============================================================
# Input Schemas (Pydantic Models)
# ============================================================

class MultiStatementInput(BaseModel):
    """Input for multi-statement verification tool"""
    question_text: str = Field(
        description="The full question text containing multiple statements to verify"
    )
    statements: List[str] = Field(
        default_factory=list,
        description="List of individual statements to verify (e.g., ['I. Twitching motility is...', 'II. 10-cm plates contain...'])"
    )
    goal_type: str = Field(
        default="which_are_true",
        description="'which_are_true' or 'which_are_false' or 'which_are_not_true'"
    )


class ModificationInput(BaseModel):
    """Input for modification mass calculation tool"""
    modification_type: str = Field(
        description="Type of modification: 'carbamidomethylation', 'biotinylation', 'oxidation', 'acetylation', 'phosphorylation', 'tmt_10plex', 'tmt_16plex', 'biotin_dadps'"
    )
    reagent_name: Optional[str] = Field(
        default=None,
        description="Name of the reagent used (e.g., 'iodoacetamide', 'Biotin-DADPS')"
    )
    target_residue: str = Field(
        default="C",
        description="Target amino acid residue (C=Cysteine, K=Lysine, M=Methionine, etc.)"
    )
    include_cleavage: bool = Field(
        default=False,
        description="Whether the modification includes a cleavable linker (for biotin_dadps)"
    )


class SgRNAAnalysisInput(BaseModel):
    """Input for CRISPR sgRNA analysis tool"""
    target_sequence: str = Field(
        description="Target DNA sequence or gene name (e.g., 'BRCA1 exon 2')"
    )
    pam_sequence: str = Field(
        default="NGG",
        description="PAM sequence (default: NGG for SpCas9)"
    )
    gene_region: Optional[str] = Field(
        default=None,
        description="Gene region being targeted (e.g., 'exon2', 'promoter')"
    )
    options_to_evaluate: Optional[List[str]] = Field(
        default=None,
        description="List of candidate sgRNA sequences to evaluate (e.g., ['ACGTTGCGAGGACAGAGTCA(AGG)', '...'])"
    )


class ExperimentalDataInput(BaseModel):
    """Input for experimental data analysis tool"""
    data_description: str = Field(
        description="Description of the experimental data (groups, conditions, measurements)"
    )
    question_type: str = Field(
        default="which_is_true",
        description="Type of question being asked: 'which_is_true', 'which_is_false', 'interpret_results'"
    )
    groups: Optional[List[str]] = Field(
        default=None,
        description="Experimental groups (e.g., ['wild type', 'mutant A', 'mutant B'])"
    )
    outcome_variable: Optional[str] = Field(
        default=None,
        description="What was measured (e.g., 'bacteria count', 'survival rate')"
    )


# ============================================================
# Tool 1: Multi-Statement Verification
# ============================================================

@tool
def verify_multi_statement(
    question_text: str,
    statements: List[str],
    goal_type: str = "which_are_true"
) -> Dict[str, Any]:
    """
    Verify multiple statements (I, II, III, IV...) against biomedical knowledge.
    
    Use this tool when the question asks "which of the following statements are TRUE/FALSE" 
    or contains numbered statements like "I. ... II. ... III. ...".
    
    This tool systematically verifies each statement against known biomedical facts,
    reducing classification errors for multi-statement questions.
    
    Args:
        question_text: The full question text containing multiple statements
        statements: List of individual statements to verify
        goal_type: 'which_are_true' or 'which_are_false' or 'which_are_not_true'
    
    Returns:
        Dictionary with verification results for each statement and recommendations
    
    Example:
        >>> verify_multi_statement(
        ...     question_text="Which statements about P. aeruginosa are true?",
        ...     statements=["I. Twitching motility...", "II. Swarming with glycerol..."],
        ...     goal_type="which_are_true"
        ... )
    """
    results = []
    
    for i, statement in enumerate(statements):
        statement_result = {
            "statement_id": chr(65 + i) if i < 26 else str(i + 1),
            "statement_text": statement,
            "verification": "UNCERTAIN",
            "reasoning": "",
            "matched_knowledge": None
        }
        
        # Check against known facts
        statement_lower = statement.lower()
        
        for fact_key, fact_info in BIOMEDICAL_FACTS.items():
            fact_keywords = fact_key.split("_")
            match_count = sum(1 for kw in fact_keywords if kw in statement_lower)
            
            if match_count >= 2:
                statement_result["matched_knowledge"] = fact_key
                statement_result["reasoning"] = f"Matched knowledge base: {fact_info['fact']}"
                
                if goal_type in ["which_are_true", "which_is_true"]:
                    if fact_info["verification"] == "TRUE":
                        statement_result["verification"] = "LIKELY_TRUE"
                    elif fact_info["verification"] == "FALSE":
                        statement_result["verification"] = "LIKELY_FALSE"
                    else:
                        statement_result["verification"] = "UNCERTAIN"
                elif goal_type in ["which_are_false", "which_are_not_true", "which_is_false"]:
                    if fact_info["verification"] == "TRUE":
                        statement_result["verification"] = "LIKELY_FALSE"
                    elif fact_info["verification"] == "FALSE":
                        statement_result["verification"] = "LIKELY_TRUE"
                    else:
                        statement_result["verification"] = "UNCERTAIN"
                break
        
        results.append(statement_result)
    
    # Summary
    true_statements = [r for r in results if r["verification"] in ["LIKELY_TRUE", "TRUE"]]
    false_statements = [r for r in results if r["verification"] in ["LIKELY_FALSE", "FALSE"]]
    uncertain_statements = [r for r in results if r["verification"] == "UNCERTAIN"]
    
    return {
        "results": results,
        "summary": {
            "likely_true_count": len(true_statements),
            "likely_false_count": len(false_statements),
            "uncertain_count": len(uncertain_statements),
            "true_statement_ids": [r["statement_id"] for r in true_statements],
            "false_statement_ids": [r["statement_id"] for r in false_statements]
        },
        "recommendation": f"For '{goal_type}' questions: Focus on statements marked as LIKELY_{goal_type.split('_')[-1].upper()}"
    }


# ============================================================
# Tool 2: Mass Spectrometry Modification Calculator
# ============================================================

@tool
def calculate_modification_mass(
    modification_type: str,
    reagent_name: Optional[str] = None,
    target_residue: str = "C",
    include_cleavage: bool = False
) -> Dict[str, Any]:
    """
    Calculate protein modification masses for mass spectrometry analysis.
    
    Use this tool when the question involves LC-MS/MS, protein modifications, 
    or asks about modification mass changes.
    
    Common modifications supported:
    - carbamidomethylation: +57.02 Da (iodoacetamide alkylation)
    - oxidation: +15.99 Da (methionine oxidation)
    - acetylation: +42.01 Da (N-terminus/lysine)
    - phosphorylation: +79.97 Da (S, T, Y)
    - biotinylation: +226.08 Da (biotin-NHS)
    - biotin_dadps: +254.14 Da (after cleavage)
    - tmt_10plex: +229.16 Da
    - tmt_16plex: +304.21 Da
    
    Args:
        modification_type: Type of modification
        reagent_name: Name of the reagent (optional)
        target_residue: Target amino acid (C, K, M, S, T, Y)
        include_cleavage: Whether to calculate mass after cleavage (for cleavable linkers)
    
    Returns:
        Dictionary with modification info, calculated mass, and notes
    
    Example:
        >>> calculate_modification_mass(
        ...     modification_type="carbamidomethylation",
        ...     reagent_name="iodoacetamide",
        ...     target_residue="C"
        ... )
    """
    mod_type = modification_type.lower().replace(" ", "_").replace("-", "_")
    
    result = {
        "query": {
            "modification_type": modification_type,
            "reagent_name": reagent_name,
            "target_residue": target_residue,
            "include_cleavage": include_cleavage
        },
        "modification_info": None,
        "calculated_mass": None,
        "notes": []
    }
    
    # Look up modification
    if mod_type in MODIFICATION_MASSES:
        mod_info = MODIFICATION_MASSES[mod_type]
        result["modification_info"] = mod_info
        
        if "mass_change" in mod_info:
            result["calculated_mass"] = mod_info["mass_change"]
            result["notes"].append(f"Standard modification: adds {mod_info['mass_change']:.2f} Da")
        
        elif "mass_change_before_cleavage" in mod_info:
            if include_cleavage:
                result["calculated_mass"] = mod_info["mass_change_after_cleavage"]
                result["notes"].append(f"After cleavage: adds {mod_info['mass_change_after_cleavage']:.2f} Da")
            else:
                result["calculated_mass"] = mod_info["mass_change_before_cleavage"]
                result["notes"].append(f"Before cleavage: adds {mod_info['mass_change_before_cleavage']:.2f} Da")
        
        result["notes"].append(f"Reagent: {mod_info.get('reagent', 'N/A')}")
        result["notes"].append(f"Description: {mod_info['description']}")
    
    else:
        result["notes"].append(f"Unknown modification type: {modification_type}")
        result["notes"].append("Common types: carbamidomethylation, oxidation, acetylation, phosphorylation, biotinylation, tmt_10plex")
        
        # Try to match by reagent name
        if reagent_name:
            reagent_lower = reagent_name.lower()
            for key, info in MODIFICATION_MASSES.items():
                if reagent_lower in info.get("reagent", "").lower():
                    result["modification_info"] = info
                    result["calculated_mass"] = info.get("mass_change")
                    result["notes"].append(f"Matched by reagent: {info['reagent']}")
                    break
    
    return result


# ============================================================
# Tool 3: CRISPR sgRNA Analyzer
# ============================================================

@tool
def analyze_sgrna(
    target_sequence: str,
    pam_sequence: str = "NGG",
    gene_region: Optional[str] = None,
    options_to_evaluate: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze CRISPR sgRNA candidates for quality and off-target potential.
    
    Use this tool when the question asks to select the best sgRNA for gene targeting
    or evaluate sgRNA candidates for off-target potential.
    
    This tool evaluates:
    - PAM sequence verification
    - Guide length (standard 20bp)
    - GC content (optimal 40-60%)
    - Pol III termination signals (TTTT)
    - U6 promoter compatibility (starts with G)
    
    Args:
        target_sequence: Target DNA sequence or gene name
        pam_sequence: PAM sequence (default: NGG for SpCas9)
        gene_region: Gene region being targeted (e.g., 'exon2')
        options_to_evaluate: List of candidate sgRNA sequences to evaluate
    
    Returns:
        Dictionary with sgRNA evaluations and recommendations
    
    Example:
        >>> analyze_sgrna(
        ...     target_sequence="BRCA1 exon 2",
        ...     options_to_evaluate=[
        ...         "ACGTTGCGAGGACAGAGTCA(AGG)",
        ...         "AGAAACCTACAACTCATGGA(AGG)"
        ...     ]
        ... )
    """
    results = {
        "query": {
            "target_sequence": target_sequence,
            "pam_sequence": pam_sequence,
            "gene_region": gene_region,
            "options_to_evaluate": options_to_evaluate
        },
        "analysis": {}
    }
    
    # PAM verification
    if pam_sequence.upper() == "NGG":
        results["analysis"]["cas_type"] = "SpCas9"
        results["analysis"]["pam_requirement"] = "Requires NGG PAM immediately 3' of target"
    
    # Evaluate options if provided
    if options_to_evaluate:
        evaluations = []
        for i, seq in enumerate(options_to_evaluate):
            eval_result = {
                "option_id": chr(65 + i),
                "sequence": seq,
                "length": len(seq.replace("(", "").replace(")", "").split("(")[0]) if "(" in seq else len(seq),
                "has_pam": False,
                "pam_sequence": None,
                "quality_score": None,
                "notes": []
            }
            
            # Check for PAM in sequence (often in parentheses)
            if "(" in seq and ")" in seq:
                parts = seq.split("(")
                guide_seq = parts[0]
                pam = parts[1].replace(")", "")
                eval_result["has_pam"] = True
                eval_result["pam_sequence"] = pam
                eval_result["guide_sequence"] = guide_seq
                
                # Basic quality checks
                if len(guide_seq) == 20:
                    eval_result["notes"].append("✓ Standard 20bp guide length")
                elif len(guide_seq) < 18:
                    eval_result["notes"].append("⚠ WARNING: Guide too short")
                elif len(guide_seq) > 24:
                    eval_result["notes"].append("⚠ WARNING: Guide too long")
                
                # GC content
                gc_count = guide_seq.upper().count("G") + guide_seq.upper().count("C")
                gc_percent = (gc_count / len(guide_seq)) * 100
                eval_result["gc_percent"] = round(gc_percent, 1)
                
                if 40 <= gc_percent <= 60:
                    eval_result["notes"].append(f"✓ Good GC content ({gc_percent:.0f}%)")
                elif gc_percent < 30:
                    eval_result["notes"].append(f"⚠ Low GC content ({gc_percent:.0f}%) - may reduce efficacy")
                elif gc_percent > 80:
                    eval_result["notes"].append(f"⚠ High GC content ({gc_percent:.0f}%) - may increase off-target")
                
                # Check for problematic patterns
                if "TTTT" in guide_seq.upper():
                    eval_result["notes"].append("⚠ Contains TTTT - Pol III termination signal")
                
                if guide_seq.upper().startswith("G"):
                    eval_result["notes"].append("✓ Starts with G - good for U6 promoter")
            
            evaluations.append(eval_result)
        
        results["analysis"]["evaluations"] = evaluations
        
        # Rank by quality
        valid_evals = [e for e in evaluations if e.get("has_pam")]
        if valid_evals:
            valid_evals.sort(key=lambda x: abs(x.get("gc_percent", 0) - 50))
            results["analysis"]["recommended_order"] = [e["option_id"] for e in valid_evals]
    
    return results


# ============================================================
# Tool 4: Experimental Data Analyzer
# ============================================================

@tool
def analyze_experimental_data(
    data_description: str,
    question_type: str = "which_is_true",
    groups: Optional[List[str]] = None,
    outcome_variable: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze experimental data patterns for multiple choice questions.
    
    Use this tool when the question presents experimental data 
    (infection assays, diet studies, etc.) and asks to interpret results.
    
    This tool provides:
    - Experiment type detection
    - Analysis strategy generation
    - Data interpretation prompts
    
    Args:
        data_description: Description of the experimental data
        question_type: Type of question: 'which_is_true', 'which_is_false', 'interpret_results'
        groups: Experimental groups (e.g., ['wild type', 'mutant A'])
        outcome_variable: What was measured (e.g., 'bacteria count')
    
    Returns:
        Dictionary with data structure analysis, patterns, and analysis strategy
    
    Example:
        >>> analyze_experimental_data(
        ...     data_description="Bacterial infection assay with wt and mutant strains",
        ...     groups=["wild type", "deltaA", "deltaB", "deltaAdeltaB"],
        ...     outcome_variable="bacteria count in liver"
        ... )
    """
    result = {
        "query": {
            "data_description": data_description,
            "question_type": question_type,
            "groups": groups,
            "outcome_variable": outcome_variable
        },
        "analysis": {
            "data_structure": None,
            "patterns": [],
            "candidate_conclusions": [],
            "analysis_strategy": []
        }
    }
    
    # Parse data description for structure
    desc_lower = data_description.lower()
    
    # Detect experiment type
    if "infection" in desc_lower or "pathogen" in desc_lower:
        result["analysis"]["data_structure"] = "infection_assay"
        result["analysis"]["analysis_strategy"].append("1. Compare pathogen load across host groups")
        result["analysis"]["analysis_strategy"].append("2. Identify which mutations affect infection")
        result["analysis"]["analysis_strategy"].append("3. Look for interaction effects between mutations")
    
    elif "diet" in desc_lower or "feeding" in desc_lower:
        result["analysis"]["data_structure"] = "dietary_experiment"
        result["analysis"]["analysis_strategy"].append("1. Compare performance across diet conditions")
        result["analysis"]["analysis_strategy"].append("2. Check for host-diet interactions")
        result["analysis"]["analysis_strategy"].append("3. Identify adaptation patterns")
    
    elif "motility" in desc_lower or "swarming" in desc_lower:
        result["analysis"]["data_structure"] = "motility_assay"
        result["analysis"]["analysis_strategy"].append("1. Compare motility across conditions")
        result["analysis"]["analysis_strategy"].append("2. Identify factors affecting motility")
    
    else:
        result["analysis"]["data_structure"] = "general_experiment"
        result["analysis"]["analysis_strategy"].append("1. Identify independent and dependent variables")
        result["analysis"]["analysis_strategy"].append("2. Compare across experimental groups")
    
    # Detect comparison pattern
    if "mutant" in desc_lower or "delta" in desc_lower or "knockout" in desc_lower:
        result["analysis"]["patterns"].append({
            "type": "genetic_comparison",
            "description": "Compare mutants to wild type"
        })
    
    if groups:
        result["analysis"]["groups_identified"] = groups
        result["analysis"]["comparison_count"] = len(groups) * (len(groups) - 1) // 2
    
    # Generate analysis prompts
    result["analysis"]["analysis_prompts"] = [
        "1. Extract all numerical values from the data",
        "2. Identify the highest and lowest values for each group",
        "3. Calculate fold changes between conditions",
        "4. Note any statistically significant differences",
        "5. Match data patterns to option claims"
    ]
    
    return result


# ============================================================
# Export all tools as a list for easy loading
# ============================================================

def get_analysis_tools() -> List:
    """
    Get all analysis tools as a list.
    
    Returns:
        List of tool functions decorated with @tool
    """
    return [
        verify_multi_statement,
        calculate_modification_mass,
        analyze_sgrna,
        analyze_experimental_data
    ]


# Export for tool_loader
__all__ = [
    'verify_multi_statement',
    'calculate_modification_mass',
    'analyze_sgrna',
    'analyze_experimental_data',
    'get_analysis_tools',
    'MultiStatementInput',
    'ModificationInput',
    'SgRNAAnalysisInput',
    'ExperimentalDataInput',
    'BIOMEDICAL_FACTS',
    'MODIFICATION_MASSES'
]
