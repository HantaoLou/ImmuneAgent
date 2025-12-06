"""
Comprehensive immunology prompts for ImmuneAgent.
Integrated from common/immunology_prompts.py and enhanced_prompts.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Define constants for backward compatibility
IMMUNOLOGY_SYSTEM_PROMPT = """You are ImmuneAgent, an expert computational immunologist with comprehensive knowledge across all immunology domains."""

THINKING_TOKENS = """
<thinking>
Let me analyze this immunology question step by step...
</thinking>
"""


@dataclass
class ImmunologyPrompts:
    """Universal immunology prompts with chain-of-thought reasoning."""

    IMMUNOLOGY_EXPERT_SYSTEM_PROMPT = """You are ImmuneAgent, a world-leading computational immunologist with comprehensive expertise across all domains of immunology:

## Core Expertise Areas:

### Adaptive Immunity
- T cell biology: CD4+, CD8+, regulatory T cells, memory, exhaustion, activation
- B cell biology: Development, activation, antibody production, memory, plasma cells
- TCR and BCR repertoires: V(D)J recombination, diversity, clonal selection
- Antigen presentation: MHC-I/II, cross-presentation, peptide processing

### Innate Immunity
- Dendritic cells: Subsets, maturation, antigen presentation, migration
- Macrophages: M1/M2 polarization, phagocytosis, cytokine production
- Neutrophils: NETs, chemotaxis, antimicrobial functions
- NK cells: Recognition, cytotoxicity, ADCC
- Pattern recognition: TLRs, NLRs, RLRs, CLRs

### Immune Metabolism
- Metabolic reprogramming: Glycolysis, OXPHOS, FAO, glutaminolysis
- Immunometabolism: Warburg effect, metabolic checkpoints
- Nutrient sensing: mTOR, AMPK, HIF-1α
- Metabolite signaling: Lactate, succinate, itaconate

### Cell Signaling & Communication
- Cytokine networks: Pro/anti-inflammatory, chemokines, growth factors
- Signal transduction: JAK-STAT, NF-κB, MAPK, PI3K/AKT
- Cell-cell interactions: Immune synapses, checkpoint molecules
- Epigenetic regulation: Chromatin remodeling, DNA methylation, histone modifications

### Tissue Immunity
- Mucosal immunity: GALT, MALT, IgA responses
- Tumor immunology: TME, TAMs, MDSCs, immune evasion
- Autoimmunity: Tolerance breakdown, autoreactive cells
- Transplant immunology: Rejection, tolerance, GvHD

### Systems Immunology
- Single-cell profiling: scRNA-seq, CyTOF, CITE-seq
- Spatial transcriptomics: Cell localization, tissue architecture
- Multi-omics integration: Transcriptome, proteome, metabolome
- Network analysis: Gene regulatory networks, protein interactions

<thinking>
When analyzing immunological questions, I will:
1. Identify the immune cell types and processes involved
2. Consider the tissue context and microenvironment
3. Evaluate metabolic and signaling pathways
4. Design appropriate experimental and computational approaches
5. Generate testable hypotheses with mechanistic insights
6. Propose validation strategies and controls
</thinking>

Your responses will be:
- Scientifically rigorous with current literature support
- Mechanistically detailed with molecular insights
- Clinically relevant when applicable
- Computationally tractable with available tools
- Hypothesis-driven with clear predictions"""

    CHAIN_OF_THOUGHT_PLANNING_TEMPLATE = """<system>
{system_prompt}
</system>

<thinking>
Let me analyze this research objective step by step:

1. **Problem Decomposition**:
   - Primary objective: {primary_objective}
   - Sub-objectives: {sub_objectives}
   - Key challenges: {challenges}

2. **Scientific Context**:
   - Relevant literature: {literature_context}
   - Current state-of-art: {sota_methods}
   - Knowledge gaps: {gaps}

3. **Tool Capability Mapping**:
   - Required analyses: {required_analyses}
   - Available tools: {available_tools}
   - Tool selection rationale: {tool_rationale}

4. **Biological Constraints**:
   - Safety considerations: {safety}
   - Feasibility assessment: {feasibility}
   - Validation requirements: {validation}

5. **Integrated Solution**:
   - Proposed approach: {approach}
   - Expected outcomes: {outcomes}
   - Success metrics: {metrics}
</thinking>

<research_objective>
{user_objective}
</research_objective>

<context>
{retrieved_context}
</context>

Generate a comprehensive research plan that includes:
1. **Hypothesis**: Clear, testable hypotheses with mechanistic basis
2. **Methodology**: Step-by-step experimental and computational protocols
3. **Tool Selection**: Specific tools from the 79+ available options
4. **Validation Strategy**: Controls, replicates, and statistical analysis
5. **Expected Outcomes**: Predicted results with confidence intervals
6. **Timeline**: Realistic execution schedule
7. **Contingency Plans**: Alternative approaches if primary fails"""

    HYPOTHESIS_GENERATION_PROMPT = """Based on the research question and scientific context, generate 3 testable hypotheses.

Research Question: {question}

Context:
{context}

For each hypothesis provide:
1. **Primary Hypothesis**: Clear statement of the expected relationship
2. **Mechanistic Basis**: Molecular/cellular mechanisms underlying the hypothesis
3. **Testable Predictions**: 3-5 specific, measurable predictions
4. **Key Experiments**: 2-3 critical experiments to test the hypothesis
5. **Expected Outcomes**: What results would support or refute the hypothesis
6. **Confidence Level**: 0-1 score based on literature support
7. **Novelty Score**: 0-1 score for innovation beyond current knowledge

Format as structured JSON."""

    TOOL_SELECTION_PROMPT = """Select the most appropriate tools for this research objective.

Objective: {objective}
Analysis Type: {analysis_type}

Available Tool Categories:
- Antibody Discovery & Engineering (10 tools)
- Protein Structure Prediction (12 tools)
- Single-Cell Analysis (14 tools)
- TCR/BCR Repertoire Analysis (8 tools)
- Epitope & MHC Prediction (7 tools)
- Molecular Dynamics & Docking (7 tools)
- Genomics & Variant Analysis (8 tools)
- Machine Learning & AI (5 tools)
- Spatial Transcriptomics (5 tools)
- Proteomics & Mass Spec (4 tools)
- Flow & Mass Cytometry (4 tools)

Select tools based on:
1. Relevance to research question
2. Compatibility with data types
3. Computational requirements
4. Expected runtime
5. Output format compatibility

Return as list of tool names with justification."""

    VALIDATION_SYNTHESIS_PROMPT = """Synthesize the research results into a comprehensive report.

Research Question: {question}

Hypotheses Tested:
{hypotheses}

Tools Executed:
{tools}

Key Results:
{results}

Generate a synthesis that includes:
1. **Hypothesis Evaluation**: Which hypotheses were supported/refuted
2. **Integrated Findings**: How results from different tools complement each other
3. **Mechanistic Insights**: New understanding of biological mechanisms
4. **Clinical Relevance**: Potential therapeutic implications
5. **Limitations**: Technical and biological constraints
6. **Future Directions**: Next experiments and analyses
7. **Confidence Assessment**: Overall confidence in conclusions"""


class ImmunologyTools:
    """Tool selection and categorization for immunology research."""

    @staticmethod
    def select_tools_for_question(question: str) -> List[str]:
        """Select appropriate tools based on research question keywords."""

        tools = []
        question_lower = question.lower()

        # Antibody-related
        if any(
            term in question_lower
            for term in [
                "antibody",
                "antibodies",
                "mab",
                "igg",
                "fab",
                "cdr",
                "humanization",
            ]
        ):
            tools.extend(["metabcr", "sapiens", "abnumber", "antiberty", "igfold"])

        # T cell related
        if any(
            term in question_lower
            for term in ["t cell", "tcr", "cd4", "cd8", "exhaustion", "car-t"]
        ):
            tools.extend(["mixcr", "tcrdist3", "gliph2", "immunarch", "deeptcr"])

        # B cell related
        if any(
            term in question_lower
            for term in ["b cell", "bcr", "plasma", "germinal", "antibody"]
        ):
            tools.extend(["mixcr", "changeo", "dandelion", "scirpy"])

        # Structure prediction
        if any(
            term in question_lower
            for term in ["structure", "fold", "3d", "binding", "interface"]
        ):
            tools.extend(["alphafold3", "igfold", "rosettaantibody", "haddock"])

        # Single-cell analysis
        if any(
            term in question_lower
            for term in ["single-cell", "scrna", "scrnaseq", "cluster", "trajectory"]
        ):
            tools.extend(["scanpy", "seurat", "celltypist", "monocle3", "velocyto"])

        # Epitope/MHC
        if any(
            term in question_lower
            for term in ["epitope", "mhc", "hla", "peptide", "neoantigen"]
        ):
            tools.extend(["netmhcpan", "pvactools", "mixmhc2pred", "iedb"])

        # Metabolism
        if any(
            term in question_lower
            for term in ["metabol", "glycolysis", "oxphos", "warburg"]
        ):
            tools.extend(["compass", "scmetabolism"])

        # Spatial
        if any(
            term in question_lower
            for term in ["spatial", "tissue", "location", "microenvironment"]
        ):
            tools.extend(["squidpy", "stlearn", "giotto"])

        # Cell-cell interaction
        if any(
            term in question_lower
            for term in ["interaction", "communication", "ligand", "receptor"]
        ):
            tools.extend(["cellphonedb", "cellchat", "nichenet"])

        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)

        return unique_tools


class ImmunologyHypothesisGenerator:
    """Generate hypothesis components for immunology research."""

    @staticmethod
    def generate_hypothesis_components(question: str) -> Dict[str, Any]:
        """Generate structured hypothesis components."""

        components = {
            "phenomenon": "",
            "cell_types": [],
            "pathways": [],
            "predictions": [],
            "validation_approaches": [],
        }

        question_lower = question.lower()

        # Identify phenomenon
        if "exhaustion" in question_lower:
            components["phenomenon"] = "T cell exhaustion and dysfunction"
            components["cell_types"] = ["CD8+ T cells", "CD4+ T cells", "Tregs"]
            components["pathways"] = [
                "PD-1 signaling",
                "TOX/NFAT",
                "metabolic dysfunction",
            ]

        elif "car-t" in question_lower:
            components["phenomenon"] = "CAR-T cell engineering and function"
            components["cell_types"] = ["CAR-T cells", "tumor cells", "TAMs"]
            components["pathways"] = ["CAR signaling", "cytokine release", "exhaustion"]

        elif "antibody" in question_lower:
            components["phenomenon"] = "Antibody generation and optimization"
            components["cell_types"] = ["B cells", "plasma cells", "memory B cells"]
            components["pathways"] = [
                "BCR signaling",
                "germinal center reactions",
                "SHM",
            ]

        elif "tumor" in question_lower or "cancer" in question_lower:
            components["phenomenon"] = "Tumor immune evasion and microenvironment"
            components["cell_types"] = ["TILs", "TAMs", "MDSCs", "Tregs"]
            components["pathways"] = [
                "checkpoint signaling",
                "metabolic competition",
                "angiogenesis",
            ]

        else:
            components["phenomenon"] = "Immune response regulation"
            components["cell_types"] = ["T cells", "B cells", "dendritic cells"]
            components["pathways"] = [
                "cytokine signaling",
                "costimulation",
                "tolerance",
            ]

        # Generate predictions
        components["predictions"] = [
            f"Changes in {components['pathways'][0]} will alter cell function",
            f"{components['cell_types'][0]} will show distinct transcriptional profiles",
            "Intervention will modify disease progression",
        ]

        # Validation approaches
        components["validation_approaches"] = [
            "Flow cytometry for phenotyping",
            "scRNA-seq for transcriptional analysis",
            "Functional assays for validation",
            "In vivo models for therapeutic testing",
        ]

        return components


# Export all classes
__all__ = ["ImmunologyPrompts", "ImmunologyTools", "ImmunologyHypothesisGenerator"]
