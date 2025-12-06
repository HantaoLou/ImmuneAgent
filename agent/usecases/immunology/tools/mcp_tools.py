"""
MCP Tool Integrations for ImmuneAgent
Real tool implementations using MCP (Model Context Protocol)
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# Add parent paths for imports
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))
# Also add agent path for common module
agent_path = root_path / "agent"
if str(agent_path) not in sys.path:
    sys.path.insert(0, str(agent_path))
from common.util.mcp_utils import mcp_tool_async

# ============= METABCR TOOLS =============


@tool
def metabcr_predict(input_file_path: str = None) -> str:
    """
    MetaBCR: Deep Learning Framework for Antibody-Antigen Interaction Prediction
    Predicts binding affinity between antibodies and antigens.

    Args:
        input_file_path: Optional CSV file with antibody sequences.
                        If not provided, uses default test dataset.

    Returns:
        Predicted binding affinities saved as Excel file path
    """
    params = {"input_file_path": input_file_path} if input_file_path else {}
    return asyncio.run(mcp_tool_async("metabcr", "metabcr", params))


@tool
def bcr_standardize(bcr_file_path: str, combine_fields: List[str]) -> str:
    """
    Standardize BCR file by combining fields to generate barcode

    Args:
        bcr_file_path: Path to BCR file
        combine_fields: List of field names to combine

    Returns:
        Output file path with standardized BCR data
    """
    params = {"bcr_file_path": bcr_file_path, "combine_fields": combine_fields}
    return asyncio.run(mcp_tool_async("metabcr", "run_bcr_standardize", params))


@tool
def extract_bcr_info(bcr_file_path: str, n_rows: int = 10) -> str:
    """
    Extract barcode, heavy chain, and light chain field names from BCR file

    Args:
        bcr_file_path: Path to BCR file
        n_rows: Number of rows to extract for analysis

    Returns:
        Field information (barcode, Heavy, Light field lists)
    """
    params = {"bcr_file_path": bcr_file_path, "n_rows": n_rows}
    return asyncio.run(mcp_tool_async("metabcr", "run_extract_bcr_info", params))


@tool
def process_antibody_csv(
    csv_file_path: str,
    bar_code: str,
    heavy: str,
    light: str,
    variant_seq: str,
    experiment: str,
    output_path: str = None,
) -> str:
    """
    Process antibody CSV file to standard format for MetaBCR

    Args:
        csv_file_path: Input CSV file path
        bar_code: Barcode field name
        heavy: Heavy chain field name
        light: Light chain field name
        variant_seq: Variant sequence value
        experiment: Experiment identifier
        output_path: Optional output path

    Returns:
        Path to processed standard format CSV
    """
    params = {
        "csv_file_path": csv_file_path,
        "bar_code": bar_code,
        "heavy": heavy,
        "light": light,
        "variant_seq": variant_seq,
        "experiment": experiment,
        "output_path": output_path,
    }
    return asyncio.run(mcp_tool_async("metabcr", "run_process_csv_to_standard", params))


# ============= R ANALYSIS TOOLS =============


@tool
def run_figure2_analysis(input_file: str) -> str:
    """
    Run Figure 2 analysis: Basic statistics and quality control
    Performs QC metrics, cell type distribution, and basic visualizations

    Args:
        input_file: Path to RDS file with single-cell data

    Returns:
        Analysis results with generated plots and statistics
    """
    params = {"input_file": input_file}
    return asyncio.run(mcp_tool_async("r_analysis", "run_figure2_analysis", params))


@tool
def run_figure3_analysis(input_file: str) -> str:
    """
    Run Figure 3 analysis: Clustering and cell type identification
    Performs clustering, UMAP, and cell type annotation

    Args:
        input_file: Path to RDS file with single-cell data

    Returns:
        Clustering results with cell type annotations and UMAP plots
    """
    params = {"input_file": input_file}
    return asyncio.run(mcp_tool_async("r_analysis", "run_figure3_analysis", params))


@tool
def run_figure4_analysis(input_file: str) -> str:
    """
    Run Figure 4 analysis: Differential expression and pathway analysis
    Identifies differentially expressed genes and enriched pathways

    Args:
        input_file: Path to RDS file with single-cell data

    Returns:
        DEG results, pathway enrichment, and volcano plots
    """
    params = {"input_file": input_file}
    return asyncio.run(mcp_tool_async("r_analysis", "run_figure4_analysis", params))


@tool
def run_figure5_analysis(input_file: str) -> str:
    """
    Run Figure 5 analysis: Trajectory and pseudotime analysis
    Performs trajectory inference and developmental dynamics

    Args:
        input_file: Path to RDS file with single-cell data

    Returns:
        Trajectory analysis with pseudotime ordering and branch points
    """
    params = {"input_file": input_file}
    return asyncio.run(mcp_tool_async("r_analysis", "run_figure5_analysis", params))


# ============= IMMGPT ANALYSIS TOOLS =============


@tool
def analyze_fdg_results() -> List[Dict]:
    """
    Analyze FDG (Foldx, DDG, GearBind) results using ImmGPT
    Performs AI-powered analysis to select top antibody candidates

    Returns:
        Table with analysis results and top 5 antibody recommendations
    """
    return asyncio.run(mcp_tool_async("immgpt", "analyse_fdg_result", {}))


@tool
def select_top_antibodies(
    results_file: str, selection_criteria: str = "broadly neutralizing", top_n: int = 5
) -> Dict[str, Any]:
    """
    Select top antibodies based on specified criteria using ImmGPT

    Args:
        results_file: Path to antibody analysis results
        selection_criteria: Criteria for selection (e.g., "broadly neutralizing")
        top_n: Number of top candidates to select

    Returns:
        Selected antibodies with ranking and rationale
    """
    params = {
        "results_file": results_file,
        "selection_criteria": selection_criteria,
        "top_n": top_n,
    }
    return asyncio.run(mcp_tool_async("immgpt", "select_antibodies", params))


# ============= ALPHAFOLD3 TOOLS =============


@tool
def predict_antibody_structure(
    heavy_chain: str, light_chain: str, antigen_sequence: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict antibody structure using AlphaFold3

    Args:
        heavy_chain: Heavy chain amino acid sequence
        light_chain: Light chain amino acid sequence
        antigen_sequence: Optional antigen sequence for complex prediction

    Returns:
        Dictionary with PDB structure, pLDDT scores, and PAE matrix
    """
    params = {"sequences": [heavy_chain, light_chain], "mode": "antibody"}
    if antigen_sequence:
        params["sequences"].append(antigen_sequence)
        params["mode"] = "antibody_antigen_complex"

    return asyncio.run(mcp_tool_async("af3", "predict_structure", params))


@tool
def predict_protein_complex(sequences: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Predict protein complex structure using AlphaFold3

    Args:
        sequences: List of amino acid sequences
        labels: List of chain labels/descriptions

    Returns:
        Complex structure with confidence metrics
    """
    params = {"sequences": sequences, "labels": labels, "mode": "complex"}
    return asyncio.run(mcp_tool_async("af3", "predict_complex", params))


# ============= FDG (FOLDX/DDG) TOOLS =============


@tool
def calculate_ddg_mutations(
    pdb_file: str, mutations: List[str], chain_id: str = "A"
) -> Dict[str, float]:
    """
    Calculate ΔΔG for mutations using FoldX

    Args:
        pdb_file: Path to PDB structure file
        mutations: List of mutations (e.g., ["A123T", "K456E"])
        chain_id: Chain identifier

    Returns:
        Dictionary of mutations and their ΔΔG values
    """
    params = {"pdb_file": pdb_file, "mutations": mutations, "chain_id": chain_id}
    return asyncio.run(mcp_tool_async("fdg", "calculate_ddg", params))


@tool
def optimize_antibody_stability(
    pdb_file: str, target_regions: List[str] = ["CDR1", "CDR2", "CDR3"]
) -> Dict[str, Any]:
    """
    Optimize antibody stability using FoldX design

    Args:
        pdb_file: Antibody structure PDB file
        target_regions: Regions to optimize

    Returns:
        Optimized sequences with stability improvements
    """
    params = {
        "pdb_file": pdb_file,
        "target_regions": target_regions,
        "mode": "stability_optimization",
    }
    return asyncio.run(mcp_tool_async("fdg", "optimize_stability", params))


# ============= INTEGRATION TOOLS =============


@tool
def integrate_multimodal_data(
    bcr_file: str, rna_file: str, protein_file: Optional[str] = None
) -> str:
    """
    Integrate BCR, RNA-seq, and optionally protein data

    Args:
        bcr_file: BCR sequencing file path
        rna_file: scRNA-seq file path
        protein_file: Optional CITE-seq protein data

    Returns:
        Path to integrated multimodal dataset
    """
    params = {"bcr_file": bcr_file, "rna_file": rna_file}
    if protein_file:
        params["protein_file"] = protein_file

    return asyncio.run(mcp_tool_async("integration", "integrate_multimodal", params))


# ============= TOOL COLLECTIONS =============

# Antibody discovery tools
antibody_discovery_tools = [
    metabcr_predict,
    extract_bcr_info,
    process_antibody_csv,
    predict_antibody_structure,
    calculate_ddg_mutations,
    optimize_antibody_stability,
]

# Single-cell analysis tools
single_cell_tools = [
    run_figure2_analysis,
    run_figure3_analysis,
    run_figure4_analysis,
    run_figure5_analysis,
    integrate_multimodal_data,
]

# AI analysis tools
ai_analysis_tools = [analyze_fdg_results, select_top_antibodies]

# Structure prediction tools
structure_tools = [predict_antibody_structure, predict_protein_complex]

# All MCP tools
all_mcp_tools = (
    antibody_discovery_tools + single_cell_tools + ai_analysis_tools + structure_tools
)


def get_mcp_tools_by_category(category: str) -> List:
    """
    Get MCP tools by category

    Args:
        category: Tool category name

    Returns:
        List of tools in that category
    """
    categories = {
        "antibody_discovery": antibody_discovery_tools,
        "single_cell": single_cell_tools,
        "ai_analysis": ai_analysis_tools,
        "structure": structure_tools,
        "all": all_mcp_tools,
    }
    return categories.get(category, [])


# Create tool dictionary for easy access
mcp_tools_dict = {
    "metabcr_predict": metabcr_predict,
    "bcr_standardize": bcr_standardize,
    "extract_bcr_info": extract_bcr_info,
    "process_antibody_csv": process_antibody_csv,
    "run_figure2_analysis": run_figure2_analysis,
    "run_figure3_analysis": run_figure3_analysis,
    "run_figure4_analysis": run_figure4_analysis,
    "run_figure5_analysis": run_figure5_analysis,
    "analyze_fdg_results": analyze_fdg_results,
    "select_top_antibodies": select_top_antibodies,
    "predict_antibody_structure": predict_antibody_structure,
    "predict_protein_complex": predict_protein_complex,
    "calculate_ddg_mutations": calculate_ddg_mutations,
    "optimize_antibody_stability": optimize_antibody_stability,
    "integrate_multimodal_data": integrate_multimodal_data,
}

# Export all tools
__all__ = [
    # MetaBCR tools
    "metabcr_predict",
    "bcr_standardize",
    "extract_bcr_info",
    "process_antibody_csv",
    # R analysis tools
    "run_figure2_analysis",
    "run_figure3_analysis",
    "run_figure4_analysis",
    "run_figure5_analysis",
    # ImmGPT tools
    "analyze_fdg_results",
    "select_top_antibodies",
    # AlphaFold3 tools
    "predict_antibody_structure",
    "predict_protein_complex",
    # FDG tools
    "calculate_ddg_mutations",
    "optimize_antibody_stability",
    # Integration tools
    "integrate_multimodal_data",
    # Tool collections
    "all_mcp_tools",
    "get_mcp_tools_by_category",
    "mcp_tools_dict",
]
