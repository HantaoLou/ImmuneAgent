"""
R Analysis MCP Server

This server provides R analysis tools for Figure 2-5 RSV data analysis.
"""

import os
import json
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("R Analysis Server")

def run_r_script(figure_name: str, input_file: str, base_dir: str) -> str:
    """
    Universal R script execution function for single-cell RNA-seq analysis
    
    Args:
        figure_name: Analysis figure name (e.g., "Figure2_Common")
        input_file: Input RDS file path containing Seurat object with scRNA-seq data
        base_dir: Base output directory path for analysis results
        
    Returns:
        Analysis execution result string with generated file paths
    """
    # Check input file existence - 抛出异常而不是返回错误字符串
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    working_dir = Path(__file__).parent
    base_dir = Path(base_dir)  # Convert to Path object
    
    # R script path
    r_script_path = working_dir / "scripts/common" / f"{figure_name}.R"
    
    # Check R script existence - 抛出异常而不是返回错误字符串
    if not r_script_path.exists():
        raise FileNotFoundError(f"R script does not exist: {r_script_path}")
    
    # Set timeout (Figure4 requires longer execution time)
    timeout = 7200 if figure_name == "Figure4" else 3600
    
    try:
        # Execute R script (R script handles configuration and directory creation)
        result = subprocess.run(
            ["Rscript", str(r_script_path), input_file, base_dir],

            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        
        # Check execution results - 抛出异常而不是返回错误字符串
        if result.returncode != 0:
            raise RuntimeError(f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}")
        
        # Collect generated files using precise paths from config
        output_dir = base_dir / figure_name
        generated_files = []
        
        if output_dir.exists():
            # CSV files
            csv_files = list((output_dir / "files").glob("*.csv")) if (output_dir / "files").exists() else []
            generated_files.extend([str(f) for f in csv_files])
            
            # PDF files
            pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
            generated_files.extend([str(f) for f in pdf_files])
            
            # Other files
            other_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.RData"))
            generated_files.extend([str(f) for f in other_files])
        
        success_msg = f"{figure_name} analysis executed successfully!\n"
        if generated_files:
            success_msg += f"Generated files ({len(generated_files)} files):\n"
            for file in generated_files:
                success_msg += f"  - {file}\n"
        else:
            success_msg += f"Analysis completed, please check output directory: {output_dir}\n"
        
        return success_msg
        
    except subprocess.TimeoutExpired:
        return f"R script execution timeout (exceeded {timeout} seconds)"
    except Exception as e:
        return f"Error occurred during R script execution: {str(e)}"

@mcp.tool()
def run_figure2_analysis(input_file: str, base_dir: str) -> str:
    """Single-cell RNA-seq differential gene expression analysis and visualization
    
    Performs comprehensive differential expression analysis on single-cell B-cell data:
    - Cell type mapping and annotation for King dataset B-cell populations
    - Antigen binding prediction value detection and processing (multiple prediction column formats)
    - Differential gene expression analysis using Seurat FindMarkers function
    - Volcano plot generation with statistical visualization
    - P-value adjustment and percentage difference calculations
    - Export differential gene results to CSV files for downstream analysis
    
    Domains: ["single_cell", "expression", "B-cell", "differential_analysis"]
    Inputs: ["scRNA-seq RDS file", "Seurat object", "B-cell expression data"]
    Outputs: ["DE genes", "Volcano plots", "Statistical results", "CSV files"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat object with scRNA-seq data
        base_dir: Base output directory absolute path for analysis results and plots
        
    Returns:
        Analysis results summary with generated file list and statistical outcomes
    """
    return run_r_script("Figure2_Common", input_file, base_dir)

@mcp.tool()
def run_figure3_analysis(input_file: str, base_dir: str) -> str:
    """Single-cell antigen binding prediction visualization and UMAP density analysis
    
    Generates comprehensive visualization of antigen binding predictions on single-cell data:
    - Cell type mapping and annotation for King dataset B-cell populations
    - Automatic detection and processing of multiple binding prediction column formats
    - Numerical conversion of binding prediction values with NA value handling
    - UMAP coordinate extraction and visualization preparation
    - Prediction score density plot generation with spatial mapping
    - Gradient color mapping visualization (transparent → coral → brown)
    - Correlation analysis between different antigen binding predictions
    
    Domains: ["single_cell", "visualization", "B-cell", "antigen_binding", "spatial"]
    Inputs: ["scRNA-seq RDS file", "UMAP coordinates", "Binding predictions"]
    Outputs: ["Density plots", "UMAP visualizations", "Correlation plots", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat object with binding predictions
        base_dir: Base output directory absolute path for visualization results
        
    Returns:
        Visualization analysis results with generated plot file paths
    """
    return run_r_script("Figure3_Common", input_file, base_dir)

@mcp.tool()
def run_figure4_analysis(input_file: str, base_dir: str) -> str:
    """Single-cell trajectory analysis and gene module scoring with pseudotime inference (computationally intensive)
    
    Performs comprehensive trajectory and temporal analysis on B-cell differentiation:
    - Cell type mapping and annotation for King dataset B-cell populations
    - Antigen binding prediction detection and H1N1 binding/neutralization value calculation
    - Multiple gene module scoring (high affinity, low affinity, exhaustion, activation, CSR pathways)
    - Monocle3 trajectory analysis with CDS object construction for developmental inference
    - Pseudotime gene expression pattern analysis across B-cell differentiation stages
    - Combined visualization generation with integrated trajectory plots
    - Supplementary figure generation for different gene set pseudotime expressions
    - Execution time typically requires 60-120 minutes for complete analysis
    
    Domains: ["single_cell", "trajectory", "B-cell", "pseudotime", "gene_modules", "differentiation"]
    Inputs: ["scRNA-seq RDS file", "Gene expression matrix", "Cell metadata"]
    Outputs: ["Trajectory plots", "Pseudotime analysis", "Gene module scores", "PDF visualizations"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat object with temporal data
        base_dir: Base output directory absolute path for trajectory analysis results
        
    Returns:
        Trajectory analysis results with generated visualization files and module scores
    """
    return run_r_script("Figure4_Common", input_file, base_dir)

@mcp.tool()
def run_figure5_analysis(input_file: str, base_dir: str) -> str:
    """B-cell receptor isotype distribution and somatic hypermutation (SHM) rate analysis
    
    Performs comprehensive BCR repertoire analysis focusing on isotype switching and affinity maturation:
    - Data loading with duplicate column name repair and quality control
    - IGH isotype mapping and classification (IgM, IgD, IgG1-4, IgA1-2, IgE subclasses)
    - BCR isotype distribution statistical analysis and visualization
    - Somatic hypermutation (SHM) rate estimation and comparative analysis
    - SHM rate distribution plotting with statistical comparisons
    - Broadly reactive BCR isotype bias analysis across antigen specificities
    - Heavy chain and light chain SHM rate comparative analysis
    - Memory B-cell subset-specific repertoire characterization
    
    Domains: ["B-cell", "BCR_analysis", "isotype", "somatic_hypermutation", "repertoire"]
    Inputs: ["scRNA-seq RDS file", "BCR sequences", "Isotype annotations", "SHM data"]
    Outputs: ["Isotype distribution plots", "SHM analysis", "Statistical comparisons", "PDF reports"]
    
    Args:
        input_file: Complete path to input RDS file containing BCR repertoire and isotype data
        base_dir: Base output directory absolute path for BCR analysis results
        
    Returns:
        BCR repertoire analysis results with isotype and SHM statistical summaries
    """
    return run_r_script("Figure5_Common", input_file, base_dir)


# Add lifecycle management
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def figure_analysis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown lifecycle"""
    print("R Analysis MCP Server is initializing...")
    
    try:
        yield {"initialized": True}
    finally:
        print("R Analysis MCP Server is shutting down...")

# Set lifecycle
mcp.lifespan = figure_analysis_lifespan

if __name__ == "__main__":
    print("Starting R Analysis MCP Server...")
    
    # Set network parameters
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8090
    
    # Start using SSE mode
    mcp.run(transport="sse")
