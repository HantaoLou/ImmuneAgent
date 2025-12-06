"""
R Analysis MCP Server

This server provides R analysis tools for Figure 2-5 RSV data analysis.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("R Analysis Server")

@mcp.tool()
def run_bcr_standardize(bcr_file_path: str, combine_fields: List[str], output_path: str) -> str:

    """
    Standardize B-cell receptor (BCR) data files by combining specified fields to create a unified barcode identifier.
    
    This tool is essential for BCR repertoire analysis workflows where multiple identifier fields need to be 
    consolidated into a single 'combine_barcode' field for downstream analysis. Commonly used to merge cell 
    identifiers, sample origins, and experimental conditions into a standardized format.
    
    Use cases:
    - Preparing BCR data for single-cell analysis pipelines
    - Standardizing multi-sample BCR datasets
    - Creating unified cell identifiers across experimental batches
    
    Args:
        bcr_file_path (str): Path to the input BCR data file (CSV format)
        combine_fields (List[str]): List of column names to combine (e.g., ['orig.ident', 'cell', 'sample_id'])
        output_path (str): Path where the standardized file will be saved

    Returns:
        str: Path to the standardized output file with combine_barcode field added
        
    Example:
        run_bcr_standardize("/data/bcr_raw.csv", ["orig.ident", "cell"], "/output/bcr_standardized.csv")
    """
    from scripts.combine.standardize_csv import standardize_csv
    output_path = standardize_csv(bcr_file_path=bcr_file_path, combine_fields=combine_fields, output_path=output_path)



    return output_path


@mcp.tool()
def run_rds_standardize(rds_file_path: str, combine_fields: List[str], output_path: str) -> str:


    """
    Standardize R data structure (RDS) files containing single-cell or bulk sequencing data by creating unified cell identifiers.
    
    This tool processes Seurat objects or other R data structures stored in RDS format, combining multiple metadata 
    fields into a standardized 'combine_barcode' identifier. Essential for integrating datasets from different 
    experiments, batches, or processing pipelines in immunology research.
    
    Use cases:
    - Standardizing Seurat objects for multi-sample integration
    - Preparing single-cell RNA-seq data for BCR/TCR analysis
    - Creating consistent cell identifiers across experimental conditions
    - Preprocessing data for spatial transcriptomics analysis
    
    Args:
        rds_file_path (str): Path to input RDS file containing single-cell or bulk data
        combine_fields (List[str]): Metadata column names to combine (typically ['orig.ident', 'cell'])
        output_path (str): Path where the standardized RDS file will be saved

        
    Returns:
        str: Detailed execution status with success/error messages and R script output
        
    Note:
        Commonly uses 'orig.ident,cell' combination to create unique cell identifiers across samples
        
    Example:
        run_rds_standardize("/data/seurat_obj.rds", ["orig.ident", "cell"], "/output/standardized.rds")
    """
    try:
        # Call standardize_rds function with default field combination
        working_dir = Path(__file__).parent
        result = subprocess.run(
                ["Rscript", "scripts/combine/standardize_rds.R", rds_file_path, ",".join(combine_fields), output_path],        
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                cwd=str(working_dir)
            )
        # Check execution result
        if result.returncode != 0:
            return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
        # Execution successful, return output result
        return f"R script executed successfully\nOutput information: {result.stdout}"
    except subprocess.TimeoutExpired:
        return f"R script execution timeout (exceeded {timeout} seconds)"
    except Exception as e:
        return f"Error occurred while executing R script: {str(e)}"


@mcp.tool()
def run_extract_bcr_info(bcr_file_path: str, n_rows: int = 5) -> Dict[str, Any]:
    """
    Intelligently extract and identify B-cell receptor (BCR) data structure using large language model analysis.
    
    This tool automatically analyzes BCR data files to identify key columns and data structure patterns. 
    It uses AI to recognize common BCR data formats and extract essential field mappings including barcode 
    identifiers, heavy chain sequences, and light chain sequences. Particularly useful when working with 
    datasets from different sources or with non-standard column naming conventions.
    
    Use cases:
    - Analyzing unknown BCR dataset structures before processing
    - Identifying column mappings in multi-source BCR data
    - Quality assessment of BCR sequencing data
    - Automated field detection for downstream analysis pipelines
    
    Args:
        bcr_file_path (str): Path to BCR data file (CSV, Excel, or TSV format)
        n_rows (int): Number of sample rows to analyze for structure detection (default: 5)
        
    Returns:
        Dict[str, Any]: Structured analysis result containing:
            - status: "success" or "error"
            - message: Human-readable status description
            - result: Extracted field mappings (bar_code, Heavy, Light chain columns) if successful
            - error: Detailed error information if failed
            
    Example:
        run_extract_bcr_info("/data/bcr_sequences.csv", 10)
    """
    try:
        from scripts.combine.bcr_extractor import extract_bcr_info_with_llm
        # Call extract_bcr_info_with_llm function
        result = extract_bcr_info_with_llm(bcr_file_path, n_rows)
        return {
            "status": "success",
            "message": "BCR information extraction successful",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "BCR information extraction failed",
            "error": str(e)
        }


@mcp.tool()
def run_process_csv_to_standard(csv_file_path: str, bar_code: str, heavy: str, light: str, 
                                   variant_seq: str, experiment: str, output_path: str = None) -> str:
    """
    Extract and standardize BCR sequence data with quality filtering and metadata annotation.
    
    This tool processes raw BCR CSV files by extracting specified columns (barcode, heavy chain, light chain), 
    renaming them to standard format (combine_barcode, Heavy, Light), and applying quality filters. It removes 
    records with missing sequences, filters out sequences longer than 235 characters, and adds experimental 
    metadata fields. The output is a clean, standardized CSV ready for downstream BCR analysis tools.
    
    Quality control features:
    - Removes rows with empty heavy or light chain sequences
    - Filters sequences exceeding 235 characters (quality threshold)
    - Standardizes column names for consistent downstream processing
    - Adds experimental metadata (variant_seq, experiment, Label fields)
    
    Use cases:
    - Preparing raw BCR data for MetaBCR antigen specificity analysis
    - Standardizing 10x Genomics VDJ output for repertoire analysis
    - Quality filtering BCR sequences before clonotype analysis
    - Creating analysis-ready datasets with experimental annotations
    
    Args:
        csv_file_path: Path to input CSV file with raw BCR sequence data
        bar_code: Source column name for cell/sequence identifiers
        heavy: Source column name for heavy chain variable region sequences
        light: Source column name for light chain variable region sequences
        variant_seq: Experimental variant identifier (added as metadata)
        experiment: Experimental condition/batch identifier (added as metadata)
        output_path: Output file path (auto-generated as *_processed.csv if None)
    
    Returns:
        str: Processing status with output file path, record count, and column information
        
    Example:
        run_process_csv_to_standard("/data/raw_bcr.csv", "cell_barcode", "VH_sequence", "VL_sequence", "variant_A", "batch_1")
    """
    try:
        from scripts.combine.bcr_extractor import process_csv_to_standard_format
        # Call extract_bcr_info_with_llm function
        result = process_csv_to_standard_format(csv_file_path, bar_code, heavy, light, variant_seq, experiment, output_path)
        return f"CSV processing successful\nExtraction result: {result}"
    except Exception as e:
        return f"CSV processing failed: {str(e)}"


@mcp.tool()
def run_integrate_rds_bcr_data(bcr_file_path: str, rds_file_path: str, output_path: str) -> str:
    """
    Integrate B-cell receptor repertoire data with single-cell RNA sequencing data for comprehensive immunological analysis.
    
    This tool performs sophisticated data integration by matching BCR sequence information with corresponding 
    single-cell transcriptomic profiles. It links BCR clonotype data (heavy/light chain sequences, CDR3 regions) 
    with gene expression profiles from the same cells, enabling paired BCR-transcriptome analysis essential for 
    understanding B-cell differentiation, activation states, and antigen specificity.
    
    Use cases:
    - Linking BCR repertoire with single-cell gene expression
    - Creating paired datasets for B-cell functional analysis
    - Integrating 10x Genomics VDJ and gene expression data
    - Preparing data for clonal evolution and lineage tracing studies
    - Combining BCR specificity with transcriptional states
    
    Args:
        bcr_file_path (str): Path to standardized BCR data file (CSV format with combine_barcode)
        rds_file_path (str): Path to single-cell RNA-seq data (RDS format, typically Seurat object)
        output_path (str): Path for the integrated output file (RDS format with BCR annotations)

        
    Returns:
        str: Detailed integration status including cell matching statistics, data quality metrics, and file paths
        
    Example:
        run_integrate_rds_bcr_data("/data/bcr_standardized.csv", "/data/scrna_seurat.rds", "/output/integrated_bcr_scrna.rds")
    """
    try:
        working_dir = Path(__file__).parent
        result = subprocess.run(
                ["Rscript", "scripts/combine/integrate_bcr_data.R", bcr_file_path, rds_file_path, output_path],        
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,
                cwd=str(working_dir)
            )
        # Check execution result
        if result.returncode != 0:
            return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
        # Execution successful, return output result
        return f"R script executed successfully\nOutput information: {result.stdout}"
    except subprocess.TimeoutExpired:
        return f"R script execution timeout (exceeded {timeout} seconds)"
    except Exception as e:
        return f"Error occurred while executing R script: {str(e)}"


# Add lifecycle management
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def figure_analysis_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown"""
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
    mcp.settings.port = 8091
    
    # Start using SSE mode
    mcp.run(transport="sse")
