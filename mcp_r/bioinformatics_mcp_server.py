"""
Bioinformatics Analysis Modular MCP Server

This server provides Figure2-Figure5 related bioinformatics analysis tools, 
with each tool corresponding to a specific analysis function.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Bioinformatics Analysis Modular Server")

def run_bioinformatics_module_script(module_name: str, input_file: str, base_dir: str, figure_type: str = "figure2", **kwargs) -> str:
    """
    Generic function for executing bioinformatics analysis modular R scripts
    
    Args:
        module_name: Module name (e.g., "Figure2_A2_Binding", "Figure3_A_Density")
        input_file: Input file path containing Seurat object and single-cell RNA-seq data
        base_dir: Output directory base path
        figure_type: Figure type ("figure2", "figure3", "figure4", "figure5")
        **kwargs: Additional parameters
        
    Returns:
        Analysis execution result string, including generated file paths
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        return f"Error: Input file does not exist: {input_file}"
    
    working_dir = Path(__file__).parent
    base_dir = Path(base_dir)
    
    # R script path
    r_script_path = working_dir / f"scripts/common/{figure_type}_modules" / f"{module_name}.R"
    
    # Check if R script exists
    if not r_script_path.exists():
        return f"Error: R script does not exist: {r_script_path}"
    
    try:
        # Build command arguments
        cmd_args = ["Rscript", str(r_script_path), input_file, str(base_dir)]
        
        # Add additional parameters
        for key, value in kwargs.items():
            if value is not None:
                cmd_args.append(str(value))
        
        # Execute R script
        result = subprocess.run(
            cmd_args,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800  # 30 minutes timeout
        )
        
        # Check execution result
        if result.returncode != 0:
            return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
        # Collect generated files
        generated_files = []
        
        # Find generated files (support Figure2 and Figure3 etc.)
        figure_pattern = figure_type.replace("figure", "Figure").capitalize()
        for output_dir in base_dir.glob(f"{figure_pattern}*"):
            if output_dir.is_dir():
                # CSV files
                csv_files = list((output_dir / "files").glob("*.csv")) if (output_dir / "files").exists() else []
                generated_files.extend([str(f) for f in csv_files])
                
                # PDF files
                pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
                generated_files.extend([str(f) for f in pdf_files])
                
                # Other files
                other_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.RData"))
                generated_files.extend([str(f) for f in other_files])
        
        success_msg = f"{module_name} bioinformatics analysis executed successfully!\n"
        if generated_files:
            success_msg += f"Generated files ({len(generated_files)} files):\n"
            for file in generated_files:
                success_msg += f"  - {file}\n"
        else:
            success_msg += f"Analysis completed, please check output directory: {base_dir}\n"
        
        return success_msg
        
    except subprocess.TimeoutExpired:
        return f"R script execution timeout (exceeded 1800 seconds)"
    except Exception as e:
        return f"Error occurred during R script execution: {str(e)}"

def run_figure2_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure2 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure2", **kwargs)

def run_figure3_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure3 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure3", **kwargs)

def run_figure4_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure4 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure4", **kwargs)

def run_figure5_module_script(module_name: str, input_file: str, base_dir: str, **kwargs) -> str:
    """Function for executing Figure5 modular R scripts"""
    return run_bioinformatics_module_script(module_name, input_file, base_dir, "figure5", **kwargs)

@mcp.tool()
def antigen_binding_prediction_visualization(
    input_file: str, 
    base_dir: str, 
    binding_threshold: Optional[float] = 0.5
) -> str:
    """Single-cell B cell antigen binding prediction visualization analysis
    
    Performs visualization analysis of antigen binding prediction for single-cell B cell data:
    - Automatically detects and processes multiple binding prediction column formats (bind_predict, bind_output, etc.)
    - Numerical conversion and NA value handling to ensure data quality
    - Broad reactivity threshold classification and statistical analysis
    - Binding prediction value distribution visualization and density plot generation
    - Cell type-specific binding pattern analysis
    - Export binding prediction statistical results to CSV files
    
    Bioinformatics domains: ["single-cell", "B-cell", "antigen binding", "prediction analysis", "visualization"]
    Input data: ["Single-cell RNA-seq RDS files", "Seurat objects", "Binding prediction data"]
    Output results: ["Binding prediction plots", "Statistical analysis", "CSV files", "Visualization charts"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat object and binding prediction data
        base_dir: Absolute path to output directory for saving analysis results and charts
        binding_threshold: Broad reactivity threshold (between 0-1, default 0.5)
        
    Returns:
        Analysis result summary including generated file list and statistical results
    """
    return run_figure2_module_script(
        "Figure2_A2_Binding", 
        input_file, 
        base_dir, 
        binding_threshold=binding_threshold
    )

@mcp.tool()
def bcell_celltype_distribution_analysis(input_file: str, base_dir: str) -> str:
    """Single-cell B cell subtype distribution visualization analysis
    
    Performs visualization analysis of cell type distribution for single-cell B cell data:
    - King dataset cell type mapping and standardized annotation
    - B cell subtype classification statistics (Naive, Memory, Germinal Center, Plasma, etc.)
    - Cell type proportion distribution calculation and visualization
    - Multi-color palette cell type coloring scheme
    - Cell type distribution pie charts and bar chart generation
    - Export cell type statistical data to CSV files
    
    Bioinformatics domains: ["single-cell", "B-cell", "cell type", "distribution analysis", "visualization"]
    Input data: ["Single-cell RNA-seq RDS files", "Seurat objects", "Cell type annotations"]
    Output results: ["Distribution plots", "Statistical charts", "CSV files", "Cell type analysis"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat object and cell type annotations
        base_dir: Absolute path to output directory for saving analysis results and charts
        
    Returns:
        Analysis result summary including cell type distribution statistics and generated visualization files
    """
    return run_figure2_module_script("Figure2_B1_CellType", input_file, base_dir)

@mcp.tool()
def binding_prediction_interval_distribution_analysis(
    input_file: str, 
    base_dir: str, 
    interval_step: Optional[float] = 0.1,
    data_min: Optional[float] = 0.0,
    data_max: Optional[float] = 1.0
) -> str:
    """Single-cell antigen binding prediction value interval distribution analysis
    
    Analyzes antigen binding prediction value in single-cell data:
    - Customize interval step and data range flexibility
    - Generate antigen binding prediction value interval distribution histogram
    - Calculate number of cells and percentage in each interval
    - Cumulative distribution function(CDF) calculation and visualization
    - Quantile analysis and outlier detection
    - Export interval statistics to CSV file for further analysis
    
    Bioinformatics domains: ["single-cell", "statistics analysis", "distribution analysis", "data mining", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "antigen binding data", "numerical prediction score"]
    Output results: ["distribution histogram", "statistics analysis", "CSV data", "quantile analysis"]
    
    Args:
        input_file: Complete path to input RDS file containing antigen binding values
        base_dir: Absolute path to output directory for saving statistics analysis results
        interval_step: Interval step (default 0.1)
        data_min: Data minimum value (default 0.0)
        data_max: Data maximum value (default 1.0)
        
    Returns:
        Interval distribution statistics result summary, including distribution features and generated analysis files
    """
    return run_figure2_module_script(
        "Figure2_B2_Intervals", 
        input_file, 
        base_dir, 
        interval_step=interval_step,
        data_min=data_min,
        data_max=data_max
    )

@mcp.tool()
def differential_gene_expression_volcano_analysis(
    input_file: str, 
    base_dir: str, 
    logfc_threshold: Optional[float] = 0.0,
    min_pct: Optional[float] = 0.2,
    analysis_strategy: Optional[str] = "both"
) -> str:
    """Single-cell differential gene expression and volcano plot visualization
    
    Analyzes single-cell B cell data for differential gene expression and volcano plot visualization:
    - Smart threshold setting, based on data distribution dynamics classification
    - Broad reaction vs specific B cell differential expression analysis
    - Seurat FindMarkers function for statistical test
    - Volcano plot generation, containing significant gene annotation and statistical information
    - Multiple analysis strategy support (broad, specific, both)
    - P value adjustment and multiple change threshold filtering
    - Export differential gene list to CSV file
    
    Bioinformatics domains: ["single-cell", "differential expression", "statistics analysis", "gene expression", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell division information"]
    Output results: ["volcano plot", "differential gene list", "statistics results", "CSV files"]
    
    Args:
        input_file: Complete path to input RDS file containing gene expression data
        base_dir: Absolute path to output directory for saving analysis results
        logfc_threshold: log2 multiple change threshold (default 0.0)
        min_pct: Minimum expression percent (default 0.2, i.e., 20% cells expression)
        analysis_strategy: Analysis strategy ("both", "broad", "specific"，default"both")
        
    Returns:
        Differential expression analysis result summary, including significant gene count and generated visualization files
    """
    return run_figure2_module_script(
        "Figure2_C_Volcano", 
        input_file, 
        base_dir, 
        logfc_threshold=logfc_threshold,
        min_pct=min_pct,
        analysis_strategy=analysis_strategy
    )

@mcp.tool()
def umap_dimensionality_reduction_visualization(input_file: str, base_dir: str) -> str:
    """Single-cell B cell UMAP reduction and cell type visualization analysis
    
    Analyzes single-cell B cell data for UMAP reduction and cell type visualization:
    - UMAP coordinate extraction and two-dimensional space mapping
    - B cell type in UMAP space distribution visualization
    - Cell type specific color encoding and figure legend
    - High quality UMAP plot generation suitable for publication use
    - Cell density distribution and cluster boundary visualization
    - Support King dataset's cell type mapping
    - Export UMAP coordinate and cell type information to CSV file
    
    Bioinformatics domains: ["single-cell", "reduction analysis", "UMAP", "visualization", "cell group"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "cell type annotations"]
    Output results: ["UMAP plot", "cell distribution plot", "coordinate data", "visualization file"]
    
    Args:
        input_file: Complete path to input RDS file containing UMAP coordinates and cell annotations
        base_dir: Absolute path to output directory for saving visualization results
        
    Returns:
        UMAP visualization analysis result summary, including cell distribution features and generated charts file
    """
    return run_figure2_module_script("Figure2_S2A_UMAP", input_file, base_dir)

@mcp.tool()
def bcell_marker_gene_dotplot_analysis(
    input_file: str, 
    base_dir: str, 
    min_pct: Optional[float] = 0.1,
    min_expression: Optional[float] = 0.25
) -> str:
    """B cell type specific gene expression dotplot analysis
    
    Analyzes B cell type specific gene expression dotplot:
    - B cell type specific gene expression set definition and detection
    - Gene expression level and expression ratio's double visualization
    - Dotplot size represents expression ratio, color represents expression strength
    - Expression threshold filtering, ensuring biological significance
    - Multiple B cell type specific gene expression comparison
    - Auto detect data available gene markers
    - Export gene expression statistics to CSV file
    
    Bioinformatics domains: ["single-cell", "gene expression", "gene expression", "cell type", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell type annotations"]
    Output results: ["dotplot visualization", "expression statistics", "gene markers list", "CSV files"]
    
    Args:
        input_file: Complete path to input RDS file containing gene expression and cell type data
        base_dir: Absolute path to output directory for saving analysis results
        min_pct: Minimum expression percent threshold (default 0.1, i.e., 10% cells expression)
        min_expression: Minimum average expression level (default 0.25)
        
    Returns:
        Gene marker dotplot analysis result summary, including expression features and generated visualization files
    """
    return run_figure2_module_script(
        "Figure2_S2C_DotPlot", 
        input_file, 
        base_dir, 
        min_pct=min_pct,
        min_expression=min_expression
    )

@mcp.tool()
def antigen_binding_neutralization_density_visualization(
    input_file: str, 
    base_dir: str, 
    prediction_keywords: Optional[str] = "neut,bind,average,predict,output",
    na_strategy: Optional[str] = "exclude_cells",
    feature_priority: Optional[str] = "neutralization_first"
) -> str:
    """Single-cell antigen binding and neutralization prediction density plot visualization analysis
    
    Performs UMAP density plot visualization of antigen binding and neutralization predictions for single-cell data:
    - Automatically detects multiple prediction field formats (neut, bind, predict, etc.)
    - Flexible NA value handling strategies (exclude cells, replace with zero, replace with median)
    - Feature selection priority configuration (neutralization first, binding first, highest value first)
    - Nebulosa density plot generation showing prediction value distribution in UMAP space
    - Gradient color mapping visualization (transparent→coral→brown)
    - Supports King dataset cell type mapping
    - Export prediction value statistics and UMAP coordinate data
    
    Bioinformatics domains: ["single-cell", "UMAP", "density visualization", "antigen binding", "neutralization prediction"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "Prediction value data"]
    Output results: ["Density plots", "UMAP visualization", "Prediction statistics", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing UMAP coordinates and prediction data
        base_dir: Absolute path to output directory for saving visualization results
        prediction_keywords: Prediction field detection keywords, comma-separated (default "neut,bind,average,predict,output")
        na_strategy: NA value handling strategy ("exclude_cells", "replace_zero", "replace_median", default "exclude_cells")
        feature_priority: Feature selection priority ("neutralization_first", "binding_first", "highest_value", default "neutralization_first")
        
    Returns:
        Density plot visualization analysis result summary including prediction value distribution characteristics and generated chart files
    """
    return run_figure3_module_script(
        "Figure3_A_Density", 
        input_file, 
        base_dir, 
        prediction_keywords=prediction_keywords,
        na_strategy=na_strategy,
        feature_priority=feature_priority
    )

@mcp.tool()
def bcell_celltype_umap_visualization(
    input_file: str, 
    base_dir: str, 
    celltype_column: Optional[str] = "CellType"
) -> str:
    """Single-cell B cell type UMAP space distribution visualization analysis
    
    Analyzes single-cell B cell data for cell type in UMAP space distribution visualization:
    - King data set cell type mapping and standardized annotation
    - B cell type in UMAP two-dimensional space distribution visualization
    - 36 tone color palette for cell type specific reactivity
    - High quality UMAP plot generation suitable for publication use
    - Cell type cluster boundary and density distribution visualization
    - Support custom cell type field name
    - Export UMAP coordinate and cell type statistics data
    
    Bioinformatics domains: ["single-cell", "UMAP", "cell type", "space distribution", "visualization"]
    Input data: ["single-cell RNA-seq RDS files", "UMAP coordinates", "cell type annotations"]
    Output results: ["UMAP plot", "cell type distribution", "statistics data", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing UMAP coordinates and cell type annotations
        base_dir: Absolute path to output directory for saving visualization results
        celltype_column: Cell type field name (default "CellType")
        
    Returns:
        UMAP cell type distribution visualization result summary, including distribution features and generated charts file
    """
    return run_figure3_module_script(
        "Figure3_C_CellType", 
        input_file, 
        base_dir, 
        celltype_column=celltype_column
    )

@mcp.tool()
def bcell_marker_gene_expression_dotplot(
    input_file: str, 
    base_dir: str, 
    celltype_column: Optional[str] = "CellType"
) -> str:
    """B cell type specific marker gene expression dotplot visualization analysis
    
    Analyzes B cell type specific marker gene expression dotplot:
    - B cell type specific marker gene expression set definition and detection
    - Gene expression level and expression ratio's double visualization
    - Dotplot size represents expression ratio, color represents expression strength
    - Multiple B cell type specific marker gene expression comparison
    - Auto detect data available gene markers
    - Support custom cell type field name
    - Export marker gene expression statistics and visualization result
    
    Bioinformatics domains: ["single-cell", "marker gene", "gene expression", "dotplot visualization", "cell type"]
    Input data: ["single-cell RNA-seq RDS files", "gene expression matrix", "cell type annotations"]
    Output results: ["dotplot visualization", "expression statistics", "marker gene information", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing gene expression and cell type data
        base_dir: Absolute path to output directory for saving analysis results
        celltype_column: Cell type field name (default "CellType")
        
    Returns:
        Marker gene dotplot analysis result summary, including expression features and generated visualization files
    """
    return run_figure3_module_script(
        "Figure3_D_DotPlot", 
        input_file, 
        base_dir, 
        celltype_column=celltype_column
    )

@mcp.tool()
def differential_gene_correlation_analysis(
    deg_file1: str, 
    deg_file2: str, 
    base_dir: str, 
    dataset1_name: str, 
    dataset2_name: str,
    p_value_threshold: Optional[float] = 0.05,
    min_common_genes: Optional[int] = 10,
    highlight_genes: Optional[str] = "ITGAX,FGR,FCRL4,FCRL5,CD68,TNFRSF1B,JCHAIN,MZB1,XBP1,MARCKSL1"
) -> str:
    """Differential gene correlation analysis and scatter plot visualization
    
    Analyzes two data sets for differential gene correlation:
    - Automatically validate input DEG file format and necessary fields
    - Filter significant differential genes with p value threshold
    - Compute Pearson correlation coefficient between two data sets
    - Generate correlation scatter plot, containing statistical significant information
    - Support custom highlight genes annotation and visualization
    - Ensure statistical significance of minimum common genes requirement
    - Export correlation data and statistical results
    
    Bioinformatics domains: ["differential expression", "correlation analysis", "statistics analysis", "gene expression", "comparison analysis"]
    Input data: ["DEG result CSV files", "differential gene list", "statistics test results"]
    Output results: ["correlation scatter plot", "statistics results", "correlation data", "PDF files"]
    
    Args:
        deg_file1: First DEG result file path
        deg_file2: Second DEG result file path
        base_dir: Output directory absolute path, used for saving analysis results
        dataset1_name: First data set name
        dataset2_name: Second data set name
        p_value_threshold: Significance p value threshold (default 0.05)
        min_common_genes: Minimum common gene count (default 10)
        highlight_genes: Highlight genes list, comma separated (default B cell related genes)
        
    Returns:
        Differential gene correlation analysis result summary, including correlation coefficient and generated visualization files
    """
    # Figure3_F_Correlation需要特殊的参数传递方式
    # 直接调用R脚本，因为它需要两个DEG文件作为输入
    working_dir = Path(__file__).parent
    base_dir_path = Path(base_dir)
    
    # R脚本路径
    r_script_path = working_dir / "scripts/common/figure3_modules/Figure3_F_Correlation.R"
    
    # Check if R script exists
    if not r_script_path.exists():
        return f"Error: R script does not exist: {r_script_path}"
    
    # Check if input files exist
    if not os.path.exists(deg_file1):
        return f"Error: First DEG file does not exist: {deg_file1}"
    if not os.path.exists(deg_file2):
        return f"Error: Second DEG file does not exist: {deg_file2}"
    
    try:
        # Build command arguments - in the order required by R script
        cmd_args = [
            "Rscript", str(r_script_path), 
            deg_file1, deg_file2, str(base_dir_path),
            dataset1_name, dataset2_name
        ]
        
        # Add optional parameters
        if p_value_threshold is not None:
            cmd_args.append(str(p_value_threshold))
        if min_common_genes is not None:
            cmd_args.append(str(min_common_genes))
        if highlight_genes is not None:
            cmd_args.append(str(highlight_genes))
        
        # Execute R script
        result = subprocess.run(
            cmd_args,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800  # 30 minutes timeout
        )
        
        # Check execution result
        if result.returncode != 0:
            return f"R script execution failed (return code: {result.returncode})\nError message: {result.stderr}"
        
        # Collect generated files
        generated_files = []
        
        # Find generated files
        for output_dir in base_dir_path.glob("Figure3*"):
            if output_dir.is_dir():
                # CSV files
                csv_files = list((output_dir / "files").glob("*.csv")) if (output_dir / "files").exists() else []
                generated_files.extend([str(f) for f in csv_files])
                
                # PDF files
                pdf_files = list((output_dir / "plots").glob("*.pdf")) if (output_dir / "plots").exists() else []
                generated_files.extend([str(f) for f in pdf_files])
                
                # Other files
                other_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.RData"))
                generated_files.extend([str(f) for f in other_files])
        
        success_msg = f"Figure3_F_Correlation differential gene correlation analysis executed successfully!\n"
        if generated_files:
            success_msg += f"Generated files ({len(generated_files)} files):\n"
            for file in generated_files:
                success_msg += f"  - {file}\n"
        else:
            success_msg += f"Analysis completed, please check output directory: {base_dir_path}\n"
        
        return success_msg
        
    except subprocess.TimeoutExpired:
        return f"R script execution timeout (exceeded 1800 seconds)"
    except Exception as e:
        return f"Error occurred during R script execution: {str(e)}"

@mcp.tool()
def prediction_value_density_visualization(
    input_file: str, 
    base_dir: str, 
    prediction_keywords: Optional[str] = "bind,predict,output,average,score",
    prediction_threshold: Optional[float] = 0.5
) -> str:
    """Prediction value UMAP density plot visualization analysis
    
    Analyzes single-cell data for prediction value density plot visualization:
    - Automatically detect multiple prediction field formats (bind, predict, output etc.)
    - Based on prediction value threshold for cell classification and statistics
    - Nebulosa density plot generation, showing prediction value space distribution
    - Gradient color mapping visualization prediction strength
    - Support custom prediction field detection keywords
    - Prediction value distribution statistics and threshold analysis
    - Export prediction value data and UMAP coordinate information
    
    Bioinformatics domains: ["single-cell", "prediction analysis", "UMAP", "density visualization", "threshold analysis"]
    Input data: ["single-cell RNA-seq RDS files", "prediction value data", "UMAP coordinates"]
    Output results: ["density plot", "prediction distribution plot", "statistics analysis", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing prediction value and UMAP coordinates
        base_dir: Absolute path to output directory for saving visualization results
        prediction_keywords: Prediction field detection keywords, comma-separated (default"bind,predict,output,average,score")
        prediction_threshold: Prediction value classification threshold (default 0.5)
        
    Returns:
        Prediction value density plot visualization result summary, including distribution features and generated charts file
    """
    return run_figure3_module_script(
        "Figure3_G_Prediction", 
        input_file, 
        base_dir, 
        prediction_keywords=prediction_keywords,
        prediction_threshold=prediction_threshold
    )

@mcp.tool()
def pseudotime_trajectory_analysis(
    input_file: str, 
    base_dir: str, 
    num_dim: Optional[int] = 50,
    cluster_resolution: Optional[float] = 0.001,
    min_gene_cells: Optional[int] = 3,
    root_celltype: Optional[str] = "Naive"
) -> str:
    """Single-cell B cell pseudotime trajectory and UMAP visualization
    
    Analyzes single-cell B cell data for pseudotime trajectory and UMAP visualization:
    - Use monocle3 for trajectory segmentation and pseudotime calculation
    - Automatically select root cell type as trajectory start (default Naive B cell)
    - Principal component analysis and reduction quality control
    - Cluster resolution optimization, suitable for trajectory analysis low resolution setting
    - Gene quality control and filtering, ensuring trajectory segmentation accuracy
    - Generate high quality pseudotime trajectory plot, suitable for publication use
    - Save monocle3 CDS object for subsequent analysis
    
    Bioinformatics domains: ["single-cell", "trajectory analysis", "pseudotime", "monocle3", "development trajectory"]
    Input data: ["single-cell RNA-seq RDS files", "Seurat objects", "cell type annotations"]
    Output results: ["trajectory plot", "CDS objects", "pseudotime data", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing Seurat objects and cell type annotations
        base_dir: Absolute path to output directory for saving analysis results
        num_dim: Principal component dimension, affecting reduction quality (default 50)
        cluster_resolution: Cluster resolution, affecting cell group granularity (default 0.001, suitable for trajectory analysis)
        min_gene_cells: Gene filtering threshold, minimum expressing cells (default 3)
        root_celltype: Root cell type, trajectory start (default"Naive")
        
    Returns:
        Pseudotime trajectory analysis result summary, including trajectory features and generated visualization file
    """
    return run_figure4_module_script(
        "Figure4_A_Trajectory", 
        input_file, 
        base_dir, 
        num_dim=num_dim,
        cluster_resolution=cluster_resolution,
        min_gene_cells=min_gene_cells,
        root_celltype=root_celltype
    )

@mcp.tool()
def pseudotime_celltype_boxplot_analysis(
    input_file: str, 
    base_dir: str, 
    celltype_column: Optional[str] = ""
) -> str:
    """Pseudotime and cell type distribution boxplot analysis
    
    Analyzes single-cell data for pseudotime and cell type distribution boxplot analysis:
    - Depends on trajectory analysis generated CDS objects and pseudotime data
    - Automatically detect cell type field, supporting various naming formats
    - Calculate different cell types' pseudotime distribution statistics
    - Generate boxplot to show cell type along trajectory's distribution mode
    - Statistical significance test and multiple comparisons adjustment
    - Recognize developmental stage specific cell type
    - Export pseudotime statistical data and visualization result
    
    Bioinformatics domains: ["single-cell", "pseudotime", "cell type", "statistics analysis", "development stage"]
    Input data: ["single-cell RNA-seq RDS files", "pseudotime data", "cell type annotations"]
    Output results: ["boxplot", "statistics data", "pseudotime distribution", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing pseudotime and cell type data
        base_dir: Absolute path to output directory for saving analysis results
        celltype_column: Cell type field name (default empty value indicates automatic detection)
        
    Returns:
        Pseudotime boxplot analysis result summary, including distribution features and generated statistics file
    """
    return run_figure4_module_script(
        "Figure4_C_Boxplot", 
        input_file, 
        base_dir, 
        celltype_column=celltype_column
    )

@mcp.tool()
def trajectory_polynomial_regression_analysis(input_file: str, base_dir: str) -> str:
    """Trajectory polynomial regression analysis and gene module scoring
    
    Analyzes single-cell trajectory data for polynomial regression analysis and gene module scoring:
    - Calculate B cell feature gene module scores (activation, memory, germinal center, etc.)
    - Estimate somatic hypermutation (SHM) levels based on gene expression features
    - Polynomial regression fitting and trend analysis along pseudotime trajectory
    - Identify key trajectory turning points and developmental stage markers
    - Generate combined plots showing trajectory change patterns of multiple features
    - Statistical significance testing and regression model evaluation
    - Export trajectory data and regression analysis results
    
    Bioinformatics domains: ["single-cell", "trajectory analysis", "polynomial regression", "gene modules", "SHM analysis"]
    Input data: ["single-cell RNA-seq RDS files", "pseudotime data", "gene expression matrix"]
    Output results: ["regression plots", "trajectory data", "module scores", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing trajectory and gene expression data
        base_dir: Absolute path to output directory for saving analysis results
        
    Returns:
        Polynomial regression analysis result summary, including trajectory features and generated visualization files
    """
    return run_figure4_module_script("Figure4_DEFG_Polynomial", input_file, base_dir)

@mcp.tool()
def trajectory_supplementary_analysis(input_file: str, base_dir: str) -> str:
    """Trajectory analysis supplementary figure generation and transcriptional marker analysis
    
    Performs supplementary analysis and transcriptional marker visualization on single-cell trajectory data:
    - S6A: Expression patterns of B cell activation-related transcriptional markers along trajectory
    - S6B: Dynamic changes of atypical B cell-related transcriptional markers
    - S6C: Immunoglobulin expression dynamics and isotype switching analysis
    - S6D: Key transcription factor expression patterns and regulatory networks
    - Multi-gene expression heatmaps and trajectory visualization
    - Gene expression correlation analysis and co-expression module identification
    - Export gene expression data and statistical analysis results
    
    Bioinformatics domains: ["single-cell", "transcriptional markers", "gene expression", "trajectory analysis", "supplementary analysis"]
    Input data: ["single-cell RNA-seq RDS files", "trajectory data", "gene expression matrix"]
    Output results: ["supplementary figures", "gene expression data", "correlation analysis", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing trajectory and gene expression data
        base_dir: Absolute path to output directory for saving analysis results
        
    Returns:
        Trajectory supplementary analysis result summary, including transcriptional marker features and generated chart files
    """
    return run_figure4_module_script("Figure4_S6_Supplementary", input_file, base_dir)

@mcp.tool()
def bcr_isotype_distribution_shm_analysis(
    input_file: str, 
    base_dir: str, 
    binding_threshold: Optional[float] = 0.5
) -> str:
    """B cell receptor isotype distribution and somatic hypermutation rate analysis
    
    Performs comprehensive analysis of B cell receptor isotype distribution and somatic hypermutation (SHM) rates:
    - Analyze isotype distribution differences between broadly reactive BCRs and specific/non-binding BCRs
    - Compare SHM rates across different binding levels (broadly reactive, specific, non-binding)
    - Automatically detect and standardize isotype annotation formats from different datasets
    - Estimate SHM levels and affinity maturation degree based on gene expression features
    - Generate combined plots: isotype distribution bar chart + SHM level distribution + SHM boxplot
    - Statistical significance testing and multiple comparison correction
    - Export detailed analysis data and statistical results
    
    Bioinformatics domains: ["B cells", "antibodies", "isotype switching", "SHM analysis", "affinity maturation"]
    Input data: ["single-cell RNA-seq RDS files", "isotype annotations", "binding prediction data"]
    Output results: ["combined plots", "statistical analysis", "isotype distribution data", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing BCR isotype and binding prediction data
        base_dir: Absolute path to output directory for saving analysis results
        binding_threshold: Binding threshold defining classification criteria for broadly reactive BCRs (default 0.5)
        
    Returns:
        BCR isotype distribution and SHM analysis result summary, including distribution features and generated visualization files
    """
    return run_figure5_module_script(
        "Figure5_C_Isotype", 
        input_file, 
        base_dir, 
        binding_threshold=binding_threshold
    )

@mcp.tool()
def neutralizing_antibody_shm_comparison_analysis(
    input_file: str, 
    base_dir: str, 
    binding_threshold: Optional[float] = 0.5
) -> str:
    """Neutralizing antibody versus non-neutralizing antibody SHM rate comparison analysis
    
    Performs SHM rate comparison analysis between predicted neutralizing and non-neutralizing antibodies:
    - Compare SHM rate differences between predicted neutralizing and non-neutralizing antibodies
    - Focus specifically on antibody characteristics from FCRL5+ atypical B cells
    - Analyze correlation between neutralization capacity and somatic hypermutation levels
    - Isotype distribution analysis to identify dominant isotypes of neutralizing antibodies
    - Generate combined plots: isotype distribution + SHM level distribution + SHM comparison boxplot
    - Statistical significance testing and effect size calculation
    - Export neutralizing antibody characteristic data and comparative analysis results
    
    Bioinformatics domains: ["neutralizing antibodies", "SHM analysis", "antibody function", "immune protection", "viral neutralization"]
    Input data: ["single-cell RNA-seq RDS files", "neutralization prediction data", "cell type annotations"]
    Output results: ["comparison plots", "statistical analysis", "neutralizing antibody data", "PDF files"]
    
    Args:
        input_file: Complete path to input RDS file containing neutralization prediction and cell type data
        base_dir: Absolute path to output directory for saving analysis results
        binding_threshold: Binding threshold consistent with isotype analysis classification criteria (default 0.5)
        
    Returns:
        Neutralizing antibody SHM comparison analysis result summary, including functional features and generated visualization files
    """
    return run_figure5_module_script(
        "Figure5_D_Neutralization", 
        input_file, 
        base_dir, 
        binding_threshold=binding_threshold
    )

# Add lifecycle management
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def bioinformatics_modules_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Handle server startup and shutdown lifecycle"""
    print("Bioinformatics modular MCP server is initializing...")
    
    try:
        yield {"initialized": True}
    finally:
        print("Bioinformatics modular MCP server is shutting down...")

# Set lifecycle
mcp.lifespan = bioinformatics_modules_lifespan

if __name__ == "__main__":
    print("Starting bioinformatics modular MCP server...")
    
    # Set network parameters
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8091
    
    # Start using SSE mode
    mcp.run(transport="sse")