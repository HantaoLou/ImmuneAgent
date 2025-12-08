# Model configurations
EMBED_MODEL = "nomic-embed-text"
REASONING_MODEL = "deepseek_70b:latest"
QWEN_MODEL_VLLM = "qwen3-8b"
QWEN_MODEL_OLLAMA = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.2
QWEN_BASE_URL = "http://117.148.176.36:6006/v1"

# OpenAI API Configuration
# 从统一的 API keys 配置导入
from config.api_keys import APIKeys
OPENAI_API_KEY = APIKeys.OPENAI_API_KEY

# Cell Agent Constants
STANDARDIZED_WORKING_DIRECTORY = "D:/PartTimeJob/agent/"

# Qdrant Configuration
QDRANT_ENABLED = True  # Enable Qdrant for vector search
SKIP_QDRANT = False  # Use Qdrant for enhanced retrieval
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "immunology"
QDRANT_API_KEY = APIKeys.QDRANT_API_KEY  # 从统一的 API keys 配置导入

# Model Tier Configuration
DEFAULT_MODEL_TIER = "TIER2"  # Options: TIER1, TIER2, TIER3
MODEL_TIER = "TIER2"  # Current active tier
TIER1_MODELS = {
    "reasoning": "deepseek_70b:latest",
    "summarize": "qwen3:8b",
    "deep_research": "qwen3:8b",
}
TIER2_MODELS = {
    "reasoning": "gpt-4o",
    "summarize": "gpt-4o-mini",
    "deep_research": "gpt-4o",
}
TIER3_MODELS = {
    "reasoning": "o1-preview",
    "summarize": "gpt-4o",
    "deep_research": "o1-preview",
}

# Enable/Disable Features
USE_DEEP_RESEARCH = True
USE_HYPOTHESIS_GENERATION = True
USE_ENHANCED_PLANNING = True

# Workflow Configuration
NON_INTERACTIVE = True  # Run in non-interactive mode for automated testing


def get_tools_info() -> str:
    """返回XML格式的工具信息（保持向后兼容）"""
    tools = f"""
<tools>
    <tool>MetaBCR:B-cell receptor antigen specificity and neutralization analysis</tool>
    <tool>IgBlast:Immunoglobulin sequence analysis</tool>
    <tool>MiXCR:Immune repertoire extraction from sequencing data</tool>
    <tool>IMGT/V-QUEST:V(D)J gene identification and analysis</tool>
    <tool>AbStar:Antibody repertoire analysis pipeline</tool>
    <tool>DeepAIR:Deep learning for antibody repertoire analysis</tool>
    <tool>DeepAb:Deep learning antibody structure prediction</tool>
    <tool>TRUST4:TCR and BCR assembly from RNA-seq</tool>
    <tool>DeepTCR:Deep learning for TCR repertoire analysis</tool>
    <tool>TCRdist:TCR similarity and clustering analysis</tool>
    <tool>VDJdb:Database of TCR sequences with known specificity</tool>
    <tool>GLIPH2:Groups TCRs by predicted specificity</tool>
    <tool>Tetramer/Multimer Staining:Experimental method for TCR specificity</tool>
    <tool>MHC-multimer + scRNA-seq (TetTCR-Seq):High-throughput TCR specificity and transcriptomics</tool>
    <tool>Scanpy:Single-cell RNA-seq analysis</tool>
    <tool>Seurat:Comprehensive single-cell toolkit</tool>
    <tool>DESeq2:Differential expression analysis</tool>
    <tool>CellRanger:10x Genomics data processing</tool>
    <tool>scvi-tools:Deep learning for single-cell analysis</tool>
    <tool>Visium:Spatial transcriptomics analysis</tool>
    <tool>CODEX:Multiplexed tissue imaging analysis</tool>
    <tool>MERFISH:Single-cell spatial transcriptomics</tool>
    <tool>Squidpy:Spatial molecular data analysis</tool>
    <tool>stLearn:Spatial trajectory inference</tool>
    <tool>AlphaFold3:Protein structure prediction</tool>
    <tool>RoseTTAFold:Fast accurate protein structure prediction</tool>
    <tool>FoldX:Protein stability and mutation analysis</tool>
    <tool>PyMOL:Molecular visualization and analysis</tool>
    <tool>ChimeraX:Advanced molecular visualization</tool>
    <tool>FlowJo:Flow cytometry data analysis</tool>
    <tool>CytoBank:Cloud-based cytometry analysis</tool>
    <tool>FlowSOM:Self-organizing maps for cytometry</tool>
    <tool>SPADE:Tree-based analysis of cytometry</tool>
    <tool>NetMHCpan:MHC-I peptide binding prediction</tool>
    <tool>IEDB:Immune epitope database and tools</tool>
    <tool>BepiPred:B-cell epitope prediction</tool>
    <tool>DiscoTope:Discontinuous B-cell epitope prediction</tool>
    <tool>HADDOCK:Data-driven protein-protein docking</tool>
    <tool>AutoDock Vina:Molecular docking for drug discovery</tool>
    <tool>GROMACS:Molecular dynamics simulations</tool>
    <tool>R Statistical Framework:Comprehensive statistical analysis</tool>
    <tool>limma:Linear models for microarray/RNA-seq</tool>
    <tool>MAST:Single-cell differential expression</tool>
</tools>
"""
    return tools


def get_tools_json() -> str:
    """返回JSON格式的工具注册表字符串，用于Stage1提示词
    只包含igblast, metabcr, lineage_analysis, af3, bioinformatics这五个服务的工具
    """
    tools_json = """{
  "BCR_analysis": [
    {
      "name": "MetaBCR",
      "priority": 1,
      "description": "MetaBCR: A deep learning-based computational framework for predicting antibody-antigen binding affinity through multimodal feature extraction using convolutional neural networks (CNN), graph neural networks (GNN), and BERT-based language models. Employs semi-supervised learning strategies to optimize binding specificity prediction for influenza antigens. Performs quantitative assessment of antibody-antigen thermodynamic binding constants (KD) by analyzing heavy and light chain variable region sequences, complementarity-determining region (CDR) structural features, and antigenic epitope information. Incorporates cross-validation methodologies for model generalization evaluation and generates standardized reports containing binding affinity predictions, confidence intervals, and statistical significance analyses.",
      "domains": [
        "B-cell",
        "antibody",
        "binding_affinity",
        "deep_learning",
        "flu_antigen"
      ],
      "inputs": [
        "CSV datasets with antibody-antigen pairs",
        "structured binding data with sequence/structure information"
      ],
      "outputs": [
        "Excel files with predicted binding affinities",
        "CSV files with test results and fold validation",
        "Named as test_results_input_basename_task_name_config_date_foldN.xlsx"
      ],
      "tool": [
        {
          "tool_name": "metabcr",
          "description": "MetaBCR: A Deep Learning Framework for Antibody-Antigen Interaction Prediction. MetaBCR is designed to predict the binding affinity between antibodies and antigens using deep learning models. It supports multiple model architectures, including CNN, GNN, and BERT-based models, and can be configured for various tasks and datasets through command-line arguments and configuration files."
        }
      ]
    },
    {
      "name": "IgBlast",
      "priority": 2,
      "description": "IgBLAST + ChangeO: Comprehensive V(D)J recombination analysis pipeline for immunoglobulin and T-cell receptor sequences. Integrates NCBI IgBLAST with ChangeO for standardized AIRR format output. Supports multiple organisms (human, mouse, rabbit, rat, rhesus, pig), receptor types (Ig/TCR), and gene loci (IGH, IGK, IGL, TRA, TRB, TRG, TRD). Provides detailed V(D)J gene assignment, CDR3 extraction, productivity analysis, and germline database alignment.",
      "domains": [
        "B-cell",
        "antibody",
        "vdj_analysis",
        "sequence_analysis"
      ],
      "inputs": [
        "Nucleotide sequences in FASTA format",
        "List of sequence dictionaries with id and sequence",
        "Antibody/TCR sequences for V(D)J analysis",
        "Local file paths or HTTP/HTTPS URLs to CSV/FASTA files"
      ],
      "outputs": [
        "AIRR format TSV files with V(D)J assignments",
        "CDR3 nucleotide and amino acid sequences",
        "Productivity analysis and gene usage statistics",
        "Junction analysis and germline alignment results"
      ],
      "tool": [
        {
          "tool_name": "analyze_vdj_batch",
          "description": "V(D)J recombination analysis using IgBLAST + ChangeO. Returns AIRR format results. Automatically splits large files into batches for efficient processing. Supports sequences input as: (1) List of sequences with id and sequence fields, (2) Local file path (FASTA/CSV), or (3) HTTP/HTTPS URL to sequence file. URLs can be passed directly as a string (e.g., 'https://example.com/sequences.csv'), not requiring array format. CSV files will be auto-converted (columns auto-detected)."
        },
        {
          "tool_name": "extract_cdr3_from_airr",
          "description": "Extract CDR3 information from AIRR format results. Supports multiple input formats: (1) Array of AIRR records: [{\"sequence_id\": \"...\", \"junction\": \"...\", ...}, ...], (2) Local file path: \"/path/to/airr_results.csv\" or \"/path/to/airr_results.json\", (3) HTTP/HTTPS URL: \"https://example.com/airr_results.csv\""
        }
      ]
    }
  ],
  "protein_structure": [
    {
      "name": "AlphaFold3",
      "priority": 1,
      "description": "AlphaFold3: Revolutionary AI system for accurate prediction of protein structures and biomolecular complexes. Utilizes advanced transformer neural networks and diffusion models to predict atomic-level 3D structures of proteins, nucleic acids, and their interactions. Provides unprecedented accuracy in modeling protein-protein, protein-DNA, protein-RNA, and protein-ligand complexes. Essential for structural biology, drug discovery, and understanding molecular mechanisms of biological processes.",
      "domains": [
        "structural",
        "general"
      ],
      "inputs": [
        "Protein sequences",
        "Excel/CSV files with antibody sequences (clone_id, Heavy, Light, Antigen)"
      ],
      "outputs": [
        "3D structures",
        "Confidence scores",
        "PDB files"
      ],
      "tool": [
        {
          "tool_name": "alphafold3",
          "description": "Uses AlphaFold3 to predict the 3D structure of antibody sequences from an input Excel file and saves the result as a PDB file. Reads an Excel file containing antibody sequences (heavy and light chains), uses AlphaFold3 to predict the 3D structure of each antibody, and writes the predicted structures to a PDB file. File must contain columns: clone_id, Heavy, Light, and optionally Antigen."
        }
      ]
    }
  ],
  "bioinformatics": [
    {
      "name": "Bioinformatics Analysis Tools",
      "priority": 1,
      "description": "Comprehensive bioinformatics analysis tools for single-cell RNA-seq data analysis, visualization, and statistical analysis. Includes tools for antigen binding prediction, B cell type distribution, differential gene expression, UMAP dimensionality reduction, trajectory analysis, and neutralizing antibody analysis.",
      "domains": [
        "single_cell",
        "bioinformatics",
        "visualization",
        "statistics"
      ],
      "inputs": [
        "Single-cell RNA-seq RDS files",
        "Seurat objects",
        "Gene expression matrices"
      ],
      "outputs": [
        "Visualization plots",
        "Statistical analysis results",
        "CSV files",
        "PDF files"
      ],
      "tool": [
        {
          "tool_name": "antigen_binding_prediction_visualization",
          "description": "Single-cell B cell antigen binding prediction visualization analysis. Performs visualization analysis of antigen binding prediction for single-cell B cell data: - Automatically detects and processes multiple binding prediction column formats (bind_predict, bind_output, etc.) - Numerical conversion and NA value handling to ensure data quality - Broad reactivity threshold classification and statistical analysis - Binding prediction value distribution visualization and density plot generation - Cell type-specific binding pattern analysis - Export binding prediction statistical results to CSV files"
        },
        {
          "tool_name": "bcell_celltype_distribution_analysis",
          "description": "Single-cell B cell subtype distribution visualization analysis. Performs visualization analysis of cell type distribution for single-cell B cell data: - King dataset cell type mapping and standardized annotation - B cell subtype classification statistics (Naive, Memory, Germinal Center, Plasma, etc.) - Cell type proportion distribution calculation and visualization - Multi-color palette cell type coloring scheme - Cell type distribution pie charts and bar chart generation - Export cell type statistical data to CSV files"
        },
        {
          "tool_name": "binding_prediction_interval_distribution_analysis",
          "description": "Single-cell antigen binding prediction value interval distribution analysis. Analyzes antigen binding prediction value in single-cell data: - Customize interval step and data range flexibility - Generate antigen binding prediction value interval distribution histogram - Calculate number of cells and percentage in each interval - Cumulative distribution function(CDF) calculation and visualization - Quantile analysis and outlier detection - Export interval statistics to CSV file for further analysis"
        },
        {
          "tool_name": "differential_gene_expression_volcano_analysis",
          "description": "Single-cell differential gene expression and volcano plot visualization. Analyzes single-cell B cell data for differential gene expression and volcano plot visualization: - Smart threshold setting, based on data distribution dynamics classification - Broad reaction vs specific B cell differential expression analysis - Seurat FindMarkers function for statistical test - Volcano plot generation, containing significant gene annotation and statistical information - Multiple analysis strategy support (broad, specific, both) - P value adjustment and multiple change threshold filtering - Export differential gene list to CSV file"
        },
        {
          "tool_name": "umap_dimensionality_reduction_visualization",
          "description": "Single-cell B cell UMAP reduction and cell type visualization analysis. Analyzes single-cell B cell data for UMAP reduction and cell type visualization: - UMAP coordinate extraction and two-dimensional space mapping - B cell type in UMAP space distribution visualization - Cell type specific color encoding and figure legend - High quality UMAP plot generation suitable for publication use - Cell density distribution and cluster boundary visualization - Support King dataset's cell type mapping - Export UMAP coordinate and cell type information to CSV file. Supports local paths or HTTP/HTTPS URLs for input RDS files."
        },
        {
          "tool_name": "bcell_marker_gene_dotplot_analysis",
          "description": "B cell type specific gene expression dotplot analysis. Analyzes B cell type specific gene expression dotplot: - B cell type specific gene expression set definition and detection - Gene expression level and expression ratio's double visualization - Dotplot size represents expression ratio, color represents expression strength - Expression threshold filtering, ensuring biological significance - Multiple B cell type specific gene expression comparison - Auto detect data available gene markers - Export gene expression statistics to CSV file"
        },
        {
          "tool_name": "antigen_binding_neutralization_density_visualization",
          "description": "Single-cell antigen binding and neutralization prediction density plot visualization analysis. Performs UMAP density plot visualization of antigen binding and neutralization predictions for single-cell data: - Automatically detects multiple prediction field formats (neut, bind, predict, etc.) - Flexible NA value handling strategies (exclude cells, replace with zero, replace with median) - Feature selection priority configuration (neutralization first, binding first, highest value first) - Nebulosa density plot generation showing prediction value distribution in UMAP space - Gradient color mapping visualization (transparent→coral→brown) - Supports King dataset cell type mapping - Export prediction value statistics and UMAP coordinate data"
        },
        {
          "tool_name": "bcell_celltype_umap_visualization",
          "description": "Single-cell B cell type UMAP space distribution visualization analysis. Analyzes single-cell B cell data for cell type in UMAP space distribution visualization: - King data set cell type mapping and standardized annotation - B cell type in UMAP two-dimensional space distribution visualization - 36 tone color palette for cell type specific reactivity - High quality UMAP plot generation suitable for publication use - Cell type cluster boundary and density distribution visualization - Support custom cell type field name - Export UMAP coordinate and cell type statistics data"
        },
        {
          "tool_name": "bcell_marker_gene_expression_dotplot",
          "description": "B cell type specific marker gene expression dotplot visualization analysis. Analyzes B cell type specific marker gene expression dotplot: - B cell type specific marker gene expression set definition and detection - Gene expression level and expression ratio's double visualization - Dotplot size represents expression ratio, color represents expression strength - Multiple B cell type specific marker gene expression comparison - Auto detect data available gene markers - Support custom cell type field name - Export marker gene expression statistics and visualization result"
        },
        {
          "tool_name": "differential_gene_correlation_analysis",
          "description": "Differential gene correlation analysis and scatter plot visualization. Analyzes two data sets for differential gene correlation: - Automatically validate input DEG file format and necessary fields - Filter significant differential genes with p value threshold - Compute Pearson correlation coefficient between two data sets - Generate correlation scatter plot, containing statistical significant information - Support custom highlight genes annotation and visualization - Ensure statistical significance of minimum common genes requirement - Export correlation data and statistical results"
        },
        {
          "tool_name": "prediction_value_density_visualization",
          "description": "Prediction value UMAP density plot visualization analysis. Analyzes single-cell data for prediction value density plot visualization: - Automatically detect multiple prediction field formats (bind, predict, output etc.) - Based on prediction value threshold for cell classification and statistics - Nebulosa density plot generation, showing prediction value space distribution - Gradient color mapping visualization prediction strength - Support custom prediction field detection keywords - Prediction value distribution statistics and threshold analysis - Export prediction value data and UMAP coordinate information"
        },
        {
          "tool_name": "pseudotime_trajectory_analysis",
          "description": "Single-cell B cell pseudotime trajectory and UMAP visualization. Analyzes single-cell B cell data for pseudotime trajectory and UMAP visualization: - Use monocle3 for trajectory segmentation and pseudotime calculation - Automatically select root cell type as trajectory start (default Naive B cell) - Principal component analysis and reduction quality control - Cluster resolution optimization, suitable for trajectory analysis low resolution setting - Gene quality control and filtering, ensuring trajectory segmentation accuracy - Generate high quality pseudotime trajectory plot, suitable for publication use - Save monocle3 CDS object for subsequent analysis"
        },
        {
          "tool_name": "pseudotime_celltype_boxplot_analysis",
          "description": "Pseudotime and cell type distribution boxplot analysis. Analyzes single-cell data for pseudotime and cell type distribution boxplot analysis: - Depends on trajectory analysis generated CDS objects and pseudotime data - Automatically detect cell type field, supporting various naming formats - Calculate different cell types' pseudotime distribution statistics - Generate boxplot to show cell type along trajectory's distribution mode - Statistical significance test and multiple comparisons adjustment - Recognize developmental stage specific cell type - Export pseudotime statistical data and visualization result"
        },
        {
          "tool_name": "trajectory_polynomial_regression_analysis",
          "description": "Trajectory polynomial regression analysis and gene module scoring. Analyzes single-cell trajectory data for polynomial regression analysis and gene module scoring: - Calculate B cell feature gene module scores (activation, memory, germinal center, etc.) - Estimate somatic hypermutation (SHM) levels based on gene expression features - Polynomial regression fitting and trend analysis along pseudotime trajectory - Identify key trajectory turning points and developmental stage markers - Generate combined plots showing trajectory change patterns of multiple features - Statistical significance testing and regression model evaluation - Export trajectory data and regression analysis results"
        },
        {
          "tool_name": "trajectory_supplementary_analysis",
          "description": "Trajectory analysis supplementary figure generation and transcriptional marker analysis. Performs supplementary analysis and transcriptional marker visualization on single-cell trajectory data: - S6A: Expression patterns of B cell activation-related transcriptional markers along trajectory - S6B: Dynamic changes of atypical B cell-related transcriptional markers - S6C: Immunoglobulin expression dynamics and isotype switching analysis - S6D: Key transcription factor expression patterns and regulatory networks - Multi-gene expression heatmaps and trajectory visualization - Gene expression correlation analysis and co-expression module identification - Export gene expression data and statistical analysis results"
        },
        {
          "tool_name": "bcr_isotype_distribution_shm_analysis",
          "description": "B cell receptor isotype distribution and somatic hypermutation rate analysis. Performs comprehensive analysis of B cell receptor isotype distribution and somatic hypermutation (SHM) rates: - Analyze isotype distribution differences between broadly reactive BCRs and specific/non-binding BCRs - Compare SHM rates across different binding levels (broadly reactive, specific, non-binding) - Automatically detect and standardize isotype annotation formats from different datasets - Estimate SHM levels and affinity maturation degree based on gene expression features - Generate combined plots: isotype distribution bar chart + SHM level distribution + SHM boxplot - Statistical significance testing and multiple comparison correction - Export detailed analysis data and statistical results"
        },
        {
          "tool_name": "neutralizing_antibody_shm_comparison_analysis",
          "description": "Neutralizing antibody versus non-neutralizing antibody SHM rate comparison analysis. Performs SHM rate comparison analysis between predicted neutralizing and non-neutralizing antibodies: - Compare SHM rate differences between predicted neutralizing and non-neutralizing antibodies - Focus specifically on antibody characteristics from FCRL5+ atypical B cells - Analyze correlation between neutralization capacity and somatic hypermutation levels - Isotype distribution analysis to identify dominant isotypes of neutralizing antibodies - Generate combined plots: isotype distribution + SHM level distribution + SHM comparison boxplot - Statistical significance testing and effect size calculation - Export neutralizing antibody characteristic data and comparative analysis results"
        }
      ]
    }
  ],
  "lineage_analysis_service": [
    {
      "name": "Lineage Analysis Service Tools",
      "priority": 1,
      "description": "Flu-related analysis tools including data collection, ChangeO+ANARCI analysis, experiment prediction, figure generation, and cell tree preliminary analysis. These tools support comprehensive influenza antibody analysis workflows corresponding to notebook pipelines: 1. get_cell_location(R).ipynb, 2. flu_dataset_collect.ipynb, 3. ChangeO+ANARCI.ipynb, 3.5. clone_result.ipynb, 4. experiment+prediction.ipynb, 5. draw_for_mainfig.ipynb, 6. cell_tree_preliminary.ipynb, 7. cell_tree(R).ipynb. **CRITICAL EXECUTION ORDER**: When using FLU tools, follow this sequence: (1) extract_seurat_umap_metadata → (2) integrate_scbcr_bulk_bcr_data (requires step 1 output) → (3) integrate_binding_neutralization_experiments (can run in parallel with step 2) → (4) integrate_predictions_with_experimental_data (requires steps 2 and 3 outputs). Never skip steps or execute out of order, as each step depends on previous outputs.",
      "domains": [
        "flu",
        "antibody",
        "numbering",
        "visualization",
        "tree_analysis",
        "data_collection",
        "experiment"
      ],
      "inputs": [
        "RDS files",
        "CSV files",
        "FASTQ files",
        "Excel files",
        "Antibody sequences",
        "Flu datasets",
        "Experimental data"
      ],
      "outputs": [
        "UMAP coordinates",
        "Merged datasets",
        "V(D)J analysis results",
        "Clone results",
        "Prediction results",
        "Visualization plots",
        "Tree structures"
      ],
      "tool": [
        {
          "tool_name": "extract_seurat_umap_metadata",
          "description": "Extract UMAP coordinates and cellular metadata from Seurat RDS files. This tool corresponds to the functionality in notebook 1.get_cell_location(R).ipynb. Extracts from Seurat object RDS files: - UMAP coordinates (dimensionality-reduced cell positions) - Cell type annotation information - Expression values for genes of interest. Bioinformatics domains: [\"single-cell\", \"dimensionality reduction\", \"UMAP\", \"metadata extraction\"]. Input data: [\"Seurat RDS files\", \"single-cell RNA-seq data\"]. Output results: [\"UMAP coordinates CSV\", \"cell type annotations\", \"gene expression values\"]."
        },
        {
          "tool_name": "integrate_scbcr_bulk_bcr_data",
          "description": "Integrate single-cell BCR and bulk BCR sequencing data. This tool corresponds to the functionality in notebook 2.flu_dataset_collect.ipynb. Performs the following data integration steps: - Load single-cell RNA-seq BCR data - Parse FASTQ files from bulk BCR sequencing - Merge single-cell and bulk BCR sequence data - Append UMAP coordinates and cell type annotations - Standardize timepoint and cell type information. Bioinformatics domains: [\"BCR repertoire\", \"data integration\", \"single-cell\", \"bulk sequencing\"]. Input data: [\"Single-cell BCR CSV\", \"Bulk BCR FASTQ files\", \"UMAP coordinates\"]. Output results: [\"Integrated BCR dataset CSV\", \"merged sequence data\"]."
        },
        {
          "tool_name": "run_changeo_anarci",
          "description": "运行ChangeO克隆聚类和ANARCI特征提取。这个工具对应notebook 3.ChangeO+ANARCI.ipynb的功能。它执行：1. ChangeO流程：igblast -> MakeDb -> DefineClones -> CreateGermlines 2. ANARCI分析：提取基因使用、CDR区域、SHM等特征"
        },
        {
          "tool_name": "integrate_binding_neutralization_experiments",
          "description": "Integrate antibody binding and neutralization experimental measurements. This tool corresponds to the functionality in notebook 3.5.clone_result.ipynb. Performs the following data processing steps: **Experimental Data Standardization** - Load two batches of antibody functional assay data - Apply thresholds to convert continuous measurements into binary labels (binding+/-, neutralization+/-) - Process binding and neutralization data for multiple influenza strains (H1N1, H3N2) - Standardize antibody nomenclature and batch information. **Data Integration** - Merge replicate measurements from two experimental batches - Prioritize results from more recent batches - Handle missing values and conflicting data. Bioinformatics domains: [\"antibody characterization\", \"functional assays\", \"data integration\"]. Input data: [\"Binding assay Excel\", \"Neutralization assay Excel\", \"antibody annotations\"]. Output results: [\"Standardized binding/neutralization results CSV\", \"binary classifications\"]."
        },
        {
          "tool_name": "integrate_predictions_with_experimental_data",
          "description": "Integrate machine learning prediction results with laboratory measurements. This tool corresponds to the functionality in notebook 4.experiment+prediction.ipynb. Performs the following data integration workflow: **Machine Learning Prediction Data** - Load ensemble prediction results from multiple folds - Binding predictions: MetaBCR model predictions for H1N1/H3N2 strain binding - Neutralization predictions: Multi-fold cross-validated neutralization activity predictions - Pivot prediction scores into wide-format tables. **Experimental Measurement Data** - Load laboratory-measured binding and neutralization activities - Standardize experimental results into binary classifications. **Data Merging** - Merge prediction and experimental data into BCR feature dataset - Add single-cell/bulk data type labels - Retain all BCR sequence features, UMAP coordinates, and cell types. Bioinformatics domains: [\"machine learning\", \"antibody prediction\", \"experimental validation\", \"data integration\"]. Input data: [\"ML prediction CSVs\", \"experimental measurements\", \"BCR features\"]. Output results: [\"Integrated dataset CSV\", \"predictions + experiments + features\"]."
        },
        {
          "tool_name": "draw_main_figures",
          "description": "绘制主图。这个工具对应notebook 5.draw_for_mainfig.ipynb的功能。由于绘图代码复杂，建议直接使用原始notebook。"
        },
        {
          "tool_name": "prepare_cell_tree_data",
          "description": "准备细胞树数据。这个工具对应notebook 6.cell_tree_preliminary.ipynb的功能。"
        },
        {
          "tool_name": "build_cell_tree",
          "description": "构建和可视化细胞树。这个工具对应notebook 7.cell_tree(R).ipynb的功能。需要在R环境中运行。"
        }
      ]
    }
  ],
  "integrateBcrData": [
    {
      "name": "BCR Data Integration Service",
      "priority": 1,
      "description": "Complete BCR data integration workflow service providing comprehensive data integration with UMAP dimensionality reduction, clustering analysis, and cell type annotation. This service offers an all-in-one solution for integrating BCR prediction data with single-cell RNA-seq data, automatically handling Excel to CSV conversion, field standardization, barcode matching, and version control for repeated integrations.",
      "domains": [
        "bcr_integration",
        "single_cell",
        "umap",
        "clustering",
        "cell_type_annotation",
        "data_integration"
      ],
      "inputs": [
        "CSV/Excel files (BCR prediction data)",
        "RDS files (Seurat single-cell RNA-seq data)"
      ],
      "outputs": [
        "Integrated RDS files",
        "UMAP coordinates",
        "Cell clusters",
        "Cell type annotations"
      ],
      "tool": [
        {
          "tool_name": "integrate_bcr_data_complete",
          "description": "Complete BCR data integration workflow with UMAP, clustering, and cell type annotation. This tool provides a comprehensive one-step solution for integrating BCR prediction data (CSV/Excel) with single-cell RNA-seq data (RDS). **Key Features**: - Automatic Excel file detection and conversion to CSV - Intelligent field version control (protects Heavy/Light chains, versionizes prediction fields) - UMAP dimensionality reduction and visualization - FindClusters cell clustering analysis - Cell type annotation based on marker genes (with confidence scores) - Field standardization and barcode matching - Version control for repeated integrations - Complete B cell subset annotation (Naive, Memory, Plasma, GC, etc.). **Parameters**: csv_file (required): CSV/Excel file path containing BCR prediction data; rds_file (required): RDS file path containing Seurat single-cell RNA-seq data; output_file (required): Output integrated RDS file path; csv_fields (optional): CSV field combination for matching (e.g., 'BarCode' or 'Sample,Barcode'); rds_fields (optional): RDS field combination for matching (e.g., 'rownames' or 'orig.ident,barcode'); separator (optional): Field separator (default ''); skip_umap (optional): Skip UMAP dimensionality reduction (default False); skip_annotation (optional): Skip cell type annotation based on marker genes (default False). **Use Cases**: This tool can replace or complement the multi-step FLU BCR workflow (extract_seurat_umap_metadata → integrate_scbcr_bulk_bcr_data) when you need a complete integration solution with built-in UMAP and cell type annotation. It is particularly useful when you have BCR prediction data in CSV/Excel format and want to integrate it with existing Seurat RDS files in a single step."
        }
      ]
    }
  ]
}
}"""
    return tools_json


def get_intelligent_tool_mapping() -> str:
    """
    Returns intelligent tool mapping prompts for mapping generic tool names to specific MCP tool calls

    This function provides a dynamic tool mapping strategy to avoid hardcoding issues
    """
    mapping_prompt = """
## Intelligent Tool Mapping Guide

### Available MCP Tools
The current system provides the following specific MCP tools. Please select intelligently based on task requirements:

**MetaBCR Analysis Tools (bcell_mcp_server):**
- `run_figure2_deg_analysis`: Single-cell RNA-seq differential gene expression analysis
  - Function: B cell type mapping, antigen binding prediction detection, differential expression analysis, volcano plot generation
  - Input: scRNA-seq RDS file (Seurat object)
  - Output: DE gene lists, volcano plots, statistical results, CSV files
  - Corresponding generic tools: Seurat, DESeq2, R Statistical Framework

- `run_figure3_correlation_analysis`: Inter-dataset correlation analysis
  - Function: Correlation analysis of DEG results from different datasets
  - Input: Multiple DEG analysis result directories
  - Output: Correlation matrices, statistical reports
  - Corresponding generic tools: R Statistical Framework, statistical analysis tools

**Figure Analysis Tools (figure_mcp_server):**
- `run_figure2_analysis`: Single-cell differential gene expression analysis and visualization
  - Function: Comprehensive differential expression analysis including cell type annotation and statistical visualization
  - Input: scRNA-seq RDS files
  - Output: DE genes, volcano plots, statistical results
  - Corresponding generic tools: Seurat, Scanpy, DESeq2

- `run_figure3_analysis`: Antigen binding prediction visualization and UMAP density analysis
  - Function: Spatial visualization of binding predictions, density plot generation, correlation analysis
  - Input: RDS files containing binding predictions
  - Output: Density plots, UMAP visualizations, correlation plots
  - Corresponding generic tools: Seurat, Scanpy, visualization tools

- `run_figure4_analysis`: Trajectory analysis and gene module scoring (computationally intensive)
  - Function: B cell differentiation trajectory analysis, pseudotime inference, gene module scoring
  - Input: scRNA-seq RDS files
  - Output: Trajectory plots, pseudotime analysis, gene module scores
  - Corresponding generic tools: Monocle3, Seurat, trajectory analysis tools
  - Note: Execution time 60-120 minutes

- `run_figure5_analysis`: BCR isotype distribution and somatic hypermutation analysis
  - Function: BCR repertoire analysis, isotype switching, affinity maturation analysis
  - Input: RDS files containing BCR data
  - Output: Isotype distribution plots, SHM analysis, statistical comparisons
  - Corresponding generic tools: MetaBCR, BCR analysis tools, statistical analysis

**Lineage Analysis BCR Tools (lineage_analysis service) - CRITICAL EXECUTION ORDER:**

When using FLU-related tools, you MUST follow this specific execution order to ensure all required intermediate outputs are available. However, **each tool call requires user confirmation** - do NOT automatically chain multiple tool calls. Wait for user approval after each step before proceeding to the next one.

**Required Sequential Steps:**
1. **`extract_seurat_umap_metadata`** - Extract Seurat UMAP metadata
   - Function: Extract UMAP coordinates and cellular metadata from Seurat RDS files
   - Input: Seurat RDS file (rds_file_path)
   - Output: UMAP coordinates CSV file (umap_coordinates_path)
   - **MUST be executed FIRST** - This output is required by step 2

2. **`integrate_scbcr_bulk_bcr_data`** - Integrate single-cell and bulk BCR data
   - Function: Merge single-cell BCR and bulk BCR sequencing data
   - Input: 
     * sc_rna_csv_path: Single-cell RNA data CSV
     * bulk_raw_data_dir: Directory containing bulk raw data (FASTQ files)
     * umap_coordinates_path: **REQUIRED** - Output from step 1
   - Output: Integrated BCR dataset CSV
   - **MUST be executed AFTER step 1** - Requires umap_coordinates_path from step 1

3. **`integrate_binding_neutralization_experiments`** - Integrate binding/neutralization experiments
   - Function: Process and merge antibody binding and neutralization experimental measurements
   - Input: 
     * first_experiment_path: First batch experimental data Excel
     * second_experiment_path: Second batch experimental data Excel
   - Output: Standardized binding/neutralization results CSV
   - **Can be executed in parallel with step 2** (no dependencies)

4. **`integrate_predictions_with_experimental_data`** - Integrate ML predictions with experiments
   - Function: Merge machine learning prediction results with laboratory measurements
   - Input:
     * feature_data_path: **REQUIRED** - Output from step 2 (integrated BCR dataset)
     * clone_results_path: **REQUIRED** - Output from step 3 (experimental results)
     * bind_predict_dir: Directory with binding prediction results
     * neu_predict_dir: Directory with neutralization prediction results
   - Output: Complete integrated dataset CSV (predictions + experiments + features)
   - **MUST be executed AFTER steps 2 and 3** - Requires outputs from both previous steps

**Optional Step:**
- `bcr_clonal_clustering_and_feature_extraction` - BCR clonal clustering and feature extraction
  - Requires ChangeO+ANARCI analysis (currently commented out)
  - Can be executed when ChangeO+ANARCI tools are available

**Alternative Complete Integration Tool (integrateBcrData service):**

**`integrate_bcr_data_complete`** - Complete BCR data integration with UMAP, clustering, and annotation
  - **Function**: One-step comprehensive BCR data integration workflow
  - **Input**: 
    * csv_file: CSV/Excel file path (BCR prediction data) - **REQUIRED**
    * rds_file: RDS file path (Seurat single-cell RNA-seq data) - **REQUIRED**
    * output_file: Output integrated RDS file path - **REQUIRED**
    * csv_fields (optional): CSV field combination for matching (e.g., 'BarCode' or 'Sample,Barcode')
    * rds_fields (optional): RDS field combination for matching (e.g., 'rownames' or 'orig.ident,barcode')
    * separator (optional): Field separator (default '')
    * skip_umap (optional): Skip UMAP dimensionality reduction (default False)
    * skip_annotation (optional): Skip cell type annotation (default False)
  - **Output**: Integrated RDS file with UMAP coordinates, clusters, and cell type annotations
  - **Key Features**:
    * Automatic Excel to CSV conversion
    * Intelligent field version control
    * Built-in UMAP dimensionality reduction
    * FindClusters cell clustering analysis
    * Cell type annotation based on marker genes (with confidence scores)
    * Field standardization and barcode matching
    * Complete B cell subset annotation (Naive, Memory, Plasma, GC, etc.)
  - **When to Use**:
    * **Alternative to multi-step workflow**: Can replace steps 1-2 (extract_seurat_umap_metadata + integrate_scbcr_bulk_bcr_data) when you need a complete integration solution
    * **Single-step integration**: Use when you have BCR prediction data in CSV/Excel format and want to integrate with Seurat RDS in one step
    * **Built-in analysis**: Use when you need UMAP, clustering, and cell type annotation as part of the integration process
    * **Excel support**: Automatically handles Excel file conversion
  - **Comparison with FLU workflow**:
    * This tool provides a more streamlined approach for BCR data integration
    * It includes UMAP and cell type annotation built-in, eliminating the need for separate extraction steps
    * Use this tool when you want a complete integration solution rather than step-by-step processing
    * The FLU workflow (steps 1-4) is still recommended when you need fine-grained control over each step or when working with bulk BCR FASTQ files

**CRITICAL RULES:**
- **NEVER skip step 1** - extract_seurat_umap_metadata must run first
- **NEVER execute step 2 without step 1 output** - integrate_scbcr_bulk_bcr_data requires umap_coordinates_path
- **NEVER execute step 4 without steps 2 and 3 outputs** - integrate_predictions_with_experimental_data requires both feature_data_path and clone_results_path

**IMPORTANT**: Although these tools have dependencies and must be executed in order, **EACH tool call still requires user confirmation**. After executing step 1 and receiving user approval, wait for user confirmation before proceeding to step 2. Do NOT automatically chain multiple tool calls - every tool invocation must be individually confirmed by the user.
- **Always check conversation history** for previous tool outputs before calling subsequent tools
- **Use file paths from previous tool outputs** - Do not use merged_csv_result_path for non-CSV parameters (RDS files, directories, etc.)

### Intelligent Mapping Strategy

**1. Task keyword-based mapping:**
- Contains "differential expression", "DE", "differential" → `run_figure2_analysis` or `run_figure2_deg_analysis`
- Contains "visualization", "UMAP", "density", "binding prediction" → `run_figure3_analysis`
- Contains "trajectory", "pseudotime", "differentiation", "trajectory" → `run_figure4_analysis`
- Contains "BCR", "isotype", "mutation", "SHM" → `run_figure5_analysis`
- Contains "correlation", "correlation" → `run_figure3_correlation_analysis`

**2. Input data type-based mapping:**
- Seurat RDS files + expression analysis → Figure analysis tools
- Multiple DEG result directories → `run_figure3_correlation_analysis`
- Data containing BCR sequences → `run_figure5_analysis`
- CSV/Excel BCR prediction data + Seurat RDS files → `integrate_bcr_data_complete` (for complete integration with UMAP and annotation)

**3. Analysis domain-based mapping:**
- Single-cell RNA-seq analysis → Figure analysis tools
- B cell-specific analysis → MetaBCR or Figure tools
- Statistical analysis and visualization → Select based on specific requirements
- BCR data integration with UMAP/clustering/annotation → `integrate_bcr_data_complete` (complete workflow) or FLU workflow (step-by-step)

**4. Parameter adaptation rules:**
- `input_file`: Always use the provided RDS file path
- `base_dir`: Use specified output directory, or default path if not specified
- Automatically detect file formats and data types
- Adjust analysis parameters based on data content

### Tool Selection Decision Flow
1. Analyze keywords and analysis types in task descriptions
2. Determine the format and content of input data
3. Select the most appropriate MCP tool based on expected output
4. If the task involves multiple analysis steps, call multiple tools in logical order
5. Prioritize tools with the best functional match and output that meets requirements

Please intelligently select the most appropriate MCP tool based on the specific task description and ensure correct parameter settings.
"""
    return mapping_prompt


if __name__ == "__main__":
    print(get_tools_json())
