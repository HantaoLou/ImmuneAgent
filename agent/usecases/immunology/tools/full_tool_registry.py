"""
Complete Tool Registry for ImmuneAgent.
Contains all 84+ bioinformatics tools across 11 categories.
This expands the execution_tools.py registry to include all tools.
"""

from typing import Any, Dict, List

# Complete 84+ tool registry
FULL_TOOL_REGISTRY = {
    # Previous categories from execution_tools.py are maintained
    # Adding the missing categories and tools below:
    # TCR/BCR Repertoire Analysis Tools (8 tools)
    "repertoire_tools": {
        "mixcr": {
            "description": "TCR/BCR repertoire analysis from NGS",
            "input": "FASTQ/FASTA files",
            "output": "Clonotypes, V(D)J usage, diversity metrics",
            "runtime": "30-120 minutes",
            "mcp_service": "mixcr",
        },
        "igblast": {
            "description": "NCBI tool for immunoglobulin analysis",
            "input": "Ig/TCR sequences",
            "output": "V(D)J assignments, mutations",
            "runtime": "5-15 minutes",
        },
        "changeo": {
            "description": "B cell repertoire analysis toolkit",
            "input": "AIRR-seq data",
            "output": "Clonal families, SHM patterns",
            "runtime": "20-60 minutes",
        },
        "immunarch": {
            "description": "R package for repertoire analysis",
            "input": "Clonotype tables",
            "output": "Diversity, overlap, dynamics",
            "runtime": "10-30 minutes",
        },
        "tcrdist3": {
            "description": "TCR similarity and clustering",
            "input": "TCR sequences",
            "output": "Distance matrices, clusters",
            "runtime": "15-45 minutes",
        },
        "gliph2": {
            "description": "Groups lymphocyte interactions by paratope hotspots",
            "input": "TCR sequences",
            "output": "Specificity groups, motifs",
            "runtime": "10-30 minutes",
        },
        "deeptcr": {
            "description": "Deep learning for TCR analysis",
            "input": "TCR sequences",
            "output": "Repertoire classification, antigen prediction",
            "runtime": "20-60 minutes",
        },
        "trust4": {
            "description": "TCR/BCR assembly from RNA-seq",
            "input": "RNA-seq BAM/FASTQ",
            "output": "Assembled TCR/BCR sequences",
            "runtime": "1-3 hours",
        },
    },
    # Epitope & MHC Prediction Tools (7 tools)
    "epitope_tools": {
        "netmhcpan": {
            "description": "Pan-specific MHC binding prediction",
            "input": "Peptide sequences, HLA alleles",
            "output": "Binding affinity, strong/weak binders",
            "runtime": "2-10 minutes",
            "mcp_service": "netmhcpan",
        },
        "pvactools": {
            "description": "Personalized vaccine design pipeline",
            "input": "VCF files, expression data",
            "output": "Neoantigen candidates, vaccine design",
            "runtime": "1-4 hours",
        },
        "mixmhc2pred": {
            "description": "MHC-II binding prediction",
            "input": "Peptide sequences",
            "output": "MHC-II binding scores",
            "runtime": "5-15 minutes",
        },
        "iedb": {
            "description": "Immune epitope database analysis",
            "input": "Sequences or epitopes",
            "output": "Known epitopes, T/B cell assays",
            "runtime": "1-5 minutes",
        },
        "mhcflurry": {
            "description": "Neural network MHC-I prediction",
            "input": "Peptides, alleles",
            "output": "Presentation scores",
            "runtime": "2-10 minutes",
        },
        "prime": {
            "description": "Immunogenicity prediction",
            "input": "Peptide-MHC complexes",
            "output": "T cell response probability",
            "runtime": "5-15 minutes",
        },
        "repitope": {
            "description": "T cell epitope prediction from TCR",
            "input": "TCR sequences",
            "output": "Predicted epitope targets",
            "runtime": "10-30 minutes",
        },
    },
    # Molecular Dynamics & Docking Tools (7 tools)
    "dynamics_tools": {
        "haddock": {
            "description": "Data-driven protein-protein docking",
            "input": "PDB structures, restraints",
            "output": "Docked complexes, scores",
            "runtime": "2-8 hours",
            "mcp_service": "haddock",
        },
        "autodock": {
            "description": "Molecular docking suite",
            "input": "Receptor, ligand structures",
            "output": "Binding poses, affinities",
            "runtime": "30-120 minutes",
        },
        "gromacs": {
            "description": "Molecular dynamics simulations",
            "input": "PDB structures, force fields",
            "output": "Trajectories, energies",
            "runtime": "hours to days",
        },
        "amber": {
            "description": "Biomolecular simulations",
            "input": "Molecular structures",
            "output": "MD trajectories, free energies",
            "runtime": "hours to days",
        },
        "zdock": {
            "description": "Protein-protein docking",
            "input": "Two protein structures",
            "output": "Top docking poses",
            "runtime": "1-4 hours",
        },
        "piper": {
            "description": "FFT-based protein docking",
            "input": "Protein structures",
            "output": "Docked conformations",
            "runtime": "2-6 hours",
        },
        "rosetta_dock": {
            "description": "High-resolution protein docking",
            "input": "Protein structures",
            "output": "Refined complexes",
            "runtime": "4-12 hours",
        },
    },
    # Genomics & Variant Analysis Tools (8 tools)
    "genomics_tools": {
        "gatk": {
            "description": "Genome analysis toolkit",
            "input": "BAM/VCF files",
            "output": "Variant calls, annotations",
            "runtime": "1-8 hours",
        },
        "vep": {
            "description": "Variant effect predictor",
            "input": "VCF files",
            "output": "Functional consequences",
            "runtime": "30-120 minutes",
        },
        "annovar": {
            "description": "Variant annotation",
            "input": "Variant lists",
            "output": "Gene-based annotations",
            "runtime": "10-60 minutes",
        },
        "snpeff": {
            "description": "Variant annotation and effect prediction",
            "input": "VCF files",
            "output": "Annotated variants",
            "runtime": "15-60 minutes",
        },
        "mutect2": {
            "description": "Somatic variant caller",
            "input": "Tumor/normal BAMs",
            "output": "Somatic mutations",
            "runtime": "2-12 hours",
        },
        "strelka2": {
            "description": "Small variant caller",
            "input": "BAM files",
            "output": "SNVs and indels",
            "runtime": "1-6 hours",
        },
        "manta": {
            "description": "Structural variant caller",
            "input": "BAM files",
            "output": "SVs and indels",
            "runtime": "1-4 hours",
        },
        "delly": {
            "description": "Structural variant discovery",
            "input": "BAM files",
            "output": "Deletions, duplications, inversions",
            "runtime": "2-8 hours",
        },
    },
    # Machine Learning & AI Tools (5 tools)
    "ml_tools": {
        "deeplift": {
            "description": "Deep learning feature importance",
            "input": "Trained models, sequences",
            "output": "Importance scores",
            "runtime": "5-30 minutes",
        },
        "xgboost": {
            "description": "Gradient boosting framework",
            "input": "Feature matrices",
            "output": "Predictions, feature importance",
            "runtime": "5-60 minutes",
        },
        "autosklearn": {
            "description": "Automated machine learning",
            "input": "Training data",
            "output": "Optimized models",
            "runtime": "1-12 hours",
        },
        "deepimmuno": {
            "description": "Deep learning for immunogenicity",
            "input": "Peptide sequences",
            "output": "Immunogenicity predictions",
            "runtime": "10-30 minutes",
        },
        "protbert": {
            "description": "Protein language model",
            "input": "Protein sequences",
            "output": "Embeddings, predictions",
            "runtime": "5-20 minutes",
        },
    },
    # Spatial Transcriptomics Tools (5 tools)
    "spatial_tools": {
        "squidpy": {
            "description": "Spatial single-cell analysis",
            "input": "Spatial expression data",
            "output": "Spatial patterns, neighborhoods",
            "runtime": "30-120 minutes",
        },
        "stlearn": {
            "description": "Spatial trajectory inference",
            "input": "ST data",
            "output": "Spatial trajectories, clusters",
            "runtime": "20-90 minutes",
        },
        "giotto": {
            "description": "Spatial data analysis framework",
            "input": "Spatial expression matrices",
            "output": "Spatial domains, interactions",
            "runtime": "30-180 minutes",
        },
        "bayesspace": {
            "description": "Bayesian spatial clustering",
            "input": "Spatial transcriptomics",
            "output": "Enhanced resolution clusters",
            "runtime": "1-4 hours",
        },
        "spagcn": {
            "description": "Graph convolutional network for ST",
            "input": "Spatial gene expression",
            "output": "Spatial domains",
            "runtime": "30-90 minutes",
        },
    },
    # Proteomics & Mass Spec Tools (4 tools)
    "proteomics_tools": {
        "maxquant": {
            "description": "Quantitative proteomics",
            "input": "MS raw files",
            "output": "Protein identifications, quantities",
            "runtime": "2-24 hours",
        },
        "perseus": {
            "description": "Proteomics data analysis",
            "input": "MaxQuant output",
            "output": "Statistical analysis, visualizations",
            "runtime": "30-120 minutes",
        },
        "msfragger": {
            "description": "Ultrafast proteomics search",
            "input": "MS/MS spectra",
            "output": "Peptide-spectrum matches",
            "runtime": "30-180 minutes",
        },
        "spectronaut": {
            "description": "DIA proteomics analysis",
            "input": "DIA raw files",
            "output": "Protein quantities",
            "runtime": "2-12 hours",
        },
    },
    # Flow & Mass Cytometry Tools (4 tools)
    "cytometry_tools": {
        "flowsom": {
            "description": "Self-organizing maps for cytometry",
            "input": "FCS files",
            "output": "Cell clusters, metaclusters",
            "runtime": "5-30 minutes",
        },
        "cytofkit": {
            "description": "Mass cytometry analysis",
            "input": "CyTOF data",
            "output": "Clusters, visualizations",
            "runtime": "15-60 minutes",
        },
        "phenograph": {
            "description": "High-dimensional clustering",
            "input": "Single-cell data",
            "output": "Community detection",
            "runtime": "10-45 minutes",
        },
        "citrus": {
            "description": "Cluster identification and characterization",
            "input": "Cytometry data",
            "output": "Stratifying clusters",
            "runtime": "30-120 minutes",
        },
    },
    # Pathway & Network Analysis Tools (5 tools) - BONUS CATEGORY
    "pathway_tools": {
        "gsea": {
            "description": "Gene set enrichment analysis",
            "input": "Expression data, gene sets",
            "output": "Enriched pathways",
            "runtime": "10-30 minutes",
        },
        "string": {
            "description": "Protein-protein interaction networks",
            "input": "Protein/gene lists",
            "output": "Interaction networks",
            "runtime": "1-5 minutes",
        },
        "cytoscape": {
            "description": "Network visualization and analysis",
            "input": "Network files",
            "output": "Network visualizations",
            "runtime": "5-30 minutes",
        },
        "pathview": {
            "description": "Pathway visualization",
            "input": "Gene expression data",
            "output": "Pathway maps",
            "runtime": "5-15 minutes",
        },
        "reactome": {
            "description": "Pathway database analysis",
            "input": "Gene lists",
            "output": "Pathway annotations",
            "runtime": "2-10 minutes",
        },
    },
    # Metabolism Analysis Tools (2 tools) - BONUS CATEGORY
    "metabolism_tools": {
        "compass": {
            "description": "Metabolic flux analysis from scRNA-seq",
            "input": "Expression matrices",
            "output": "Metabolic states",
            "runtime": "1-4 hours",
        },
        "scmetabolism": {
            "description": "Single-cell metabolic analysis",
            "input": "scRNA-seq data",
            "output": "Metabolic pathway activities",
            "runtime": "30-90 minutes",
        },
    },
}


def merge_with_existing_registry(existing_registry: Dict) -> Dict:
    """
    Merge the full registry with existing registry from execution_tools.py

    Args:
        existing_registry: Current TOOL_REGISTRY from execution_tools.py

    Returns:
        Complete merged registry with all 84+ tools
    """
    # Start with existing registry
    merged = existing_registry.copy()

    # Add all new categories and tools
    for category, tools in FULL_TOOL_REGISTRY.items():
        if category not in merged:
            merged[category] = tools
        else:
            # Merge tools within category
            merged[category].update(tools)

    return merged


def get_registry_statistics() -> Dict[str, Any]:
    """Get statistics about the tool registry."""
    stats = {
        "total_categories": len(FULL_TOOL_REGISTRY),
        "total_tools": sum(len(tools) for tools in FULL_TOOL_REGISTRY.values()),
        "categories": {},
    }

    for category, tools in FULL_TOOL_REGISTRY.items():
        stats["categories"][category] = {
            "count": len(tools),
            "tools": list(tools.keys()),
        }

    return stats


def get_tools_for_analysis_type(analysis_type: str) -> List[str]:
    """
    Get recommended tools for a specific analysis type.

    Args:
        analysis_type: Type of analysis (e.g., "antibody_discovery", "tcr_analysis")

    Returns:
        List of recommended tool names
    """
    recommendations = {
        "antibody_discovery": [
            "metabcr",
            "sapiens",
            "abnumber",
            "igfold",
            "alphafold3",
            "haddock",
            "mixcr",
            "changeo",
        ],
        "tcr_analysis": [
            "mixcr",
            "tcrdist3",
            "gliph2",
            "immunarch",
            "deeptcr",
            "trust4",
            "netmhcpan",
            "repitope",
        ],
        "single_cell": [
            "scanpy",
            "seurat",
            "celltypist",
            "scvi_tools",
            "cellphonedb",
            "nichenet",
            "scenic",
            "scirpy",
        ],
        "tumor_immunology": [
            "pvactools",
            "netmhcpan",
            "mutect2",
            "vep",
            "mixmhc2pred",
            "mhcflurry",
            "infercnv",
        ],
        "spatial_analysis": [
            "squidpy",
            "stlearn",
            "giotto",
            "bayesspace",
            "spagcn",
            "cellphonedb",
        ],
        "protein_engineering": [
            "alphafold3",
            "rosettafold",
            "foldx",
            "haddock",
            "autodock",
            "gromacs",
            "rosetta_dock",
        ],
    }

    return recommendations.get(analysis_type, [])


# Export complete registry
__all__ = [
    "FULL_TOOL_REGISTRY",
    "merge_with_existing_registry",
    "get_registry_statistics",
    "get_tools_for_analysis_type",
]
