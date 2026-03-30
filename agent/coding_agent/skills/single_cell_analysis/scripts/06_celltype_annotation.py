"""
06_celltype_annotation.py — Cell type annotation by marker gene scoring.
=========================================================================
Scores cells against canonical marker gene sets for major immune lineages,
then assigns each cell a predicted lineage label based on the highest score.

This is a REFERENCE script. The agent should adapt marker gene lists based
on species, tissue type, or analysis focus (e.g., adding plasma cell markers
for a B-cell-focused study).

Input contract:
    --input   Path to h5ad file (typically clustered_adata.h5ad from 05_clustering)
    --output  Directory to write annotated adata
    --params  JSON string:
              {
                  "species": "human",        (### ADAPT ### human or mouse)
                  "custom_markers": {}        (optional overrides per lineage)
              }

Output contract:
    <output_dir>/annotated_adata.h5ad    — AnnData with obs["predicted_lineage"]
    <output_dir>/lineage_summary.json    — Per-lineage cell counts and per-cluster breakdown
"""

import argparse
import json
import os
import warnings

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore", category=UserWarning)

### ADAPT ### — marker gene lists are species-dependent
# Human markers use uppercase gene symbols (CD3D, MS4A1, etc.)
# Mouse markers use title-case gene symbols (Cd3d, Ms4a1, etc.)
# Tissue-specific or disease-specific markers may need to be added
HUMAN_MARKERS = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2"],
    "CD4_T_cell": ["CD4", "IL7R", "CCR7", "LEF1"],
    "CD8_T_cell": ["CD8A", "CD8B", "GZMK", "GZMB"],
    "B_cell": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Plasma_cell": ["JCHAIN", "MZB1", "SDC1", "XBP1"],
    "NK_cell": ["NCAM1", "NKG7", "GNLY", "KLRD1", "KLRB1"],
    "Monocyte": ["CD14", "LYZ", "S100A8", "S100A9", "FCGR3A"],
    "Macrophage": ["CD68", "CD163", "MRC1", "MARCO"],
    "DC": ["FCER1A", "CST3", "CLEC9A", "CD1C", "CLEC4C"],
    "Granulocyte": ["S100A8", "CSF3R", "ELANE", "MPO"],
}

MOUSE_MARKERS = {
    "T_cell": ["Cd3d", "Cd3e", "Cd3g", "Cd2"],
    "CD4_T_cell": ["Cd4", "Il7r", "Ccr7", "Lef1"],
    "CD8_T_cell": ["Cd8a", "Cd8b1", "Gzmk", "Gzmb"],
    "B_cell": ["Cd19", "Ms4a1", "Cd79a", "Cd79b"],
    "Plasma_cell": ["Jchain", "Mzb1", "Sdc1", "Xbp1"],
    "NK_cell": ["Ncam1", "Nkg7", "Gnly", "Klrd1", "Klrb1c"],
    "Monocyte": ["Cd14", "Lyz2", "S100a8", "S100a9", "Fcgr3"],
    "Macrophage": ["Cd68", "Cd163", "Mrc1", "Marco"],
    "DC": ["Fcer1a", "Cst3", "Clec9a", "Cd1d1", "Siglech"],
    "Granulocyte": ["S100a8", "Csf3r", "Elane", "Mpo"],
}


def get_markers(species: str, custom_markers: dict = None) -> dict:
    """Select marker gene set for the given species, with optional overrides."""
    if species.lower() == "mouse":
        markers = {k: list(v) for k, v in MOUSE_MARKERS.items()}
    else:
        markers = {k: list(v) for k, v in HUMAN_MARKERS.items()}

    if custom_markers:
        for lineage, genes in custom_markers.items():
            markers[lineage] = genes

    return markers


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    params = params or {}
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    print(f"Input: {adata.n_obs} cells x {adata.n_vars} genes")

    ### ADAPT ### — species selection determines marker gene casing
    species = params.get("species", "human")
    custom_markers = params.get("custom_markers", {})
    markers = get_markers(species, custom_markers)

    gene_set = set(adata.var_names)
    scored_lineages = []

    for lineage, genes in markers.items():
        present = [g for g in genes if g in gene_set]
        if len(present) < 2:
            print(f"  {lineage}: skipped ({len(present)}/{len(genes)} markers found)")
            continue

        score_name = f"score_{lineage}"
        sc.tl.score_genes(adata, gene_list=present, score_name=score_name)
        scored_lineages.append((lineage, score_name))
        print(f"  {lineage}: scored with {len(present)}/{len(genes)} markers")

    if not scored_lineages:
        print("WARNING: No lineages could be scored. Check species/marker compatibility.")
        adata.obs["predicted_lineage"] = "Unknown"
    else:
        score_cols = [s for _, s in scored_lineages]
        lineage_names = [l for l, _ in scored_lineages]

        score_matrix = adata.obs[score_cols].values
        best_idx = np.argmax(score_matrix, axis=1)
        best_score = np.max(score_matrix, axis=1)

        labels = []
        for idx, score in zip(best_idx, best_score):
            if score > 0:
                labels.append(lineage_names[idx])
            else:
                labels.append("Unknown")

        adata.obs["predicted_lineage"] = labels

    lineage_counts = adata.obs["predicted_lineage"].value_counts().to_dict()
    print(f"\nLineage distribution:")
    for lineage, count in sorted(lineage_counts.items(), key=lambda x: -x[1]):
        print(f"  {lineage}: {count} cells")

    cluster_lineage = {}
    if "leiden" in adata.obs.columns:
        for cluster in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
            mask = adata.obs["leiden"] == cluster
            dist = adata.obs.loc[mask, "predicted_lineage"].value_counts().to_dict()
            cluster_lineage[str(cluster)] = {str(k): int(v) for k, v in dist.items()}

    out_path = os.path.join(output_dir, "annotated_adata.h5ad")
    adata.write_h5ad(out_path)

    lineage_summary = {
        "lineage_counts": {str(k): int(v) for k, v in lineage_counts.items()},
        "cluster_lineage_distribution": cluster_lineage,
        "scored_lineages": [l for l, _ in scored_lineages],
        "species": species,
    }

    summary_path = os.path.join(output_dir, "lineage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(lineage_summary, f, indent=2)

    report = {
        "status": "success",
        "n_cells": int(adata.n_obs),
        "n_lineages_scored": len(scored_lineages),
        "lineage_counts": {str(k): int(v) for k, v in lineage_counts.items()},
        "species": species,
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "annotation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="06_celltype_annotation: Marker-based lineage scoring")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
