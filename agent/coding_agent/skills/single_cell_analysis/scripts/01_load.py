"""
01_load.py — Load single-cell data into AnnData format.
=========================================================
Supports h5ad, RDS (via sceasy R conversion), CSV, and 10x MTX formats.
The agent may need to adapt the format detection and loading logic when
the input doesn't match any of the expected formats.

Input contract:
    --input   Path to a single-cell data file (.h5ad, .rds, .csv, .mtx dir)
    --output  Directory to write the loaded adata (saves as loaded_adata.h5ad)
    --params  JSON string: {"format": "auto"} (optional override)

Output contract:
    <output_dir>/loaded_adata.h5ad  — AnnData object with raw expression matrix
    <output_dir>/load_report.json   — Loading summary (n_cells, n_genes, format)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import scanpy as sc


### ADAPT ### — format detection logic; add new extensions or magic-byte checks here
FORMAT_MAP = {
    ".h5ad": "h5ad",
    ".rds": "rds",
    ".robj": "rds",
    ".csv": "csv",
    ".csv.gz": "csv",
    ".tsv": "tsv",
    ".mtx": "10x_mtx",
    ".mtx.gz": "10x_mtx",
}


def detect_format(input_path: str, params: dict) -> str:
    """Detect file format from extension or explicit override."""
    override = params.get("format")
    if override and override != "auto":
        return override

    name = Path(input_path).name.lower()
    for ext in sorted(FORMAT_MAP.keys(), key=len, reverse=True):
        if name.endswith(ext):
            return FORMAT_MAP[ext]

    raise ValueError(
        f"Cannot detect format for '{input_path}'. "
        f"Supported extensions: {list(FORMAT_MAP.keys())}. "
        f"Pass --params '{{\"format\": \"h5ad\"}}' to override."
    )


### ADAPT ### — file loading; add new loaders or adjust existing ones here
def load_h5ad(input_path: str) -> sc.AnnData:
    return sc.read_h5ad(input_path)


def load_rds(input_path: str, output_dir: str) -> sc.AnnData:
    """Convert RDS (Seurat/SCE) to h5ad via sceasy, then load."""
    tmp_h5ad = os.path.join(output_dir, "_converted_from_rds.h5ad")
    r_script = f'''
library(sceasy)
library(Seurat)
obj <- readRDS("{input_path}")
if (inherits(obj, "Seurat")) {{
    sceasy::convertFormat(obj, from="seurat", to="anndata",
                          outFile="{tmp_h5ad}")
}} else if (inherits(obj, "SingleCellExperiment")) {{
    sceasy::convertFormat(obj, from="sce", to="anndata",
                          outFile="{tmp_h5ad}")
}} else {{
    stop(paste("Unsupported R object class:", class(obj)))
}}
'''
    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RDS conversion failed:\n{result.stderr[:1000]}\n"
            f"Ensure R, Seurat, and sceasy are installed."
        )
    return sc.read_h5ad(tmp_h5ad)


def load_csv(input_path: str) -> sc.AnnData:
    return sc.read_csv(input_path)


def load_10x_mtx(input_path: str) -> sc.AnnData:
    """Load 10x Genomics MTX directory.

    Expects a directory containing matrix.mtx(.gz), barcodes.tsv(.gz),
    and features.tsv(.gz) or genes.tsv(.gz).
    """
    mtx_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    return sc.read_10x_mtx(mtx_dir)


def main(input_path: str, output_dir: str, params: dict = None) -> dict:
    params = params or {}
    os.makedirs(output_dir, exist_ok=True)

    fmt = detect_format(input_path, params)
    print(f"Detected format: {fmt}")

    ### ADAPT ### — dispatch to the right loader
    if fmt == "h5ad":
        adata = load_h5ad(input_path)
    elif fmt == "rds":
        adata = load_rds(input_path, output_dir)
    elif fmt in ("csv", "tsv"):
        adata = load_csv(input_path)
    elif fmt == "10x_mtx":
        adata = load_10x_mtx(input_path)
    else:
        raise ValueError(f"No loader for format: {fmt}")

    adata.var_names_make_unique()
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    out_path = os.path.join(output_dir, "loaded_adata.h5ad")
    adata.write_h5ad(out_path)

    report = {
        "status": "success",
        "format": fmt,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "output_file": out_path,
    }

    report_path = os.path.join(output_dir, "load_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="01_load: Load single-cell data")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default="{}", help="JSON string of parameters")
    args = parser.parse_args()

    params = json.loads(args.params)
    result = main(args.input, args.output, params)
    print(json.dumps(result, indent=2))
