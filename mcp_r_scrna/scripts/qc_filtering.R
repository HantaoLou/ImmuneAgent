#!/usr/bin/env Rscript
# QC Filtering for scRNA-seq data
# Uses Seurat for quality control and cell filtering

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript qc_filtering.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
min_genes <- if(!is.null(params$min_genes)) params$min_genes else 200
max_genes <- if(!is.null(params$max_genes)) params$max_genes else 6000
min_counts <- if(!is.null(params$min_counts)) params$min_counts else 1000
mt_percent <- if(!is.null(params$mt_percent)) params$mt_percent else 20.0

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Calculate mitochondrial percentage if not already present
if (!"percent.mt" %in% colnames(seurat_obj@meta.data)) {
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
}

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "qc_filtering")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Pre-filtering statistics
n_cells_before <- ncol(seurat_obj)
cat("Cells before filtering:", n_cells_before, "\n")

# Generate pre-filtering QC plots
pdf(file.path(output_dir, "qc_prefiltering_violin.pdf"), width = 12, height = 6)
VlnPlot(seurat_obj,
        features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
        ncol = 3,
        pt.size = 0.1)
dev.off()

pdf(file.path(output_dir, "qc_prefiltering_scatter.pdf"), width = 12, height = 6)
p1 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
p2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
print(p1 + p2)
dev.off()

# Apply filtering
cat("Applying filters:\n")
cat("  - nFeature_RNA:", min_genes, "to", max_genes, "\n")
cat("  - nCount_RNA: >", min_counts, "\n")
cat("  - percent.mt: <", mt_percent, "\n")

seurat_obj <- subset(seurat_obj,
                     subset = nFeature_RNA > min_genes &
                              nFeature_RNA < max_genes &
                              nCount_RNA > min_counts &
                              percent.mt < mt_percent)

# Post-filtering statistics
n_cells_after <- ncol(seurat_obj)
cat("Cells after filtering:", n_cells_after, "\n")
cat("Cells removed:", n_cells_before - n_cells_after, "\n")
cat("Retention rate:", round(n_cells_after/n_cells_before*100, 2), "%\n")

# Generate post-filtering QC plots
pdf(file.path(output_dir, "qc_postfiltering_violin.pdf"), width = 12, height = 6)
VlnPlot(seurat_obj,
        features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
        ncol = 3,
        pt.size = 0.1)
dev.off()

pdf(file.path(output_dir, "qc_postfiltering_scatter.pdf"), width = 12, height = 6)
p1 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "percent.mt")
p2 <- FeatureScatter(seurat_obj, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
print(p1 + p2)
dev.off()

# Save QC statistics
qc_stats <- data.frame(
  metric = c("cells_before", "cells_after", "cells_removed", "retention_rate_percent",
             "min_genes", "max_genes", "min_counts", "max_mt_percent"),
  value = c(n_cells_before, n_cells_after, n_cells_before - n_cells_after,
            round(n_cells_after/n_cells_before*100, 2),
            min_genes, max_genes, min_counts, mt_percent)
)
write.csv(qc_stats, file.path(output_dir, "qc_statistics.csv"), row.names = FALSE)

# Save filtered Seurat object
output_rds <- file.path(output_dir, "seurat_filtered.rds")
saveRDS(seurat_obj, output_rds)
cat("Filtered Seurat object saved to:", output_rds, "\n")

cat("QC filtering completed successfully!\n")
