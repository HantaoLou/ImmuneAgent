#!/usr/bin/env Rscript
# Dimensionality Reduction
# Generates PCA, UMAP, and tSNE embeddings

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)
set.seed(42)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript dim_reduction.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
methods <- if(!is.null(params$methods) && length(params$methods) > 0) {
  params$methods
} else {
  c("PCA", "UMAP", "tSNE")
}
dims <- if(!is.null(params$dims)) params$dims else 30
n_neighbors <- if(!is.null(params$n_neighbors)) params$n_neighbors else 30
min_dist <- if(!is.null(params$min_dist)) params$min_dist else 0.3

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "dim_reduction")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Dimensionality reduction parameters:\n")
cat("  - Methods:", paste(methods, collapse = ", "), "\n")
cat("  - Dimensions:", dims, "\n")
cat("  - UMAP n_neighbors:", n_neighbors, "\n")
cat("  - UMAP min_dist:", min_dist, "\n")

# PCA
if ("PCA" %in% methods) {
  cat("Running PCA...\n")
  seurat_obj <- RunPCA(seurat_obj,
                      features = VariableFeatures(object = seurat_obj),
                      npcs = dims,
                      verbose = FALSE)

  # Save PCA embeddings
  pca_embeddings <- Embeddings(seurat_obj, reduction = "pca")
  write.csv(pca_embeddings, file.path(output_dir, "pca_embeddings.csv"))

  # Elbow plot
  pdf(file.path(output_dir, "pca_elbow_plot.pdf"), width = 10, height = 6)
  print(ElbowPlot(seurat_obj, ndims = dims) +
        ggtitle("PCA Elbow Plot"))
  dev.off()

  # PCA plots
  pdf(file.path(output_dir, "pca_scatter.pdf"), width = 12, height = 5)
  p1 <- DimPlot(seurat_obj, reduction = "pca", dims = c(1, 2)) + ggtitle("PCA: PC1 vs PC2")
  p2 <- DimPlot(seurat_obj, reduction = "pca", dims = c(2, 3)) + ggtitle("PCA: PC2 vs PC3")
  print(p1 + p2)
  dev.off()

  # PCA loadings
  pdf(file.path(output_dir, "pca_loadings.pdf"), width = 14, height = 10)
  print(VizDimLoadings(seurat_obj, dims = 1:9, reduction = "pca", ncol = 3))
  dev.off()

  # PCA heatmap
  pdf(file.path(output_dir, "pca_heatmap.pdf"), width = 12, height = 14)
  print(DimHeatmap(seurat_obj, dims = 1:15, cells = 500, balanced = TRUE))
  dev.off()

  cat("PCA completed\n")
}

# UMAP
if ("UMAP" %in% methods) {
  cat("Running UMAP...\n")

  # Determine which reduction to use
  reduction_use <- if ("harmony" %in% names(seurat_obj@reductions)) {
    "harmony"
  } else if ("pca" %in% names(seurat_obj@reductions)) {
    "pca"
  } else {
    stop("No PCA or Harmony reduction found")
  }

  seurat_obj <- RunUMAP(seurat_obj,
                       reduction = reduction_use,
                       dims = 1:dims,
                       n.neighbors = n_neighbors,
                       min.dist = min_dist,
                       verbose = FALSE)

  # Save UMAP embeddings
  umap_embeddings <- Embeddings(seurat_obj, reduction = "umap")
  write.csv(umap_embeddings, file.path(output_dir, "umap_embeddings.csv"))

  # UMAP plots
  pdf(file.path(output_dir, "umap_plot.pdf"), width = 10, height = 8)
  print(DimPlot(seurat_obj, reduction = "umap") +
        ggtitle(paste("UMAP (from", reduction_use, ")")))
  dev.off()

  # UMAP by sample if applicable
  if ("orig.ident" %in% colnames(seurat_obj@meta.data) &&
      length(unique(seurat_obj$orig.ident)) > 1) {
    pdf(file.path(output_dir, "umap_by_sample.pdf"), width = 12, height = 5)
    p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = "orig.ident") +
          ggtitle("UMAP by Sample")
    p2 <- DimPlot(seurat_obj, reduction = "umap", split.by = "orig.ident") +
          ggtitle("UMAP Split by Sample")
    print(p1)
    dev.off()

    pdf(file.path(output_dir, "umap_split_by_sample.pdf"), width = 14, height = 5)
    print(p2)
    dev.off()
  }

  # UMAP by clusters if present
  if ("seurat_clusters" %in% colnames(seurat_obj@meta.data)) {
    pdf(file.path(output_dir, "umap_clusters.pdf"), width = 10, height = 8)
    print(DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters", label = TRUE) +
          ggtitle("UMAP with Clusters"))
    dev.off()
  }

  cat("UMAP completed\n")
}

# tSNE
if ("tSNE" %in% methods) {
  cat("Running tSNE...\n")

  # Determine which reduction to use
  reduction_use <- if ("harmony" %in% names(seurat_obj@reductions)) {
    "harmony"
  } else if ("pca" %in% names(seurat_obj@reductions)) {
    "pca"
  } else {
    stop("No PCA or Harmony reduction found")
  }

  seurat_obj <- RunTSNE(seurat_obj,
                       reduction = reduction_use,
                       dims = 1:dims,
                       verbose = FALSE)

  # Save tSNE embeddings
  tsne_embeddings <- Embeddings(seurat_obj, reduction = "tsne")
  write.csv(tsne_embeddings, file.path(output_dir, "tsne_embeddings.csv"))

  # tSNE plots
  pdf(file.path(output_dir, "tsne_plot.pdf"), width = 10, height = 8)
  print(DimPlot(seurat_obj, reduction = "tsne") +
        ggtitle(paste("tSNE (from", reduction_use, ")")))
  dev.off()

  # tSNE by sample if applicable
  if ("orig.ident" %in% colnames(seurat_obj@meta.data) &&
      length(unique(seurat_obj$orig.ident)) > 1) {
    pdf(file.path(output_dir, "tsne_by_sample.pdf"), width = 12, height = 8)
    print(DimPlot(seurat_obj, reduction = "tsne", group.by = "orig.ident") +
          ggtitle("tSNE by Sample"))
    dev.off()
  }

  # tSNE by clusters if present
  if ("seurat_clusters" %in% colnames(seurat_obj@meta.data)) {
    pdf(file.path(output_dir, "tsne_clusters.pdf"), width = 10, height = 8)
    print(DimPlot(seurat_obj, reduction = "tsne", group.by = "seurat_clusters", label = TRUE) +
          ggtitle("tSNE with Clusters"))
    dev.off()
  }

  cat("tSNE completed\n")
}

# Comparison plot (if multiple methods)
if (length(methods) > 1 && all(c("UMAP", "tSNE") %in% methods)) {
  pdf(file.path(output_dir, "comparison_umap_tsne.pdf"), width = 14, height = 6)
  p1 <- DimPlot(seurat_obj, reduction = "umap") + ggtitle("UMAP")
  p2 <- DimPlot(seurat_obj, reduction = "tsne") + ggtitle("tSNE")
  print(p1 + p2)
  dev.off()
}

# Save reduction statistics
reduction_stats <- data.frame(
  metric = c("methods", "dims", "n_neighbors", "min_dist", "n_cells", "n_genes"),
  value = c(paste(methods, collapse = ", "), dims, n_neighbors, min_dist,
            ncol(seurat_obj), nrow(seurat_obj))
)
write.csv(reduction_stats, file.path(output_dir, "reduction_statistics.csv"), row.names = FALSE)

# Save Seurat object with embeddings
output_rds <- file.path(output_dir, "seurat_with_reductions.rds")
saveRDS(seurat_obj, output_rds)
cat("Seurat object with reductions saved to:", output_rds, "\n")

cat("Dimensionality reduction completed successfully!\n")
cat("Methods completed:", paste(methods, collapse = ", "), "\n")
