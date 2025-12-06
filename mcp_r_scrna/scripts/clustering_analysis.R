#!/usr/bin/env Rscript
# Clustering Analysis for scRNA-seq data
# Uses Seurat for graph-based clustering (Leiden or Louvain)

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
  stop("Usage: Rscript clustering_analysis.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
resolution <- if(!is.null(params$resolution)) params$resolution else 0.8
dims <- if(!is.null(params$dims)) params$dims else 30
algorithm <- if(!is.null(params$algorithm)) params$algorithm else "louvain"

# Map algorithm name to Seurat integer
algorithm_int <- switch(algorithm,
                        "louvain" = 1,
                        "louvain_refined" = 2,
                        "slm" = 3,
                        "leiden" = 4,
                        1)  # default to louvain

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "clustering_analysis")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Clustering parameters:\n")
cat("  - Algorithm:", algorithm, "\n")
cat("  - Resolution:", resolution, "\n")
cat("  - Dimensions:", dims, "\n")

# Determine which reduction to use
reduction_use <- if ("harmony" %in% names(seurat_obj@reductions)) {
  "harmony"
} else if ("pca" %in% names(seurat_obj@reductions)) {
  "pca"
} else {
  stop("No PCA or Harmony reduction found. Please run normalization or integration first.")
}
cat("Using reduction:", reduction_use, "\n")

# Find neighbors
cat("Building SNN graph...\n")
seurat_obj <- FindNeighbors(seurat_obj,
                           reduction = reduction_use,
                           dims = 1:dims,
                           verbose = FALSE)

# Find clusters
cat("Finding clusters...\n")
seurat_obj <- FindClusters(seurat_obj,
                          resolution = resolution,
                          algorithm = algorithm_int,
                          verbose = FALSE)

n_clusters <- length(unique(Idents(seurat_obj)))
cat("Number of clusters found:", n_clusters, "\n")

# Run UMAP if not present
if (!"umap" %in% names(seurat_obj@reductions)) {
  cat("Computing UMAP...\n")
  seurat_obj <- RunUMAP(seurat_obj,
                       reduction = reduction_use,
                       dims = 1:dims,
                       verbose = FALSE)
}

# Generate UMAP plots
pdf(file.path(output_dir, "umap_clusters.pdf"), width = 10, height = 8)
p1 <- DimPlot(seurat_obj,
              reduction = "umap",
              label = TRUE,
              label.size = 6) +
      ggtitle(paste0("Clustering (", algorithm, ", resolution=", resolution, ")"))
print(p1)
dev.off()

# If multiple samples, show by sample
if ("orig.ident" %in% colnames(seurat_obj@meta.data) &&
    length(unique(seurat_obj$orig.ident)) > 1) {
  pdf(file.path(output_dir, "umap_clusters_by_sample.pdf"), width = 12, height = 8)
  p2 <- DimPlot(seurat_obj,
                reduction = "umap",
                split.by = "orig.ident",
                label = TRUE) +
        ggtitle("Clusters by Sample")
  print(p2)
  dev.off()
}

# Cluster composition
pdf(file.path(output_dir, "cluster_composition.pdf"), width = 10, height = 6)
cluster_counts <- table(Idents(seurat_obj))
barplot(cluster_counts,
        main = "Cells per Cluster",
        xlab = "Cluster",
        ylab = "Number of Cells",
        col = rainbow(n_clusters))
dev.off()

# Save cluster assignments
cluster_assignments <- data.frame(
  cell_barcode = colnames(seurat_obj),
  cluster = as.character(Idents(seurat_obj))
)
write.csv(cluster_assignments, file.path(output_dir, "cluster_assignments.csv"), row.names = FALSE)

# Save cluster statistics
cluster_stats <- as.data.frame(table(Idents(seurat_obj)))
colnames(cluster_stats) <- c("cluster", "n_cells")
cluster_stats$percentage <- round(cluster_stats$n_cells / sum(cluster_stats$n_cells) * 100, 2)
write.csv(cluster_stats, file.path(output_dir, "cluster_statistics.csv"), row.names = FALSE)

# Save clustering parameters
clustering_params <- data.frame(
  metric = c("algorithm", "resolution", "dims", "reduction", "n_clusters"),
  value = c(algorithm, resolution, dims, reduction_use, n_clusters)
)
write.csv(clustering_params, file.path(output_dir, "clustering_parameters.csv"), row.names = FALSE)

# Save clustered Seurat object
output_rds <- file.path(output_dir, "seurat_clustered.rds")
saveRDS(seurat_obj, output_rds)
cat("Clustered Seurat object saved to:", output_rds, "\n")

cat("Clustering analysis completed successfully!\n")
