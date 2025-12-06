#!/usr/bin/env Rscript
# Harmony Integration for scRNA-seq data
# Uses Harmony for batch correction

suppressPackageStartupMessages({
  library(Seurat)
  library(harmony)
  library(ggplot2)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript integration_harmony.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
batch_variable <- if(!is.null(params$batch_variable)) params$batch_variable else "orig.ident"
dims <- if(!is.null(params$dims)) params$dims else 30
theta <- if(!is.null(params$theta)) params$theta else c(2.0)

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "integration_harmony")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Check if batch variable exists
if (!batch_variable %in% colnames(seurat_obj@meta.data)) {
  stop(paste("Batch variable", batch_variable, "not found in metadata"))
}

cat("Batch variable:", batch_variable, "\n")
cat("Number of batches:", length(unique(seurat_obj@meta.data[[batch_variable]])), "\n")
cat("Dimensions:", dims, "\n")
cat("Theta:", paste(theta, collapse = ", "), "\n")

# Ensure PCA is computed
if (!"pca" %in% names(seurat_obj@reductions)) {
  cat("PCA not found, computing...\n")
  seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj), verbose = FALSE)
}

# Generate pre-integration UMAP
cat("Generating pre-integration UMAP...\n")
seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:dims, verbose = FALSE)
pdf(file.path(output_dir, "umap_before_harmony.pdf"), width = 12, height = 5)
p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = batch_variable) +
      ggtitle("Before Harmony")
print(p1)
dev.off()

# Run Harmony integration
cat("Running Harmony integration...\n")
seurat_obj <- RunHarmony(seurat_obj,
                         group.by.vars = batch_variable,
                         dims.use = 1:dims,
                         theta = theta,
                         plot_convergence = FALSE,
                         verbose = FALSE)

# Generate post-integration UMAP
cat("Generating post-integration UMAP...\n")
seurat_obj <- RunUMAP(seurat_obj, reduction = "harmony", dims = 1:dims, verbose = FALSE)
pdf(file.path(output_dir, "umap_after_harmony.pdf"), width = 12, height = 5)
p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = batch_variable) +
      ggtitle("After Harmony")
print(p2)
dev.off()

# Combined before/after plot
pdf(file.path(output_dir, "umap_comparison.pdf"), width = 14, height = 5)
# Recompute pre-integration UMAP for comparison
seurat_obj_pre <- seurat_obj
DefaultAssay(seurat_obj_pre) <- "RNA"
seurat_obj_pre <- RunUMAP(seurat_obj_pre, reduction = "pca", dims = 1:dims, verbose = FALSE, reduction.name = "umap_pca")
p1 <- DimPlot(seurat_obj_pre, reduction = "umap_pca", group.by = batch_variable) + ggtitle("Before Harmony")
p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = batch_variable) + ggtitle("After Harmony")
print(p1 + p2)
dev.off()

# Save Harmony embeddings
harmony_embeddings <- Embeddings(seurat_obj, reduction = "harmony")
write.csv(harmony_embeddings[, 1:dims],
          file.path(output_dir, "harmony_embeddings.csv"))

# Save integration statistics
batch_counts <- table(seurat_obj@meta.data[[batch_variable]])
batch_stats <- data.frame(
  batch = names(batch_counts),
  n_cells = as.numeric(batch_counts)
)
write.csv(batch_stats, file.path(output_dir, "batch_statistics.csv"), row.names = FALSE)

integration_stats <- data.frame(
  metric = c("n_cells", "n_batches", "batch_variable", "dims", "theta"),
  value = c(ncol(seurat_obj), length(unique(seurat_obj@meta.data[[batch_variable]])),
            batch_variable, dims, paste(theta, collapse = ", "))
)
write.csv(integration_stats, file.path(output_dir, "integration_statistics.csv"), row.names = FALSE)

# Save integrated Seurat object
output_rds <- file.path(output_dir, "seurat_integrated.rds")
saveRDS(seurat_obj, output_rds)
cat("Integrated Seurat object saved to:", output_rds, "\n")

cat("Harmony integration completed successfully!\n")
