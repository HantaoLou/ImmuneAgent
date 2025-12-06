#!/usr/bin/env Rscript
# SCTransform Normalization for scRNA-seq data
# Uses Seurat's SCTransform for variance stabilization and normalization

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
  stop("Usage: Rscript normalization_sct.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
vars_to_regress <- if(!is.null(params$vars_to_regress) && length(params$vars_to_regress) > 0) {
  params$vars_to_regress
} else {
  NULL
}
n_variable_features <- if(!is.null(params$n_variable_features)) params$n_variable_features else 3000

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "normalization_sct")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Run SCTransform
cat("Running SCTransform normalization...\n")
cat("  - Variable features:", n_variable_features, "\n")
if (!is.null(vars_to_regress)) {
  cat("  - Regressing out:", paste(vars_to_regress, collapse = ", "), "\n")
}

seurat_obj <- SCTransform(seurat_obj,
                          vars.to.regress = vars_to_regress,
                          variable.features.n = n_variable_features,
                          verbose = FALSE)

# Run PCA
cat("Running PCA...\n")
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj), verbose = FALSE)

# Generate diagnostic plots
# Elbow plot
pdf(file.path(output_dir, "elbow_plot.pdf"), width = 8, height = 6)
print(ElbowPlot(seurat_obj, ndims = 50))
dev.off()

# Variable features plot
pdf(file.path(output_dir, "variable_features.pdf"), width = 10, height = 6)
top10 <- head(VariableFeatures(seurat_obj), 10)
plot1 <- VariableFeaturePlot(seurat_obj)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
print(plot2)
dev.off()

# PCA loadings
pdf(file.path(output_dir, "pca_loadings.pdf"), width = 12, height = 8)
print(VizDimLoadings(seurat_obj, dims = 1:6, reduction = "pca", ncol = 3))
dev.off()

# PCA heatmap
pdf(file.path(output_dir, "pca_heatmap.pdf"), width = 10, height = 12)
print(DimHeatmap(seurat_obj, dims = 1:15, cells = 500, balanced = TRUE))
dev.off()

# Save variable features
variable_features <- VariableFeatures(seurat_obj)
write.csv(data.frame(gene = variable_features),
          file.path(output_dir, "variable_features.csv"),
          row.names = FALSE)

# Save PCA results
pca_embeddings <- Embeddings(seurat_obj, reduction = "pca")
write.csv(pca_embeddings[, 1:30],
          file.path(output_dir, "pca_embeddings.csv"))

# Save normalization statistics
norm_stats <- data.frame(
  metric = c("n_cells", "n_genes", "n_variable_features", "n_pca_components",
             "vars_regressed"),
  value = c(ncol(seurat_obj), nrow(seurat_obj), n_variable_features, 50,
            ifelse(is.null(vars_to_regress), "none", paste(vars_to_regress, collapse = ", ")))
)
write.csv(norm_stats, file.path(output_dir, "normalization_statistics.csv"), row.names = FALSE)

# Save normalized Seurat object
output_rds <- file.path(output_dir, "seurat_normalized.rds")
saveRDS(seurat_obj, output_rds)
cat("Normalized Seurat object saved to:", output_rds, "\n")

cat("SCTransform normalization completed successfully!\n")
