#!/usr/bin/env Rscript
# Doublet Detection using DoubletFinder
# Identifies and removes doublets from scRNA-seq data

suppressPackageStartupMessages({
  library(Seurat)
  library(DoubletFinder)
  library(ggplot2)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)
set.seed(42)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript doublet_detection.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
expected_doublet_rate <- if(!is.null(params$expected_doublet_rate)) params$expected_doublet_rate else 0.08
pN <- if(!is.null(params$pN)) params$pN else 0.25
pK <- if(!is.null(params$pK)) params$pK else 0.09
dims <- if(!is.null(params$dims)) params$dims else 20

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "doublet_detection")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

n_cells_before <- ncol(seurat_obj)
cat("Cells before doublet removal:", n_cells_before, "\n")
cat("Expected doublet rate:", expected_doublet_rate, "\n")

# Calculate expected number of doublets
nExp_poi <- round(expected_doublet_rate * n_cells_before)
cat("Expected number of doublets:", nExp_poi, "\n")

# Ensure necessary reductions are present
if (!"pca" %in% names(seurat_obj@reductions)) {
  cat("PCA not found, computing...\n")
  seurat_obj <- RunPCA(seurat_obj, verbose = FALSE)
}

if (!"umap" %in% names(seurat_obj@reductions)) {
  cat("UMAP not found, computing...\n")
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:dims, verbose = FALSE)
}

# Generate pre-doubletfinder UMAP
pdf(file.path(output_dir, "umap_before_doublet_removal.pdf"), width = 8, height = 6)
p1 <- DimPlot(seurat_obj, reduction = "umap") + ggtitle("Before Doublet Removal")
print(p1)
dev.off()

# Run DoubletFinder
cat("Running DoubletFinder...\n")
cat("  - pN:", pN, "\n")
cat("  - pK:", pK, "\n")
cat("  - Dimensions:", dims, "\n")

seurat_obj <- doubletFinder_v3(seurat_obj,
                               PCs = 1:dims,
                               pN = pN,
                               pK = pK,
                               nExp = nExp_poi,
                               reuse.pANN = FALSE,
                               sct = FALSE)

# Find the DoubletFinder classification column
df_col <- grep("^DF.classifications", colnames(seurat_obj@meta.data), value = TRUE)
pann_col <- grep("^pANN", colnames(seurat_obj@meta.data), value = TRUE)

if (length(df_col) == 0) {
  stop("DoubletFinder classification column not found")
}

# Rename columns for easier access
seurat_obj$doublet_class <- seurat_obj[[df_col]]
seurat_obj$doublet_score <- seurat_obj[[pann_col]]
seurat_obj[[df_col]] <- NULL
seurat_obj[[pann_col]] <- NULL

# Count doublets
n_doublets <- sum(seurat_obj$doublet_class == "Doublet")
cat("Doublets detected:", n_doublets, "\n")
cat("Doublet rate:", round(n_doublets/n_cells_before*100, 2), "%\n")

# Plot with doublet labels
pdf(file.path(output_dir, "umap_doublets_labeled.pdf"), width = 9, height = 6)
p2 <- DimPlot(seurat_obj,
              reduction = "umap",
              group.by = "doublet_class",
              cols = c("Singlet" = "#272E6A", "Doublet" = "#D51F26")) +
      ggtitle("Doublet Detection Results")
print(p2)
dev.off()

# Plot doublet scores
pdf(file.path(output_dir, "umap_doublet_scores.pdf"), width = 9, height = 6)
p3 <- FeaturePlot(seurat_obj,
                  features = "doublet_score",
                  reduction = "umap") +
      scale_color_viridis_c() +
      ggtitle("Doublet Scores")
print(p3)
dev.off()

# Doublet score distribution
pdf(file.path(output_dir, "doublet_score_distribution.pdf"), width = 8, height = 6)
hist(seurat_obj$doublet_score,
     breaks = 50,
     main = "Doublet Score Distribution",
     xlab = "Doublet Score",
     ylab = "Frequency",
     col = "skyblue")
abline(v = median(seurat_obj$doublet_score[seurat_obj$doublet_class == "Doublet"]),
       col = "red", lwd = 2, lty = 2)
legend("topright",
       legend = c("Doublet threshold"),
       col = "red", lty = 2, lwd = 2)
dev.off()

# Save doublet assignments
doublet_assignments <- data.frame(
  cell_barcode = colnames(seurat_obj),
  doublet_class = seurat_obj$doublet_class,
  doublet_score = seurat_obj$doublet_score
)
write.csv(doublet_assignments, file.path(output_dir, "doublet_assignments.csv"), row.names = FALSE)

# Remove doublets
cat("Removing doublets...\n")
seurat_obj <- subset(seurat_obj, subset = doublet_class == "Singlet")

n_cells_after <- ncol(seurat_obj)
cat("Cells after doublet removal:", n_cells_after, "\n")

# Generate post-doubletfinder UMAP
pdf(file.path(output_dir, "umap_after_doublet_removal.pdf"), width = 8, height = 6)
p4 <- DimPlot(seurat_obj, reduction = "umap") + ggtitle("After Doublet Removal")
print(p4)
dev.off()

# Save doublet statistics
doublet_stats <- data.frame(
  metric = c("cells_before", "doublets_detected", "cells_after", "doublet_rate_percent",
             "expected_doublet_rate", "pN", "pK", "dims"),
  value = c(n_cells_before, n_doublets, n_cells_after,
            round(n_doublets/n_cells_before*100, 2),
            expected_doublet_rate, pN, pK, dims)
)
write.csv(doublet_stats, file.path(output_dir, "doublet_statistics.csv"), row.names = FALSE)

# Save filtered Seurat object
output_rds <- file.path(output_dir, "seurat_singlets.rds")
saveRDS(seurat_obj, output_rds)
cat("Singlet-only Seurat object saved to:", output_rds, "\n")

cat("Doublet detection completed successfully!\n")
