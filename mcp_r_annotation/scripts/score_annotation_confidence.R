#!/usr/bin/env Rscript

# Calculate Annotation Confidence Scores
# Scores based on marker expression, cluster homogeneity, and SingleR confidence

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(jsonlite)
  library(ggplot2)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("No parameters provided. Expecting JSON string.")
}

# Parse JSON parameters
params <- fromJSON(args[1])
input_file <- params$input_file
annotation_column <- params$annotation_column
marker_genes <- params$marker_genes  # Optional: {"T cells": ["CD3D", "CD3E"], ...}

# Setup output directory
script_name <- "score_annotation_confidence"
# 使用当前工作目录作为基础目录
base_dir <- getwd()
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Check if annotation column exists
if (!annotation_column %in% colnames(seurat_obj@meta.data)) {
  stop(sprintf("Annotation column '%s' not found in metadata", annotation_column))
}

# Extract annotations
annotations <- seurat_obj@meta.data[[annotation_column]]
celltypes <- unique(annotations)

cat(sprintf("Analyzing confidence for %d cell types...\n", length(celltypes)))

# Initialize confidence scores
confidence_scores <- data.frame(
  celltype = character(),
  n_cells = integer(),
  cluster_homogeneity = numeric(),
  marker_score = numeric(),
  silhouette_score = numeric(),
  overall_confidence = numeric(),
  stringsAsFactors = FALSE
)

# Calculate cluster homogeneity (entropy-based)
calculate_homogeneity <- function(annotation_vec) {
  if (length(annotation_vec) <= 1) return(1.0)

  # Calculate entropy
  props <- table(annotation_vec) / length(annotation_vec)
  entropy <- -sum(props * log2(props + 1e-10))

  # Normalize by max entropy
  max_entropy <- log2(length(unique(annotation_vec)))
  if (max_entropy == 0) return(1.0)

  homogeneity <- 1 - (entropy / max_entropy)
  return(homogeneity)
}

# Calculate marker expression score if provided
calculate_marker_score <- function(seurat_obj, cells, markers) {
  if (is.null(markers) || length(markers) == 0) {
    return(NA)
  }

  # Get expression data
  expr_data <- GetAssayData(seurat_obj, slot = "data")

  # Filter markers that exist in dataset
  available_markers <- markers[markers %in% rownames(expr_data)]

  if (length(available_markers) == 0) {
    return(NA)
  }

  # Calculate mean expression of markers in these cells
  marker_expr <- expr_data[available_markers, cells, drop = FALSE]
  mean_expr <- mean(as.matrix(marker_expr))

  # Normalize to 0-1 scale (assuming log-normalized data)
  score <- min(mean_expr / 3, 1.0)  # Cap at 1.0
  return(score)
}

# Calculate silhouette score approximation
# (simplified version based on intra vs inter-cluster distances)
calculate_silhouette_approx <- function(seurat_obj, cells, annotation) {
  if (length(cells) < 2) return(NA)

  # Get PCA embeddings
  if (!"pca" %in% names(seurat_obj@reductions)) {
    return(NA)
  }

  pca_data <- Embeddings(seurat_obj, reduction = "pca")
  cell_coords <- pca_data[cells, 1:min(30, ncol(pca_data))]

  # Calculate mean intra-cluster distance
  if (nrow(cell_coords) < 2) return(NA)

  intra_dist <- mean(dist(cell_coords))

  # Calculate mean inter-cluster distance (sample for efficiency)
  other_cells <- setdiff(colnames(seurat_obj), cells)
  if (length(other_cells) > 500) {
    other_cells <- sample(other_cells, 500)
  }

  other_coords <- pca_data[other_cells, 1:min(30, ncol(pca_data))]

  # Calculate distances to other cells
  inter_dists <- numeric()
  for (i in 1:min(100, nrow(cell_coords))) {
    dists <- sqrt(rowSums((sweep(other_coords, 2, cell_coords[i, ]))^2))
    inter_dists <- c(inter_dists, mean(dists))
  }
  inter_dist <- mean(inter_dists)

  # Silhouette approximation
  sil_score <- (inter_dist - intra_dist) / max(intra_dist, inter_dist)
  sil_score <- (sil_score + 1) / 2  # Normalize to 0-1
  return(sil_score)
}

# Analyze each cell type
for (celltype in celltypes) {
  cat(sprintf("  Analyzing: %s\n", celltype))

  # Get cells of this type
  cells <- colnames(seurat_obj)[annotations == celltype]
  n_cells <- length(cells)

  # Get cluster assignments for homogeneity
  if ("seurat_clusters" %in% colnames(seurat_obj@meta.data)) {
    cluster_assignments <- seurat_obj@meta.data[cells, "seurat_clusters"]
    homogeneity <- calculate_homogeneity(cluster_assignments)
  } else {
    homogeneity <- NA
  }

  # Calculate marker score if provided
  if (!is.null(marker_genes) && celltype %in% names(marker_genes)) {
    markers <- marker_genes[[celltype]]
    marker_score <- calculate_marker_score(seurat_obj, cells, markers)
  } else {
    marker_score <- NA
  }

  # Calculate silhouette score
  silhouette <- calculate_silhouette_approx(seurat_obj, cells, celltype)

  # Calculate overall confidence (average of available scores)
  scores <- c(homogeneity, marker_score, silhouette)
  scores <- scores[!is.na(scores)]

  if (length(scores) > 0) {
    overall <- mean(scores)
  } else {
    overall <- NA
  }

  # Add to results
  confidence_scores <- rbind(confidence_scores, data.frame(
    celltype = celltype,
    n_cells = n_cells,
    cluster_homogeneity = round(homogeneity, 3),
    marker_score = round(marker_score, 3),
    silhouette_score = round(silhouette, 3),
    overall_confidence = round(overall, 3),
    stringsAsFactors = FALSE
  ))
}

# Save confidence scores
confidence_file <- file.path(output_dir, "confidence_scores.csv")
write.csv(confidence_scores, confidence_file, row.names = FALSE)
cat(sprintf("\nSaved confidence scores to: %s\n", confidence_file))

# Add confidence scores to Seurat metadata
confidence_column <- sprintf("%s_confidence", annotation_column)
seurat_obj@meta.data[[confidence_column]] <- sapply(
  annotations,
  function(ct) {
    score <- confidence_scores$overall_confidence[confidence_scores$celltype == ct]
    ifelse(length(score) > 0, score, NA)
  }
)

# Save updated Seurat object
output_rds <- file.path(output_dir, "seurat_with_confidence.rds")
saveRDS(seurat_obj, output_rds)
cat(sprintf("Saved Seurat object with confidence scores to: %s\n", output_rds))

# Generate confidence bar plot
cat("Generating confidence visualization...\n")
pdf(file.path(output_dir, "plots", "confidence_scores.pdf"),
    width = 10, height = 6)

p <- ggplot(confidence_scores,
            aes(x = reorder(celltype, overall_confidence),
                y = overall_confidence,
                fill = overall_confidence)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.2f", overall_confidence)),
            hjust = -0.2, size = 3) +
  coord_flip() +
  scale_fill_gradient(low = "red", high = "green", limits = c(0, 1)) +
  ylim(0, 1.1) +
  labs(title = "Cell Type Annotation Confidence",
       x = "Cell Type",
       y = "Overall Confidence Score",
       fill = "Confidence") +
  theme_minimal()

print(p)
dev.off()

# Generate detailed scores heatmap
pdf(file.path(output_dir, "plots", "confidence_heatmap.pdf"),
    width = 10, height = 8)

# Prepare data for heatmap
heatmap_data <- confidence_scores[, c("celltype", "cluster_homogeneity",
                                      "marker_score", "silhouette_score")]
rownames(heatmap_data) <- heatmap_data$celltype
heatmap_data$celltype <- NULL

# Remove NA columns
heatmap_data <- heatmap_data[, colSums(!is.na(heatmap_data)) > 0, drop = FALSE]

if (ncol(heatmap_data) > 0 && nrow(heatmap_data) > 0) {
  heatmap(as.matrix(heatmap_data),
          main = "Confidence Score Components",
          scale = "none",
          col = colorRampPalette(c("red", "yellow", "green"))(100),
          margins = c(12, 10))
}
dev.off()

# Generate UMAP with confidence overlay
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("Generating UMAP with confidence overlay...\n")

  pdf(file.path(output_dir, "plots", "umap_confidence.pdf"),
      width = 12, height = 5)

  p1 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = annotation_column,
                label = TRUE,
                label.size = 3) +
    ggtitle(annotation_column)

  p2 <- FeaturePlot(seurat_obj,
                    reduction = "umap",
                    features = confidence_column) +
    scale_color_gradient(low = "red", high = "green", limits = c(0, 1)) +
    ggtitle("Confidence Score")

  print(p1 | p2)
  dev.off()
}

# Save results as JSON
result <- list(
  status = "success",
  annotation_column = annotation_column,
  confidence_column = confidence_column,
  total_celltypes = nrow(confidence_scores),
  mean_confidence = round(mean(confidence_scores$overall_confidence, na.rm = TRUE), 3),
  confidence_scores = confidence_scores
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== Confidence Scoring Complete ===\n")
cat(sprintf("Cell types analyzed: %d\n", nrow(confidence_scores)))
cat(sprintf("Mean confidence: %.3f\n",
            mean(confidence_scores$overall_confidence, na.rm = TRUE)))
cat(sprintf("Confidence column added: %s\n", confidence_column))
