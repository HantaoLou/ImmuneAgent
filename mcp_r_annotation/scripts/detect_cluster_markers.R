#!/usr/bin/env Rscript

# Detect Cluster-Specific Marker Genes
# Uses Seurat's FindAllMarkers for differential expression analysis

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
test_use <- ifelse(is.null(params$test_use), "wilcox", params$test_use)
only_pos <- ifelse(is.null(params$only_pos), TRUE, params$only_pos)
min_pct <- ifelse(is.null(params$min_pct), 0.25, params$min_pct)
logfc_threshold <- ifelse(is.null(params$logfc_threshold), 0.5, params$logfc_threshold)
top_n <- ifelse(is.null(params$top_n), 10, params$top_n)

# Setup output directory
script_name <- "detect_cluster_markers"
# 使用当前工作目录作为基础目录
base_dir <- getwd()
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Set identity to seurat_clusters
Idents(seurat_obj) <- "seurat_clusters"

# Find all markers
cat(sprintf("Finding cluster markers (test: %s)...\n", test_use))
markers <- FindAllMarkers(
  seurat_obj,
  only.pos = only_pos,
  min.pct = min_pct,
  logfc.threshold = logfc_threshold,
  test.use = test_use
)

# Save all markers
all_markers_file <- file.path(output_dir, "all_markers.csv")
write.csv(markers, all_markers_file, row.names = FALSE)
cat(sprintf("Saved all markers to: %s\n", all_markers_file))

# Get top N markers per cluster
top_markers <- markers %>%
  group_by(cluster) %>%
  top_n(n = top_n, wt = avg_log2FC) %>%
  arrange(cluster, desc(avg_log2FC))

top_markers_file <- file.path(output_dir, "top_markers.csv")
write.csv(top_markers, top_markers_file, row.names = FALSE)
cat(sprintf("Saved top %d markers per cluster to: %s\n", top_n, top_markers_file))

# Generate marker summary statistics
marker_summary <- markers %>%
  group_by(cluster) %>%
  summarise(
    total_markers = n(),
    avg_logFC = mean(avg_log2FC),
    max_logFC = max(avg_log2FC),
    avg_pct1 = mean(pct.1),
    avg_pct2 = mean(pct.2)
  )

summary_file <- file.path(output_dir, "marker_summary.csv")
write.csv(marker_summary, summary_file, row.names = FALSE)
cat(sprintf("Saved marker summary to: %s\n", summary_file))

# Generate heatmap of top markers
if (nrow(top_markers) > 0) {
  cat("Generating marker heatmap...\n")

  # Select top 5 markers per cluster for visualization
  top5_markers <- markers %>%
    group_by(cluster) %>%
    top_n(n = 5, wt = avg_log2FC) %>%
    pull(gene)

  pdf(file.path(output_dir, "plots", "marker_heatmap.pdf"),
      width = 12, height = 10)
  DoHeatmap(seurat_obj,
            features = unique(top5_markers),
            size = 3) +
    theme(axis.text.y = element_text(size = 6))
  dev.off()
}

# Generate dotplot of top markers
if (nrow(top_markers) > 0) {
  cat("Generating marker dotplot...\n")

  # Select top 3 markers per cluster
  top3_markers <- markers %>%
    group_by(cluster) %>%
    top_n(n = 3, wt = avg_log2FC) %>%
    pull(gene)

  pdf(file.path(output_dir, "plots", "marker_dotplot.pdf"),
      width = 14, height = 8)
  print(DotPlot(seurat_obj, features = unique(top3_markers)) +
    RotatedAxis() +
    theme(axis.text.x = element_text(size = 8)))
  dev.off()
}

# Generate violin plots for top marker of each cluster
cat("Generating violin plots for top markers...\n")
pdf(file.path(output_dir, "plots", "top_marker_violins.pdf"),
    width = 15, height = 12)

top_genes <- markers %>%
  group_by(cluster) %>%
  top_n(n = 1, wt = avg_log2FC) %>%
  arrange(cluster) %>%
  pull(gene)

n_clusters <- length(unique(markers$cluster))
n_cols <- ceiling(sqrt(n_clusters))
n_rows <- ceiling(n_clusters / n_cols)

par(mfrow = c(n_rows, n_cols))
for (gene in top_genes) {
  print(VlnPlot(seurat_obj, features = gene, pt.size = 0.1) +
    ggtitle(sprintf("Top marker: %s", gene)))
}
dev.off()

# Generate feature plots for select top markers
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("Generating feature plots...\n")

  select_markers <- head(top_genes, 9)  # Top marker from first 9 clusters

  pdf(file.path(output_dir, "plots", "marker_featureplots.pdf"),
      width = 15, height = 12)
  print(FeaturePlot(seurat_obj, features = select_markers, ncol = 3))
  dev.off()
}

# Save results as JSON for MCP response
result <- list(
  status = "success",
  test_method = test_use,
  total_markers = nrow(markers),
  clusters_analyzed = length(unique(markers$cluster)),
  top_n_per_cluster = top_n,
  marker_summary = marker_summary
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== Marker Detection Complete ===\n")
cat(sprintf("Test method: %s\n", test_use))
cat(sprintf("Total markers found: %d\n", nrow(markers)))
cat(sprintf("Clusters analyzed: %d\n", length(unique(markers$cluster))))
cat(sprintf("Top %d markers saved per cluster\n", top_n))
