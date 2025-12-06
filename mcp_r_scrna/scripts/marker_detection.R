#!/usr/bin/env Rscript
# Marker Gene Detection for All Clusters
# Uses Seurat's FindAllMarkers

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(dplyr)
  library(jsonlite)
  library(patchwork)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript marker_detection.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
group_by <- if(!is.null(params$group_by)) params$group_by else "seurat_clusters"
only_pos <- if(!is.null(params$only_pos)) params$only_pos else TRUE
min_pct <- if(!is.null(params$min_pct)) params$min_pct else 0.25
logfc_threshold <- if(!is.null(params$logfc_threshold)) params$logfc_threshold else 0.5
top_n <- if(!is.null(params$top_n)) params$top_n else 10

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "marker_detection")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Set identity
if (group_by %in% colnames(seurat_obj@meta.data)) {
  Idents(seurat_obj) <- group_by
  cat("Grouping by:", group_by, "\n")
} else {
  stop(paste("Column", group_by, "not found in metadata"))
}

n_groups <- length(unique(Idents(seurat_obj)))
cat("Number of groups:", n_groups, "\n")

cat("Marker detection parameters:\n")
cat("  - Only positive:", only_pos, "\n")
cat("  - Min percent:", min_pct, "\n")
cat("  - Log FC threshold:", logfc_threshold, "\n")
cat("  - Top N per group:", top_n, "\n")

# Find all markers
cat("Finding markers for all groups...\n")
all_markers <- FindAllMarkers(seurat_obj,
                             only.pos = only_pos,
                             min.pct = min_pct,
                             logfc.threshold = logfc_threshold,
                             test.use = "wilcox")

cat("Total markers found:", nrow(all_markers), "\n")

# Save all markers
all_markers$gene <- rownames(all_markers)
all_markers <- all_markers[, c("gene", "cluster", "p_val", "avg_log2FC",
                               "pct.1", "pct.2", "p_val_adj")]
write.csv(all_markers, file.path(output_dir, "all_markers.csv"), row.names = FALSE)

# Get top N markers per cluster
cat("Extracting top", top_n, "markers per group...\n")
top_markers <- all_markers %>%
  group_by(cluster) %>%
  top_n(n = top_n, wt = avg_log2FC)

write.csv(as.data.frame(top_markers),
          file.path(output_dir, paste0("top", top_n, "_markers.csv")),
          row.names = FALSE)

# Summary statistics per cluster
marker_summary <- all_markers %>%
  group_by(cluster) %>%
  summarise(
    n_markers = n(),
    n_significant = sum(p_val_adj < 0.05),
    mean_log2fc = mean(avg_log2FC),
    max_log2fc = max(avg_log2FC)
  )
write.csv(as.data.frame(marker_summary),
          file.path(output_dir, "marker_summary.csv"),
          row.names = FALSE)

# Generate dot plot
cat("Generating dot plot...\n")
top_genes <- top_markers %>%
  group_by(cluster) %>%
  top_n(n = 5, wt = avg_log2FC) %>%
  pull(gene)

pdf(file.path(output_dir, "dotplot_top_markers.pdf"), width = 14, height = 10)
print(DotPlot(seurat_obj, features = unique(top_genes)) +
      RotatedAxis() +
      ggtitle(paste("Top 5 Markers per", group_by)))
dev.off()

# Generate heatmap
cat("Generating heatmap...\n")
pdf(file.path(output_dir, "heatmap_top_markers.pdf"), width = 12, height = 14)
top_heatmap_genes <- top_markers %>%
  group_by(cluster) %>%
  top_n(n = 10, wt = avg_log2FC) %>%
  pull(gene)

# Check which genes exist in the data for heatmap
available_genes <- rownames(seurat_obj)
valid_heatmap_genes <- unique(top_heatmap_genes)[unique(top_heatmap_genes) %in% available_genes]

if (length(valid_heatmap_genes) > 0) {
  print(DoHeatmap(seurat_obj, features = valid_heatmap_genes) +
        ggtitle(paste("Top 10 Markers per", group_by)))
  cat("Generated heatmap with", length(valid_heatmap_genes), "valid genes\n")
} else {
  cat("Warning: No valid genes found for heatmap - creating empty plot\n")
  plot.new()
  text(0.5, 0.5, "No valid genes for heatmap", cex = 2)
}
dev.off()

# Generate feature plots for top markers per cluster
cat("Generating feature plots...\n")
for (clust in unique(all_markers$cluster)) {
  clust_markers <- all_markers %>%
    filter(cluster == clust) %>%
    top_n(n = 9, wt = avg_log2FC) %>%
    pull(gene)

  if (length(clust_markers) > 0) {
    # Check which genes exist in the data
    available_genes <- rownames(seurat_obj)
    valid_markers <- clust_markers[clust_markers %in% available_genes]
    
    if (length(valid_markers) > 0) {
      pdf(file.path(output_dir, paste0("features_cluster_", clust, ".pdf")),
          width = 15, height = 15)
      print(FeaturePlot(seurat_obj, features = valid_markers, ncol = 3) +
            plot_annotation(title = paste("Top Markers for", group_by, clust)))
      dev.off()
      cat("Generated feature plot for cluster", clust, "with", length(valid_markers), "valid genes\n")
    } else {
      cat("Warning: No valid genes found for cluster", clust, "- skipping feature plot\n")
    }
  }
}

# Generate violin plots for top markers
cat("Generating violin plots...\n")
pdf(file.path(output_dir, "violin_top_markers.pdf"), width = 16, height = 12)
print(VlnPlot(seurat_obj, features = head(unique(top_genes), 9), ncol = 3, pt.size = 0))
dev.off()

# Save overall statistics
overall_stats <- data.frame(
  metric = c("n_groups", "total_markers", "total_significant", "only_positive",
             "min_pct", "logfc_threshold", "top_n"),
  value = c(n_groups,
            nrow(all_markers),
            sum(all_markers$p_val_adj < 0.05),
            only_pos,
            min_pct,
            logfc_threshold,
            top_n)
)
write.csv(overall_stats, file.path(output_dir, "overall_statistics.csv"), row.names = FALSE)

cat("Marker detection completed successfully!\n")
cat("  - Total markers:", nrow(all_markers), "\n")
cat("  - Significant markers (p_adj < 0.05):", sum(all_markers$p_val_adj < 0.05), "\n")
cat("  - Top", top_n, "markers per group saved\n")
