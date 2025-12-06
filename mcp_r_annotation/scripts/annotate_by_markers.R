#!/usr/bin/env Rscript

# Manual Cell Type Annotation Based on Marker Genes
# Applies user-provided cluster-to-celltype mapping

suppressPackageStartupMessages({
  library(Seurat)
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
marker_list <- params$marker_list  # Dictionary: {"0": "T cells", "1": "B cells"}
cluster_column <- ifelse(is.null(params$cluster_column),
                         "seurat_clusters",
                         params$cluster_column)
new_column <- ifelse(is.null(params$new_column),
                     "manual_celltype",
                     params$new_column)

# Setup output directory
script_name <- "annotate_by_markers"
# 使用当前工作目录作为基础目录
base_dir <- getwd()
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Validate marker_list
if (is.null(marker_list) || length(marker_list) == 0) {
  stop("marker_list is required and cannot be empty")
}

# Convert marker_list to data frame
annotation_mapping <- data.frame(
  cluster = names(marker_list),
  celltype = unlist(marker_list),
  stringsAsFactors = FALSE
)

cat("Applying manual annotations:\n")
print(annotation_mapping)

# Initialize new annotation column
seurat_obj@meta.data[[new_column]] <- NA

# Apply annotations based on cluster mapping
clusters <- seurat_obj@meta.data[[cluster_column]]
for (i in 1:nrow(annotation_mapping)) {
  cluster_id <- annotation_mapping$cluster[i]
  celltype <- annotation_mapping$celltype[i]

  # Match cells
  matched_cells <- which(as.character(clusters) == cluster_id)

  if (length(matched_cells) > 0) {
    seurat_obj@meta.data[matched_cells, new_column] <- celltype
    cat(sprintf("  Cluster %s -> %s (%d cells)\n",
                cluster_id, celltype, length(matched_cells)))
  } else {
    cat(sprintf("  Warning: No cells found for cluster %s\n", cluster_id))
  }
}

# Handle unannotated clusters
unannotated <- sum(is.na(seurat_obj@meta.data[[new_column]]))
if (unannotated > 0) {
  cat(sprintf("\nWarning: %d cells remain unannotated\n", unannotated))
  seurat_obj@meta.data[[new_column]][is.na(seurat_obj@meta.data[[new_column]])] <- "Unannotated"
}

# Save annotation mapping
mapping_file <- file.path(output_dir, "annotation_mapping.csv")
write.csv(annotation_mapping, mapping_file, row.names = FALSE)
cat(sprintf("\nSaved annotation mapping to: %s\n", mapping_file))

# Generate annotation summary
annotation_summary <- table(seurat_obj@meta.data[[new_column]])
summary_df <- data.frame(
  CellType = names(annotation_summary),
  Count = as.numeric(annotation_summary),
  Percentage = round(as.numeric(annotation_summary) / ncol(seurat_obj) * 100, 2)
)

summary_file <- file.path(output_dir, "annotation_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("Saved annotation summary to: %s\n", summary_file))

# Save updated Seurat object
output_rds <- file.path(output_dir, "seurat_with_manual_annotation.rds")
saveRDS(seurat_obj, output_rds)
cat(sprintf("Saved annotated Seurat object to: %s\n", output_rds))

# Generate UMAP visualization
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("Generating UMAP visualization...\n")

  pdf(file.path(output_dir, "plots", "manual_annotation_umap.pdf"),
      width = 14, height = 6)

  p1 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = cluster_column,
                label = TRUE,
                label.size = 5) +
    ggtitle("Original Clusters")

  p2 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = new_column,
                label = TRUE,
                label.size = 4) +
    ggtitle("Manual Annotation")

  print(p1 | p2)
  dev.off()
}

# Generate stacked bar plot
cat("Generating cluster composition plot...\n")
pdf(file.path(output_dir, "plots", "cluster_composition.pdf"),
    width = 10, height = 6)

composition_data <- data.frame(
  cluster = seurat_obj@meta.data[[cluster_column]],
  celltype = seurat_obj@meta.data[[new_column]]
)

p <- ggplot(composition_data, aes(x = cluster, fill = celltype)) +
  geom_bar(position = "fill") +
  labs(title = "Cluster Composition by Cell Type",
       y = "Proportion",
       x = "Cluster",
       fill = "Cell Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p)
dev.off()

# Save results as JSON for MCP response
result <- list(
  status = "success",
  annotation_column = new_column,
  total_cells = ncol(seurat_obj),
  total_celltypes = length(unique(seurat_obj@meta.data[[new_column]])),
  unannotated_cells = sum(seurat_obj@meta.data[[new_column]] == "Unannotated"),
  annotation_mapping = annotation_mapping,
  summary = summary_df
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== Manual Annotation Complete ===\n")
cat(sprintf("Annotation column: %s\n", new_column))
cat(sprintf("Total cells: %d\n", ncol(seurat_obj)))
cat(sprintf("Cell types: %d\n", length(unique(seurat_obj@meta.data[[new_column]]))))
