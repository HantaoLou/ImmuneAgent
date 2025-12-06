#!/usr/bin/env Rscript
# Subset Cells by Metadata Criteria
# Filters Seurat object based on metadata values

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
  stop("Usage: Rscript subset_cells.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
subset_column <- params$subset_column
subset_values <- params$subset_values
invert <- if(!is.null(params$invert)) params$invert else FALSE

if (is.null(subset_column) || length(subset_values) == 0) {
  stop("subset_column and subset_values are required")
}

# Load Seurat object
cat("Loading Seurat object from:", input_rds, "\n")
seurat_obj <- readRDS(input_rds)

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "subset_cells")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Check if column exists
if (!subset_column %in% colnames(seurat_obj@meta.data)) {
  stop(paste("Column", subset_column, "not found in metadata"))
}

n_cells_before <- ncol(seurat_obj)
cat("Cells before subsetting:", n_cells_before, "\n")
cat("Subsetting by column:", subset_column, "\n")
cat("Values to", if(invert) "exclude:" else "keep:", paste(subset_values, collapse = ", "), "\n")

# Show value distribution before subsetting
cat("\nValue distribution before subsetting:\n")
value_counts <- table(seurat_obj@meta.data[[subset_column]])
print(value_counts)

# Generate pre-subsetting UMAP if available
if ("umap" %in% names(seurat_obj@reductions)) {
  pdf(file.path(output_dir, "umap_before_subset.pdf"), width = 10, height = 8)
  print(DimPlot(seurat_obj, reduction = "umap", group.by = subset_column) +
        ggtitle("Before Subsetting"))
  dev.off()
}

# Remove graph objects to avoid subsetting issues
# This prevents "Please provide rownames to the matrix before converting to a Graph" error
graph_names <- names(seurat_obj@graphs)
if (length(graph_names) > 0) {
  cat("Removing graph objects before subsetting:", paste(graph_names, collapse = ", "), "\n")
  for (graph_name in graph_names) {
    seurat_obj@graphs[[graph_name]] <- NULL
  }
}

# Perform subsetting
if (invert) {
  # Exclude specified values
  cells_logical <- seurat_obj@meta.data[[subset_column]] %in% subset_values
  cells_to_keep <- rownames(seurat_obj@meta.data)[!cells_logical]
} else {
  # Keep only specified values
  cells_logical <- seurat_obj@meta.data[[subset_column]] %in% subset_values
  cells_to_keep <- rownames(seurat_obj@meta.data)[cells_logical]
}

cat("Number of cells to keep:", length(cells_to_keep), "\n")
seurat_obj <- subset(seurat_obj, cells = cells_to_keep)

n_cells_after <- ncol(seurat_obj)
cat("\nCells after subsetting:", n_cells_after, "\n")
cat("Cells removed:", n_cells_before - n_cells_after, "\n")
cat("Retention rate:", round(n_cells_after/n_cells_before*100, 2), "%\n")

# Show value distribution after subsetting
cat("\nValue distribution after subsetting:\n")
value_counts_after <- table(seurat_obj@meta.data[[subset_column]])
print(value_counts_after)

# Generate post-subsetting UMAP if available
if ("umap" %in% names(seurat_obj@reductions)) {
  pdf(file.path(output_dir, "umap_after_subset.pdf"), width = 10, height = 8)
  print(DimPlot(seurat_obj, reduction = "umap", group.by = subset_column) +
        ggtitle("After Subsetting"))
  dev.off()

  # Comparison plot
  pdf(file.path(output_dir, "umap_comparison.pdf"), width = 14, height = 6)
  # Reload original for comparison
  seurat_orig <- readRDS(input_rds)
  p1 <- DimPlot(seurat_orig, reduction = "umap", group.by = subset_column) +
        ggtitle(paste("Before:", n_cells_before, "cells"))
  p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = subset_column) +
        ggtitle(paste("After:", n_cells_after, "cells"))
  print(p1 + p2)
  dev.off()
}

# Save cell barcodes
cell_barcodes <- data.frame(
  cell_barcode = colnames(seurat_obj),
  subset_value = seurat_obj@meta.data[[subset_column]]
)
write.csv(cell_barcodes, file.path(output_dir, "retained_cells.csv"), row.names = FALSE)

# Save subset statistics
subset_stats_before <- data.frame(
  value = names(value_counts),
  n_cells_before = as.numeric(value_counts)
)

subset_stats_after <- data.frame(
  value = names(value_counts_after),
  n_cells_after = as.numeric(value_counts_after)
)

subset_stats <- merge(subset_stats_before, subset_stats_after, by = "value", all = TRUE)
subset_stats[is.na(subset_stats)] <- 0
subset_stats$percentage_retained <- round(subset_stats$n_cells_after / subset_stats$n_cells_before * 100, 2)
write.csv(subset_stats, file.path(output_dir, "subset_statistics.csv"), row.names = FALSE)

# Save overall summary
summary_stats <- data.frame(
  metric = c("subset_column", "subset_values", "invert", "cells_before", "cells_after",
             "cells_removed", "retention_rate_percent"),
  value = c(subset_column,
            paste(subset_values, collapse = ", "),
            invert,
            n_cells_before,
            n_cells_after,
            n_cells_before - n_cells_after,
            round(n_cells_after/n_cells_before*100, 2))
)
write.csv(summary_stats, file.path(output_dir, "summary.csv"), row.names = FALSE)

# Generate composition barplot
pdf(file.path(output_dir, "composition_comparison.pdf"), width = 12, height = 6)
par(mfrow = c(1, 2))

# Before
barplot(value_counts,
        main = paste("Before Subsetting\n(", n_cells_before, "cells)"),
        xlab = subset_column,
        ylab = "Number of Cells",
        las = 2,
        col = rainbow(length(value_counts)))

# After
barplot(value_counts_after,
        main = paste("After Subsetting\n(", n_cells_after, "cells)"),
        xlab = subset_column,
        ylab = "Number of Cells",
        las = 2,
        col = rainbow(length(value_counts_after)))

dev.off()

# Save subsetted Seurat object
output_rds <- file.path(output_dir, "seurat_subset.rds")
saveRDS(seurat_obj, output_rds)
cat("\nSubsetted Seurat object saved to:", output_rds, "\n")

cat("\nCell subsetting completed successfully!\n")
