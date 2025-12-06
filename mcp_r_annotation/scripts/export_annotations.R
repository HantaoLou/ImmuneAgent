#!/usr/bin/env Rscript

# Export Cell Type Annotations
# Exports annotations to various formats for downstream analysis

suppressPackageStartupMessages({
  library(Seurat)
  library(jsonlite)
  
  # Try to load SeuratDisk, but don't fail if it's not available
  seuratdisk_available <- FALSE
  tryCatch({
    library(SeuratDisk)
    seuratdisk_available <- TRUE
    cat("SeuratDisk package loaded successfully\n")
  }, error = function(e) {
    cat("Warning: SeuratDisk package not available. H5AD and Loom exports will fall back to CSV.\n")
  })
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("No parameters provided. Expecting JSON string.")
}

# Parse JSON parameters
params <- fromJSON(args[1])
input_file <- params$input_file
annotation_columns <- params$annotation_columns  # List of columns to export
export_format <- ifelse(is.null(params$export_format), "csv", params$export_format)
include_umap <- ifelse(is.null(params$include_umap), TRUE, params$include_umap)

# Setup output directory
script_name <- "export_annotations"
# 使用当前工作目录作为基础目录
base_dir <- getwd()
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Determine which columns to export
if (is.null(annotation_columns) || length(annotation_columns) == 0) {
  # Export all annotation-related columns (containing "celltype", "annotation", "singler")
  all_cols <- colnames(seurat_obj@meta.data)
  annotation_keywords <- c("celltype", "annotation", "singler", "confidence")
  annotation_columns <- all_cols[grepl(paste(annotation_keywords, collapse = "|"),
                                       all_cols, ignore.case = TRUE)]

  if (length(annotation_columns) == 0) {
    # If no annotation columns found, use basic metadata
    annotation_columns <- c("seurat_clusters")
    if ("orig.ident" %in% all_cols) {
      annotation_columns <- c(annotation_columns, "orig.ident")
    }
  }
}

cat(sprintf("Exporting %d annotation columns:\n", length(annotation_columns)))
for (col in annotation_columns) {
  cat(sprintf("  - %s\n", col))
}

# Prepare metadata for export
export_data <- seurat_obj@meta.data[, annotation_columns, drop = FALSE]

# Add cell barcodes as first column
export_data <- cbind(cell_barcode = rownames(export_data), export_data)

# Add UMAP coordinates if requested
if (include_umap && "umap" %in% names(seurat_obj@reductions)) {
  cat("Including UMAP coordinates...\n")
  umap_coords <- Embeddings(seurat_obj, reduction = "umap")
  colnames(umap_coords) <- c("UMAP_1", "UMAP_2")
  export_data <- cbind(export_data, umap_coords)
}

# Export based on format
cat(sprintf("Exporting to %s format...\n", export_format))

if (export_format == "csv") {
  # CSV export
  output_file <- file.path(output_dir, "annotations.csv")
  write.csv(export_data, output_file, row.names = FALSE)
  cat(sprintf("Saved annotations to: %s\n", output_file))

} else if (export_format == "tsv") {
  # TSV export
  output_file <- file.path(output_dir, "annotations.tsv")
  write.table(export_data, output_file, sep = "\t",
              row.names = FALSE, quote = FALSE)
  cat(sprintf("Saved annotations to: %s\n", output_file))

} else if (export_format == "h5ad") {
  # H5AD export for scanpy
  if (!seuratdisk_available) {
    cat("Error: SeuratDisk package not available for h5ad export\n")
    cat("Falling back to CSV export\n")
    output_file <- file.path(output_dir, "annotations.csv")
    write.csv(export_data, output_file, row.names = FALSE)
  } else {
    tryCatch({
      output_file <- file.path(output_dir, "seurat_annotated.h5ad")

      # Convert to h5Seurat first
      h5seurat_file <- file.path(output_dir, "temp.h5seurat")
      SaveH5Seurat(seurat_obj, filename = h5seurat_file, overwrite = TRUE)

      # Convert to h5ad
      Convert(h5seurat_file, dest = "h5ad", overwrite = TRUE)

      # Clean up temp file
      file.remove(h5seurat_file)

      cat(sprintf("Saved h5ad file to: %s\n", output_file))
    }, error = function(e) {
      cat("Error: h5ad conversion failed\n")
      cat("Falling back to CSV export\n")
      output_file <- file.path(output_dir, "annotations.csv")
      write.csv(export_data, output_file, row.names = FALSE)
    })
  }

} else if (export_format == "loom") {
  # Loom export
  if (!seuratdisk_available) {
    cat("Error: SeuratDisk package not available for loom export\n")
    cat("Falling back to CSV export\n")
    output_file <- file.path(output_dir, "annotations.csv")
    write.csv(export_data, output_file, row.names = FALSE)
  } else {
    tryCatch({
      output_file <- file.path(output_dir, "seurat_annotated.loom")
      as.loom(seurat_obj, filename = output_file, overwrite = TRUE)
      cat(sprintf("Saved loom file to: %s\n", output_file))
    }, error = function(e) {
      cat("Error: Loom export failed\n")
      cat("Falling back to CSV export\n")
      output_file <- file.path(output_dir, "annotations.csv")
      write.csv(export_data, output_file, row.names = FALSE)
    })
  }

} else {
  stop(sprintf("Unknown export format: %s", export_format))
}

# Generate export summary
export_summary <- data.frame(
  metric = c("Total cells", "Annotation columns", "Total columns exported",
             "Export format", "Includes UMAP"),
  value = c(nrow(export_data),
            length(annotation_columns),
            ncol(export_data),
            export_format,
            as.character(include_umap))
)

summary_file <- file.path(output_dir, "export_summary.csv")
write.csv(export_summary, summary_file, row.names = FALSE)
cat(sprintf("Saved export summary to: %s\n", summary_file))

# Generate annotation statistics for each column
cat("\nAnnotation Statistics:\n")
for (col in annotation_columns) {
  if (col %in% colnames(seurat_obj@meta.data)) {
    col_data <- seurat_obj@meta.data[[col]]

    # Statistics
    unique_vals <- length(unique(col_data))
    cat(sprintf("  %s: %d unique values\n", col, unique_vals))

    # Save detailed statistics
    stats <- as.data.frame(table(col_data))
    colnames(stats) <- c(col, "Count")
    stats$Percentage <- round(stats$Count / sum(stats$Count) * 100, 2)

    stats_file <- file.path(output_dir, sprintf("%s_distribution.csv", col))
    write.csv(stats, stats_file, row.names = FALSE)
  }
}

# Save results as JSON
result <- list(
  status = "success",
  export_format = export_format,
  total_cells = nrow(export_data),
  annotation_columns = annotation_columns,
  total_columns = ncol(export_data),
  includes_umap = include_umap,
  export_summary = export_summary
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== Export Complete ===\n")
cat(sprintf("Format: %s\n", export_format))
cat(sprintf("Total cells: %d\n", nrow(export_data)))
cat(sprintf("Columns exported: %d\n", ncol(export_data)))
cat(sprintf("Output directory: %s\n", output_dir))
