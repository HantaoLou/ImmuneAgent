#!/usr/bin/env Rscript

# SingleR Automated Cell Type Annotation
# Uses celldex reference datasets for automated annotation

# Install missing packages if needed
required_packages <- c("Seurat", "SingleR", "celldex", "jsonlite", "SingleCellExperiment", "scuttle", "ggplot2")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing missing package: %s\n", pkg))
    if (pkg %in% c("SingleR", "celldex", "SingleCellExperiment", "scuttle", "scrapper")) {
      # Bioconductor packages
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      # CRAN packages
      install.packages(pkg, repos = "https://cran.rstudio.com/")
    }
  }
}

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleR)
  library(celldex)
  library(jsonlite)
  library(SingleCellExperiment)
  library(scuttle)
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
reference_dataset <- ifelse(is.null(params$reference_dataset),
                            "HumanPrimaryCellAtlasData",
                            params$reference_dataset)
label_type <- ifelse(is.null(params$label_type), "label.main", params$label_type)
cluster_column <- ifelse(is.null(params$cluster_column),
                         "seurat_clusters",
                         params$cluster_column)

# Setup output directory
script_name <- "run_singler_annotation"

# Get base directory in a command-line friendly way
# Try to get script path first, fallback to working directory
script_path <- tryCatch({
  # Method 1: Use sys.frame to get script path
  frame_files <- lapply(sys.frames(), function(x) x$ofile)
  script_file <- Filter(Negate(is.null), frame_files)
  if (length(script_file) > 0) {
    dirname(dirname(script_file[[1]]))
  } else {
    # Method 2: Use commandArgs to get script path
    args_all <- commandArgs(trailingOnly = FALSE)
    script_arg <- grep("--file=", args_all, value = TRUE)
    if (length(script_arg) > 0) {
      script_file <- sub("--file=", "", script_arg[1])
      dirname(dirname(script_file))
    } else {
      NULL
    }
  }
}, error = function(e) NULL)

# Fallback to working directory if script path detection fails
base_dir <- if (!is.null(script_path) && script_path != "") {
  script_path
} else {
  getwd()
}
output_dir <- file.path(base_dir, "output", script_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

# Load Seurat object
cat(sprintf("Loading Seurat object from: %s\n", input_file))
seurat_obj <- readRDS(input_file)

# Load reference dataset
cat(sprintf("Loading reference dataset: %s\n", reference_dataset))
ref_data <- switch(reference_dataset,
  "HumanPrimaryCellAtlasData" = celldex::HumanPrimaryCellAtlasData(),
  "BlueprintEncodeData" = celldex::BlueprintEncodeData(),
  "MonacoImmuneData" = celldex::MonacoImmuneData(),
  "DatabaseImmuneCellExpressionData" = celldex::DatabaseImmuneCellExpressionData(),
  stop(sprintf("Unknown reference dataset: %s", reference_dataset))
)

# Extract test data (normalized counts)
cat("Extracting normalized expression data...\n")
# Use layer parameter instead of deprecated slot parameter
test_data <- GetAssayData(seurat_obj, layer = "data")

# Extract cluster information
clusters <- seurat_obj@meta.data[[cluster_column]]

# Run SingleR annotation (cluster-level)
cat(sprintf("Running SingleR annotation with %s labels...\n", label_type))
# Remove deprecated method parameter when clusters is specified
cell_pred <- SingleR(
  test = test_data,
  ref = ref_data,
  labels = ref_data[[label_type]],
  clusters = clusters,
  assay.type.test = "logcounts",
  assay.type.ref = "logcounts"
)

# Create annotation mapping
celltype_mapping <- data.frame(
  ClusterID = rownames(cell_pred),
  celltype = cell_pred$labels,
  confidence = cell_pred$pruned.labels,
  stringsAsFactors = FALSE
)

# Save annotation mapping
csv_file <- file.path(output_dir, "singler_annotation.csv")
write.csv(celltype_mapping, csv_file, row.names = FALSE)
cat(sprintf("Saved annotation mapping to: %s\n", csv_file))

# Add annotations to Seurat object metadata
annotation_col <- sprintf("singler_%s", reference_dataset)
seurat_obj@meta.data[[annotation_col]] <- celltype_mapping[
  match(clusters, celltype_mapping$ClusterID),
  "celltype"
]

# Add confidence scores
confidence_col <- sprintf("singler_%s_confidence", reference_dataset)
seurat_obj@meta.data[[confidence_col]] <- celltype_mapping[
  match(clusters, celltype_mapping$ClusterID),
  "confidence"
]

# Save updated Seurat object
output_rds <- file.path(output_dir, "seurat_with_singler.rds")
saveRDS(seurat_obj, output_rds)
cat(sprintf("Saved annotated Seurat object to: %s\n", output_rds))

# Generate annotation summary
annotation_summary <- table(seurat_obj@meta.data[[annotation_col]])
summary_df <- data.frame(
  CellType = names(annotation_summary),
  Count = as.numeric(annotation_summary),
  Percentage = round(as.numeric(annotation_summary) / ncol(seurat_obj) * 100, 2)
)

summary_file <- file.path(output_dir, "annotation_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("Saved annotation summary to: %s\n", summary_file))

# Generate UMAP visualization
if ("umap" %in% names(seurat_obj@reductions)) {
  cat("Generating UMAP visualization...\n")

  pdf(file.path(output_dir, "plots", "singler_umap.pdf"), width = 12, height = 5)
  p1 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = cluster_column,
                label = TRUE,
                label.size = 5) +
    ggtitle("Original Clusters")

  p2 <- DimPlot(seurat_obj,
                reduction = "umap",
                group.by = annotation_col,
                label = TRUE,
                label.size = 3) +
    ggtitle(sprintf("SingleR: %s", reference_dataset))

  print(p1 | p2)
  dev.off()
}

# Save results as JSON for MCP response
result <- list(
  status = "success",
  reference_dataset = reference_dataset,
  label_type = label_type,
  total_cells = ncol(seurat_obj),
  total_celltypes = length(unique(celltype_mapping$celltype)),
  annotation_column = annotation_col,
  confidence_column = confidence_col,
  summary = summary_df
)

result_json <- file.path(output_dir, "result.json")
write_json(result, result_json, pretty = TRUE)

cat("\n=== SingleR Annotation Complete ===\n")
cat(sprintf("Reference: %s\n", reference_dataset))
cat(sprintf("Total cells: %d\n", ncol(seurat_obj)))
cat(sprintf("Cell types identified: %d\n", length(unique(celltype_mapping$celltype))))
cat(sprintf("Annotation column: %s\n", annotation_col))
