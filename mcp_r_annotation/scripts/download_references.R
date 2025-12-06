#!/usr/bin/env Rscript

# Download and Cache SingleR Reference Datasets
# Manages celldex reference datasets for automated annotation

suppressPackageStartupMessages({
  library(celldex)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Configuration
# 使用当前工作目录作为基础目录
REFERENCE_DIR <- file.path(getwd(), "reference_data")

# Create reference directory
dir.create(REFERENCE_DIR, recursive = TRUE, showWarnings = FALSE)

# Available reference datasets
REFERENCES <- list(
  HumanPrimaryCellAtlasData = list(
    func = celldex::HumanPrimaryCellAtlasData,
    species = "human",
    description = "General human cell types from primary cells",
    size_mb = 113
  ),
  BlueprintEncodeData = list(
    func = celldex::BlueprintEncodeData,
    species = "human",
    description = "Pure stroma and immune cells",
    size_mb = 95
  ),
  MonacoImmuneData = list(
    func = celldex::MonacoImmuneData,
    species = "human",
    description = "Detailed immune cell types (29 types)",
    size_mb = 87,
    recommended = TRUE
  ),
  DatabaseImmuneCellExpressionData = list(
    func = celldex::DatabaseImmuneCellExpressionData,
    species = "human",
    description = "DICE immune cell populations",
    size_mb = 72
  ),
  NovershternHematopoieticData = list(
    func = celldex::NovershternHematopoieticData,
    species = "human",
    description = "Hematopoietic differentiation",
    size_mb = 68
  ),
  MouseRNAseqData = list(
    func = celldex::MouseRNAseqData,
    species = "mouse",
    description = "Mouse cell types from various tissues",
    size_mb = 105
  )
)

# Helper function to download and save reference
download_reference <- function(name, ref_info, force = FALSE) {
  cat(sprintf("\n=== Processing: %s ===\n", name))
  cat(sprintf("Description: %s\n", ref_info$description))
  cat(sprintf("Species: %s\n", ref_info$species))
  cat(sprintf("Estimated size: ~%d MB\n", ref_info$size_mb))

  # Check if already cached
  cache_file <- file.path(REFERENCE_DIR, paste0(name, ".rdata"))

  if (file.exists(cache_file) && !force) {
    cat(sprintf("✓ Already cached: %s\n", cache_file))
    cat("  Use --force to re-download\n")
    return(TRUE)
  }

  # Download reference
  cat("Downloading reference data...\n")
  tryCatch({
    ref_data <- ref_info$func()

    # Save to cache
    cat(sprintf("Saving to cache: %s\n", cache_file))
    save(ref_data, file = cache_file)

    # Verify file size
    file_size_mb <- file.size(cache_file) / 1024^2
    cat(sprintf("✓ Downloaded successfully (%.1f MB)\n", file_size_mb))

    # Print metadata
    cat(sprintf("  Cell types (label.main): %d\n",
                length(unique(ref_data$label.main))))
    if ("label.fine" %in% colnames(colData(ref_data))) {
      cat(sprintf("  Cell types (label.fine): %d\n",
                  length(unique(ref_data$label.fine))))
    }
    cat(sprintf("  Number of samples: %d\n", ncol(ref_data)))
    cat(sprintf("  Number of genes: %d\n", nrow(ref_data)))

    return(TRUE)
  }, error = function(e) {
    cat(sprintf("✗ Error downloading %s:\n", name))
    cat(sprintf("  %s\n", e$message))
    return(FALSE)
  })
}

# Check status of reference datasets
check_references <- function() {
  cat("\n=== Reference Dataset Status ===\n\n")

  status_list <- list()

  for (name in names(REFERENCES)) {
    ref_info <- REFERENCES[[name]]
    cache_file <- file.path(REFERENCE_DIR, paste0(name, ".rdata"))

    status <- list(
      name = name,
      species = ref_info$species,
      description = ref_info$description,
      cached = file.exists(cache_file),
      size_mb = if (file.exists(cache_file)) {
        round(file.size(cache_file) / 1024^2, 1)
      } else {
        ref_info$size_mb
      },
      recommended = ifelse(is.null(ref_info$recommended), FALSE, ref_info$recommended)
    )

    status_list[[name]] <- status

    # Print status
    status_symbol <- ifelse(status$cached, "✓", "✗")
    rec_symbol <- ifelse(status$recommended, "⭐", "")

    cat(sprintf("%s %s %s\n", status_symbol, name, rec_symbol))
    cat(sprintf("   %s (%s)\n", status$description, status$species))
    cat(sprintf("   Size: %.1f MB | Cached: %s\n",
                status$size_mb, ifelse(status$cached, "YES", "NO")))
    if (status$cached) {
      cat(sprintf("   Location: %s\n", cache_file))
    }
    cat("\n")
  }

  # Summary
  total_refs <- length(REFERENCES)
  cached_refs <- sum(sapply(status_list, function(x) x$cached))
  total_size_mb <- sum(sapply(status_list, function(x) x$size_mb))

  cat(sprintf("Summary: %d/%d references cached (%.1f MB total)\n",
              cached_refs, total_refs, total_size_mb))

  if (cached_refs < total_refs) {
    cat("\nTo download missing references, run:\n")
    cat("  Rscript download_references.R\n")
  }

  # Return status as JSON
  invisible(status_list)
}

# Load reference from cache
load_reference <- function(name) {
  cache_file <- file.path(REFERENCE_DIR, paste0(name, ".rdata"))

  if (!file.exists(cache_file)) {
    stop(sprintf("Reference not cached: %s\nRun download_references.R first", name))
  }

  cat(sprintf("Loading reference: %s\n", name))
  load(cache_file)

  if (!exists("ref_data")) {
    stop("Cache file corrupted. Re-download with --force")
  }

  cat(sprintf("✓ Loaded successfully (%d samples, %d genes)\n",
              ncol(ref_data), nrow(ref_data)))

  return(ref_data)
}

# Main execution
main <- function() {
  # Parse arguments
  force_download <- "--force" %in% args
  check_only <- "--check" %in% args
  specific_dataset <- NULL

  for (arg in args) {
    if (arg %in% names(REFERENCES)) {
      specific_dataset <- arg
    }
  }

  # Check only
  if (check_only) {
    status <- check_references()
    return(invisible(status))
  }

  # Download references
  cat("\n╔════════════════════════════════════════════════════════╗")
  cat("\n║  SingleR Reference Dataset Download & Cache Manager   ║")
  cat("\n╚════════════════════════════════════════════════════════╝\n")

  cat(sprintf("\nReference directory: %s\n", REFERENCE_DIR))

  # Download specific dataset or all
  if (!is.null(specific_dataset)) {
    cat(sprintf("\nDownloading specific dataset: %s\n", specific_dataset))
    success <- download_reference(specific_dataset,
                                  REFERENCES[[specific_dataset]],
                                  force = force_download)
  } else {
    cat("\nDownloading all reference datasets...\n")
    cat("This will take 10-20 minutes and ~600 MB of disk space.\n")

    # Download human references first (most commonly used)
    human_refs <- names(REFERENCES)[sapply(REFERENCES,
                                           function(x) x$species == "human")]

    cat("\n--- Human References ---\n")
    for (name in human_refs) {
      download_reference(name, REFERENCES[[name]], force = force_download)
    }

    # Download mouse references
    cat("\n--- Mouse References ---\n")
    mouse_refs <- names(REFERENCES)[sapply(REFERENCES,
                                           function(x) x$species == "mouse")]
    for (name in mouse_refs) {
      download_reference(name, REFERENCES[[name]], force = force_download)
    }
  }

  # Final status
  cat("\n")
  check_references()

  cat("\n✓ Reference download complete!\n")
  cat("\nRecommended reference for immune cells: MonacoImmuneData ⭐\n")
  cat("\nUsage in SingleR:\n")
  cat('  load(file.path(REFERENCE_DIR, "MonacoImmuneData.rdata"))\n')
  cat('  cell_pred <- SingleR(test = test_data, ref = ref_data,\n')
  cat('                       labels = ref_data$label.main)\n')
}

# Help message
if ("--help" %in% args || "-h" %in% args) {
  cat("\nSingleR Reference Dataset Downloader\n")
  cat("\nUsage:\n")
  cat("  Rscript download_references.R [OPTIONS] [DATASET]\n")
  cat("\nOptions:\n")
  cat("  --check          Check status of cached references (no download)\n")
  cat("  --force          Force re-download even if cached\n")
  cat("  --help, -h       Show this help message\n")
  cat("\nDatasets:\n")
  for (name in names(REFERENCES)) {
    ref_info <- REFERENCES[[name]]
    rec <- ifelse(is.null(ref_info$recommended), "", " ⭐ RECOMMENDED")
    cat(sprintf("  %s%s\n", name, rec))
    cat(sprintf("    %s (%s, ~%d MB)\n",
                ref_info$description, ref_info$species, ref_info$size_mb))
  }
  cat("\nExamples:\n")
  cat("  # Download all references\n")
  cat("  Rscript download_references.R\n\n")
  cat("  # Check status only\n")
  cat("  Rscript download_references.R --check\n\n")
  cat("  # Download specific reference\n")
  cat("  Rscript download_references.R MonacoImmuneData\n\n")
  cat("  # Force re-download\n")
  cat("  Rscript download_references.R --force\n")
  quit(save = "no")
}

# Run main
main()
