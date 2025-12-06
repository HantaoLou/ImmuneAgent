#!/usr/bin/env Rscript
# Install required Bioconductor packages for pathway enrichment analysis

cat("Installing Bioconductor packages for pathway enrichment analysis...\n")

# Remove lock directory if it exists
lock_dir <- file.path(Sys.getenv("R_LIBS_USER"), "00LOCK")
if (dir.exists(lock_dir)) {
  cat("Removing lock directory:", lock_dir, "\n")
  unlink(lock_dir, recursive = TRUE, force = TRUE)
}

# Install BiocManager if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  cat("Installing BiocManager...\n")
  install.packages("BiocManager", repos = "https://cran.rstudio.com/")
}

# Load BiocManager
library(BiocManager)

# List of required packages
bioc_packages <- c(
  "clusterProfiler",
  "org.Hs.eg.db",
  "org.Mm.eg.db", 
  "enrichplot"
)

# Install packages one by one
for (pkg in bioc_packages) {
  cat("Installing", pkg, "...\n")
  tryCatch({
    # Check if package is already installed
    if (!requireNamespace(pkg, quietly = TRUE)) {
      BiocManager::install(pkg, update = FALSE, ask = FALSE, force = TRUE)
      cat("  ✓", pkg, "installed successfully\n")
    } else {
      cat("  ✓", pkg, "already installed\n")
    }
  }, error = function(e) {
    cat("  ✗", pkg, "installation failed:", e$message, "\n")
    # Try to remove lock and retry once
    lock_dir <- file.path(Sys.getenv("R_LIBS_USER"), "00LOCK")
    if (dir.exists(lock_dir)) {
      unlink(lock_dir, recursive = TRUE, force = TRUE)
      Sys.sleep(2)
      tryCatch({
        BiocManager::install(pkg, update = FALSE, ask = FALSE, force = TRUE)
        cat("  ✓", pkg, "installed successfully on retry\n")
      }, error = function(e2) {
        cat("  ✗", pkg, "retry failed:", e2$message, "\n")
      })
    }
  })
}

# Verify installation
cat("\nVerifying package installation...\n")
for (pkg in bioc_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("  ✓", pkg, "is available\n")
  } else {
    cat("  ✗", pkg, "is not available\n")
  }
}

cat("\nBioconductor package installation completed.\n")