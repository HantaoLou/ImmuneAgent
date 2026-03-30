# Setup Environment for Bioinformatics Skill
# Mirror: Using TUNA (Tsinghua University) for accelerated downloads
repos <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN/" 

if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("Installing BiocManager...\n")
    install.packages("BiocManager", repos = repos)
}

# List of required R packages
packages <- c("optparse", "ggplot2", "dplyr", "magrittr", "patchwork", "Seurat", "monocle3")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    cat(paste0("Package not found. Installing: ", pkg, "...\n"))
    
    if (pkg == "monocle3") {
      # Monocle3 requires installation from the developer's GitHub/Bioconductor
      BiocManager::install("cole-trapnell-lab/monocle3", update = FALSE, ask = FALSE)
    } else {
      BiocManager::install(pkg, update = FALSE, ask = FALSE)
    }
  } else {
    cat(paste0("Package already installed: ", pkg, "\n"))
  }
}

cat("\n--------------------------------------------\n")
cat("Environment configuration complete!\n")
cat("--------------------------------------------\n")