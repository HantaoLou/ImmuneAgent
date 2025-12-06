#!/usr/bin/env Rscript
# Test R environment and package installation

cat("Testing R environment...\n")

# Test package loading
packages <- c("clusterProfiler", "org.Hs.eg.db", "org.Mm.eg.db", "ggplot2", "enrichplot", "jsonlite")

for (pkg in packages) {
  cat("Testing package:", pkg, "\n")
  tryCatch({
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
    cat("  ✓", pkg, "loaded successfully\n")
  }, error = function(e) {
    cat("  ✗", pkg, "failed to load:", e$message, "\n")
  })
}

# Test basic functionality
cat("\nTesting basic functionality...\n")

# Test bitr function
tryCatch({
  test_genes <- c("TP53", "BRCA1", "EGFR")
  result <- bitr(test_genes, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  cat("  ✓ bitr function works, converted", nrow(result), "genes\n")
}, error = function(e) {
  cat("  ✗ bitr function failed:", e$message, "\n")
})

# Test enrichGO function
tryCatch({
  test_entrez <- c("7157", "672", "1956")  # TP53, BRCA1, EGFR
  go_test <- enrichGO(gene = test_entrez,
                     OrgDb = org.Hs.eg.db,
                     ont = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff = 0.05,
                     qvalueCutoff = 0.2,
                     readable = TRUE)
  cat("  ✓ enrichGO function works\n")
}, error = function(e) {
  cat("  ✗ enrichGO function failed:", e$message, "\n")
})

cat("\nR environment test completed.\n")