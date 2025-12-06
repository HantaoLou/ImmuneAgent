#!/usr/bin/env Rscript
# Pathway Enrichment Analysis
# Uses clusterProfiler for GO and KEGG enrichment

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
  library(ggplot2)
  library(enrichplot)
  library(jsonlite)
})

# Set language to English for error messages
Sys.setenv(LANGUAGE = "en")
options(stringsAsFactors = FALSE)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript pathway_enrichment.R <input_rds> <params_json>")
}

input_rds <- args[1]
params_json <- args[2]

# Parse parameters
params <- fromJSON(params_json)
deg_csv <- params$deg_csv
organism <- if(!is.null(params$organism)) params$organism else "human"
ontology <- if(!is.null(params$ontology)) params$ontology else "BP"
pvalue_cutoff <- if(!is.null(params$pvalue_cutoff)) params$pvalue_cutoff else 0.05
qvalue_cutoff <- if(!is.null(params$qvalue_cutoff)) params$qvalue_cutoff else 0.2

# Validate DEG CSV file
if (is.null(deg_csv) || !file.exists(deg_csv)) {
  stop("DEG CSV file not provided or does not exist")
}

# Setup output directory
base_dir <- getwd()  # Server already sets cwd correctly
config_path <- file.path(base_dir, "config.json")
config <- fromJSON(config_path)
output_dir <- file.path(config$base_dir, config$output_dir, "pathway_enrichment")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Select organism database
orgdb <- switch(organism,
                "human" = org.Hs.eg.db,
                "mouse" = org.Mm.eg.db,
                org.Hs.eg.db)

kegg_organism <- switch(organism,
                       "human" = "hsa",
                       "mouse" = "mmu",
                       "hsa")

cat("Pathway enrichment parameters:\n")
cat("  - Organism:", organism, "\n")
cat("  - Ontology:", ontology, "\n")
cat("  - P-value cutoff:", pvalue_cutoff, "\n")
cat("  - Q-value cutoff:", qvalue_cutoff, "\n")

# Read DEG results
cat("Reading DEG results from:", deg_csv, "\n")
deg_data <- read.csv(deg_csv, stringsAsFactors = FALSE)

# Filter significant genes
sig_genes <- deg_data[deg_data$p_val_adj < 0.05, ]
cat("Significant DEGs:", nrow(sig_genes), "\n")

if (nrow(sig_genes) == 0) {
  cat("No significant genes found. Exiting.\n")
  quit(save = "no", status = 0)
}

# Separate up and down regulated genes
up_genes <- sig_genes[sig_genes$avg_log2FC > 0, "gene"]
down_genes <- sig_genes[sig_genes$avg_log2FC < 0, "gene"]

cat("Upregulated genes:", length(up_genes), "\n")
cat("Downregulated genes:", length(down_genes), "\n")

# Convert gene symbols to Entrez IDs
cat("Converting gene symbols to Entrez IDs...\n")
all_genes_entrez <- bitr(sig_genes$gene,
                        fromType = "SYMBOL",
                        toType = "ENTREZID",
                        OrgDb = orgdb)

up_genes_entrez <- bitr(up_genes,
                       fromType = "SYMBOL",
                       toType = "ENTREZID",
                       OrgDb = orgdb)

down_genes_entrez <- bitr(down_genes,
                         fromType = "SYMBOL",
                         toType = "ENTREZID",
                         OrgDb = orgdb)

cat("Converted", nrow(all_genes_entrez), "genes to Entrez IDs\n")

# GO Enrichment - All significant genes
cat("Running GO enrichment analysis...\n")
if (nrow(all_genes_entrez) > 0) {
  go_enrich <- enrichGO(gene = all_genes_entrez$ENTREZID,
                       OrgDb = orgdb,
                       ont = ontology,
                       pAdjustMethod = "BH",
                       pvalueCutoff = pvalue_cutoff,
                       qvalueCutoff = qvalue_cutoff,
                       readable = TRUE)

  if (!is.null(go_enrich) && nrow(as.data.frame(go_enrich)) > 0) {
    cat("GO terms enriched:", nrow(as.data.frame(go_enrich)), "\n")

    # Save GO results
    write.csv(as.data.frame(go_enrich),
              file.path(output_dir, "go_enrichment_all.csv"),
              row.names = FALSE)

    # GO dot plot
    pdf(file.path(output_dir, "go_dotplot_all.pdf"), width = 10, height = 8)
    print(dotplot(go_enrich, showCategory = 20) +
          ggtitle(paste("GO", ontology, "Enrichment")))
    dev.off()

    # GO bar plot
    pdf(file.path(output_dir, "go_barplot_all.pdf"), width = 10, height = 8)
    print(barplot(go_enrich, showCategory = 20) +
          ggtitle(paste("GO", ontology, "Enrichment")))
    dev.off()

    # GO network plot
    if (nrow(as.data.frame(go_enrich)) >= 5) {
      pdf(file.path(output_dir, "go_network_all.pdf"), width = 12, height = 10)
      tryCatch({
        print(cnetplot(go_enrich, categorySize = "pvalue", foldChange = NULL))
      }, error = function(e) {
        cat("Network plot failed:", e$message, "\n")
      })
      dev.off()
    }
  } else {
    cat("No significant GO terms found\n")
  }
}

# GO Enrichment - Upregulated genes
if (length(up_genes_entrez$ENTREZID) > 0) {
  cat("Running GO enrichment for upregulated genes...\n")
  go_up <- enrichGO(gene = up_genes_entrez$ENTREZID,
                   OrgDb = orgdb,
                   ont = ontology,
                   pAdjustMethod = "BH",
                   pvalueCutoff = pvalue_cutoff,
                   qvalueCutoff = qvalue_cutoff,
                   readable = TRUE)

  if (!is.null(go_up) && nrow(as.data.frame(go_up)) > 0) {
    write.csv(as.data.frame(go_up),
              file.path(output_dir, "go_enrichment_upregulated.csv"),
              row.names = FALSE)

    pdf(file.path(output_dir, "go_dotplot_upregulated.pdf"), width = 10, height = 8)
    print(dotplot(go_up, showCategory = 20) +
          ggtitle("GO Enrichment - Upregulated Genes"))
    dev.off()
  }
}

# GO Enrichment - Downregulated genes
if (length(down_genes_entrez$ENTREZID) > 0) {
  cat("Running GO enrichment for downregulated genes...\n")
  go_down <- enrichGO(gene = down_genes_entrez$ENTREZID,
                     OrgDb = orgdb,
                     ont = ontology,
                     pAdjustMethod = "BH",
                     pvalueCutoff = pvalue_cutoff,
                     qvalueCutoff = qvalue_cutoff,
                     readable = TRUE)

  if (!is.null(go_down) && nrow(as.data.frame(go_down)) > 0) {
    write.csv(as.data.frame(go_down),
              file.path(output_dir, "go_enrichment_downregulated.csv"),
              row.names = FALSE)

    pdf(file.path(output_dir, "go_dotplot_downregulated.pdf"), width = 10, height = 8)
    print(dotplot(go_down, showCategory = 20) +
          ggtitle("GO Enrichment - Downregulated Genes"))
    dev.off()
  }
}

# KEGG Enrichment
cat("Running KEGG pathway enrichment...\n")
if (nrow(all_genes_entrez) > 0) {
  kegg_enrich <- enrichKEGG(gene = all_genes_entrez$ENTREZID,
                           organism = kegg_organism,
                           pvalueCutoff = pvalue_cutoff,
                           qvalueCutoff = qvalue_cutoff)

  if (!is.null(kegg_enrich) && nrow(as.data.frame(kegg_enrich)) > 0) {
    cat("KEGG pathways enriched:", nrow(as.data.frame(kegg_enrich)), "\n")

    # Convert Entrez IDs back to symbols for readability
    kegg_readable <- setReadable(kegg_enrich, OrgDb = orgdb, keyType = "ENTREZID")

    # Save KEGG results
    write.csv(as.data.frame(kegg_readable),
              file.path(output_dir, "kegg_enrichment.csv"),
              row.names = FALSE)

    # KEGG dot plot
    pdf(file.path(output_dir, "kegg_dotplot.pdf"), width = 10, height = 8)
    print(dotplot(kegg_readable, showCategory = 20) +
          ggtitle("KEGG Pathway Enrichment"))
    dev.off()

    # KEGG bar plot
    pdf(file.path(output_dir, "kegg_barplot.pdf"), width = 10, height = 8)
    print(barplot(kegg_readable, showCategory = 20) +
          ggtitle("KEGG Pathway Enrichment"))
    dev.off()
  } else {
    cat("No significant KEGG pathways found\n")
  }
}

# Save summary statistics
enrichment_summary <- data.frame(
  metric = c("organism", "ontology", "total_sig_genes", "upregulated", "downregulated",
             "genes_converted", "pvalue_cutoff", "qvalue_cutoff"),
  value = c(organism, ontology, nrow(sig_genes), length(up_genes), length(down_genes),
            nrow(all_genes_entrez), pvalue_cutoff, qvalue_cutoff)
)
write.csv(enrichment_summary,
          file.path(output_dir, "enrichment_summary.csv"),
          row.names = FALSE)

cat("Pathway enrichment analysis completed successfully!\n")
