# r_scripts/utils.R
# General Utility Functions for Single-Cell Trajectory Analysis

#' Synchronize Seurat Coordinates to Monocle3 CDS Object
#' @param cds Monocle3 cell_data_set object
#' @param seurat_obj Source Seurat object
#' @param reduction Reduction type, defaults to "umap"
sync_coordinates <- function(cds, seurat_obj, reduction = "umap") {
  # Extract coordinates from Seurat (ensure only cells present in CDS are included)
  embeds <- Embeddings(seurat_obj, reduction = reduction)
  common_cells <- intersect(colnames(cds), rownames(embeds))
  
  if (length(common_cells) == 0) {
    stop("Error: No common cell IDs found between CDS and Seurat objects.")
  }
  
  # Inject coordinates into CDS (Monocle3 requires uppercase: UMAP/TSNE/PCA)
  red_name <- toupper(reduction)
  cds@int_colData$reducedDims[[red_name]] <- embeds[common_cells, ]
  
  return(cds)
}

#' Automatically Identify Trajectory Root Nodes Based on Cell Type
#' @param cds CDS object after running learn_graph
#' @param label_col Column name for cell labels (metadata)
#' @param root_type Name of the starting cell type (e.g., "Basal")
#' @param reduction Name of the reduction space used
get_root_nodes <- function(cds, label_col, root_type, reduction = "UMAP") {
  # 1. Identify target cell IDs
  all_labels <- colData(cds)[[label_col]]
  root_cells <- colnames(cds)[all_labels == root_type]
  
  if (length(root_cells) == 0) {
    stop(paste0("Error: Cell type '", root_type, "' not found in column '", label_col, "'."))
  }
  
  # 2. Retrieve the closest vertices on the principal graph
  # pr_graph_cell_proj_closest_vertex maps each cell to its nearest trajectory node
  closest_vertex <- cds@principal_graph_aux[[reduction]]$pr_graph_cell_proj_closest_vertex
  
  # 3. Identify the node with the highest density of the target cell type
  root_node_counts <- table(closest_vertex[root_cells, ])
  root_node <- names(which.max(root_node_counts))
  
  cat(paste0("Trajectory root identified as Node: ", root_node, " (based on ", root_type, ").\n"))
  return(root_node)
}

#' Custom Color Palette (Optimized for skin or general multi-omic visualization)
get_custom_colors <- function() {
  # Defined colors for standard skin cell types or general use
  colors <- c(
    "Basal" = "#D6E7A3", 
    "Spinous" = "#E4C755", 
    "Granular" = "#EFAD57",
    "Mitotic" = "#73A056", 
    "Channel" = "#D6C0B0", 
    "T_cell" = "#A3C1AD"
  )
  return(colors)
}