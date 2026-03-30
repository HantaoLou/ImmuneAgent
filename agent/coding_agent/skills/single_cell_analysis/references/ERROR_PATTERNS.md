## Seurat Load Error
- match: "Error in validObject"
- failure_kind: MCP_TOOL_ERROR
- description: Seurat object version mismatch
- recovery: Run UpdateSeuratObject() on the RDS file, or convert to h5ad format

## Missing Package
- match: "there is no package called"
- failure_kind: MCP_TOOL_ERROR
- description: Required R package not installed
- recovery: Install the missing package in the R environment (renv::install)

## Memory Error
- match: "cannot allocate vector of size"
- failure_kind: PROCESS_CRASH
- description: R ran out of memory processing the dataset
- recovery: Subsample the data or increase available RAM. For >100k cells, consider downsampling to 50k

## Monocle3 Error
- match: "Error in learn_graph"
- failure_kind: MCP_TOOL_ERROR
- description: Monocle3 trajectory learning failed, often due to too few cells
- recovery: Increase the cell population by loosening filters or choosing a broader lineage

## No Cells After Filter
- match: "0 cells passed"
- failure_kind: MCP_TOOL_ERROR
- description: QC or subsetting removed all cells
- recovery: Check species-specific mitochondrial prefix (MT- vs mt-) and adjust QC thresholds

## Empty Cluster
- match: "not enough cells in cluster"
- failure_kind: MCP_TOOL_ERROR
- description: Some clusters have too few cells for analysis
- recovery: Lower clustering resolution or merge small clusters

## CellChat Insufficient Data
- match: "not enough cell types"
- failure_kind: MCP_TOOL_ERROR
- description: CellChat requires at least 2 cell types with sufficient cells
- recovery: Check that cell type annotation succeeded and that the dataset has diverse cell populations
