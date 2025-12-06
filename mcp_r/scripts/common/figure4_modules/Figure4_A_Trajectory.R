# Figure4 A UMAP伪时间轨迹分析模块 - 独立运行版本
# UMAP - the pseudotime trajectory analysis of the single-cell RNA sequencing (scRNA-seq) data from Influenza

################ Figure 4A - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure4_A_Trajectory.R <input_rds_file> <base_dir> [num_dim] [cluster_resolution] [min_gene_cells] [root_celltype]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数1: 主成分维度数（影响降维质量）
num_dim <- if (length(args) >= 3 && args[3] != "") {
  as.numeric(args[3])
} else {
  50  # 默认50个主成分，适合大多数单细胞数据
}

# 生信核心参数2: 聚类分辨率（影响细胞群体粒度）
cluster_resolution <- if (length(args) >= 4 && args[4] != "") {
  as.numeric(args[4])
} else {
  0.001  # 默认低分辨率，适合轨迹分析
}

# 生信核心参数3: 基因过滤阈值（影响基因质量控制）
min_gene_cells <- if (length(args) >= 5 && args[5] != "") {
  as.numeric(args[5])
} else {
  3  # 默认至少在3个细胞中表达
}

# 生信核心参数4: 根细胞类型（影响轨迹起点选择）
root_celltype <- if (length(args) >= 6 && args[6] != "") {
  args[6]
} else {
  "Naive"  # 默认使用Naive B细胞作为起点
}

# 获取当前脚本所在目录并加载工具函数
get_script_dir <- function() {
  # 方法1：尝试使用commandArgs获取脚本路径
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("--file=", "", file_arg[1])
    return(dirname(script_path))
  }
  
  # 方法2：尝试使用sys.frame（在某些环境下可用）
  tryCatch({
    script_path <- sys.frame(1)$ofile
    if (!is.null(script_path)) {
      return(dirname(script_path))
    }
  }, error = function(e) {})
  
  # 方法3：回退到当前工作目录
  return(getwd())
}

script_dir <- get_script_dir()
source(file.path(script_dir, "Figure4_Utils.R"))

# 加载必需的R包
load_required_packages()

# 创建输出目录
output_dirs <- create_output_directories(base_dir)
plots_dir <- output_dirs$plots_dir
files_dir <- output_dirs$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

###############################################################################
#'                          Manuscipt: figure4A                              '#
###############################################################################

## Figure 4A; 
## UMAP - the pseudotime trajectory analysis of the single-cell RNA sequencing (scRNA-seq) data from Influenza

# 检查是否有必要的包
if (!requireNamespace("monocle3", quietly = TRUE)) {
  cat("Warning: monocle3 package not available, skipping Figure 4A\n")
} else {
  # 检查数据是否适合进行轨迹分析
  if ("combined_cluster" %in% colnames(cell_obj@meta.data) || "seurat_clusters" %in% colnames(cell_obj@meta.data) || "CellType" %in% colnames(cell_obj@meta.data)) {
    
    sub <- cell_obj
    
    # monocle3分析
    cat("Starting monocle3 trajectory analysis...\n")
    
    # 提取counts数据
    data_m <- GetAssayData(sub, assay = "RNA", layer = "counts")
    # 过滤低表达基因（使用参数化阈值）
    genes_before <- nrow(data_m)
    data_m <- data_m[rowSums(data_m > 0) >= min_gene_cells, ]
    genes_after <- nrow(data_m)
    cat("基因过滤: 从", genes_before, "个基因过滤到", genes_after, "个基因 (阈值:", min_gene_cells, "个细胞)\n")
    
    cell_metadata <- sub@meta.data
    gene_annotation <- data.frame(gene_short_name = rownames(data_m))
    rownames(gene_annotation) <- rownames(data_m)
    
    # 创建monocle3对象
    cds <- monocle3::new_cell_data_set(data_m,
                                       cell_metadata = cell_metadata,
                                       gene_metadata = gene_annotation)
    
    # 数据预处理（使用参数化的主成分数量）
    cat("使用", num_dim, "个主成分进行降维\n")
    cds <- monocle3::preprocess_cds(cds, num_dim = num_dim)
    cds <- monocle3::reduce_dimension(cds, preprocess_method = "PCA")
    
    # 使用Seurat的UMAP坐标（如果可用）
    if ("UMAP_1" %in% colnames(sub@meta.data) && "UMAP_2" %in% colnames(sub@meta.data)) {
      cds.embed <- cds@int_colData$reducedDims$UMAP
      int.embed <- sub@meta.data[, c("UMAP_1", "UMAP_2")]
      colnames(int.embed) <- c("UMAP_1", "UMAP_2")
      int.embed <- as.matrix(int.embed[rownames(cds.embed), ])
      cds@int_colData$reducedDims$UMAP <- int.embed
      cat("Using existing UMAP coordinates\n")
    } else if ("umap" %in% names(sub@reductions)) {
      cds.embed <- cds@int_colData$reducedDims$UMAP
      int.embed <- Embeddings(sub, reduction = "umap")
      int.embed <- int.embed[rownames(cds.embed), ]
      cds@int_colData$reducedDims$UMAP <- int.embed
      cat("Using Seurat UMAP reduction\n")
    }
    
    # 细胞聚类（使用参数化的分辨率）
    cat("使用聚类分辨率:", cluster_resolution, "\n")
    cds <- monocle3::cluster_cells(cds, resolution = cluster_resolution, k = 40, random_seed = 18, verbose = TRUE)
    
    # 使用适当的细胞类型注释字段
    if ("combined_cluster" %in% colnames(sub@meta.data)) {
      cds@clusters$UMAP$clusters <- sub$combined_cluster
      cat("Using combined_cluster for cell type annotation\n")
    } else if ("seurat_clusters" %in% colnames(sub@meta.data)) {
      cds@clusters$UMAP$clusters <- sub$seurat_clusters
      cat("Using seurat_clusters for cell type annotation\n")
    } else if ("CellType" %in% colnames(sub@meta.data)) {
      cds@clusters$UMAP$clusters <- sub$CellType
      cat("Using CellType for cell type annotation\n")
    }
    
    # 学习轨迹图谱
    cds <- monocle3::learn_graph(cds, verbose = TRUE,
                                use_partition = TRUE, close_loop = FALSE,
                                learn_graph_control = list(minimal_branch_len = 30, rann.k = 10))
    
    # 自动选择根细胞（使用参数化的细胞类型作为起点）
    cat("寻找根细胞类型:", root_celltype, "\n")
    
    root_cell_found <- FALSE
    
    if ("combined_cluster" %in% colnames(sub@meta.data)) {
      # 在combined_cluster中查找指定的细胞类型
      # 支持模糊匹配（如"naive B"匹配"Naive"）
      target_cells <- which(grepl(root_celltype, sub$combined_cluster, ignore.case = TRUE))
      if (length(target_cells) > 0) {
        root_cell <- colnames(sub)[target_cells[1]]
        cds <- monocle3::order_cells(cds, root_cells = root_cell)
        cat("根细胞自动选择成功 (", root_celltype, " from combined_cluster)\n")
        root_cell_found <- TRUE
      }
    }
    
    if (!root_cell_found && "CellType" %in% colnames(sub@meta.data)) {
      # 在CellType中查找指定的细胞类型
      target_cells <- which(sub$CellType == root_celltype)
      if (length(target_cells) > 0) {
        root_cell <- colnames(sub)[target_cells[1]]
        cds <- monocle3::order_cells(cds, root_cells = root_cell)
        cat("根细胞自动选择成功 (", root_celltype, " from CellType)\n")
        root_cell_found <- TRUE
      }
    }
    
    if (!root_cell_found) {
      # 如果没有找到指定的细胞类型，手动选择
      cds <- monocle3::order_cells(cds)
      cat("警告: 未找到", root_celltype, "细胞类型，需要手动选择根细胞\n")
    }
    
    # 绘制伪时间图
    p <- monocle3::plot_cells(cds = cds,
                             color_cells_by = "pseudotime",
                             show_trajectory_graph = FALSE,
                             trajectory_graph_color = "white",
                             trajectory_graph_segment_size = 0.5,
                             graph_label_size = 2,
                             cell_size = 1,
                             label_cell_groups = FALSE,
                             label_groups_by_cluster = FALSE,
                             label_branch_points = FALSE,
                             label_roots = FALSE,
                             label_leaves = FALSE)
    
    # 保存图片和数据
    ggsave(file.path(plots_dir, "Figure_4A.pdf"), plot = p, width = 9, height = 8)
    save(cds, file = file.path(files_dir, "flu_B_monocle_cds.RData"))
    
    cat("Figure 4A generated successfully\n")
    cat("Monocle3 CDS object saved to:", file.path(files_dir, "flu_B_monocle_cds.RData"), "\n")
    
  } else {
    cat("Warning: No suitable cell type annotation found (combined_cluster, seurat_clusters, or CellType), skipping Figure 4A\n")
  }
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure4A_session_info.txt"))  # 将输出重定向到文件
cat("Figure 4A Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Parameters:\n")
cat("  - Number of dimensions:", num_dim, "\n")
cat("  - Cluster resolution:", cluster_resolution, "\n")
cat("  - Min gene cells:", min_gene_cells, "\n")
cat("  - Root cell type:", root_celltype, "\n")
sessionInfo()
sink()  # 关闭重定向

cat("Figure 4A 模块运行完成！\n")