# Figure4 C 伪时间与细胞类型箱线图模块 - 独立运行版本
# 伪时间与细胞类型的箱线图

################ Figure 4C - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure4_C_Boxplot.R <input_rds_file> <base_dir> [celltype_column]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数: 细胞类型字段名（与Figure4_A保持一致的字段选择策略）
celltype_column <- if (length(args) >= 3 && args[3] != "") {
  args[3]
} else {
  ""  # 空值表示使用自动检测
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

# 获取B细胞颜色配置
B_cell_color_panel <- get_bcell_color_panel()

###########################################################
# Figure 4C: 伪时间与细胞类型的箱线图
###########################################################

# 尝试加载CDS对象
cds <- NULL
rdata_path <- file.path(files_dir, "flu_B_monocle_cds.RData")

if (file.exists(rdata_path)) {
  cat("加载外部monocle3 CDS数据...\n")
  load(rdata_path)
} else {
  cat("警告：未找到monocle3 CDS对象文件，需要先运行Figure4_A_Trajectory.R\n")
  cat("尝试使用当前数据创建简化的伪时间分析...\n")
  
  # 如果没有CDS对象，创建一个简化版本
  if (requireNamespace("monocle3", quietly = TRUE)) {
    # 检查数据是否适合进行轨迹分析
    if ("combined_cluster" %in% colnames(cell_obj@meta.data) || "seurat_clusters" %in% colnames(cell_obj@meta.data) || "CellType" %in% colnames(cell_obj@meta.data)) {
      
      sub <- cell_obj
      
      # 提取counts数据
      data_m <- GetAssayData(sub, assay = "RNA", layer = "counts")
      # 过滤低表达基因（至少在3个细胞中表达）
      data_m <- data_m[rowSums(data_m > 0) >= 3, ]
      
      cell_metadata <- sub@meta.data
      gene_annotation <- data.frame(gene_short_name = rownames(data_m))
      rownames(gene_annotation) <- rownames(data_m)
      
      # 创建monocle3对象
      cds <- monocle3::new_cell_data_set(data_m,
                                         cell_metadata = cell_metadata,
                                         gene_metadata = gene_annotation)
      
      # 数据预处理
      cds <- monocle3::preprocess_cds(cds, num_dim = 50)
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
      
      # 细胞聚类
      cds <- monocle3::cluster_cells(cds, resolution = 0.001, k = 40, random_seed = 18, verbose = TRUE)
      
      # 使用适当的细胞类型注释字段
      if ("combined_cluster" %in% colnames(sub@meta.data)) {
        cds@clusters$UMAP$clusters <- sub$combined_cluster
      } else if ("seurat_clusters" %in% colnames(sub@meta.data)) {
        cds@clusters$UMAP$clusters <- sub$seurat_clusters
      } else if ("CellType" %in% colnames(sub@meta.data)) {
        cds@clusters$UMAP$clusters <- sub$CellType
      }
      
      # 学习轨迹图谱
      cds <- monocle3::learn_graph(cds, verbose = TRUE,
                                  use_partition = TRUE, close_loop = FALSE,
                                  learn_graph_control = list(minimal_branch_len = 30, rann.k = 10))
      
      # 自动选择根细胞
      if ("CellType" %in% colnames(sub@meta.data)) {
        naive_cells <- which(sub$CellType == "Naive")
        if (length(naive_cells) > 0) {
          root_cell <- colnames(sub)[naive_cells[1]]
          cds <- monocle3::order_cells(cds, root_cells = root_cell)
        } else {
          cds <- monocle3::order_cells(cds)
        }
      } else {
        cds <- monocle3::order_cells(cds)
      }
      
      cat("简化的CDS对象创建完成\n")
    }
  }
}

# 检查是否存在cds对象
if (exists("cds") && !is.null(cds)) {
  cat("\nGenerating Figure 4C...\n")
  
  # 确定使用哪个细胞类型字段（支持用户指定）
  celltype_field <- NULL
  
  if (celltype_column != "" && celltype_column %in% colnames(cds@colData)) {
    # 用户指定的字段存在，优先使用
    celltype_field <- celltype_column
    cat("使用用户指定的细胞类型字段:", celltype_field, "\n")
  } else {
    # 自动检测字段（与Figure4_A保持一致的优先级）
    if ("annotation_final" %in% colnames(cds@colData)) {
      celltype_field <- "annotation_final"
    } else if ("combined_cluster" %in% colnames(cds@colData)) {
      celltype_field <- "combined_cluster"
    } else if ("CellType" %in% colnames(cds@colData)) {
      celltype_field <- "CellType"
    }
    
    if (!is.null(celltype_field)) {
      cat("自动检测到细胞类型字段:", celltype_field, "\n")
    }
  }
  
  if (!is.null(celltype_field)) {
    # 创建数据框
    df <- data.frame(
      pseudotime = monocle3::pseudotime(cds),
      celltype = cds@colData[[celltype_field]]
    )
    
    # 移除NA值
    df <- df[!is.na(df$pseudotime) & !is.na(df$celltype), ]
    
    if (nrow(df) > 0) {
      # 创建箱线图
      p_boxplot <- ggplot2::ggplot(df, ggplot2::aes(y = celltype, x = pseudotime, color = celltype)) +
        ggplot2::geom_boxplot() +
        ggplot2::theme_bw() +
        ggplot2::scale_color_manual(values = B_cell_color_panel) +
        ggplot2::labs(title = "Pseudotime Distribution by Cell Type",
                     x = "Pseudotime",
                     y = "Cell Type") +
        ggplot2::theme(legend.position = "none")
      
      # 保存图片
      ggsave(file.path(plots_dir, "Figure_4C.pdf"), plot = p_boxplot, width = 9, height = 6)
      
      # 保存统计数据
      pseudotime_stats <- df %>%
        group_by(celltype) %>%
        summarise(
          count = n(),
          mean_pseudotime = mean(pseudotime, na.rm = TRUE),
          median_pseudotime = median(pseudotime, na.rm = TRUE),
          sd_pseudotime = sd(pseudotime, na.rm = TRUE),
          .groups = 'drop'
        )
      
      write.csv(pseudotime_stats, file.path(files_dir, "Figure4C_pseudotime_stats.csv"), row.names = FALSE)
      
      cat("Figure 4C generated successfully\n")
      cat("细胞类型分布:\n")
      print(table(df$celltype))
      
    } else {
      cat("Warning: No valid data for Figure 4C\n")
    }
  } else {
    cat("Warning: No suitable cell type annotation found for Figure 4C\n")
  }
} else {
  cat("Warning: CDS object not found, skipping Figure 4C\n")
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure4C_session_info.txt"))  # 将输出重定向到文件
cat("Figure 4C Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Parameters:\n")
cat("  - Celltype column parameter:", ifelse(celltype_column == "", "auto-detect", celltype_column), "\n")
if (exists("celltype_field") && !is.null(celltype_field)) {
  cat("  - Celltype field used:", celltype_field, "\n")
}
sessionInfo()
sink()  # 关闭重定向

cat("Figure 4C 模块运行完成！\n")