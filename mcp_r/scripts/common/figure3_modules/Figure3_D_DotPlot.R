# Figure3 D DotPlot标记基因表达图模块 - 独立运行版本
# Dotplot representation of marker expression across different B cell subsets

################ Figure 3D - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure3_D_DotPlot.R <input_rds_file> <base_dir> [celltype_column]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 生信核心参数：细胞类型字段名
celltype_column <- if (length(args) >= 3 && args[3] != "") {
  args[3]
} else {
  "CellType"  # 默认字段名
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
source(file.path(script_dir, "Figure3_Utils.R"))

# 加载必需的R包
load_required_packages()

# 创建输出目录
output_dirs <- create_output_directories(base_dir)
plots_dir <- output_dirs$plots_dir
files_dir <- output_dirs$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

# 获取B细胞marker基因
markers_a1 <- get_bcell_markers()

###############################################################################
#'                          Manuscipt: figure3D                              '#
###############################################################################

## Figure 3D; 
## Dotplot representation of marker expression across different B cell subsets. 

# 检查细胞类型字段是否存在
if (!celltype_column %in% colnames(cell_obj@meta.data)) {
  stop(paste("细胞类型字段", celltype_column, "不存在。可用字段:", paste(colnames(cell_obj@meta.data), collapse = ", ")))
}

# 清理指定的细胞类型字段，去除重复的因子水平
cell_obj@meta.data[[celltype_column]] <- as.character(cell_obj@meta.data[[celltype_column]])
# 获取唯一的细胞类型并重新创建因子
unique_celltypes <- unique(cell_obj@meta.data[[celltype_column]])
cell_obj@meta.data[[celltype_column]] <- factor(cell_obj@meta.data[[celltype_column]], levels = unique_celltypes)

# 设置细胞类型为Idents
Idents(cell_obj) <- cell_obj@meta.data[[celltype_column]]

# 显示细胞类型统计
cat("使用细胞类型字段:", celltype_column, "\n")
cat("细胞类型分布:\n")
print(table(cell_obj@meta.data[[celltype_column]]))

# 将markers_a1列表转换为基因向量
all_markers <- unique(unlist(markers_a1))

# 检查哪些基因在数据中存在
available_genes <- rownames(cell_obj)
valid_markers <- all_markers[all_markers %in% available_genes]

cat("总标记基因数量:", length(all_markers), "\n")
cat("数据中可用的标记基因数量:", length(valid_markers), "\n")

if(length(valid_markers) > 0) {
  cat("使用的标记基因:", paste(valid_markers, collapse = ", "), "\n")
  
  # 创建DotPlot
  p <- DotPlot(object = cell_obj, features = valid_markers, scale = T) + 
    scale_colour_gradientn(colors = brewer.pal(9, "YlGnBu")) + 
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    labs(title = "B Cell Marker Expression",
         x = "Genes",
         y = "Cell Types")
  
  # 保存图片
  ggsave(file.path(plots_dir, "Figure_3D.pdf"), 
         plot = p, 
         width = 18, 
         height = 6,
         dpi = 300)
  
  cat("Figure 3D 已成功生成\n")
  
  # 保存标记基因信息
  marker_info <- data.frame(
    Gene = valid_markers,
    Available = TRUE,
    stringsAsFactors = FALSE
  )
  
  # 添加缺失的基因信息
  missing_markers <- all_markers[!all_markers %in% available_genes]
  if(length(missing_markers) > 0) {
    missing_info <- data.frame(
      Gene = missing_markers,
      Available = FALSE,
      stringsAsFactors = FALSE
    )
    marker_info <- rbind(marker_info, missing_info)
  }
  
  write.csv(marker_info, file.path(files_dir, "Figure3D_marker_genes.csv"), row.names = FALSE)
  
  # 保存每个细胞类型的标记基因列表
  markers_by_celltype <- data.frame(
    CellType = rep(names(markers_a1), sapply(markers_a1, length)),
    Gene = unlist(markers_a1),
    Available = unlist(markers_a1) %in% available_genes,
    stringsAsFactors = FALSE
  )
  write.csv(markers_by_celltype, file.path(files_dir, "Figure3D_markers_by_celltype.csv"), row.names = FALSE)
  
} else {
  cat("警告：没有找到有效的标记基因，跳过Figure 3D生成\n")
  
  # 保存缺失基因信息用于调试
  missing_genes_info <- data.frame(
    Gene = all_markers,
    Available = FALSE,
    stringsAsFactors = FALSE
  )
  write.csv(missing_genes_info, file.path(files_dir, "Figure3D_missing_genes.csv"), row.names = FALSE)
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure3D_session_info.txt"))  # 将输出重定向到文件
cat("Figure 3D Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Celltype column:", celltype_column, "\n")
cat("Total marker genes:", length(all_markers), "\n")
cat("Available marker genes:", length(valid_markers), "\n")
cat("Cell types:", paste(unique_celltypes, collapse = ", "), "\n")
sessionInfo()
sink()  # 关闭重定向

cat("Figure 3D 模块运行完成！\n")