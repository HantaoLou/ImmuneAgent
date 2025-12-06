# Figure3 C B细胞类型分布图模块 - 独立运行版本
# UMAP - B cells derived from flu BCells cohort

################ Figure 3C - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure3_C_CellType.R <input_rds_file> <base_dir> [celltype_column]")
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

# 获取颜色配置
my36colors <- get_color_palette()

###############################################################################
#'                          Manuscipt: figure3C                              '#
###############################################################################

## Figure 3C; 
## UMAP - B cells derived from flu BCells cohort

meta <- cell_obj@meta.data

# 使用metadata中的UMAP坐标
if("UMAP_1" %in% colnames(meta) && "UMAP_2" %in% colnames(meta)) {
  meta$umap_1 <- meta$UMAP_1
  meta$umap_2 <- meta$UMAP_2
  cat("成功获取UMAP坐标\n")
} else {
  cat("Warning: UMAP coordinates not found in metadata\n")
  # 如果没有UMAP坐标，尝试从Seurat对象的reduction中获取
  if ("umap" %in% names(cell_obj@reductions)) {
    umap_coords <- Embeddings(cell_obj, reduction = "umap")
    meta$umap_1 <- umap_coords[, 1]
    meta$umap_2 <- umap_coords[, 2]
    cat("从Seurat对象的reduction中获取UMAP坐标\n")
  } else {
    stop("无法找到UMAP坐标，请确保数据包含UMAP_1和UMAP_2列或umap reduction")
  }
}

# 检查细胞类型字段是否存在
if (!celltype_column %in% colnames(meta)) {
  stop(paste("细胞类型字段", celltype_column, "不存在。可用字段:", paste(colnames(meta), collapse = ", ")))
}

# 使用指定的细胞类型字段作为分组变量
meta$plot_celltype <- as.factor(meta[[celltype_column]])

# 显示细胞类型统计
cat("使用细胞类型字段:", celltype_column, "\n")
cat("细胞类型分布:\n")
print(table(meta$plot_celltype))

# 创建UMAP图
p <- ggplot(meta, aes(x = umap_1, y = umap_2, color = plot_celltype)) +
  geom_point(size = 0.15,
             shape = 16,
             stroke = 0) +
  theme_void() +
  scale_color_manual(values = my36colors, name = '') +
  theme(aspect.ratio = 1,
        legend.position = "")

# 保存图片
ggsave(
  file.path(plots_dir, "Figure_3C.pdf"), 
  plot = p,
  width = 2,
  height = 2,
  dpi = 300
)

cat("Figure 3C generated successfully\n")
cat("细胞总数:", nrow(meta), "\n")
cat("细胞类型数量:", length(unique(meta$plot_celltype)), "\n")

# 保存细胞类型统计信息
celltype_stats <- table(meta$plot_celltype)
write.csv(celltype_stats, file.path(files_dir, "Figure3C_celltype_stats.csv"))

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure3C_session_info.txt"))  # 将输出重定向到文件
cat("Figure 3C Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Celltype column:", celltype_column, "\n")
cat("Cell types found:", paste(names(celltype_stats), collapse = ", "), "\n")
cat("Total cells:", nrow(meta), "\n")
sessionInfo()
sink()  # 关闭重定向

cat("Figure 3C 模块运行完成！\n")