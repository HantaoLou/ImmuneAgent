# Figure2 B1细胞类型分布模块 - 独立运行版本
# 细胞类型分布可视化

################ Figure 2 B1 - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_B1_CellType.R <input_rds_file> <base_dir>")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 获取当前脚本所在目录并加载工具函数
# 使用更可靠的方法获取脚本路径
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
source(file.path(script_dir, "Figure2_Utils.R"))

# 初始化环境和创建输出目录
paths <- initialize_figure2_environment(input_rds_file, base_dir, "B1")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

# 获取颜色方案
my36colors <- get_color_palette()

###############################################################################
#'                          Manuscipt: figure2B1                            '#
###############################################################################

## Figure 2B.1; 
## Cell type distribution

if("CellType" %in% colnames(cell_obj@meta.data)) {
  # 手动创建DimPlot效果
  umap_data <- data.frame(
    UMAP_1 = cell_obj@meta.data$UMAP_1,
    UMAP_2 = cell_obj@meta.data$UMAP_2,
    CellType = cell_obj@meta.data$CellType
  )
  
  p <- ggplot(umap_data, aes(x = UMAP_1, y = UMAP_2, color = CellType)) +
    geom_point(size = 0.5, alpha = 0.7) +
    scale_color_manual(values = my36colors[1:length(unique(umap_data$CellType))]) +
    labs(title = "UMAP - B cell subtypes",
         x = "UMAP_1", y = "UMAP_2") +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10))
  
  ggsave(file.path(plots_dir, "Figure_2B1_celltype_distribution.pdf"), p, width = 10, height = 8)
  
  print("Cell type distribution UMAP saved successfully!")
}

cat("Figure 2B1细胞类型分布模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")