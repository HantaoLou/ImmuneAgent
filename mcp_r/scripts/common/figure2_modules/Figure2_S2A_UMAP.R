# Figure2 S2A UMAP可视化模块 - 独立运行版本
# UMAP - B细胞亚群可视化

################ Figure 2 S2A - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_S2A_UMAP.R <input_rds_file> <base_dir>")
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
paths <- initialize_figure2_environment(input_rds_file, base_dir, "S2A")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

# 获取颜色方案
my36colors <- get_color_palette()

###############################################################################
#'                          Manuscipt: figureS2A                            '#
###############################################################################

## Figure S2A; 
## UMAP - The B cell subsets of PBMC derived from Priest, et al

# 检查数据结构
print(paste("Number of cells:", ncol(cell_obj)))
print(paste("Available metadata columns:", paste(colnames(cell_obj@meta.data), collapse=", ")))
print(paste("Unique cell types:", paste(unique(cell_obj@meta.data$CellType), collapse=", ")))

# 创建UMAP可视化图
# 由于fluBcells数据中UMAP坐标存储在metadata中，需要手动创建可视化
if("UMAP_1" %in% colnames(cell_obj@meta.data) && "UMAP_2" %in% colnames(cell_obj@meta.data)) {
  # 提取UMAP坐标和元数据
  umap_data <- data.frame(
    UMAP_1 = cell_obj@meta.data$UMAP_1,
    UMAP_2 = cell_obj@meta.data$UMAP_2,
    CellType = cell_obj@meta.data$CellType
  )
  
  # 生成UMAP图展示B细胞亚群
  p <- ggplot(umap_data, aes(x = UMAP_1, y = UMAP_2, color = CellType)) +
    geom_point(size = 0.5, alpha = 0.7) +
    scale_color_manual(values = my36colors[1:length(unique(umap_data$CellType))]) +
    labs(title = "UMAP - The B cell subsets from flu BCells data",
         x = "UMAP_1", y = "UMAP_2") +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 10)) +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1)))
  
  # 保存图片
  ggsave(file.path(plots_dir, "Figure_S2A_fluBcells_UMAP.pdf"), p, width = 12, height = 8, create.dir = TRUE)
  
  print("UMAP visualization saved successfully!")
  
} else {
  print("UMAP coordinates not found in metadata")
}

cat("Figure S2A UMAP可视化模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")