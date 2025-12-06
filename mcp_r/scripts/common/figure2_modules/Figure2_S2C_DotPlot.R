# Figure2 S2C Marker基因点图模块 - 独立运行版本
# Marker基因点图可视化

################ Figure 2 S2C - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure2_S2C_DotPlot.R <input_rds_file> <base_dir> [min_pct] [min_expression]")
}

input_rds_file <- args[1]
base_dir <- args[2]

# 表达阈值过滤参数（可选，保持向后兼容的默认值）
min_pct <- 0.1          # 默认值：10%细胞表达
min_expression <- 0.25  # 默认值：平均表达0.25

# 解析可选参数
if (length(args) >= 3) {
  min_pct <- as.numeric(args[3])
  if (is.na(min_pct) || min_pct < 0 || min_pct > 1) {
    stop("Error: min_pct must be between 0 and 1")
  }
}

if (length(args) >= 4) {
  min_expression <- as.numeric(args[4])
  if (is.na(min_expression) || min_expression < 0) {
    stop("Error: min_expression must be >= 0")
  }
}

# 输出参数信息
cat("表达阈值过滤参数:\n")
cat("  min_pct: ", min_pct, "\n")
cat("  min_expression: ", min_expression, "\n")

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
paths <- initialize_figure2_environment(input_rds_file, base_dir, "S2C")
output_dir <- paths$output_dir
plots_dir <- paths$plots_dir
files_dir <- paths$files_dir

# 加载和预处理数据
cell_obj <- load_and_preprocess_data(input_rds_file)

# 获取B细胞marker基因
markers_a1 <- get_bcell_markers()

###############################################################################
#'                    Manuscipt: figureS2C                                   '#
###############################################################################

## Figure S2C; 
## Dot plot of marker genes relating to the cell types

# 检查是否存在基因表达数据和标记基因
if(exists("cell_obj") && exists("markers_a1")) {
  
  # 过滤掉不存在的基因
  all_markers <- unlist(markers_a1)
  available_markers <- intersect(all_markers, rownames(cell_obj))
  
  if(length(available_markers) > 0) {
    # 计算每个基因的表达统计并应用阈值过滤
    cat("\n正在计算基因表达统计...\n")
    
    gene_stats <- data.frame(
      gene = available_markers,
      mean_expr = numeric(length(available_markers)),
      pct_expr = numeric(length(available_markers))
    )
    
    # 获取表达数据
    expr_data <- GetAssayData(cell_obj, slot = "data")
    
    for(i in 1:length(available_markers)) {
      gene <- available_markers[i]
      gene_expr <- expr_data[gene, ]
      
      # 计算平均表达水平
      gene_stats$mean_expr[i] <- mean(gene_expr)
      
      # 计算表达细胞比例
      gene_stats$pct_expr[i] <- sum(gene_expr > 0) / length(gene_expr)
    }
    
    # 应用阈值过滤
    filtered_genes <- gene_stats[
      gene_stats$mean_expr >= min_expression & 
      gene_stats$pct_expr >= min_pct, 
      "gene"
    ]
    
    # 输出过滤统计信息
    cat("\n表达阈值过滤统计:\n")
    cat("原始marker基因数:", length(available_markers), "\n")
    cat("过滤后基因数:", length(filtered_genes), "\n")
    cat("过滤掉的基因数:", length(available_markers) - length(filtered_genes), "\n")
    
    # 输出被过滤掉的基因
    filtered_out_genes <- setdiff(available_markers, filtered_genes)
    if(length(filtered_out_genes) > 0) {
      cat("被过滤的基因:", paste(filtered_out_genes, collapse=", "), "\n")
    }
    
    if(length(filtered_genes) > 0) {
      # 使用过滤后的基因生成点图
      p <- DotPlot(object = cell_obj, features = filtered_genes, scale = T) + 
        scale_colour_gradientn(colors=brewer.pal(9, "YlGnBu")) + theme_bw() +
        theme(axis.text.x = element_text(angle = 90)) 
      
      ggsave(file.path(plots_dir, "Figure_S2C_marker_genes_dotplot.pdf"), p, width=16, height=8)
      
      print("Marker genes dot plot saved successfully!")
      
    } else {
      stop("No genes passed the expression thresholds. Consider lowering min_pct or min_expression.")
    }
    
    # 输出缺失的基因（不在数据中的基因）
    missing_markers <- setdiff(all_markers, rownames(cell_obj))
    if(length(missing_markers) > 0) {
      cat("数据中缺失的marker基因:", paste(missing_markers, collapse=", "), "\n")
    }
    
  } else {
    print("No marker genes found in the dataset")
  }
  
} else {
  print("Marker genes dot plot skipped: cell_obj or markers_a1 not found")
}

cat("Figure S2C Marker基因点图模块执行完成\n")
cat("输出目录:", output_dir, "\n")
cat("图片保存至:", plots_dir, "\n")