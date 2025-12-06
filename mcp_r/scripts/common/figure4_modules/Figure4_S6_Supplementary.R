# Figure4 S6A/B/C/D 补充图形模块 - 独立运行版本
# Supplementary figures for trajectory analysis

################ Figure S6A/B/C/D - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure4_S6_Supplementary.R <input_rds_file> <base_dir>")
}

input_rds_file <- args[1]
base_dir <- args[2]

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

# 获取轨迹路径定义
trajectory_paths <- get_trajectory_paths()
flu_path2 <- trajectory_paths$path2

###############################################################################
#'                     Manuscipt: figureS6A/B/C/D                            '#
###############################################################################

## Figure S6A/B/C/D
## (A) B cell activation related transcriptional markers (CD86, ITGB2, ITGB2-AS1, SOX5, TNFRSF1B and FAS) across pseudotime trajectory 2. 
## (B) Atypical B cell related transcriptional markers (ITGAX, ZEB2, FCRL3, FCRL4, FCRL5) across pseudotime trajectory 2.  
## (C) Isotype-specific immunoglobulin expression dynamics (IGHA1, IGHD, IGHG1, IGHG2, IGHG3, and IGHM) along the pseudotime trajectory 2
## (D) Expression patterns of transcription factors (AFF3, BACH2, IRF8) along trajectory 2.

# 尝试加载CDS对象
cds <- NULL
rdata_path <- file.path(files_dir, "flu_B_monocle_cds.RData")

if (file.exists(rdata_path)) {
  cat("加载外部monocle3 CDS数据...\n")
  load(rdata_path)
} else {
  cat("警告：未找到monocle3 CDS对象，需要先运行Figure4_A_Trajectory.R\n")
}

# 检查是否存在cds对象并创建子集
flu_cds_subset_2 <- NULL

if (!is.null(cds)) {
  # 检查annotation_final列是否存在
  has_annotation_final <- "annotation_final" %in% colnames(cds@colData)
  has_celltype <- "CellType" %in% colnames(cds@colData)
  
  if (has_annotation_final) {
    flu_cds_subset_2 <- cds[, cds@colData$annotation_final %in% flu_path2]
  } else if (has_celltype) {
    flu_cds_subset_2 <- cds[, cds@colData$CellType %in% flu_path2]
  } else {
    cat("警告：未找到annotation_final或CellType列，使用完整CDS对象\n")
    flu_cds_subset_2 <- cds
  }
}

if (exists("flu_cds_subset_2") && !is.null(flu_cds_subset_2)) {
  cat("开始生成FigureS6A/B/C/D...\n")
  
  # 设置细胞类型信息
  if ("annotation_final" %in% colnames(flu_cds_subset_2@colData)) {
    flu_cds_subset_2$celltype <- flu_cds_subset_2@colData$annotation_final
  } else if ("CellType" %in% colnames(flu_cds_subset_2@colData)) {
    flu_cds_subset_2$celltype <- flu_cds_subset_2@colData$CellType
  } else if (exists("cell_obj") && "CellType" %in% colnames(cell_obj@meta.data)) {
    flu_cds_subset_2$celltype <- cell_obj@meta.data[colnames(flu_cds_subset_2), "CellType"]
  } else {
    cat("警告：未找到细胞类型注释，使用默认分组\n")
    flu_cds_subset_2$celltype <- "Unknown"
  }
  
  # 根据路径过滤细胞（如果flu_path2已定义）
  if (exists("flu_path2") && length(flu_path2) > 0) {
    flu_cds_subset_2 <- flu_cds_subset_2[, flu_cds_subset_2$celltype %in% flu_path2]
    cat("使用flu_path2过滤细胞，剩余细胞数:", ncol(flu_cds_subset_2), "\n")
  }
  
  # 定义基因集
  genes.1 <- c('CD86','ITGB2','ITGB2-AS1','SOX5','TNFRSF1B','FAS')
  genes.2 <- c('ITGAX','ZEB2','FCRL3','FCRL4','FCRL5')
  genes.3 <- c('IGHA1','IGHD','IGHG1','IGHG2','IGHG3','IGHM')
  genes.4 <- c('AFF3','BACH2','IRF8')
  
  # 检查基因可用性并过滤
  available_genes <- rownames(flu_cds_subset_2)
  genes.1 <- genes.1[genes.1 %in% available_genes]
  genes.2 <- genes.2[genes.2 %in% available_genes]
  genes.3 <- genes.3[genes.3 %in% available_genes]
  genes.4 <- genes.4[genes.4 %in% available_genes]
  
  cat("基因可用性检查:\n")
  cat("FigureS6A基因:", length(genes.1), "个可用\n")
  cat("FigureS6B基因:", length(genes.2), "个可用\n")
  cat("FigureS6C基因:", length(genes.3), "个可用\n")
  cat("FigureS6D基因:", length(genes.4), "个可用\n")
  
  # 生成FigureS6A - B细胞激活相关标记基因
  if (length(genes.1) > 0) {
    tryCatch({
      p <- plot_genes_in_pseudotime(flu_cds_subset_2[genes.1,], color_cells_by="celltype", ncol = 6)
      ggsave(file.path(plots_dir, "FigureS6A.pdf"), plot = p, width = 16, height = 3)
      cat("FigureS6A生成成功\n")
      
      # 保存基因表达数据（使用monocle3兼容的方法）
      tryCatch({
        gene_expr_data <- as.data.frame(t(as.matrix(exprs(flu_cds_subset_2[genes.1,]))))
        gene_expr_data$pseudotime <- pseudotime(flu_cds_subset_2)
        gene_expr_data$celltype <- flu_cds_subset_2$celltype
        write.csv(gene_expr_data, file.path(files_dir, "FigureS6A_gene_expression.csv"), row.names = TRUE)
      }, error = function(e) {
        cat("保存FigureS6A基因表达数据失败:", e$message, "\n")
      })
      
    }, error = function(e) {
      cat("FigureS6A生成失败:", e$message, "\n")
    })
  } else {
    cat("警告：FigureS6A无可用基因\n")
  }
  
  # 生成FigureS6B - 非典型B细胞相关标记基因
  if (length(genes.2) > 0) {
    tryCatch({
      p <- plot_genes_in_pseudotime(flu_cds_subset_2[genes.2,], color_cells_by="celltype", ncol = 6)
      ggsave(file.path(plots_dir, "FigureS6B.pdf"), plot = p, width = 16, height = 3)
      cat("FigureS6B生成成功\n")
      
      # 保存基因表达数据（使用monocle3兼容的方法）
      tryCatch({
        gene_expr_data <- as.data.frame(t(as.matrix(exprs(flu_cds_subset_2[genes.2,]))))
        gene_expr_data$pseudotime <- pseudotime(flu_cds_subset_2)
        gene_expr_data$celltype <- flu_cds_subset_2$celltype
        write.csv(gene_expr_data, file.path(files_dir, "FigureS6B_gene_expression.csv"), row.names = TRUE)
      }, error = function(e) {
        cat("保存FigureS6B基因表达数据失败:", e$message, "\n")
      })
      
    }, error = function(e) {
      cat("FigureS6B生成失败:", e$message, "\n")
    })
  } else {
    cat("警告：FigureS6B无可用基因\n")
  }
  
  # 生成FigureS6C - 免疫球蛋白表达动态
  if (length(genes.3) > 0) {
    tryCatch({
      p <- plot_genes_in_pseudotime(flu_cds_subset_2[genes.3,], color_cells_by="celltype", ncol = 6)
      ggsave(file.path(plots_dir, "FigureS6C.pdf"), plot = p, width = 16, height = 3)
      cat("FigureS6C生成成功\n")
      
      # 保存基因表达数据（使用monocle3兼容的方法）
      tryCatch({
        gene_expr_data <- as.data.frame(t(as.matrix(exprs(flu_cds_subset_2[genes.3,]))))
        gene_expr_data$pseudotime <- pseudotime(flu_cds_subset_2)
        gene_expr_data$celltype <- flu_cds_subset_2$celltype
        write.csv(gene_expr_data, file.path(files_dir, "FigureS6C_gene_expression.csv"), row.names = TRUE)
      }, error = function(e) {
        cat("保存FigureS6C基因表达数据失败:", e$message, "\n")
      })
      
    }, error = function(e) {
      cat("FigureS6C生成失败:", e$message, "\n")
    })
  } else {
    cat("警告：FigureS6C无可用基因\n")
  }
  
  # 生成FigureS6D - 转录因子表达模式
  if (length(genes.4) > 0) {
    tryCatch({
      p <- plot_genes_in_pseudotime(flu_cds_subset_2[genes.4,], color_cells_by="celltype", ncol = 6)
      ggsave(file.path(plots_dir, "FigureS6D.pdf"), plot = p, width = 9, height = 3)
      cat("FigureS6D生成成功\n")
      
      # 保存基因表达数据（使用monocle3兼容的方法）
      tryCatch({
        gene_expr_data <- as.data.frame(t(as.matrix(exprs(flu_cds_subset_2[genes.4,]))))
        gene_expr_data$pseudotime <- pseudotime(flu_cds_subset_2)
        gene_expr_data$celltype <- flu_cds_subset_2$celltype
        write.csv(gene_expr_data, file.path(files_dir, "FigureS6D_gene_expression.csv"), row.names = TRUE)
      }, error = function(e) {
        cat("保存FigureS6D基因表达数据失败:", e$message, "\n")
      })
      
    }, error = function(e) {
      cat("FigureS6D生成失败:", e$message, "\n")
    })
  } else {
    cat("警告：FigureS6D无可用基因\n")
  }
  
  # 保存基因可用性统计
  gene_availability <- data.frame(
    Figure = c("S6A", "S6B", "S6C", "S6D"),
    Total_Genes = c(6, 5, 6, 3),
    Available_Genes = c(length(genes.1), length(genes.2), length(genes.3), length(genes.4)),
    Available_Gene_List = c(
      paste(genes.1, collapse = ", "),
      paste(genes.2, collapse = ", "),
      paste(genes.3, collapse = ", "),
      paste(genes.4, collapse = ", ")
    ),
    stringsAsFactors = FALSE
  )
  
  write.csv(gene_availability, file.path(files_dir, "FigureS6_gene_availability.csv"), row.names = FALSE)
  
} else {
  cat("警告：flu_cds_subset_2不存在，跳过FigureS6A/B/C/D生成\n")
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "FigureS6_session_info.txt"))  # 将输出重定向到文件
cat("Figure S6A/B/C/D Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Trajectory path 2:", paste(flu_path2, collapse = ", "), "\n")
if (exists("flu_cds_subset_2") && !is.null(flu_cds_subset_2)) {
  cat("CDS subset 2 cells:", ncol(flu_cds_subset_2), "\n")
}
sessionInfo()
sink()  # 关闭重定向

cat("Figure S6A/B/C/D 模块运行完成！\n")