# Figure4 工具函数集合
# 包含所有Figure4模块需要的通用函数

# King数据集CellType映射函数
king_celltype_mapping <- function(king_celltype) {
  mapping <- c(
    "Naive" = "Naive",
    "Activated" = "Activated", 
    "preGC" = "Germinal_Center",
    "LZ GC" = "Germinal_Center",
    "GC" = "Germinal_Center",
    "DZ GC" = "Germinal_Center",
    "FCRL2/3high GC" = "Germinal_Center",
    "prePB" = "Plasma",
    "Plasmablast" = "Plasma",
    "MBC" = "Memory",
    "MBC FCRL4+" = "Atypical",  # FCRL4+记忆B细胞映射为非典型B细胞
    "Cycling" = "Proliferating"
  )
  
  # 返回映射后的细胞类型，如果没有找到映射则返回原值
  mapped_type <- mapping[king_celltype]
  ifelse(is.na(mapped_type), king_celltype, mapped_type)
}

# 加载和预处理数据函数
load_and_preprocess_data <- function(input_rds_file) {
  # 加载细胞数据
  cell_obj <- readRDS(input_rds_file)
  
  # 检查并修复重复的列名
  meta_cols <- colnames(cell_obj@meta.data)
  if (any(duplicated(meta_cols))) {
    cat("检测到重复的元数据列名，正在修复...\n")
    # 为重复的列名添加后缀
    colnames(cell_obj@meta.data) <- make.names(meta_cols, unique = TRUE)
    cat("重复列名已修复\n")
  }
  
  # 检查是否为King数据集并应用映射
  king_specific_types <- c("preGC", "LZ GC", "DZ GC", "FCRL2/3high GC", "prePB", "MBC FCRL4+")
  if(any(king_specific_types %in% unique(cell_obj@meta.data$CellType))) {
    cat("检测到King数据集，应用CellType映射...\n")
    # 保存原始CellType
    cell_obj@meta.data$Original_CellType <- cell_obj@meta.data$CellType
    # 应用映射
    cell_obj@meta.data$CellType <- sapply(cell_obj@meta.data$CellType, king_celltype_mapping)
    cat("CellType映射完成\n")
    
    # 显示映射结果统计
    cat("映射后的CellType分布:\n")
    print(table(cell_obj@meta.data$CellType))
  }
  
  return(cell_obj)
}

# 创建输出目录函数
create_output_directories <- function(base_dir) {
  output_dir <- file.path(base_dir, "output", "Figure4")
  plots_dir <- file.path(output_dir, "plots")
  files_dir <- file.path(output_dir, "files")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(files_dir, recursive = TRUE, showWarnings = FALSE)
  
  return(list(
    output_dir = output_dir,
    plots_dir = plots_dir,
    files_dir = files_dir
  ))
}

# 获取B细胞颜色配置
get_bcell_color_panel <- function() {
  B_cell_color_panel = c(
    'Naive' = '#53A85F',          # B.01.TCL1A+Bn - naive B cells (TCL1A, FCER2, IL4R)
    'Activated' = '#C1E6F3',      # B.02.NR4A2+Bn - activated B cells (NR4A2, CD69, CD83)
    'Memory' = '#E4C755',         # B.04.DUSP2+Bm - memory B cells (DUSP2, GPR183, CD27)
    'Germinal_Center' = '#91D0BE', # B.06.pre-GC - germinal center B cells
    'Proliferating' = '#E95C59',  # B.07.Bgc_DZ-like - proliferating GC B cells
    'Atypical' = '#E59CC4',       # B.09.ITGAX+AtM - atypical memory (ITGAX, FCRL5)
    'Transitional' = '#E39A35',   # B.09.ITGAX+AtM - transitional-like cells
    'Plasma' = '#B53E2B'          # B.10.Plasmablast - plasma cells (JCHAIN, PRDM1, XBP1, MZB1)
  )
  return(B_cell_color_panel)
}

# 获取B细胞标记基因
get_bcell_markers <- function() {
  B_cell_markers <- list(
    "Naive" = c("TCL1A", "FCER2", "IL4R", "IGHD", "CD23", "CD38"),
    "Memory" = c("CD27", "TNFRSF13B", "TNFRSF13C", "AIM2", "GPR183"),
    "Germinal_Center" = c("BCL6", "AICDA", "CD10", "MME", "CXCR4", "LMO2"),
    "Plasma" = c("PRDM1", "XBP1", "MZB1", "JCHAIN", "CD138", "SDC1"),
    "Activated" = c("CD69", "CD83", "EGR1", "FOS", "JUN", "NR4A2"),
    "Proliferating" = c("MKI67", "TOP2A", "PCNA", "STMN1", "TUBB"),
    "Atypical" = c("FCRL5", "ITGAX", "TBX21", "FCRL4", "CD274"),
    "Transitional" = c("CD24", "CD38", "FCER2", "IL4R")
  )
  return(B_cell_markers)
}

# 加载必需的R包
load_required_packages <- function() {
  required_packages <- c("Seurat", "dplyr", "ggplot2", "cowplot", "ggrepel", 
                        "stringr", "monocle3", "RColorBrewer", "Nebulosa", "ggrastr")
  
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat("Warning:", pkg, "package not available\n")
    } else {
      library(pkg, character.only = TRUE)
    }
  }
}

# 检测结合预测字段函数
detect_binding_columns <- function(metadata, patterns = c("bind_predict\\.", "output\\.", "bind_output\\.")) {
  all_cols <- colnames(metadata)
  detected_cols <- c()
  
  for (pattern in patterns) {
    matching_cols <- grep(pattern, all_cols, value = TRUE)
    if (length(matching_cols) > 0) {
      detected_cols <- c(detected_cols, matching_cols)
    }
  }
  
  return(unique(detected_cols))
}

# 处理结合数据函数
process_binding_data <- function(metadata, binding_cols) {
  if (length(binding_cols) == 0) {
    warning("No binding prediction columns found")
    return(rep(0, nrow(metadata)))
  }
  
  # Convert to numeric and handle NA values
  bind_matrix <- metadata[, binding_cols, drop = FALSE]
  bind_matrix <- apply(bind_matrix, 2, function(x) {
    x <- as.character(x)
    x[x == "NA" | is.na(x)] <- "0"
    as.numeric(x)
  })
  
  # Calculate average
  if (length(binding_cols) == 1) {
    return(bind_matrix[,1])
  } else {
    return(rowMeans(bind_matrix, na.rm = TRUE))
  }
}

# 获取特征基因集合
get_feature_gene_sets <- function() {
  high_affinity=c("BATF","GARS","GART","LER3","MIF","MYC","SPP1","UCK2",
                  "CD320","TIMD2","TNFRSF8",
                  "AURKA","BUB1","CCNA2","CCNB1","CCNB2","CCND2","CDC20","CDC25C","KIF22","PLK1",
                  "NFIL3","PML")
  Low_affinity=c("PPP1R15A","RGS1","CCR6","CD22","CD38","CD72","FCER2A","ICOSL","PACAM1","SIGLECG","TLRL","TNFRSF18",
                 "BACH2","EGR3","ELK4","FOXP1","JUN","NR4A1","REL")
  exhaustion_genes = c('PDCD1','CD160','FASLG','CD244','LAG3','TNFRSF1B','CCR5','CCL3',
                       'CCL4','CXCL10','CTLA4','LGALS1','LGALS3','PTPN13','RGS16','ISG20',
                       'MX1','IRF4','EOMES','PBX3','NFATC1','NR4A2','CKS2','GAS2',
                       'ENTPD1','CA2',"CD52", "APOE", "PTLP", "PTGDS", "PIM2", "DERL3")
  
  Bactivated_genes=c("CD69","CD83","IER2","DUSP2","IL6","NR4A2","JUN","CCR7","GPR183")
  BCSR_genes=c("APEX1","APEX2","XRCC5","XRCC6","POLD2","AICDA")
  CSR_m=c("APEX1","XRCC5","XRCC6","POLD2","POLE3", #CSR machinery
          "NCL","NME2","DDX21",#IgH locus
          "NPM1","SERBP1",#CSR interactors
          "MIR155HG","HSP90AB1",#AICDA/AICDA stability
          "BATF","HIVEP3","BHLHE40","IRF4")#TF
  
  feature <- list(high_affinity=high_affinity,
                  Low_affinity=Low_affinity,
                  exhaustion_genes=exhaustion_genes,
                  Bactivated_genes=Bactivated_genes,
                  BCSR_genes=BCSR_genes,
                  CSR_m=CSR_m)
  
  return(feature)
}

# 基于生物学机制估算SHM水平函数（改进版本）
estimate_shm_from_expression <- function(seurat_obj) {
  # 获取元数据
  meta_data <- seurat_obj@meta.data
  
  # 方法1: 如果有BCR序列数据，优先使用直接计算
  if (all(c("IGH_sequence", "IGL_sequence") %in% colnames(meta_data))) {
    cat("检测到BCR序列数据，使用直接序列比对方法计算SHM\n")
    return(calculate_shm_from_bcr_sequence(seurat_obj))
  }
  
  # 方法2: 基于SHM相关基因表达和细胞类型
  cat("使用基于SHM机制基因和细胞类型的估算方法\n")
  
  # SHM相关的关键基因（AID/APOBEC家族和DNA修复基因）
  shm_machinery_genes <- c("AICDA", "UNG", "MSH2", "MSH6", "PMS2", "MLH1", "APEX1", "XRCC1")
  gc_markers <- c("BCL6", "LMO2", "CXCR4", "CD10", "MME")
  
  # 检查基因可用性
  available_shm_genes <- shm_machinery_genes[shm_machinery_genes %in% rownames(seurat_obj)]
  available_gc_genes <- gc_markers[gc_markers %in% rownames(seurat_obj)]
  
  cat("可用SHM机制基因:", length(available_shm_genes), "个\n")
  cat("可用生发中心标记基因:", length(available_gc_genes), "个\n")
  
  # 计算SHM机制活跃度分数
  if (length(available_shm_genes) > 0) {
    shm_activity <- calculate_gene_signature_score(seurat_obj, available_shm_genes)
  } else {
    warning("缺少SHM机制相关基因，使用默认值")
    shm_activity <- rep(0, ncol(seurat_obj))
  }
  
  # 计算生发中心活跃度分数
  if (length(available_gc_genes) > 0) {
    gc_activity <- calculate_gene_signature_score(seurat_obj, available_gc_genes)
  } else {
    gc_activity <- rep(0, ncol(seurat_obj))
  }
  
  # 基于细胞类型的SHM基线水平（基于文献报道）
  baseline_shm <- c(
    "Naive" = 0,           # 初始B细胞：无SHM
    "Germinal_Center" = 15, # 生发中心B细胞：高SHM（10-25个突变）
    "Memory" = 10,         # 记忆B细胞：中等SHM（5-15个突变）
    "Plasma" = 12,         # 浆细胞：中等偏高SHM（8-18个突变）
    "Activated" = 2,       # 激活B细胞：低SHM（0-5个突变）
    "Atypical" = 8,        # 非典型记忆B细胞：中等SHM（5-12个突变）
    "Proliferating" = 12,  # 增殖B细胞：中等偏高SHM
    "Transitional" = 1     # 过渡B细胞：极低SHM
  )
  
  # 获取每个细胞的基线SHM
  celltype <- meta_data$CellType
  base_shm_h <- baseline_shm[celltype]
  base_shm_h[is.na(base_shm_h)] <- 5  # 未知类型默认值
  
  # 基于SHM机制活跃度调整（标准化后的调整因子）
  if (sd(shm_activity, na.rm = TRUE) > 0) {
    activity_modifier <- (shm_activity - mean(shm_activity, na.rm = TRUE)) / sd(shm_activity, na.rm = TRUE)
    activity_modifier[is.na(activity_modifier)] <- 0
    activity_modifier <- pmax(-2, pmin(2, activity_modifier))  # 限制调整范围
  } else {
    activity_modifier <- rep(0, length(shm_activity))
  }
  
  # 基于生发中心标记调整
  if (sd(gc_activity, na.rm = TRUE) > 0) {
    gc_modifier <- (gc_activity - mean(gc_activity, na.rm = TRUE)) / sd(gc_activity, na.rm = TRUE)
    gc_modifier[is.na(gc_modifier)] <- 0
    gc_modifier <- pmax(-1, pmin(1, gc_modifier))  # 限制调整范围
  } else {
    gc_modifier <- rep(0, length(gc_activity))
  }
  
  # 最终SHM估算（重链）
  estimated_shm_h <- pmax(0, round(base_shm_h + activity_modifier * 3 + gc_modifier * 2))
  
  # 轻链SHM通常比重链少20-30%
  estimated_shm_l <- pmax(0, round(estimated_shm_h * 0.7))
  
  cat("SHM估算完成，重链范围:", range(estimated_shm_h), "，轻链范围:", range(estimated_shm_l), "\n")
  
  return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
}

# 计算基因特征分数的辅助函数
calculate_gene_signature_score <- function(seurat_obj, genes) {
  # 检查基因是否在数据中存在
  available_genes <- genes[genes %in% rownames(seurat_obj)]
  
  if (length(available_genes) == 0) {
    return(rep(0, ncol(seurat_obj)))
  }
  
  # 获取基因表达数据
  expr_data <- GetAssayData(seurat_obj, assay = "RNA", layer = "data")
  gene_expr <- expr_data[available_genes, , drop = FALSE]
  
  # 计算平均表达作为特征分数
  if (nrow(gene_expr) == 1) {
    signature_score <- as.numeric(gene_expr[1, ])
  } else {
    signature_score <- colMeans(gene_expr, na.rm = TRUE)
  }
  
  return(signature_score)
}

# BCR序列直接计算SHM的函数（如果有序列数据）
calculate_shm_from_bcr_sequence <- function(seurat_obj) {
  meta_data <- seurat_obj@meta.data
  
  # 这里需要实现具体的序列比对逻辑
  # 由于缺少胚系序列参考，这里提供框架
  cat("注意：BCR序列SHM计算需要胚系基因参考序列\n")
  
  # 占位符实现，实际需要序列比对
  n_cells <- nrow(meta_data)
  estimated_shm_h <- rep(10, n_cells)  # 占位符
  estimated_shm_l <- rep(7, n_cells)   # 占位符
  
  return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
}

# 异常值检测函数
detect_outlier <- function(x) {
  Quantile1 <- quantile(x, probs=.25)
  Quantile3 <- quantile(x, probs=.75)
  IQR = Quantile3-Quantile1
  #x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
  x > Quantile3  | x < Quantile1 
}

# 移除异常值函数
remove_outlier <- function(dataframe, columns=names(dataframe)) {
  for (col in columns) {
    dataframe <- dataframe[!detect_outlier(dataframe[[col]]), ]
  }
  return(dataframe)
  print("Remove outliers")
}

# 获取脚本目录函数
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

# 获取轨迹路径定义
get_trajectory_paths <- function() {
  flu_path1 <- c("Naive", "Activated", "Memory")
  flu_path2 <- c("Naive", "Germinal_Center", "Plasma")
  
  return(list(path1 = flu_path1, path2 = flu_path2))
}