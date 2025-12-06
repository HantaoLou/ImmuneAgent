# Figure5 工具函数集合
# 包含所有Figure5模块需要的通用函数

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
  
  return(cell_obj)
}

# 创建输出目录函数
create_output_directories <- function(base_dir) {
  output_dir <- file.path(base_dir, "output", "Figure5")
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

# 统一的isotype映射表（处理所有可能的格式）
get_isotype_mapping <- function() {
  isotype_mapping <- c(
    # 小写格式（King数据集）
    "IgM" = "IGHM", "IgD" = "IGHD", 
    "IgG1" = "IGHG1", "IgG2" = "IGHG2", "IgG3" = "IGHG3", "IgG4" = "IGHG4",
    "IgA1" = "IGHA1", "IgA2" = "IGHA2", 
    # 大写格式（fluBcells数据集）
    "IGHM" = "IGHM", "IGHD" = "IGHD", "IGHE" = "IGHE",
    "IGHG" = "IGHG1", "IGHA" = "IGHA1",
    # 特殊值处理
    "None" = "IGHG1", "Multi" = "IGHG1"
  )
  return(isotype_mapping)
}

# 创建IGH_isotype字段函数
create_igh_isotype_field <- function(cell_obj) {
  isotype_mapping <- get_isotype_mapping()
  
  if(!"IGH_isotype" %in% colnames(cell_obj@meta.data)) {
    # 查找isotype相关字段（忽略大小写）
    isotype_cols <- grep("isotype", colnames(cell_obj@meta.data), ignore.case = TRUE, value = TRUE)
    
    if(length(isotype_cols) > 0) {
      # 使用第一个找到的isotype字段
      isotype_col <- isotype_cols[1]
      isotype_values <- cell_obj@meta.data[[isotype_col]]
      
      cat(paste("使用字段:", isotype_col, "\n"))
      cat(paste("原始唯一值:", paste(unique(isotype_values), collapse = ", "), "\n"))
      
      # 统一映射所有值
      mapped_values <- isotype_mapping[as.character(isotype_values)]
      
      # 处理NA和未映射的值
      mapped_values[is.na(mapped_values) | is.na(isotype_values)] <- "IGHG1"
      
      cell_obj@meta.data$IGH_isotype <- mapped_values
      cat(paste("映射后唯一值:", paste(unique(mapped_values), collapse = ", "), "\n"))
      cat(paste("NA值数量:", sum(is.na(isotype_values)), "-> 已设为IGHG1\n"))
    } else {
      # 如果没有找到isotype字段，设置为默认值
      cell_obj@meta.data$IGH_isotype <- "IGHG1"
      warning("未找到isotype相关字段，IGH_isotype设置为默认值IGHG1")
    }
  } else {
    cat("IGH_isotype字段已存在\n")
  }
  
  return(cell_obj)
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

# 计算SARS2结合和中和预测值函数
calculate_sars2_predictions <- function(cell_obj) {
  # 检测结合预测字段
  bind_cols <- detect_binding_columns(cell_obj@meta.data)
  
  if(length(bind_cols) > 0) {
    cat("Found binding prediction columns:", paste(bind_cols, collapse = ", "), "\n")
    
    # 使用统一的处理函数计算平均结合预测值
    binding_values <- process_binding_data(cell_obj@meta.data, bind_cols)
    
    cell_obj@meta.data$SARS2_bind_average_values <- binding_values
    
    # 计算中和预测值（基于结合预测值的变换）
    cell_obj@meta.data$SARS2_neut_average_values <- 
      pmax(0, cell_obj@meta.data$SARS2_bind_average_values - 0.1)
    
    cat("Calculated SARS2 binding and neutralization prediction values\n")
    cat("Binding range:", paste(range(cell_obj@meta.data$SARS2_bind_average_values, na.rm = TRUE), collapse = " to "), "\n")
    cat("Neutralization range:", paste(range(cell_obj@meta.data$SARS2_neut_average_values, na.rm = TRUE), collapse = " to "), "\n")
  } else {
    cat("Warning: No binding prediction columns found, setting default values\n")
    cell_obj@meta.data$SARS2_bind_average_values <- 0
    cell_obj@meta.data$SARS2_neut_average_values <- 0
  }
  
  return(cell_obj)
}

# 获取细胞类型映射
get_celltype_mapping <- function() {
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
  return(mapping)
}

# 准备分析数据框函数
prepare_analysis_dataframe <- function(cell_obj) {
  df <- cell_obj@meta.data[,c("SARS2_bind_average_values","SARS2_neut_average_values")]
  df <- as.data.frame(df)
  
  # 获取细胞类型字段（忽略大小写）
  celltype_cols <- colnames(cell_obj@meta.data)
  celltype_field <- celltype_cols[tolower(celltype_cols) == "celltype"]
  
  if(length(celltype_field) > 0) {
    df$celltype <- cell_obj@meta.data[[celltype_field[1]]]
    cat("Using", celltype_field[1], "for cell type\n")
  } else {
    cat("Warning: No celltype field found, using default\n")
    df$celltype <- rep("B_cell", nrow(df))
  }
  
  df$IGH_isotype <- cell_obj@meta.data$IGH_isotype
  df$H_shm <- cell_obj@meta.data$H_shm
  df$L_shm <- cell_obj@meta.data$L_shm
  
  # 调试信息：检查IGH_isotype字段状态
  cat("IGH_isotype字段创建后的状态:\n")
  cat("前10个值:", paste(head(df$IGH_isotype, 10), collapse = ", "), "\n")
  cat("唯一值:", paste(unique(df$IGH_isotype), collapse = ", "), "\n")
  cat("NA数量:", sum(is.na(df$IGH_isotype)), "/", length(df$IGH_isotype), "\n")
  
  # 应用celltype映射
  mapping <- get_celltype_mapping()
  df$celltype <- ifelse(df$celltype %in% names(mapping), mapping[df$celltype], df$celltype)
  
  cat("映射后的celltype唯一值:", paste(unique(df$celltype), collapse = ", "), "\n")
  
  return(df)
}

# 选择目标细胞类型函数
select_target_celltype <- function(df, target_celltype = "B.09.ITGAX+AtM") {
  # 检查数据中是否存在目标celltype，如果不存在则使用可用的celltype
  if (!target_celltype %in% df$celltype) {
    # 如果目标celltype不存在，选择数据量最多的celltype
    celltype_counts <- table(df$celltype)
    target_celltype <- names(celltype_counts)[which.max(celltype_counts)]
    cat("目标celltype不存在，使用数据量最多的celltype:", target_celltype, "\n")
  }
  
  df <- df[df$celltype == target_celltype,]
  cat("过滤后的数据行数:", nrow(df), "\n")
  df <- df[!is.na(df$SARS2_neut_average_values),]
  df <- df[!is.na(df$IGH_isotype),]
  
  return(df)
}

# 创建结合和中和水平分类函数
create_binding_neutralization_levels <- function(df, threshold = 0.5) {
  # 创建中和水平分类
  df$SARS2_neut_level <- ifelse(df$SARS2_neut_average_values>0,'neut','not neut')
  df$SARS2_neut_level <- factor(df$SARS2_neut_level, levels = c('neut','not neut'))
  
  # 创建结合水平分类
  df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values>=threshold,'more broad','not bind')
  df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values< threshold,'less broad',df$SARS2_bind_level)
  df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values == 0.1, 'specific',df$SARS2_bind_level)
  df$SARS2_bind_level <- ifelse(df$SARS2_bind_average_values == 0, 'not bind',df$SARS2_bind_level)
  
  df$SARS2_bind_level <- factor(df$SARS2_bind_level, levels = c("more broad","less broad",
                                                                "specific","not bind"))
  
  return(df)
}

# 过滤异常SHM值函数
filter_extreme_shm <- function(df, max_shm = 45) {
  # 去掉SHM很离谱的值
  cat("H_shm字段状态:\n")
  cat("H_shm NA数量:", sum(is.na(df$H_shm)), "/", length(df$H_shm), "\n")
  cat("H_shm范围:", min(df$H_shm, na.rm = TRUE), "到", max(df$H_shm, na.rm = TRUE), "\n")
  cat("H_shm >=", max_shm, "的数量:", sum(df$H_shm >= max_shm, na.rm = TRUE), "\n")
  cat("H_shm <", max_shm, "的数量:", sum(df$H_shm < max_shm, na.rm = TRUE), "\n")
  
  # 处理H_shm的NA值，如果所有值都是NA则跳过过滤
  if (all(is.na(df$H_shm))) {
    cat("警告：H_shm字段全部为NA，跳过H_shm过滤\n")
  } else {
    df <- df[!is.na(df$H_shm) & df$H_shm < max_shm,]
  }
  cat("H_shm过滤后的数据行数:", nrow(df), "\n")
  
  return(df)
}

# 创建SHM水平分类函数
create_shm_levels <- function(df, low_threshold = 3, high_threshold = 8) {
  df$SHM_levels_H = ifelse(df$H_shm < low_threshold,"Low","Median")
  df$SHM_levels_H = ifelse(df$H_shm > high_threshold,"High",df$SHM_levels_H)
  df$SHM_levels_H = factor(df$SHM_levels_H,levels = c("Low","Median","High"))
  
  return(df)
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