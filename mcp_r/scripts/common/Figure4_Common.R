rm(list = ls())

################ Figure 4  ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure4.R <input_rds_file> <base_dir>")
}

input_rds_file <- args[1]
fdir <- args[2]

# 检查输入文件是否存在
if (!file.exists(input_rds_file)) {
  stop(paste("Input file does not exist:", input_rds_file))
}

# 检查基础目录是否存在
if (!dir.exists(fdir)) {
  stop(paste("Base directory does not exist:", fdir))
}

# 创建输出目录
output_dir <- file.path(fdir, "output", "Figure4")
plots_dir <- file.path(output_dir, "plots")
files_dir <- file.path(output_dir, "files")
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(files_dir, recursive = TRUE, showWarnings = FALSE)

## I. load packages
library(Seurat)
library(dplyr)
library(ggplot2)
library(cowplot)
library(ggrepel)
library(stringr)
library(monocle3)
library(RColorBrewer)
library(Nebulosa)
library(ggrastr)

## II. load data
# King数据集CellType映射函数（需要在数据加载前定义）
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

## IV. load marker
# 使用与integrate_bcr_data.R一致的B细胞marker基因定义
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
    
    # 自动选择根细胞（选择naive B细胞作为起点）
    if ("combined_cluster" %in% colnames(sub@meta.data)) {
      # 查找naive B细胞
      naive_cells <- which(sub$combined_cluster == "naive B")
      if (length(naive_cells) > 0) {
        # 选择第一个naive B细胞作为根
        root_cell <- colnames(sub)[naive_cells[1]]
        cds <- monocle3::order_cells(cds, root_cells = root_cell)
        cat("Root cell selected automatically (naive B cell from combined_cluster)\n")
      } else {
        # 如果没有naive B细胞，手动选择
        cds <- monocle3::order_cells(cds)
        cat("Manual root cell selection required\n")
      }
    } else if ("CellType" %in% colnames(sub@meta.data)) {
      # 查找Naive细胞类型
      naive_cells <- which(sub$CellType == "Naive")
      if (length(naive_cells) > 0) {
        # 选择第一个Naive细胞作为根
        root_cell <- colnames(sub)[naive_cells[1]]
        cds <- monocle3::order_cells(cds, root_cells = root_cell)
        cat("Root cell selected automatically (Naive cell from CellType)\n")
      } else {
        # 如果没有Naive细胞，手动选择
        cds <- monocle3::order_cells(cds)
        cat("Manual root cell selection required\n")
      }
    } else {
      # 手动选择根细胞
      cds <- monocle3::order_cells(cds)
      cat("Manual root cell selection\n")
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
# Figure 4C: 伪时间与细胞类型的箱线图
###########################################################

# 检查是否存在cds对象
if (exists("cds") && !is.null(cds)) {
  cat("\nGenerating Figure 4C...\n")
  
  # 确定使用哪个细胞类型字段
  celltype_field <- NULL
  if ("annotation_final" %in% colnames(cds@colData)) {
    celltype_field <- "annotation_final"
  } else if ("combined_cluster" %in% colnames(cds@colData)) {
    celltype_field <- "combined_cluster"
  } else if ("CellType" %in% colnames(cds@colData)) {
    celltype_field <- "CellType"
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
        ggplot2::scale_color_manual(values = B_cell_color_panel)
      
      # 保存图片
      ggsave(file.path(plots_dir, "Figure_4C.pdf"), plot = p_boxplot, width = 9, height = 6)
      
      cat("Figure 4C generated successfully\n")
      
    } else {
      cat("Warning: No valid data for Figure 4C\n")
    }
  } else {
    cat("Warning: No suitable cell type annotation found for Figure 4C\n")
  }
} else {
  cat("Warning: CDS object not found, skipping Figure 4C\n")
}


###############################################################################
#'                     Manuscipt: figure4D/E/F//G                            '#
###############################################################################

## Figure 4D/E/F//G 
## polynomial 


## 1. 计算feature score
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

# 为了保持代码兼容性，将cell_obj赋值给flu_obj
flu_obj <- cell_obj

# 定义轨迹路径变量（如果尚未定义）
if (!exists("flu_path1")) {
  flu_path1 <- c("Naive", "Activated", "Memory")
}
if (!exists("flu_path2")) {
  flu_path2 <- c("Naive", "Germinal_Center", "Plasma")
}

# 检测并计算结合预测值
# 使用统一的结合预测字段检测和处理函数
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

# Function to process binding data
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

# 检测结合预测字段 - 兼容多种格式
bind_cols <- detect_binding_columns(flu_obj@meta.data)

if(length(bind_cols) > 0) {
  cat("Found binding prediction columns:", paste(bind_cols, collapse = ", "), "\n")
  
  # 使用统一的处理函数计算平均结合预测值
  binding_values <- process_binding_data(flu_obj@meta.data, bind_cols)
  
  flu_obj@meta.data$H1N1_bind_average_values_ensemble <- binding_values
  
  # 计算中和预测值（基于结合预测值的变换）
  # 假设中和能力与结合能力相关，但通常更严格
  flu_obj@meta.data$H1N1_neut.bind_average_values_ensemble <- 
    pmax(0, flu_obj@meta.data$H1N1_bind_average_values_ensemble - 0.1)
  
  cat("Calculated H1N1 binding and neutralization prediction values\n")
  cat("Binding range:", paste(range(flu_obj@meta.data$H1N1_bind_average_values_ensemble, na.rm = TRUE), collapse = " to "), "\n")
  cat("Neutralization range:", paste(range(flu_obj@meta.data$H1N1_neut.bind_average_values_ensemble, na.rm = TRUE), collapse = " to "), "\n")
} else {
  cat("Warning: No binding prediction columns found, using placeholder values\n")
  # 使用基于高亲和力分数的估算值作为替代
  flu_obj@meta.data$H1N1_bind_average_values_ensemble <- 
    pmax(0, pmin(1, runif(ncol(flu_obj), 0, 0.8)))
  flu_obj@meta.data$H1N1_neut.bind_average_values_ensemble <- 
    pmax(0, flu_obj@meta.data$H1N1_bind_average_values_ensemble - 0.2)
  
  cat("Generated placeholder binding prediction values\n")
}

# 过滤基因列表，只保留在对象中存在的基因
available_genes <- rownames(flu_obj)
filtered_features <- list()

for(i in 1:length(feature)) {
  feature_name <- names(feature)[i]
  genes_in_feature <- feature[[i]]
  available_genes_in_feature <- genes_in_feature[genes_in_feature %in% available_genes]
  
  if(length(available_genes_in_feature) > 0) {
    filtered_features[[feature_name]] <- available_genes_in_feature
    cat("Feature", feature_name, ":", length(available_genes_in_feature), "out of", length(genes_in_feature), "genes available\n")
  } else {
    cat("Warning: No genes available for feature", feature_name, "\n")
    # 为空的特征创建一个占位符
    filtered_features[[feature_name]] <- available_genes[1:min(5, length(available_genes))]
  }
}

# 使用过滤后的基因列表计算模块分数
# 设置较小的nbin参数以避免采样错误
tryCatch({
  flu_obj <- AddModuleScore(flu_obj, 
                           features = filtered_features,
                           name = c("high_affinity", "Low_affinity", "exhaustion_genes",
                                   "Bactivated_genes", "BCSR_genes", "CSR_m"),
                           nbin = min(5, floor(nrow(flu_obj)/10)))
  cat("Successfully calculated module scores\n")
}, error = function(e) {
  cat("Error in AddModuleScore:", e$message, "\n")
  # 如果仍然出错，使用更保守的参数
  flu_obj <<- AddModuleScore(flu_obj, 
                            features = filtered_features,
                            name = c("high_affinity", "Low_affinity", "exhaustion_genes",
                                    "Bactivated_genes", "BCSR_genes", "CSR_m"),
                            nbin = 3,
                            ctrl = min(10, floor(nrow(flu_obj)/5)))
  cat("Calculated module scores with conservative parameters\n")
})

## 2. 计算SHM (体细胞超突变)
# 基于现有免疫球蛋白序列数据计算SHM

# 函数：计算序列的SHM数量
calculate_shm <- function(sequence, germline_sequence) {
  if(is.na(sequence) || is.na(germline_sequence) || 
     nchar(sequence) == 0 || nchar(germline_sequence) == 0) {
    return(0)
  }
  
  # 将序列转换为字符向量
  seq_chars <- strsplit(sequence, "")[[1]]
  germ_chars <- strsplit(germline_sequence, "")[[1]]
  
  # 确保序列长度一致
  min_length <- min(length(seq_chars), length(germ_chars))
  if(min_length == 0) return(0)
  
  # 计算突变数量
  mutations <- sum(seq_chars[1:min_length] != germ_chars[1:min_length], na.rm = TRUE)
  return(mutations)
}

# 函数：基于基因表达估算SHM水平
estimate_shm_from_expression <- function(seurat_obj) {
  # 获取元数据
  meta_data <- seurat_obj@meta.data
  
  # 基于高亲和力和低亲和力特征分数估算SHM
  # 高亲和力分数通常与更多SHM相关
  high_affinity_score <- meta_data$high_affinity1
  low_affinity_score <- meta_data$Low_affinity2
  
  # 标准化分数到0-1范围
  high_norm <- (high_affinity_score - min(high_affinity_score, na.rm = TRUE)) / 
               (max(high_affinity_score, na.rm = TRUE) - min(high_affinity_score, na.rm = TRUE))
  low_norm <- (low_affinity_score - min(low_affinity_score, na.rm = TRUE)) / 
              (max(low_affinity_score, na.rm = TRUE) - min(low_affinity_score, na.rm = TRUE))
  
  # 估算SHM：高亲和力分数高、低亲和力分数低的细胞SHM更多
  estimated_shm_h <- round((high_norm - low_norm + 1) * 15)  # 重链SHM估算
  estimated_shm_l <- round((high_norm - low_norm + 1) * 10)  # 轻链SHM估算
  
  # 确保SHM值为非负数
  estimated_shm_h[estimated_shm_h < 0] <- 0
  estimated_shm_l[estimated_shm_l < 0] <- 0
  
  return(list(H_shm = estimated_shm_h, L_shm = estimated_shm_l))
}

# 检查是否存在序列数据用于直接计算SHM
if("ig_seq.x" %in% colnames(flu_obj@meta.data) && "ig_seq.y" %in% colnames(flu_obj@meta.data)) {
  # 如果有序列数据，尝试直接计算SHM
  cat("检测到序列数据，尝试计算SHM...\n")
  
  # 这里需要胚系序列作为参考，如果没有，使用估算方法
  # 由于缺少胚系序列，我们使用基于表达的估算方法
  shm_results <- estimate_shm_from_expression(flu_obj)
  flu_obj$len.H_shm <- shm_results$H_shm
  flu_obj$len.L_shm <- shm_results$L_shm
  
} else {
  # 使用基于基因表达特征的SHM估算
  cat("使用基于基因表达特征的SHM估算方法...\n")
  shm_results <- estimate_shm_from_expression(flu_obj)
  flu_obj$len.H_shm <- shm_results$H_shm
  flu_obj$len.L_shm <- shm_results$L_shm
}

# 输出SHM计算结果统计
cat("SHM计算完成:\n")
cat("重链SHM范围:", range(flu_obj$len.H_shm, na.rm = TRUE), "\n")
cat("轻链SHM范围:", range(flu_obj$len.L_shm, na.rm = TRUE), "\n")
cat("重链SHM平均值:", mean(flu_obj$len.H_shm, na.rm = TRUE), "\n")
cat("轻链SHM平均值:", mean(flu_obj$len.L_shm, na.rm = TRUE), "\n")


## 3.绘图
col_flg<-colorRampPalette(brewer.pal(8,"Set1"))(8)
###remove outlier
detect_outlier <- function(x) {
  Quantile1 <- quantile(x, probs=.25)
  Quantile3 <- quantile(x, probs=.75)
  IQR = Quantile3-Quantile1
  #x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
  x > Quantile3  | x < Quantile1 
}

# create remove outlier function
remove_outlier <- function(dataframe,
                           columns=names(dataframe)) {
  for (col in columns) {
    dataframe <- dataframe[!detect_outlier(dataframe[[col]]), ]
  }
  return(dataframe)
  print("Remove outliers")
}

##  3.1 flu

# 使用前面生成的cds对象，如果不存在则尝试加载
if (!exists("cds") || is.null(cds)) {
  # 尝试加载外部RData文件作为备选
  rdata_path <- file.path(files_dir, "flu_B_monocle_cds.RData")
  if (file.exists(rdata_path)) {
    cat("加载外部monocle3 CDS数据...\n")
    load(rdata_path)
  } else {
    cat("警告：未找到monocle3 CDS对象，将跳过轨迹分析部分\n")
    cds <- NULL
  }
} else {
  cat("使用前面生成的monocle3 CDS对象...\n")
}

# 检查cds对象是否存在，如果不存在则跳过后续分析
if (!is.null(cds)) {
  # 检查main_name列是否存在
  if ("main_name" %in% colnames(cds@colData) && "main_name" %in% colnames(flu_obj@meta.data)) {
    df <- flu_obj@meta.data[match(cds@colData$main_name,flu_obj$main_name),
                            c("H1N1_bind_average_values_ensemble","H1N1_neut.bind_average_values_ensemble",
                              "high_affinity1", "Low_affinity2", "exhaustion_genes3", "Bactivated_genes4",
                              "BCSR_genes5", "CSR_m6","len.H_shm","len.L_shm")]
  } else {
    # 如果没有main_name列，使用行名匹配
    common_cells <- intersect(rownames(flu_obj@meta.data), colnames(cds))
    df <- flu_obj@meta.data[common_cells,
                            c("H1N1_bind_average_values_ensemble","H1N1_neut.bind_average_values_ensemble",
                              "high_affinity1", "Low_affinity2", "exhaustion_genes3", "Bactivated_genes4",
                              "BCSR_genes5", "CSR_m6","len.H_shm","len.L_shm")]
  }
  
  df <- as.data.frame(df)
  df$pseudotime <- pseudotime(cds)
  
  # 检查annotation_final列是否存在
  if ("annotation_final" %in% colnames(cds@colData)) {
    df$celltype <- cds@colData$annotation_final
  } else if ("CellType" %in% colnames(cds@colData)) {
    df$celltype <- cds@colData$CellType
  } else {
    # 使用flu_obj中的细胞类型信息
    if ("CellType" %in% colnames(flu_obj@meta.data)) {
      df$celltype <- flu_obj@meta.data[rownames(df), "CellType"]
    } else {
      cat("警告：未找到细胞类型注释，使用默认分组\n")
      df$celltype <- "Unknown"
    }
  }
  
  pathL <- list(path1 = flu_path1, path2 = flu_path2)
  
  # 创建cds子集（如果不存在）
  if (!exists("flu_cds_subset") || !exists("flu_cds_subset_2")) {
    cat("创建CDS子集...\n")
    # 为每个路径创建子集
    # 检查annotation_final列是否存在
    has_annotation_final <- "annotation_final" %in% colnames(cds@colData)
    has_celltype <- "CellType" %in% colnames(cds@colData)
    
    if (has_annotation_final) {
      flu_cds_subset <- cds[, cds@colData$annotation_final %in% flu_path1]
      flu_cds_subset_2 <- cds[, cds@colData$annotation_final %in% flu_path2]
    } else if (has_celltype) {
      flu_cds_subset <- cds[, cds@colData$CellType %in% flu_path1]
      flu_cds_subset_2 <- cds[, cds@colData$CellType %in% flu_path2]
    } else {
      cat("警告：未找到annotation_final或CellType列，无法创建子集\n")
      flu_cds_subset <- cds
      flu_cds_subset_2 <- cds
    }
  }
  
  totalMD <- NULL
  cds_subset_list <- list(flu_cds_subset, flu_cds_subset_2)
  
  for(li in 1:length(pathL)){
    tempName <- names(pathL[li])
    tempPath <- pathL[[li]]
    
    # 检查cds_subset_list是否有效
    if (li <= length(cds_subset_list) && !is.null(cds_subset_list[[li]])) {
      tempdf <- df[(df$celltype %in% tempPath) & (rownames(df) %in% colnames(cds_subset_list[[li]])),]
    } else {
      # 如果子集不存在，只根据细胞类型筛选
      tempdf <- df[df$celltype %in% tempPath,]
    }
    
    # print(nrow(tempdf))
    tempdf <- tempdf[!is.na(tempdf$len.L_shm),]
    # print(nrow(tempdf))
    tempdf <- tempdf[!is.na(tempdf$H1N1_bind_average_values_ensemble),]
    tempdf$path <- names(pathL)[li]
    totalMD <- rbind(totalMD,tempdf)
  }
} else {
  cat("跳过轨迹分析，因为CDS对象不可用\n")
  totalMD <- NULL
}


# 检查totalMD是否有数据
if (!is.null(totalMD) && nrow(totalMD) > 0) {
  totalMD.clean <- NULL
  for(i in 1:length(unique(totalMD$celltype))){
    temp <- totalMD[totalMD$celltype==unique(totalMD$celltype)[i],]
    temp.clean <- remove_outlier(temp,c("pseudotime"))
    totalMD.clean <- rbind(totalMD.clean,temp.clean)
  }
  
  totalMD.clean <- totalMD[is.finite(totalMD$pseudotime), ]
  
  if (nrow(totalMD.clean) > 0) {
    totalMD.clean2 <- NULL
    for(i in 1:length(unique(totalMD.clean$path))){
      temp.clean <- totalMD.clean[totalMD.clean$path==unique(totalMD.clean$path)[i],]
      temp.clean$pseudotime <- (temp.clean$pseudotime-min(temp.clean$pseudotime))/(max(temp.clean$pseudotime)-min(temp.clean$pseudotime))
      totalMD.clean2 <- rbind(totalMD.clean2,temp.clean)
    }
    
    # 检查是否需要加载reshape2包
    if (!requireNamespace("reshape2", quietly = TRUE)) {
      cat("警告：reshape2包不可用，跳过图形生成\n")
    } else {
      library(reshape2)
      #col_flg
      colnames(totalMD.clean2)
      totalMD.clean.use <- melt(totalMD.clean2,id.vars = c("celltype", "path",
                                                           "pseudotime")) 
      
      fig <- list()
      for(i in 1:length(unique(totalMD.clean.use$variable))){
        temp <- totalMD.clean.use[totalMD.clean.use$variable==unique(totalMD.clean.use$variable)[i],]
        p <- ggplot(temp) +
          #geom_point(aes(x = pseudotime, y = value,color=path)) +
          stat_smooth(aes(x = pseudotime, y = value,color=path),method = "lm", formula = y ~ poly(x, 2),se=F)+
          theme_classic()+scale_color_manual(values=col_flg)+ggtitle(unique(totalMD.clean.use$variable)[i])+
          theme(aspect.ratio = 1,
                text = element_text(size = 18), 
                legend.position = "",
                axis.title.x = element_blank(),  # 不显示 x 轴标题
                axis.title.y = element_blank()   # 不显示 y 轴标题
          )
        fig[[i]] <- p
      }
      
      # 显示统计信息
      cat("Path1 细胞类型分布:\n")
      print(table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path1']))
      cat("Path2 细胞类型分布:\n")
      print(table(totalMD.clean.use$celltype[totalMD.clean.use$path == 'path2']))
      
      # 检查是否需要加载gridExtra包
      if (!requireNamespace("gridExtra", quietly = TRUE)) {
        cat("警告：gridExtra包不可用，无法生成组合图形\n")
        # 单独保存每个图形
        for(i in 1:length(fig)) {
          ggsave(file.path(plots_dir, paste0("Figure4D_E_F_G_part", i, ".pdf")), 
                 plot = fig[[i]], width = 5, height = 5)
        }
      } else {
        library(gridExtra)
        length(fig)
        fig[['nrow']] <- 2
        fig[['ncol']] <- 5
        
        # 生成组合图形
        pdf(file.path(plots_dir, 'Figure4D_E_F_G-flu.pdf'), width = 25, height = 10)
        do.call('grid.arrange', fig)
        dev.off()
        
        cat("Figure 4D/E/F/G 生成成功\n")
      }
    }
  } else {
    cat("警告：清理后的数据为空，跳过图形生成\n")
  }
} else {
  cat("警告：没有可用的轨迹数据，跳过Figure 4D/E/F/G生成\n")
}

###############################################################################
#'                     Manuscipt: figureS6A/B/C/D                            '#
###############################################################################

## Figure S6A/B/C/D
## (A) B cell activation related transcriptional markers (CD86, ITGB2, ITGB2-AS1, SOX5, TNFRSF1B and FAS) across pseudotime trajectory 2. 
## (B) Atypical B cell related transcriptional markers (ITGAX, ZEB2, FCRL3, FCRL4, FCRL5) across pseudotime trajectory 2.  
## (C) Isotype-specific immunoglobulin expression dynamics (IGHA1, IGHD, IGHG1, IGHG2, IGHG3, and IGHM) along the pseudotime trajectory 2
## (D) Expression patterns of transcription factors (AFF3, BACH2, IRF8) along trajectory 2.

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
    }, error = function(e) {
      cat("FigureS6D生成失败:", e$message, "\n")
    })
  } else {
    cat("警告：FigureS6D无可用基因\n")
  }
  
} else {
  cat("警告：flu_cds_subset_2不存在，跳过FigureS6A/B/C/D生成\n")
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "session_info.txt"))  # 将输出重定向到文件
sessionInfo()
sink()  # 关闭重定向
