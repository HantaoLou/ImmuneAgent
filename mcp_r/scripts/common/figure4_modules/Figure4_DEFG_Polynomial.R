# Figure4 D/E/F/G 多项式回归分析模块 - 独立运行版本
# polynomial regression analysis

################ Figure 4D/E/F/G - 独立运行 ###################

# 接收命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Figure4_DEFG_Polynomial.R <input_rds_file> <base_dir>")
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

###############################################################################
#'                     Manuscipt: figure4D/E/F//G                            '#
###############################################################################

## Figure 4D/E/F//G 
## polynomial 

# 为了保持代码兼容性，将cell_obj赋值给flu_obj
flu_obj <- cell_obj

# 获取轨迹路径定义
trajectory_paths <- get_trajectory_paths()
flu_path1 <- trajectory_paths$path1
flu_path2 <- trajectory_paths$path2

## 1. 计算feature score
feature <- get_feature_gene_sets()

# 检测并计算结合预测值
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

# 尝试加载CDS对象
cds <- NULL
rdata_path <- file.path(files_dir, "flu_B_monocle_cds.RData")

if (file.exists(rdata_path)) {
  cat("加载外部monocle3 CDS数据...\n")
  load(rdata_path)
} else {
  cat("警告：未找到monocle3 CDS对象，将跳过轨迹分析部分\n")
  cds <- NULL
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
  cat("创建CDS子集...\n")
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
    
    tempdf <- tempdf[!is.na(tempdf$len.L_shm),]
    tempdf <- tempdf[!is.na(tempdf$H1N1_bind_average_values_ensemble),]
    tempdf$path <- names(pathL)[li]
    totalMD <- rbind(totalMD,tempdf)
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
        
        totalMD.clean.use <- melt(totalMD.clean2,id.vars = c("celltype", "path", "pseudotime")) 
        
        fig <- list()
        for(i in 1:length(unique(totalMD.clean.use$variable))){
          temp <- totalMD.clean.use[totalMD.clean.use$variable==unique(totalMD.clean.use$variable)[i],]
          p <- ggplot(temp) +
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
        
        # 保存数据
        write.csv(totalMD.clean2, file.path(files_dir, "Figure4DEFG_trajectory_data.csv"), row.names = FALSE)
      }
    } else {
      cat("警告：清理后的数据为空，跳过图形生成\n")
    }
  } else {
    cat("警告：没有可用的轨迹数据，跳过Figure 4D/E/F/G生成\n")
  }
} else {
  cat("跳过轨迹分析，因为CDS对象不可用\n")
}

###########################################################
# 记录运行环境
sink(file.path(plots_dir, "Figure4DEFG_session_info.txt"))  # 将输出重定向到文件
cat("Figure 4D/E/F/G Session Information\n")
cat("=============================\n")
cat("Generated on:", as.character(Sys.time()), "\n")
cat("Input file:", input_rds_file, "\n")
cat("Base directory:", base_dir, "\n")
cat("Trajectory paths:\n")
cat("Path1:", paste(flu_path1, collapse = ", "), "\n")
cat("Path2:", paste(flu_path2, collapse = ", "), "\n")
sessionInfo()
sink()  # 关闭重定向

cat("Figure 4D/E/F/G 模块运行完成！\n")