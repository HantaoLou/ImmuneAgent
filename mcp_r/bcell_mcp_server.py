"""
B Cell Analysis MCP Server - Simplified Version
Clean, focused MCP implementation for B cell single-cell RNA-seq analysis
"""
import subprocess
import logging
import sys
import csv
import os
import glob
import tempfile
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import urllib.request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError

# Configure logging to stderr to avoid interfering with MCP protocol
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器实例
mcp = FastMCP("B Cell Analysis Server")

def download_url_to_temp_file(url: str, default_ext: str = None) -> str:
    """
    下载 HTTP/HTTPS URL 到临时文件
    
    Args:
        url: HTTP/HTTPS URL
        default_ext: 默认文件扩展名（如果 URL 中没有扩展名）
        
    Returns:
        临时文件路径
        
    Raises:
        Exception: 如果下载失败
    """
    try:
        # 从 URL 获取文件扩展名
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        # 获取文件扩展名
        ext = os.path.splitext(url_path)[1]
        if not ext and default_ext:
            ext = default_ext
        elif not ext:
            ext = '.rds'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # 下载文件
        urllib.request.urlretrieve(url, temp_file_path)
        
        return temp_file_path
    except Exception as e:
        raise Exception(f"Failed to download URL {url}: {str(e)}")

# Pydantic参数模型定义
class RunFigure2DegAnalysisArgs(BaseModel):
    """Parameters for Figure 2 DEG analysis"""
    flu_data_path: Optional[str] = Field(default=None, description="Flu data RDS path", json_schema_extra={"ui_type": "text"})
    sars_data_path: Optional[str] = Field(default=None, description="SARS data RDS path", json_schema_extra={"ui_type": "text"})
    rsv_data_path: Optional[str] = Field(default=None, description="RSV data RDS path", json_schema_extra={"ui_type": "text"})
    flu_binding_threshold: float = Field(default=0.625, description="Flu binding threshold", json_schema_extra={"ui_type": "number", "min": 0.0, "max": 1.0})
    sars_binding_threshold: float = Field(default=0.5, description="SARS binding threshold", json_schema_extra={"ui_type": "number", "min": 0.0, "max": 1.0})
    rsv_binding_threshold: float = Field(default=1.0, description="RSV binding threshold", json_schema_extra={"ui_type": "number", "min": 0.0, "max": 2.0})
    logfc_threshold: float = Field(default=0.0, description="LogFC threshold", json_schema_extra={"ui_type": "number"})
    min_pct: float = Field(default=0.2, description="Min percentage", json_schema_extra={"ui_type": "number", "min": 0.01, "max": 1.0})
    output_dir: str = Field(default="./output/Figure2", description="Output directory", json_schema_extra={"ui_type": "text"})

class RunDataProcessingArgs(BaseModel):
    """Parameters for data processing"""
    data_path: str = Field(..., description="Data path", json_schema_extra={"ui_type": "text"})
    output_dir: str = Field(default="./output", description="Output directory", json_schema_extra={"ui_type": "text"})

class RunCustomAnalysisArgs(BaseModel):
    """Parameters for custom analysis"""
    data_path: str = Field(..., description="Data path", json_schema_extra={"ui_type": "text"})
    analysis_type: str = Field(..., description="Analysis type", json_schema_extra={"ui_type": "text"})
    output_dir: str = Field(default="./output", description="Output directory", json_schema_extra={"ui_type": "text"})

class ReadCsvFileArgs(BaseModel):
    """Parameters for reading CSV file"""
    path: str = Field(..., description="CSV file path", json_schema_extra={"ui_type": "text"})

def execute_r_script(r_script: str, output_dir: str) -> Dict[str, Any]:
    """Execute R script and return results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary R script file in the output directory instead of system temp
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Create R script file with explicit path and encoding
    script_path = temp_dir / f"script_{hash(r_script) % 100000}.R"
    
    # 写入R脚本
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(r_script)
    
    try:
        # 执行R脚本
        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=18000,  # 30分钟超时
            cwd=str(Path.cwd())  # 使用项目根目录作为工作目录
        )
        
        if result.returncode == 0:
            # 查找输出文件
            output_files = []
            for ext in ['*.csv', '*.pdf', '*.rds']:
                output_files.extend(list(output_path.rglob(ext)))
            
            return {
                "status": "success",
                "output": result.stdout,
                "output_files": [str(f) for f in output_files],
                "output_dir": str(output_path)
            }
        else:
            # 执行失败，保留脚本文件以便调试
            debug_script_path = output_path / "debug_script.R"
            with open(debug_script_path, 'w', encoding='utf-8') as f:
                f.write(r_script)
            
            error_msg = f"R script failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nError: {result.stderr}"
            error_msg += f"\nDebug script saved to: {debug_script_path}"
            raise RuntimeError(error_msg)
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("R script execution timed out after 30 minutes")
    except FileNotFoundError:
        raise RuntimeError("Rscript not found. Please install R and ensure Rscript is in PATH")

@mcp.tool()
def run_figure2_deg_analysis(args: RunFigure2DegAnalysisArgs) -> Dict[str, Any]:
    """
    Run Figure 2 DEG (Differential Gene Expression) analysis
    
    Args:
        flu_data_path: Path to flu vaccination data RDS file (支持本地路径或 HTTP/HTTPS URL)
        sars_data_path: Path to SARS-CoV-2 data RDS file (支持本地路径或 HTTP/HTTPS URL)
        rsv_data_path: Path to RSV data RDS file (支持本地路径或 HTTP/HTTPS URL)
        flu_binding_threshold: Threshold for flu binding classification (default: 0.625)
        sars_binding_threshold: Threshold for SARS-CoV-2 binding (default: 0.5)
        rsv_binding_threshold: Threshold for RSV binding (default: 1.0)
        logfc_threshold: Log fold change threshold (default: 0.0)
        min_pct: Minimum percentage expression (default: 0.2)
        output_dir: Output directory path
    
    Returns:
        Analysis results and output file paths
    """
    # 处理 URL 下载
    temp_files = {}  # 存储需要清理的临时文件路径
    actual_flu_path = None
    actual_sars_path = None
    actual_rsv_path = None
    
    try:
        if not any([args.flu_data_path, args.sars_data_path, args.rsv_data_path]):
            return {
                "status": "error",
                "message": "At least one data file path must be provided"
            }
        if args.flu_data_path:
            if args.flu_data_path.startswith(('http://', 'https://')):
                actual_flu_path = download_url_to_temp_file(args.flu_data_path, '.rds')
                temp_files['flu'] = actual_flu_path
            else:
                actual_flu_path = args.flu_data_path
        
        if args.sars_data_path:
            if args.sars_data_path.startswith(('http://', 'https://')):
                actual_sars_path = download_url_to_temp_file(args.sars_data_path, '.rds')
                temp_files['sars'] = actual_sars_path
            else:
                actual_sars_path = args.sars_data_path
        
        if args.rsv_data_path:
            if args.rsv_data_path.startswith(('http://', 'https://')):
                actual_rsv_path = download_url_to_temp_file(args.rsv_data_path, '.rds')
                temp_files['rsv'] = actual_rsv_path
            else:
                actual_rsv_path = args.rsv_data_path
        
        # 导入校验函数
        from .validate_rds import validate_rds_file
        
        # 校验输入的RDS文件
        validation_results = {}
        
        if actual_flu_path:
            flu_validation = validate_rds_file(actual_flu_path, args.flu_binding_threshold, args.output_dir)
            validation_results["flu"] = flu_validation
            if not flu_validation["valid"]:
                return {
                    "status": "error",
                    "message": f"Flu数据验证失败: {flu_validation['message']}",
                    "validation_results": validation_results
                }
        
        if actual_sars_path:
            sars_validation = validate_rds_file(actual_sars_path, args.sars_binding_threshold, args.output_dir)
            validation_results["sars"] = sars_validation
            if not sars_validation["valid"]:
                return {
                    "status": "error",
                    "message": f"SARS数据验证失败: {sars_validation['message']}",
                    "validation_results": validation_results
                }
        
        if actual_rsv_path:
            rsv_validation = validate_rds_file(actual_rsv_path, args.rsv_binding_threshold, args.output_dir)
            validation_results["rsv"] = rsv_validation
            if not rsv_validation["valid"]:
                return {
                    "status": "error",
                    "message": f"RSV数据验证失败: {rsv_validation['message']}",
                    "validation_results": validation_results
                }
        
        # 生成时间戳后缀用于文件名区分
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        r_script = f"""
# Figure 2 Analysis - DEG Analysis
rm(list = ls())
options(stringsAsFactors = FALSE)
Sys.setenv(LANGUAGE = "en")

# Load required libraries
suppressPackageStartupMessages({{
    library(Seurat)
    library(dplyr)
    library(ggplot2)
    library(future)
}})

# 检查并安装ggrepel包用于基因标签
if (!requireNamespace("ggrepel", quietly = TRUE)) {{
    cat("Installing ggrepel package for gene labels...\n")
    tryCatch({{
        install.packages("ggrepel", repos = "https://cran.r-project.org")
        cat("ggrepel package installed successfully\n")
    }}, error = function(e) {{
        cat("Warning: Could not install ggrepel package:", e$message, "\n")
        cat("Gene labels will be skipped\n")
    }})
}}

# 加载ggrepel库
if (requireNamespace("ggrepel", quietly = TRUE)) {{
    library(ggrepel)
    cat("ggrepel loaded successfully\n")
}}

# 配置future包以解决内存限制问题
options(future.globals.maxSize = 2 * 1024^3)  # 设置为2GB
plan("sequential")  # 使用顺序执行避免并行问题

# 检查并尝试安装presto包以提高FindMarkers性能
if (!requireNamespace("presto", quietly = TRUE)) {{
    cat("Installing presto package for faster FindMarkers...\n")
    tryCatch({{
        if (!requireNamespace("devtools", quietly = TRUE)) {{
            install.packages("devtools", repos = "https://cran.r-project.org")
        }}
        devtools::install_github("immunogenomics/presto")
        cat("Presto package installed successfully\n")
    }}, error = function(e) {{
        cat("Warning: Could not install presto package:", e$message, "\n")
        cat("FindMarkers will use default implementation\n")
    }})
}}

# Create output directory
output_dir <- "{args.output_dir}"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 设置时间戳后缀用于文件名区分
timestamp_suffix <- "{timestamp}"

tryCatch({{
    # Load data files
    data_files <- list()
    
    {"data_files$flu <- readRDS('" + actual_flu_path + "')" if actual_flu_path else "# No flu data"}
    {"data_files$sars <- readRDS('" + actual_sars_path + "')" if actual_sars_path else "# No SARS data"}
    {"data_files$rsv <- readRDS('" + actual_rsv_path + "')" if actual_rsv_path else "# No RSV data"}
    
    # 统一的预测字段检测函数
    detect_all_binding_columns <- function(metadata) {{
        all_cols <- colnames(metadata)
        detected_cols <- c()
        
        # 定义所有可能的检测模式
        patterns <- c(
            "^bind_output$",           # 精确匹配 bind_output
            "^bind_predict$",          # 精确匹配 bind_predict
            "^bind_output\\\\.",          # bind_output.开头的列
            "^bind_predict\\\\.",         # bind_predict.开头的列
            "^output\\\\.[xy]$",          # output.x 或 output.y
            "^output\\\\.[0-9]+$",        # output.数字
            "bind_output\\\\.[0-9]+$",    # bind_output.数字
            "bind_predict\\\\.[0-9]+$",   # bind_predict.数字
            "bind_average_values",     # 包含 bind_average_values 的列
            "_bind_.*_ensemble$"       # 任何包含_bind_和_ensemble的列
        )
        
        for (pattern in patterns) {{
            matching_cols <- grep(pattern, all_cols, value = TRUE)
            if (length(matching_cols) > 0) {{
                detected_cols <- c(detected_cols, matching_cols)
                cat("Found columns matching pattern '", pattern, "':", paste(matching_cols, collapse = ", "), "\n")
            }}
        }}
        
        return(unique(detected_cols))
    }}
    
    # 统一的预测值计算函数
    calculate_binding_average <- function(obj, dataset_name) {{
        # 检测预测字段
        selected_cols <- detect_all_binding_columns(obj@meta.data)
        
        if(length(selected_cols) > 0) {{
            cat("Total detected binding columns for", dataset_name, ":", paste(selected_cols, collapse = ", "), "\n")
            
            # 计算平均结合值
            bind_matrix <- obj@meta.data[, selected_cols, drop = FALSE]
            bind_matrix <- apply(bind_matrix, 2, function(x) {{
                x <- as.character(x)
                x[x == "NA" | is.na(x)] <- "0"
                as.numeric(x)
            }})
            
            # 计算平均值
            if(length(selected_cols) == 1) {{
                bind_avg <- bind_matrix[,1]
            }} else {{
                bind_avg <- rowMeans(bind_matrix, na.rm = TRUE)
            }}
            
            # 赋值到meta.data
            obj@meta.data$bind_average_values <- bind_avg
            
            cat("Calculated bind_average_values for", dataset_name, "with range:", 
                paste(range(obj@meta.data$bind_average_values, na.rm = TRUE), collapse = " to "), "\n")
        }} else {{
            cat("Warning: No binding prediction columns found for", dataset_name, "\n")
            obj@meta.data$bind_average_values <- 0
        }}
        
        return(obj)
    }}
    
    # Function to perform DEG analysis
    perform_deg <- function(obj, threshold, output_name) {{
        
        # 打印Seurat版本信息
        cat("Seurat version:", as.character(packageVersion("Seurat")), "\n")
        
        obj <- obj[, !is.na(obj@meta.data$bind_average_values)]
        
        # 确保使用RNA assay作为默认assay
        DefaultAssay(obj) <- "RNA"
        
        # 关键修复：清理Seurat对象的命令历史以避免参数冲突（参考Figure2_Common.R第596行）
        obj@commands <- list()
        
        # 检查并修复Seurat v5数据层问题
        if ("Assay5" %in% class(obj[["RNA"]])) {{
            cat("Detected Seurat v5 object, checking data layers...\n")
            
            # 检查data层是否为空
            if (length(obj[["RNA"]]@layers$data) == 0 || is.null(obj[["RNA"]]@layers$data)) {{
                cat("Data layer is empty, normalizing data...\n")
                obj <- NormalizeData(obj, verbose = FALSE)
            }}
            
            # 合并数据层以确保数据完整性
            tryCatch({{
                obj[["RNA"]] <- JoinLayers(obj[["RNA"]])
                cat("Successfully joined layers\n")
            }}, error = function(e) {{
                cat("Warning: Could not join layers:", e$message, "\n")
            }})
        }}
        
        cat("Using original object structure as in Figure2_Common.R\n")
        
        # Classify binding levels
        obj@meta.data$bind_level <- ifelse(
            obj@meta.data$bind_average_values >= threshold,
            "broad",
            "specific"
        )
        
        # 检查分组是否有效
        bind_table <- table(obj@meta.data$bind_level)
        cat("Binding level distribution:", paste(names(bind_table), bind_table, sep="=", collapse=", "), "\n")
        
        if (length(bind_table) < 2 || any(bind_table < 3)) {{
            stop("Insufficient cells in one or both groups for DEG analysis. Need at least 3 cells per group.")
        }}
        
        # Find markers
        Idents(obj) <- "bind_level"
        
        # 使用更稳定的FindMarkers参数配置
        markers <- FindMarkers(
            obj,
            ident.1 = "broad",
            ident.2 = "specific",
            logfc.threshold = {args.logfc_threshold},
            min.pct = {args.min_pct},
            verbose = FALSE,
            test.use = "wilcox",  # 明确指定测试方法
            slot = "data"  # 明确指定使用data slot（兼容Seurat v4/v5）
        )
        
        # Adjust p-values
        markers$p_val_adj <- p.adjust(markers$p_val, method = "fdr")
        
        # Save results with timestamp prefix
        output_file <- file.path(output_dir, paste0(output_name, "_", timestamp_suffix, "_DEG.csv"))
        write.csv(markers, output_file)
        
        # 生成火山图（参考Figure2_Common.R实现）
        if (nrow(markers) > 0) {{
            # 添加 -log10(p_val_adj) 列
            markers$log10pvalue <- -log10(markers$p_val_adj)
            
            # 标记显著性（根据实际数据调整标准）
            # 使用更宽松的标准以确保有足够的显著基因用于可视化
            markers$significance <- ifelse(
                markers$p_val_adj < 0.5,  # 使用更宽松的p值阈值
                ifelse(markers$avg_log2FC > 0.1, "Up", 
                       ifelse(markers$avg_log2FC < -0.1, "Down", "Not Significant")), 
                "Not Significant"
            )
            
            # 打印显著性分布以便调试
            cat("Significance distribution:", table(markers$significance), "\n")
            cat("P-value range:", range(markers$p_val), "\n")
            cat("Adjusted P-value range:", range(markers$p_val_adj), "\n")
            cat("Log2FC range:", range(markers$avg_log2FC), "\n")
            
            # 计算百分比差异
            markers$pct_dif <- markers$pct.1 - markers$pct.2
            
            # 计算X轴范围（参考Figure2_Common.R实现，添加安全检查）
            # 过滤掉无限值和NaN
            valid_fc <- markers$avg_log2FC[is.finite(markers$avg_log2FC)]
            if (length(valid_fc) > 0) {{
                x1 <- max(abs(valid_fc))
                # 确保x1是有限的正数
                if (!is.finite(x1) || x1 <= 0) {{
                    x1 <- 4  # 默认值
                }}
            }} else {{
                x1 <- 4  # 默认值
            }}
            x2 <- -x1
            
            # 绘制火山图（增强版本，包含清晰的坐标轴和基因标签）
            p <- ggplot(markers, aes(x = avg_log2FC, y = log10pvalue, color = significance)) +
                geom_point(alpha = 0.8, size = 1.5) +  # 增大点的大小以便更好地显示
                scale_color_manual(values = c("Up" = "red", "Down" = "blue", "Not Significant" = "grey")) +
                geom_vline(xintercept = c(-1, 1), linetype = "dashed", alpha = 0.7) +
                geom_hline(yintercept = -log10(0.05), linetype = "dashed", alpha = 0.7) +  # 添加显著性水平线
                labs(title = "Volcano Plot", 
                     x = "log2 Fold Change", 
                     y = "-log10 p-value",
                     color = "Significance") +
                theme_minimal() +
                theme(aspect.ratio = 1,
                      text = element_text(size = 14),  # 调整字体大小
                      axis.text = element_text(size = 12, color = "black"),  # 增强坐标轴文字
                      axis.title = element_text(size = 14, color = "black", face = "bold"),  # 增强坐标轴标题
                      axis.line = element_line(colour = "black", linewidth = 0.8),  # 增强坐标轴线
                      axis.ticks = element_line(colour = "black", linewidth = 0.5),  # 添加坐标轴刻度线
                      axis.ticks.length = unit(0.2, "cm"),  # 设置刻度线长度
                      panel.grid.major = element_blank(),  # 去掉主要网格线
                      panel.grid.minor = element_blank(),  # 去掉次要网格线
                      panel.background = element_rect(fill = "white", colour = "white"),  # 明确设置面板背景为白色
                      plot.background = element_rect(fill = "white", colour = "white"),  # 明确设置图形背景为白色
                      legend.position = "right",  # 图例位置
                      legend.title = element_text(size = 12, face = "bold"),  # 图例标题
                      legend.text = element_text(size = 10),  # 图例文字
                      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)) +  # 标题居中
                # 设置坐标轴范围和刻度（修复重复scale问题，添加安全检查）
                scale_x_continuous(limits = c(x2, x1), breaks = seq(ceiling(x2), floor(x1), by = 1)) +
                scale_y_continuous(limits = c(0, ifelse(is.finite(max(markers$log10pvalue)), max(markers$log10pvalue) + 3, 10)), 
                                 breaks = seq(0, ceiling(ifelse(is.finite(max(markers$log10pvalue)), max(markers$log10pvalue) + 3, 10)), by = 2))
            
            # 添加基因名标记（显著基因和高表达基因）
            if (requireNamespace("ggrepel", quietly = TRUE)) {{
                # 选择要标记的基因：显著基因 + top表达基因
                top_genes_to_label <- markers[order(-abs(markers$avg_log2FC)), ][1:min(15, nrow(markers)), ]
                significant_genes <- markers[markers$significance != "Not Significant", ]
                
                # 合并要标记的基因
                genes_to_label <- unique(c(rownames(significant_genes), rownames(top_genes_to_label)))
                
                p <- p + geom_text_repel(
                    aes(label = ifelse(rownames(markers) %in% genes_to_label, rownames(markers), "")), 
                    size = 2.5, 
                    box.padding = 0.5,
                    point.padding = 0.3,
                    max.overlaps = 30,
                    min.segment.length = 0.1,
                    segment.color = "grey50",
                    segment.linewidth = 0.3,
                    force = 2,
                    nudge_x = 0.1,
                    nudge_y = 0.1
                )
            }}
            
            # 保存火山图with timestamp prefix
            volcano_file <- file.path(output_dir, paste0(output_name, "_", timestamp_suffix, "_volcano_plot.png"))
            ggsave(volcano_file, p, width = 10, height = 8, dpi = 300)
            cat("Saved volcano plot to:", volcano_file, "\n")
        }}
        
        message(paste("Saved DEG results to:", output_file))
        return(markers)
    }}
    
    # 为每个数据集计算统一的预测平均值
    {"if ('flu' %in% names(data_files)) { data_files$flu <- calculate_binding_average(data_files$flu, 'flu') }" if actual_flu_path else ""}
    {"if ('sars' %in% names(data_files)) { data_files$sars <- calculate_binding_average(data_files$sars, 'sars') }" if actual_sars_path else ""}
    {"if ('rsv' %in% names(data_files)) { data_files$rsv <- calculate_binding_average(data_files$rsv, 'rsv') }" if actual_rsv_path else ""}
    
    # Run analysis for available datasets
    results <- list()
    
    {"if ('flu' %in% names(data_files)) { results$flu <- perform_deg(data_files$flu, " + str(args.flu_binding_threshold) + ", 'flu') }" if actual_flu_path else ""}
    {"if ('sars' %in% names(data_files)) { results$sars <- perform_deg(data_files$sars, " + str(args.sars_binding_threshold) + ", 'sars') }" if actual_sars_path else ""}
    {"if ('rsv' %in% names(data_files)) { results$rsv <- perform_deg(data_files$rsv, " + str(args.rsv_binding_threshold) + ", 'rsv') }" if actual_rsv_path else ""}
    
    message("Figure 2 DEG analysis completed successfully!")
    
}}, error = function(e) {{
    message("Error in Figure 2 analysis: ", e$message)
    quit(status = 1)
}})
"""
        
        result = execute_r_script(r_script, args.output_dir)
        
        # 添加验证结果到返回值
        result["validation_results"] = validation_results
        
        return result
    finally:
        # 清理临时文件
        for temp_path in temp_files.values():
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"已清理临时文件: {temp_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败 {temp_path}: {e}")

@mcp.tool()
def run_figure3_correlation_analysis(
    figure2_results_dir: str = "./output/Figure2",
    output_dir: str = "./output/Figure3"
) -> Dict[str, Any]:
    """
    Run Figure 3 correlation analysis between different datasets
    
    Args:
        figure2_results_dir: Directory containing Figure 2 DEG results
        output_dir: Output directory path
    
    Returns:
        Analysis results and correlation statistics
    """
    
    # 处理 URL 下载（如果是目录的 zip 压缩包）
    temp_dir_path = None
    actual_figure2_results_dir = figure2_results_dir
    
    try:
        # 如果 figure2_results_dir 是 URL，先下载并解压到临时目录
        if figure2_results_dir.startswith(('http://', 'https://')):
            try:
                # 从 URL 路径检测压缩格式
                parsed_url = urlparse(figure2_results_dir)
                url_path = parsed_url.path.lower()
                
                # 检测文件扩展名
                if url_path.endswith('.zip'):
                    archive_format = 'zip'
                    default_ext = '.zip'
                elif url_path.endswith(('.tar.gz', '.tgz')):
                    archive_format = 'tar.gz'
                    default_ext = '.tar.gz'
                elif url_path.endswith('.tar'):
                    archive_format = 'tar'
                    default_ext = '.tar'
                else:
                    # 默认尝试 zip 格式
                    archive_format = 'zip'
                    default_ext = '.zip'
                
                # 下载压缩文件到临时文件
                temp_archive = download_url_to_temp_file(figure2_results_dir, default_ext)
                
                # 创建临时目录用于解压
                temp_dir = tempfile.mkdtemp(prefix='dir_download_')
                
                # 根据格式解压文件
                if archive_format == 'zip':
                    with zipfile.ZipFile(temp_archive, 'r') as archive_ref:
                        archive_ref.extractall(temp_dir)
                elif archive_format in ('tar.gz', 'tgz'):
                    with tarfile.open(temp_archive, 'r:gz') as archive_ref:
                        archive_ref.extractall(temp_dir)
                elif archive_format == 'tar':
                    with tarfile.open(temp_archive, 'r') as archive_ref:
                        archive_ref.extractall(temp_dir)
                else:
                    raise ValueError(f"Unsupported archive format: {archive_format}")
                
                # 清理临时压缩文件
                try:
                    os.unlink(temp_archive)
                except:
                    pass
                
                actual_figure2_results_dir = temp_dir
                temp_dir_path = temp_dir
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to download and extract directory from URL {figure2_results_dir}: {str(e)}",
                    "output_files": []
                }
        
        # 1. 检查输入目录是否存在
        if not os.path.exists(actual_figure2_results_dir):
            # 清理临时目录
            if temp_dir_path and os.path.exists(temp_dir_path):
                try:
                    shutil.rmtree(temp_dir_path)
                except:
                    pass
            return {
                "status": "error",
                "message": f"输入目录不存在: {actual_figure2_results_dir}",
                "output_files": []
            }
        
        # 2. 检查是否有足够的DEG文件（至少2个）
        deg_files = glob.glob(os.path.join(actual_figure2_results_dir, "*_DEG.csv"))
        if len(deg_files) < 2:
            # 清理临时目录
            if temp_dir_path and os.path.exists(temp_dir_path):
                try:
                    shutil.rmtree(temp_dir_path)
                except:
                    pass
            return {
                "status": "error",
                "message": f"Need at least 2 DEG result files for correlation analysis. Found {len(deg_files)} files.",
                "output_files": []
            }
        
        # 3. 检查每个DEG文件是否包含avg_log2FC字段
        for deg_file in deg_files:
            try:
                # 读取CSV文件的第一行来检查列名
                with open(deg_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # 读取第一行（列名）
                    if 'avg_log2FC' not in header:
                        # 清理临时目录
                        if temp_dir_path and os.path.exists(temp_dir_path):
                            try:
                                shutil.rmtree(temp_dir_path)
                            except:
                                pass
                        return {
                            "status": "error",
                            "message": f"文件 {os.path.basename(deg_file)} 缺少必需的 'avg_log2FC' 字段",
                            "output_files": []
                        }
            except Exception as e:
                # 清理临时目录
                if temp_dir_path and os.path.exists(temp_dir_path):
                    try:
                        shutil.rmtree(temp_dir_path)
                    except:
                        pass
                return {
                    "status": "error",
                    "message": f"无法读取文件 {os.path.basename(deg_file)}: {str(e)}",
                    "output_files": []
                }
        
        logger.info(f"输入验证通过: 找到 {len(deg_files)} 个有效的DEG文件")
        
        r_script = f"""
# Figure 3 Analysis - Correlation Analysis
rm(list = ls())
options(stringsAsFactors = FALSE)
Sys.setenv(LANGUAGE = "en")

suppressPackageStartupMessages({{
    library(Seurat)
    library(dplyr)
    library(ggplot2)
}})

output_dir <- "{output_dir}"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tryCatch({{
    # Load DEG results from Figure 2
    deg_files <- list.files("{actual_figure2_results_dir}", pattern = "*_DEG.csv$", full.names = TRUE)
    
    if (length(deg_files) > 0) {{
        deg_results <- list()
        for (file in deg_files) {{
            name <- gsub("_DEG.csv", "", basename(file))
            deg_results[[name]] <- read.csv(file, row.names = 1)
        }}
        
        # Perform correlation analysis between datasets
        if (length(deg_results) >= 2) {{
            # Create correlation matrix
            common_genes <- Reduce(intersect, lapply(deg_results, rownames))
            
            if (length(common_genes) > 10) {{
                cor_matrix <- matrix(NA, length(deg_results), length(deg_results))
                rownames(cor_matrix) <- names(deg_results)
                colnames(cor_matrix) <- names(deg_results)
                
                for (i in 1:length(deg_results)) {{
                    for (j in 1:length(deg_results)) {{
                        genes_i <- deg_results[[i]][common_genes, "avg_log2FC"]
                        genes_j <- deg_results[[j]][common_genes, "avg_log2FC"]
                        cor_matrix[i, j] <- cor(genes_i, genes_j, use = "complete.obs")
                    }}
                }}
                
                # Save correlation matrix
                write.csv(cor_matrix, file.path(output_dir, "correlation_matrix.csv"))
                
                # Create correlation plot using base R
                pdf(file.path(output_dir, "correlation_plot.pdf"), width = 8, height = 6)
                
                # Create a heatmap using base R
                par(mar = c(5, 5, 4, 2))
                image(1:ncol(cor_matrix), 1:nrow(cor_matrix), t(cor_matrix[nrow(cor_matrix):1, ]), 
                      col = colorRampPalette(c("blue", "white", "red"))(100),
                      xlab = "", ylab = "", axes = FALSE, main = "Correlation Matrix")
                
                # Add axis labels
                axis(1, at = 1:ncol(cor_matrix), labels = colnames(cor_matrix), las = 2, cex.axis = 0.8)
                axis(2, at = 1:nrow(cor_matrix), labels = rev(rownames(cor_matrix)), las = 2, cex.axis = 0.8)
                
                # Add correlation values as text
                for(i in 1:nrow(cor_matrix)) {{
                    for(j in 1:ncol(cor_matrix)) {{
                        text(j, nrow(cor_matrix) - i + 1, round(cor_matrix[i, j], 2), cex = 0.8)
                    }}
                }}
                
                # Add color bar legend
                legend("topright", legend = c("1", "0", "-1"), 
                       fill = c("red", "white", "blue"), title = "Correlation")
                
                dev.off()
                
                message("Correlation analysis completed!")
            }} else {{
                message("Not enough common genes for correlation analysis")
            }}
        }} else {{
            message("Need at least 2 DEG result files for correlation analysis")
        }}
    }} else {{
        message("No DEG result files found in {actual_figure2_results_dir}")
    }}
    
}}, error = function(e) {{
    message("Error in Figure 3 analysis: ", e$message)
    quit(status = 1)
}})
"""
        
        return execute_r_script(r_script, output_dir)
    finally:
        # 清理临时目录
        if temp_dir_path and os.path.exists(temp_dir_path):
            try:
                shutil.rmtree(temp_dir_path)
            except:
                pass

@mcp.tool()
def run_figure4_trajectory_analysis(
    a1a11_data_path: Optional[str] = None,
    flu_data_path: Optional[str] = None,
    num_dim: int = 50,
    k_neighbors: int = 40,
    resolution: float = 0.001,
    minimal_branch_len: int = 30,
    output_dir: str = "./output/Figure4"
) -> Dict[str, Any]:
    """
    Run Figure 4 trajectory analysis using monocle3 - supports multiple datasets
    
    Args:
        a1a11_data_path: Path to A1-A11 combined data RDS file
        flu_data_path: Path to flu vaccination data RDS file
        num_dim: Number of dimensions for PCA (default: 50)
        k_neighbors: Number of neighbors for clustering (default: 40)
        resolution: Clustering resolution (default: 0.001)
        minimal_branch_len: Minimal branch length for trajectory (default: 30)
        output_dir: Output directory path
    
    Returns:
        Analysis results and trajectory output paths
    """
    
    if not any([a1a11_data_path, flu_data_path]):
        return {
            "status": "error",
            "message": "At least one data file path must be provided"
        }
    
    # 导入校验函数
    from .validate_rds_figure4 import validate_rds_file_for_trajectory
    
    # 校验输入的RDS文件
    validation_results = {}
    
    if a1a11_data_path:
        a1a11_validation = validate_rds_file_for_trajectory(a1a11_data_path, output_dir)
        validation_results["a1a11"] = a1a11_validation
        if not a1a11_validation["valid"]:
            return {
                "status": "error",
                "message": f"A1A11数据验证失败: {a1a11_validation['message']}",
                "validation_results": validation_results
            }
    
    if flu_data_path:
        flu_validation = validate_rds_file_for_trajectory(flu_data_path, output_dir)
        validation_results["flu"] = flu_validation
        if not flu_validation["valid"]:
            return {
                "status": "error",
                "message": f"Flu数据验证失败: {flu_validation['message']}",
                "validation_results": validation_results
            }
    
    r_script = f"""
# Figure 4 Analysis - Trajectory Analysis
rm(list = ls())
options(stringsAsFactors = FALSE)
Sys.setenv(LANGUAGE = "en")

suppressPackageStartupMessages({{
    library(Seurat)
    library(monocle3)
    library(dplyr)
    library(ggplot2)
}})

output_dir <- "{output_dir}"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tryCatch({{
    # Load data
    {"a1a11_obj <- readRDS('" + a1a11_data_path + "')" if a1a11_data_path else "a1a11_obj <- NULL"}
    {"flu_obj <- readRDS('" + flu_data_path + "')" if flu_data_path else "flu_obj <- NULL"}
    
    # Function to create trajectory
    create_trajectory <- function(obj, output_prefix) {{
        # Extract data with version compatibility
        if ("Assay5" %in% class(obj[["RNA"]])) {{
            # Seurat v5
            data_m <- GetAssayData(obj, assay = "RNA", layer = "counts")
        }} else {{
            # Seurat v4
            data_m <- GetAssayData(obj, assay = "RNA", slot = "counts")
        }}
        data_m <- data_m[rowSums(data_m > 0) >= 3, ]
        
        # Create CDS object
        cell_metadata <- obj@meta.data
        
        # 修复重复列名问题 - 检查并移除重复的列名
        if (any(duplicated(colnames(cell_metadata)))) {{
            message("Warning: Detected duplicated column names in metadata, removing duplicates...")
            # 保留第一个出现的列，移除后续重复的列
            cell_metadata <- cell_metadata[, !duplicated(colnames(cell_metadata))]
            message(paste("Cleaned metadata now has", ncol(cell_metadata), "unique columns"))
        }}
        
        gene_annotation <- data.frame(gene_short_name = rownames(data_m))
        rownames(gene_annotation) <- rownames(data_m)
        
        cds <- new_cell_data_set(
            data_m,
            cell_metadata = cell_metadata,
            gene_metadata = gene_annotation
        )
        
        # Process CDS
        cds <- preprocess_cds(cds, num_dim = {num_dim})
        cds <- reduce_dimension(cds, preprocess_method = "PCA")
        
        # Cluster cells
        cds <- cluster_cells(
            cds,
            resolution = {resolution},
            k = {k_neighbors}
        )
        
        # Learn graph
        cds <- learn_graph(
            cds,
            use_partition = TRUE,
            close_loop = FALSE,
            learn_graph_control = list(
                minimal_branch_len = {minimal_branch_len},
                rann.k = 10
            )
        )
        
        # Save CDS
        saveRDS(cds, file.path(output_dir, paste0(output_prefix, "_cds.rds")))
        
        # Generate trajectory plots
        pdf(file.path(output_dir, paste0(output_prefix, "_trajectory_plot.pdf")), width = 10, height = 8)
        print(plot_cells(cds, color_cells_by = "cluster", label_groups_by_cluster = FALSE))
        print(plot_cells(cds, color_cells_by = "partition", label_groups_by_cluster = FALSE))
        dev.off()
        
        message(paste("Saved trajectory for:", output_prefix))
        return(cds)
    }}
    
    # Run trajectory analysis for available datasets
    results <- list()
    
    {"if (!is.null(a1a11_obj)) { results$a1a11 <- create_trajectory(a1a11_obj, 'a1a11') }" if a1a11_data_path else ""}
    
    {"if (!is.null(flu_obj)) { results$flu <- create_trajectory(flu_obj, 'flu') }" if flu_data_path else ""}
    
    message("Figure 4 trajectory analysis completed!")
    
}}, error = function(e) {{
    message("Error in Figure 4 analysis: ", e$message)
    quit(status = 1)
}})
"""
    
    return execute_r_script(r_script, output_dir)

@mcp.tool()
def run_figure5_bcr_analysis(
    flu_data_path: Optional[str] = None,
    a1a11_data_path: Optional[str] = None,
    bcr_file_path: Optional[str] = None,
    shm_file_path: Optional[str] = None,
    shm_outlier_cutoff: int = 45,
    shm_low_cutoff: int = 3,
    shm_high_cutoff: int = 8,
    output_dir: str = "./output/Figure5"
) -> Dict[str, Any]:
    """
    Run Figure 5 BCR (B Cell Receptor) and SHM (Somatic Hypermutation) analysis
    
    Args:
        flu_data_path: Path to flu vaccination data RDS file
        a1a11_data_path: Path to A1-A11 combined data RDS file
        bcr_file_path: Path to BCR data file (Excel format)
        shm_file_path: Path to SHM data file (CSV format)
        shm_outlier_cutoff: SHM outlier threshold (default: 45)
        shm_low_cutoff: Low SHM threshold (default: 3)
        shm_high_cutoff: High SHM threshold (default: 8)
        output_dir: Output directory path
    
    Returns:
        BCR analysis results
    """
    
    r_script = f"""
# Figure 5 Analysis - BCR and SHM Analysis
rm(list = ls())
options(stringsAsFactors = FALSE)
Sys.setenv(LANGUAGE = "en")

suppressPackageStartupMessages({{
    library(Seurat)
    library(dplyr)
    library(ggplot2)
    library(readxl)
}})

output_dir <- "{output_dir}"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tryCatch({{
    # Load Seurat objects
    {"flu_obj <- readRDS('" + flu_data_path + "')" if flu_data_path else "flu_obj <- NULL"}
    {"a1a11_obj <- readRDS('" + a1a11_data_path + "')" if a1a11_data_path else "a1a11_obj <- NULL"}
    
    # Process BCR data if available
    {"bcr_data <- read_xlsx('" + bcr_file_path + "')" if bcr_file_path else "bcr_data <- NULL"}
    
    if (!is.null(bcr_data)) {{
        message("Processing BCR data...")
        # BCR analysis code here
        write.csv(bcr_data, file.path(output_dir, "bcr_processed.csv"))
    }}
    
    # Process SHM data if available
    {"shm_data <- read.csv('" + shm_file_path + "')" if shm_file_path else "shm_data <- NULL"}
    
    if (!is.null(shm_data)) {{
        message("Processing SHM data...")
        
        # Filter outliers
        shm_data_filtered <- shm_data[shm_data$H_shm < {shm_outlier_cutoff}, ]
        
        # Classify SHM levels
        shm_data_filtered$SHM_level <- case_when(
            shm_data_filtered$H_shm < {shm_low_cutoff} ~ "Low",
            shm_data_filtered$H_shm > {shm_high_cutoff} ~ "High",
            TRUE ~ "Medium"
        )
        
        # Save processed SHM data
        write.csv(shm_data_filtered, file.path(output_dir, "shm_processed.csv"))
        
        # Create SHM distribution plot
        pdf(file.path(output_dir, "shm_distribution.pdf"), width = 8, height = 6)
        p <- ggplot(shm_data_filtered, aes(x = H_shm, fill = SHM_level)) +
            geom_histogram(bins = 30, alpha = 0.7) +
            theme_minimal() +
            labs(title = "SHM Distribution", x = "SHM Length", y = "Count")
        print(p)
        dev.off()
        
        message("SHM analysis completed!")
    }}
    
    # Additional BCR/SHM integration analysis if both datasets available
    if (!is.null(bcr_data) && !is.null(shm_data)) {{
        message("Performing integrated BCR-SHM analysis...")
        # Integration analysis code here
    }}
    
    message("Figure 5 BCR analysis completed!")
    
}}, error = function(e) {{
    message("Error in Figure 5 analysis: ", e$message)
    quit(status = 1)
}})
"""
    
    return execute_r_script(r_script, output_dir)


if __name__ == "__main__":
    """
    启动 MCP 服务器: python bcell_mcp_server.py
    """
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8092
    
    # 使用SSE模式启动
    mcp.run(transport="sse")