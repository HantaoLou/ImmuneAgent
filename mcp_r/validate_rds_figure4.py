"""
RDS文件校验模块 - Figure4轨迹分析

用于验证RDS文件是否满足Figure4轨迹分析的要求
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any


def validate_rds_file_for_trajectory(rds_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    验证RDS文件是否满足轨迹分析的基本要求
    
    Args:
        rds_path: RDS文件路径
        output_dir: 输出目录，如果为None则使用默认目录
        
    Returns:
        dict: 包含验证结果和详细信息
    """
    # 检查文件是否存在
    if not os.path.exists(rds_path):
        return {
            "valid": False, 
            "message": f"文件不存在: {rds_path}"
        }
    
    # 创建R验证脚本
    r_script = f"""
# RDS文件验证脚本 - Figure4轨迹分析
options(stringsAsFactors = FALSE)

# 加载Seurat包
tryCatch({{
    library(Seurat)
}}, error = function(e) {{
    cat("错误: 无法加载Seurat包，请确保已安装。\\n")
    quit(status = 1)
}})

# 验证结果
result <- list(
    valid = TRUE,
    message = "RDS文件满足轨迹分析要求",
    details = list()
)

# 尝试读取RDS文件
tryCatch({{
    # 读取RDS文件
    obj <- readRDS("{rds_path}")
    
    # 检查是否为Seurat对象
    if (!inherits(obj, "Seurat")) {{
        result$valid <- FALSE
        result$message <- "文件不是有效的Seurat对象"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查是否包含RNA assay
    if (!"RNA" %in% Seurat::Assays(obj)) {{
        result$valid <- FALSE
        result$message <- "Seurat对象缺少RNA assay"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查counts数据
    DefaultAssay(obj) <- "RNA"
    has_counts <- FALSE
    
    if ("Assay5" %in% class(obj[["RNA"]])) {{
        # Seurat v5
        has_counts <- length(obj[["RNA"]]@layers$counts) > 0 && !is.null(obj[["RNA"]]@layers$counts)
    }} else {{
        # Seurat v3/v4
        has_counts <- dim(obj[["RNA"]]@counts)[1] > 0 && !is.null(obj[["RNA"]]@counts)
    }}
    
    if (!has_counts) {{
        result$valid <- FALSE
        result$message <- "RNA assay缺少counts数据（轨迹分析必需）"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查细胞数量
    cell_count <- ncol(obj)
    if (cell_count < 100) {{
        result$valid <- FALSE
        result$message <- paste("细胞数量不足，轨迹分析至少需要100个细胞，当前只有", cell_count, "个")
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查基因数量（应用过滤条件）
    if ("Assay5" %in% class(obj[["RNA"]])) {{
        data_m <- GetAssayData(obj, assay = "RNA", layer = "counts")
    }} else {{
        data_m <- GetAssayData(obj, assay = "RNA", slot = "counts")
    }}
    
    # 应用基因过滤条件（至少在3个细胞中表达）
    filtered_genes <- rowSums(data_m > 0) >= 3
    gene_count <- sum(filtered_genes)
    
    if (gene_count < 500) {{
        result$valid <- FALSE
        result$message <- paste("过滤后基因数量不足，需要至少500个基因，当前只有", gene_count, "个")
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查元数据
    if (nrow(obj@meta.data) == 0) {{
        result$valid <- FALSE
        result$message <- "元数据为空"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 添加详细信息
    result$details$cell_count <- cell_count
    result$details$total_genes <- nrow(data_m)
    result$details$filtered_genes <- gene_count
    result$details$metadata_columns <- ncol(obj@meta.data)
    
    # 检查重复列名
    if (any(duplicated(colnames(obj@meta.data)))) {{
        result$details$has_duplicate_columns <- TRUE
        result$details$duplicate_warning <- "检测到重复列名，分析时将自动处理"
    }} else {{
        result$details$has_duplicate_columns <- FALSE
    }}
    
    # 输出JSON结果
    cat(jsonlite::toJSON(result, auto_unbox = TRUE))
    
}}, error = function(e) {{
    result$valid <- FALSE
    result$message <- paste("验证过程出错:", e$message)
    cat(jsonlite::toJSON(result, auto_unbox = TRUE))
    quit(status = 1)
}})
"""
    
    # 创建临时目录
    if output_dir:
        # 使用指定的输出目录
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时R脚本文件，使用与figure4相似的命名方式
        script_path = temp_dir / f"validate_figure4_trajectory_{Path(rds_path).stem}.R"
    else:
        # 使用默认的临时目录
        temp_dir = Path("./temp_validation_figure4")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时R脚本文件
        script_path = temp_dir / f"validate_trajectory_{Path(rds_path).stem}.R"
    
    try:
        # 导入execute_r_script函数
        from .bcell_mcp_server import execute_r_script
        
        # 使用execute_r_script执行R脚本
        if output_dir:
            # 使用指定的输出目录
            exec_result = execute_r_script(r_script, output_dir)
        else:
            # 使用临时目录
            temp_output_dir = str(temp_dir)
            exec_result = execute_r_script(r_script, temp_output_dir)
        
        # 检查执行结果
        if exec_result["status"] != "success":
            return {
                "valid": False,
                "message": f"验证脚本执行失败: {exec_result.get('message', '未知错误')}"
            }
        
        # 解析JSON输出
        import json
        try:
            # 提取JSON部分，去掉R环境配置信息
            output_lines = exec_result["output"].strip().split('\n')
            json_line = None
            for line in output_lines:
                if line.startswith('{"valid"'):
                    json_line = line
                    break
            
            if json_line:
                validation_result = json.loads(json_line)
                return validation_result
            else:
                return {
                    "valid": False,
                    "message": f"无法找到有效的JSON结果: {exec_result['output']}"
                }
        except json.JSONDecodeError:
            return {
                "valid": False,
                "message": f"无法解析验证结果: {exec_result['output']}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "message": "验证脚本执行超时"
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"验证过程出错: {str(e)}"
        }
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)