"""
RDS文件校验模块

用于验证RDS文件是否满足Figure2 DEG分析的要求
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any


def validate_rds_file(rds_path: str, binding_threshold: float = 0.5, output_dir: str = None) -> Dict[str, Any]:
    """
    验证RDS文件是否满足DEG分析的基本要求
    
    Args:
        rds_path: RDS文件路径
        binding_threshold: 绑定阈值，用于分组
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
# RDS文件验证脚本
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
    message = "RDS文件满足DEG分析要求",
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
    
    # 检测绑定预测相关列
    metadata <- obj@meta.data
    all_cols <- colnames(metadata)
    
    # 定义所有可能的检测模式
    patterns <- c(
        "^bind_output$",
        "^bind_predict$",
        "^bind_output\\\\.",
        "^bind_predict\\\\.",
        "^output\\\\.[xy]$",
        "^output\\\\.[0-9]+$",
        "bind_output\\\\.[0-9]+$",
        "bind_predict\\\\.[0-9]+$",
        "bind_average_values",
        "_bind_.*_ensemble$"
    )
    
    # 查找匹配的列
    detected_cols <- c()
    for (pattern in patterns) {{
        matching_cols <- grep(pattern, all_cols, value = TRUE)
        if (length(matching_cols) > 0) {{
            detected_cols <- c(detected_cols, matching_cols)
        }}
    }}
    detected_cols <- unique(detected_cols)
    
    # 检查是否找到绑定预测列
    if (length(detected_cols) == 0) {{
        result$valid <- FALSE
        result$message <- "元数据中缺少绑定预测相关列"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 计算平均绑定值
    bind_matrix <- metadata[, detected_cols, drop = FALSE]
    bind_matrix <- apply(bind_matrix, 2, function(x) {{
        x <- as.character(x)
        x[x == "NA" | is.na(x)] <- "0"
        as.numeric(x)
    }})
    
    # 计算平均值
    if (length(detected_cols) == 1) {{
        bind_avg <- bind_matrix[,1]
    }} else {{
        bind_avg <- rowMeans(bind_matrix, na.rm = TRUE)
    }}
    
    # 分组
    bind_level <- ifelse(bind_avg >= {binding_threshold}, "broad", "specific")
    bind_table <- table(bind_level)
    
    # 检查分组是否有效
    if (length(bind_table) < 2 || any(bind_table < 3)) {{
        result$valid <- FALSE
        result$message <- "分组后细胞数量不足，每组至少需要3个细胞"
        result$details$group_counts <- as.list(bind_table)
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 检查表达矩阵
    DefaultAssay(obj) <- "RNA"
    
    # 检查数据层
    has_data <- FALSE
    if ("Assay5" %in% class(obj[["RNA"]])) {{
        # Seurat v5
        has_data <- length(obj[["RNA"]]@layers$data) > 0 && !is.null(obj[["RNA"]]@layers$data)
        if (!has_data) {{
            has_data <- length(obj[["RNA"]]@layers$counts) > 0 && !is.null(obj[["RNA"]]@layers$counts)
        }}
    }} else {{
        # Seurat v3/v4
        has_data <- dim(obj[["RNA"]]@data)[1] > 0 && !is.null(obj[["RNA"]]@data)
        if (!has_data) {{
            has_data <- dim(obj[["RNA"]]@counts)[1] > 0 && !is.null(obj[["RNA"]]@counts)
        }}
    }}
    
    if (!has_data) {{
        result$valid <- FALSE
        result$message <- "RNA assay缺少表达数据"
        cat(jsonlite::toJSON(result, auto_unbox = TRUE))
        quit(status = 0)
    }}
    
    # 添加详细信息
    result$details$binding_cols <- detected_cols
    result$details$group_counts <- as.list(bind_table)
    
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
        
        # 创建临时R脚本文件，使用与figure2相似的命名方式
        script_path = temp_dir / f"validate_figure2_deg_{Path(rds_path).stem}.R"
    else:
        # 使用默认的临时目录
        temp_dir = Path("./temp_validation")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时R脚本文件
        script_path = temp_dir / f"validate_{Path(rds_path).stem}.R"
    
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