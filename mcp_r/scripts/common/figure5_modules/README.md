# Figure5 模块化代码说明

本目录包含从 `Figure5_Common.R` 拆分出来的独立模块，每个模块对应原始代码中的一个 Manuscript 块。

## 目录结构

```
figure5_modules/
├── Figure5_Utils.R              # 工具函数集合
├── Figure5_C_Isotype.R          # Figure 5C - Isotype分布和SHM率分析
├── Figure5_D_Neutralization.R   # Figure 5D - 中和抗体SHM率比较
└── README.md                    # 本说明文件
```

## 模块功能说明

### 1. Figure5_Utils.R
包含所有模块需要的通用函数：
- `load_and_preprocess_data()` - 数据加载和预处理
- `create_output_directories()` - 创建输出目录
- `load_required_packages()` - 加载必需的R包
- `get_isotype_mapping()` - 获取isotype映射表
- `create_igh_isotype_field()` - 创建IGH_isotype字段
- `estimate_shm_from_expression()` - 基于表达估算SHM水平
- `detect_binding_columns()` - 检测结合预测字段
- `process_binding_data()` - 处理结合数据
- `calculate_sars2_predictions()` - 计算SARS2结合和中和预测值
- `prepare_analysis_dataframe()` - 准备分析数据框
- `select_target_celltype()` - 选择目标细胞类型
- `create_binding_neutralization_levels()` - 创建结合和中和水平分类
- `filter_extreme_shm()` - 过滤异常SHM值
- `create_shm_levels()` - 创建SHM水平分类

### 2. Figure5_C_Isotype.R
**功能**: Isotype分布和SHM率分析
- 分析广泛反应性BCR与特异性和非结合BCR的isotype分布
- 比较不同结合水平的SHM率
- 生成条形图和箱线图组合

**输出**:
- `Figure5C.pdf` - 组合图形（isotype分布 + SHM水平分布 + SHM箱线图）
- `Figure5C_analysis_data.csv` - 分析数据
- `Figure5C_statistics.csv` - 统计数据

### 3. Figure5_D_Neutralization.R
**功能**: 中和抗体SHM率比较
- 比较预测中和抗体与非中和抗体的SHM率
- 分析来自FCRL5+非典型B细胞的抗体
- 生成条形图和箱线图组合

**输出**:
- `Figure5D.pdf` - 组合图形（isotype分布 + SHM水平分布 + SHM箱线图）
- `Figure5D_analysis_data.csv` - 分析数据
- `Figure5D_statistics.csv` - 统计数据

## 使用方法

### 基本用法
每个模块都可以独立运行：

```bash
# 运行Figure 5C
Rscript Figure5_C_Isotype.R <input_rds_file> <base_dir>

# 运行Figure 5D
Rscript Figure5_D_Neutralization.R <input_rds_file> <base_dir>
```

### 参数说明
- `<input_rds_file>`: 输入的RDS文件路径
- `<base_dir>`: 基础输出目录

### 使用示例
```bash
# 运行Figure 5C分析
Rscript Figure5_C_Isotype.R data/cells.rds output/

# 运行Figure 5D分析
Rscript Figure5_D_Neutralization.R data/cells.rds output/
```

## 输入数据要求

### 必需字段
- 细胞类型注释：`celltype` 或 `CellType`
- Isotype信息：包含 "isotype" 的字段（忽略大小写）

### 可选字段
- 结合预测数据：包含 "bind_predict"、"output" 或 "bind_output" 的列
- 亲和力分数：`high_affinity1`、`Low_affinity2`（用于SHM估算）

### 数据处理特性
- **自动字段检测**: 自动检测和映射不同数据集的字段名
- **Isotype标准化**: 统一不同格式的isotype注释
- **细胞类型映射**: 支持King数据集到标准格式的映射
- **SHM估算**: 基于基因表达特征估算SHM水平
- **异常值过滤**: 自动过滤极端SHM值

## 依赖包
- Seurat
- dplyr
- ggplot2
- cowplot
- ggrepel
- stringr
- monocle3
- RColorBrewer
- Nebulosa
- ggrastr

## 特性
- **独立运行**: 每个模块都可以独立执行
- **错误处理**: 包含完善的错误处理和警告信息
- **自动检测**: 自动检测数据格式和可用字段
- **向后兼容**: 保持与原始Figure5_Common.R相同的逻辑
- **完整输出**: 生成图片和相关数据文件

## 分析流程

### Figure 5C 分析流程
1. 数据加载和预处理
2. 创建IGH_isotype字段
3. 估算SHM水平
4. 计算SARS2结合和中和预测值
5. 选择目标细胞类型
6. 创建结合水平分类
7. 过滤异常SHM值
8. 生成组合图形

### Figure 5D 分析流程
1. 数据加载和预处理
2. 创建IGH_isotype字段
3. 估算SHM水平
4. 计算SARS2结合和中和预测值
5. 选择目标细胞类型
6. 创建中和水平分类
7. 过滤异常SHM值
8. 生成组合图形

## 与原始代码的对应关系
- **Figure5_C_Isotype.R** ← Figure5_Common.R 第67-447行 (figure5C)
- **Figure5_D_Neutralization.R** ← Figure5_Common.R 第449-540行 (figure5D)

## 输出目录结构
```
output/
└── Figure5/
    ├── plots/
    │   ├── Figure5C.pdf
    │   ├── Figure5D.pdf
    │   ├── Figure5C_session_info.txt
    │   └── Figure5D_session_info.txt
    └── files/
        ├── Figure5C_analysis_data.csv
        ├── Figure5C_statistics.csv
        ├── Figure5D_analysis_data.csv
        └── Figure5D_statistics.csv
```