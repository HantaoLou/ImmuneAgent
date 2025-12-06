# 生信分析模块化MCP服务器测试文件

本目录包含生信分析模块化MCP服务器中所有工具的测试文件。

## 文件结构

```
test/
├── README.md                    # 本说明文件
├── test_all_tools.py           # 运行所有工具测试的主文件
├── test_figure2_tools.py       # Figure2相关工具测试
├── test_figure3_tools.py       # Figure3相关工具测试
├── test_figure4_tools.py       # Figure4相关工具测试
└── test_figure5_tools.py       # Figure5相关工具测试
```

## 测试配置

### 输入数据
- **RDS文件**: `D:\data\test_data_20251001\Age_Bcells.rds`
- **输出目录**: `D:\data\test_data_20251001`

### 测试工具列表

#### Figure2工具 (6个)
1. `antigen_binding_prediction_visualization` - 单细胞B细胞抗原结合预测可视化分析
2. `bcell_celltype_distribution_analysis` - 单细胞B细胞亚群类型分布可视化分析
3. `binding_prediction_interval_distribution_analysis` - 单细胞抗原结合预测值区间分布统计分析
4. `differential_gene_expression_volcano_analysis` - 单细胞差异表达基因分析和火山图可视化
5. `umap_dimensionality_reduction_visualization` - 单细胞B细胞UMAP降维和细胞亚群可视化分析
6. `bcell_marker_gene_dotplot_analysis` - B细胞亚群特异性标记基因表达点图分析

#### Figure3工具 (5个)
1. `antigen_binding_neutralization_density_visualization` - 单细胞抗原结合和中和预测密度图可视化分析
2. `bcell_celltype_umap_visualization` - 单细胞B细胞类型UMAP空间分布可视化分析
3. `bcell_marker_gene_expression_dotplot` - B细胞亚群标记基因表达模式点图可视化分析
4. `differential_gene_correlation_analysis` - 差异表达基因相关性分析和散点图可视化
5. `prediction_value_density_visualization` - 预测值UMAP密度图可视化分析

#### Figure4工具 (4个)
1. `pseudotime_trajectory_analysis` - 单细胞B细胞伪时间轨迹分析和UMAP可视化
2. `pseudotime_celltype_boxplot_analysis` - 伪时间与细胞类型分布箱线图分析
3. `trajectory_polynomial_regression_analysis` - 轨迹多项式回归分析和特征基因模块评分
4. `trajectory_supplementary_analysis` - 轨迹分析补充图形生成和转录标记分析

#### Figure5工具 (2个)
1. `bcr_isotype_distribution_shm_analysis` - B细胞受体同型分布和体细胞超突变率分析
2. `neutralizing_antibody_shm_comparison_analysis` - 中和抗体与非中和抗体SHM率比较分析

## 使用方法

### 运行所有测试
```bash
cd d:\PartTimeJob\hd\antibody_gen\mcp_r\test
python test_all_tools.py
```

### 运行特定Figure的测试
```bash
# 运行Figure2工具测试
python test_all_tools.py 2

# 运行Figure3工具测试
python test_all_tools.py 3

# 运行Figure4工具测试
python test_all_tools.py 4

# 运行Figure5工具测试
python test_all_tools.py 5
```

### 运行单个Figure的测试文件
```bash
# 单独运行Figure2测试
python test_figure2_tools.py

# 单独运行Figure3测试
python test_figure3_tools.py

# 单独运行Figure4测试
python test_figure4_tools.py

# 单独运行Figure5测试
python test_figure5_tools.py
```

## 注意事项

### 环境要求
1. **Python环境**: 确保已安装Python 3.7+
2. **R环境**: 确保已安装R和必需的R包
3. **输入文件**: 确保测试数据文件存在于指定路径
4. **输出目录**: 确保输出目录存在且有写入权限

### 依赖关系
- **Figure4工具**: 有执行顺序依赖，建议先运行轨迹分析
- **Figure3相关性分析**: 需要先有DEG结果文件

### 测试参数
- 所有参数都使用合理的默认值
- 可以根据实际数据特征调整参数
- 结合阈值默认为0.5，适合大多数分析

## 故障排除

### 常见问题
1. **文件不存在错误**: 检查输入RDS文件路径是否正确
2. **R脚本执行失败**: 检查R环境和R包是否正确安装
3. **权限错误**: 检查输出目录的写入权限
4. **内存不足**: 对于大数据集，可能需要调整R的内存设置

### 调试建议
1. 先运行单个工具测试，确认基本功能正常
2. 检查R脚本的详细错误信息
3. 确认数据格式符合工具要求
4. 逐步增加测试复杂度

## 输出结果

测试成功后，会在输出目录生成以下结构：
```
D:\data\test_data_20251001\
├── output/
│   ├── Figure2/
│   │   ├── plots/          # PDF图表文件
│   │   └── files/          # CSV数据文件
│   ├── Figure3/
│   │   ├── plots/
│   │   └── files/
│   ├── Figure4/
│   │   ├── plots/
│   │   └── files/
│   └── Figure5/
│       ├── plots/
│       └── files/
```

每个工具会生成相应的分析结果文件，包括可视化图表和数据文件。