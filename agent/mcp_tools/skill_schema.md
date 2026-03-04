# NetTCR Skill YAML Schema 规范

本文档定义 `skill.yaml` 的结构规范，确保命名统一，便于硬编码提取。

---

## 顶层结构

```yaml
meta:              # 元信息（必需）
workflow:          # 工作流程图示（可选）
tools:             # 工具列表（必需）
input_formats:     # 输入格式定义（可选）
constraints:       # 约束条件（可选）
external_refs:     # 外部引用（可选）
```

---

## 字段命名规范

### 通用规则
- 使用 `snake_case` 命名
- 布尔值字段以 `is_` 或 `has_` 前缀
- 列表字段使用复数形式
- 时间/大小字段带单位后缀

---

## 详细 Schema 定义

### 1. meta 元信息

```yaml
meta:
  name: "nettcr"                    # 服务名称（必需）
  version: "2.2"                    # 版本号（必需）
  summary: "一句话简介"               # 简短描述（必需）
  description: "详细描述"            # 详细描述（必需）
  tags:                             # 标签列表（必需）
    - tcr
    - peptide-binding
  capabilities:                     # 能力列表（必需）
    - "能力1"
    - "能力2"
```

### 2. workflow 工作流程（可选）

```yaml
workflow:
  description: "流程描述"
  steps:
    - step: 1
      name: "信息查询"
      tools: ["list_available_peptides", "check_peptide_support"]
      optional: true
    - step: 2
      name: "输入验证"
      tools: ["validate_tcr_input"]
      optional: false
    - step: 3
      name: "执行预测"
      tools: ["predict_tcr_binding_fast", "predict_tcr_binding_complete"]
      optional: false
```

### 3. tools 工具列表

```yaml
tools:
  - name: "tool_name"               # 工具名称（必需）
    summary: "一句话简介"             # 简短描述（必需）
    description: "详细描述"          # 详细描述（必需）
    
    category: "prediction"          # 分类（必需）
    priority: "core"                # 优先级: core/high/medium/low/legacy
    execution_order: 1              # 执行顺序（数字）
    
    dependencies:                   # 依赖关系
      prerequisites: []             # 前置依赖（必需）
      recommended: []               # 推荐前置（可选）
      enables: []                   # 启用的后续工具（可选）
      alternative_to: []            # 替代工具（可选）
    
    when_to_use:                    # 使用场景
      - "场景1"
      - "场景2"
    
    parameters:                     # 参数列表
      - name: "param_name"
        type: "string"              # string/integer/number/boolean/array/object
        required: true
        description: "参数描述"
        default: null               # 默认值（可选）
        example: "示例值"           # 示例（可选）
        constraints: []             # 约束条件（可选）
    
    returns:                        # 返回值定义
      type: "csv"                   # csv/json/object/array/file
      description: "返回描述"
      schema:                       # 统一用 schema 描述结构
        - name: "column1"
          type: "string"
          description: "列描述"
        - name: "column2"
          type: "number"
          description: "列描述"
    
    performance:                    # 性能指标（可选）
      typical_runtime: "1-5 seconds"
      max_batch_size: 10000
```

### 4. input_formats 输入格式（可选）

```yaml
input_formats:
  description: "输入文件格式说明"
  reference: "file_convert.md"      # 外部参考文档
  
  formats:
    - name: "native"
      description: "格式描述"
      columns: ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]
      is_recommended: true
    
    - name: "legacy"
      description: "格式描述"
      columns: ["peptide", "CDR3a", "CDR3b", "TRA_v_gene", "TRB_v_gene"]
      is_recommended: false
```

### 5. constraints 约束条件（可选）

```yaml
constraints:
  pretrained_peptides:
    count: 26
    description: "预训练肽段列表"
    items:
      - "GILGFVFTL"
      - "NLVPMVATV"
  
  sequence_limits:
    peptide_length: "8-15"
    cdr3_length: "5-25"
  
  supported_chains:
    - "alpha"
    - "beta"
    # 不支持: gamma, delta
```

### 6. external_refs 外部引用（可选）

```yaml
external_refs:
  dependencies:
    - service: "tcell"
      description: "T cell 分析服务"
  
  references:
    - title: "论文标题"
      authors: "作者"
      journal: "期刊"
      year: 2024
      doi: "10.xxx/xxx"
```

---

## 类型枚举值

### priority 优先级
- `core` - 核心工具
- `high` - 高优先级
- `medium` - 中优先级
- `low` - 低优先级
- `legacy` - 遗留/兼容工具

### category 分类
- `information` - 信息查询
- `validation` - 验证工具
- `prediction` - 预测工具
- `pipeline` - 流程工具
- `conversion` - 转换工具

### parameter.type 参数类型
- `string` - 字符串
- `integer` - 整数
- `number` - 数字（含小数）
- `boolean` - 布尔值
- `array` - 数组
- `object` - 对象
- `file` - 文件路径

### returns.type 返回类型
- `csv` - CSV 文件
- `json` - JSON 数据
- `object` - 对象
- `array` - 数组
- `file` - 单个文件
- `files` - 多个文件

---

## 示例：规范化的工具定义

```yaml
tools:
  - name: "predict_tcr_binding_fast"
    summary: "快速 TCR-肽段结合预测"
    description: |
      使用 NetTCR-2.2 模型进行快速预测。
      支持 20 个模型集成和百分位排名。
    
    category: "prediction"
    priority: "core"
    execution_order: 3
    
    dependencies:
      prerequisites: []
      recommended: ["validate_tcr_input", "check_peptide_support"]
      enables: []
    
    when_to_use:
      - "需要快速预测"
      - "已有正确格式的数据"
    
    parameters:
      - name: "test_file"
        type: "string"
        required: true
        description: "TCR 数据 CSV 文件路径"
        example: "/data/sessions/{session_id}/input/tcr_data.csv"
      
      - name: "output_file"
        type: "string"
        required: false
        description: "输出文件路径"
        default: null
      
      - name: "percentile_rank"
        type: "boolean"
        required: false
        description: "是否包含百分位排名"
        default: true
    
    returns:
      type: "csv"
      description: "预测结果 CSV"
      schema:
        - name: "peptide"
          type: "string"
          description: "目标肽段"
        - name: "score"
          type: "number"
          description: "结合分数 (0-1)"
        - name: "percentile_rank"
          type: "number"
          description: "百分位排名"
    
    performance:
      typical_runtime: "1-5 seconds per 100 TCRs"
      max_batch_size: 10000
```

