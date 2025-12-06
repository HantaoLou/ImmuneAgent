# IgBLAST MCP Server 配置说明

## 临时文件路径配置

### 概述
IgBLAST MCP Server 现在支持配置临时文件的存储路径。默认情况下，所有临时文件都存储在 `/tmp/` 目录中，但您可以根据需要修改为任何具有写权限的绝对路径。

### 配置方法

#### 方法1：修改 config.py 文件
编辑 `config.py` 文件中的 `TEMP_DIR` 变量：

```python
# 默认配置
TEMP_DIR = Path("/tmp")

# 自定义配置示例
TEMP_DIR = Path("/data/temp")           # 自定义临时目录
TEMP_DIR = Path("/home/user/igblast_temp")  # 用户特定目录
```

#### 方法2：环境变量（未来支持）
未来版本将支持通过环境变量配置：
```bash
export IGBLAST_TEMP_DIR="/custom/temp/path"
```

### 临时文件说明

服务器在处理过程中会创建以下临时文件：

1. **输入FASTA文件**: `igblast_input_{session_id}.fasta`
   - 包含待分析的序列数据
   
2. **IgBLAST输出文件**: `igblast_output_{session_id}.txt`
   - IgBLAST分析的原始输出结果
   
3. **ChangeO输出文件**: `changeo_{session_id}_db-pass.tsv`
   - ChangeO处理后的AIRR格式结果

### 路径要求

选择的临时文件目录必须满足以下条件：

1. **绝对路径**: 必须使用完整的绝对路径
2. **写权限**: 运行服务器的用户必须对该目录有写权限
3. **足够空间**: 确保有足够的磁盘空间存储临时文件
4. **目录存在**: 目录必须已经存在（服务器不会自动创建）

### 测试配置

修改配置后，可以通过以下方式测试：

```bash
# 测试目录写权限
touch /your/custom/path/test_file && rm /your/custom/path/test_file

# 测试服务器导入
python -c "import igblast_mcp_server; print('配置成功')"
```

### 清理机制

服务器会自动清理临时文件：
- 处理完成后立即删除所有临时文件
- 使用 `unlink(missing_ok=True)` 确保安全删除
- 每个会话使用唯一的 session_id 避免文件冲突

### 注意事项

1. **性能考虑**: 选择快速的存储设备（如SSD）可以提高处理速度
2. **安全考虑**: 避免使用共享目录，确保文件访问安全
3. **监控空间**: 定期检查临时目录的磁盘使用情况
4. **备份策略**: 临时文件会被自动删除，无需备份