# Knowledge Base
知识库包含
- 向量库
- 文档库
- embedding 工具
- 其他和向量库交互的命令行工具

## 向量库
启动向量库
```shell
docker compose up -d
```

## 加载文档

```shell
# 把 library 目录的文档加载到 collection2 中
uv run kb load-doc --path ./library --collection_name collection2
uv run kb query --query "gearbind" --collection_name immune
# 把 https://foldxsuite.crg.eu/products 下的文档，以及指向同个域名的链接加载到 protein 中
uv run kb load-url --url https://foldxsuite.crg.eu/products --collection_name protein

# 重新加载文档
# --reload 会删除已经加载过的同 source 的文档
uv run kb load-doc --path ./library --collection_name collection2 --reload

```

## 依赖这个项目
### 配置 uv
只需要uv配置源码依赖即可
```toml
dependencies = [
    "kb",
]

[tool.uv.sources]
kb = { path = "../kb" }
```

### 更新依赖
```shell
uv update -n
```

### 导入

```python
from kb.vectorstore import get_vector_store
```