# Foundry
维护模型微调相关代码。

## deepspeed 数据并行
```sh
deepspeed --include="localhost:2,3,4,5"  finetune.py 
```

## 普通模式
```
uv run finetune.py
```