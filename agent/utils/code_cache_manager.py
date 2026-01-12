from typing import Dict, Any, Optional
from pydantic import BaseModel
import hashlib
import json
import os
from datetime import datetime

# 代码缓存数据模型
class CodeCache(BaseModel):
    cache_key: str
    task_type: str
    core_params: Dict[str, Any]
    executable_code: str
    code_description: str
    create_time: str
    reuse_count: int = 0
    related_mcp_tool: Optional[str] = None  # 新增：关联的MCP工具名称（若为MCP调用代码）

class CodeCacheManager:
    """代码缓存管理工具类（统一处理所有代码缓存的读写与复用）"""
    @staticmethod
    def generate_cache_key(task_type: str, core_params: Dict[str, Any]) -> str:
        """生成唯一缓存键：任务类型 + 核心参数哈希"""
        sorted_params = json.dumps(core_params, sort_keys=True, ensure_ascii=False)
        param_hash = hashlib.md5(sorted_params.encode("utf-8")).hexdigest()[:16]
        return f"{task_type}_{param_hash}"
    
    @staticmethod
    def load_persist_cache(cache_path: str = "./code_cache.json") -> Dict[str, CodeCache]:
        """从项目根目录加载持久化代码缓存"""
        if not os.path.exists(cache_path):
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
            return {k: CodeCache(**v) for k, v in cache_dict.items()}
        except Exception as e:
            print(f"[CodeCache] 加载缓存失败：{e}")
            return {}
    
    @staticmethod
    def save_persist_cache(
        cache: Dict[str, CodeCache],
        cache_path: str = "./code_cache.json"
    ) -> None:
        """将代码缓存持久化到项目根目录"""
        cache_dict = {k: v.dict() for k, v in cache.items()}
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=2)
            print(f"[CodeCache] 缓存已持久化到{cache_path}，共{len(cache)}条记录")
        except Exception as e:
            print(f"[CodeCache] 保存缓存失败：{e}")
    
    @staticmethod
    def get_cached_code(
        task_type: str,
        core_params: Dict[str, Any],
        cache_path: str = "./code_cache.json"
    ) -> Optional[CodeCache]:
        """查询指定任务的可复用代码缓存"""
        cache = CodeCacheManager.load_persist_cache(cache_path)
        cache_key = CodeCacheManager.generate_cache_key(task_type, core_params)
        return cache.get(cache_key, None)
    
    @staticmethod
    def add_cached_code(
        task_type: str,
        core_params: Dict[str, Any],
        executable_code: str,
        code_description: str,
        related_mcp_tool: Optional[str] = None,
        cache_path: str = "./code_cache.json"
    ) -> None:
        """添加新的代码缓存（自动持久化）"""
        # 1. 加载现有缓存
        cache = CodeCacheManager.load_persist_cache(cache_path)
        # 2. 生成缓存键与缓存对象
        cache_key = CodeCacheManager.generate_cache_key(task_type, core_params)
        code_cache = CodeCache(
            cache_key=cache_key,
            task_type=task_type,
            core_params=core_params,
            executable_code=executable_code,
            code_description=code_description,
            create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reuse_count=0,
            related_mcp_tool=related_mcp_tool
        )
        # 3. 更新缓存并持久化
        cache[cache_key] = code_cache
        CodeCacheManager.save_persist_cache(cache, cache_path)