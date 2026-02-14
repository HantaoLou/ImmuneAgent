"""
Query Cache - P3 Priority Optimization

Implements caching for repeated queries:
1. In-memory cache with TTL
2. Disk persistence for long-term cache
3. Cache hit/miss statistics
4. Smart cache invalidation
"""

import json
import hashlib
import time
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps
import threading


@dataclass
class CacheEntry:
    """A single cache entry"""
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        return cls(
            key=data['key'],
            value=data['value'],
            created_at=data['created_at'],
            expires_at=data['expires_at'],
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', 0),
            tags=data.get('tags', [])
        )


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class QueryCache:
    """
    Query cache with TTL and persistence
    """
    
    def __init__(self,
                 max_size: int = 1000,
                 default_ttl: float = 3600.0,  # 1 hour
                 cache_dir: Optional[str] = None,
                 persist: bool = True):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'query_cache')
        self.persist = persist
        
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Load from disk if persisting
        if self.persist:
            self._load_from_disk()
    
    def _generate_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key from query and context"""
        content = query
        if context:
            # Sort context keys for consistent hashing
            sorted_context = json.dumps(context, sort_keys=True)
            content += sorted_context
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, query: str, context: Optional[Dict] = None) -> Optional[Any]:
        """
        Get cached result for query
        
        Args:
            query: The query string
            context: Optional context for key generation
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._generate_key(query, context)
        
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            self._stats.hits += 1
            return entry.value
    
    def set(self, 
            query: str, 
            value: Any, 
            context: Optional[Dict] = None,
            ttl: Optional[float] = None,
            tags: Optional[List[str]] = None):
        """
        Set cached value for query
        
        Args:
            query: The query string
            value: Value to cache
            context: Optional context for key generation
            ttl: Time-to-live in seconds (uses default if not provided)
            tags: Optional tags for group invalidation
        """
        key = self._generate_key(query, context)
        ttl = ttl if ttl is not None else self.default_ttl
        tags = tags or []
        
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            tags=tags
        )
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = entry
            self._stats.total_entries = len(self._cache)
            
            # Persist to disk
            if self.persist:
                self._save_entry_to_disk(entry)
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self._cache:
            return
        
        # Remove expired entries first
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
            self._stats.evictions += 1
        
        # If still over limit, remove LRU
        while len(self._cache) >= self.max_size:
            # Find LRU entry
            lru_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at)
            del self._cache[lru_key]
            self._stats.evictions += 1
    
    def invalidate(self, query: str, context: Optional[Dict] = None) -> bool:
        """Invalidate a specific cache entry"""
        key = self._generate_key(query, context)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.total_entries = len(self._cache)
                
                # Remove from disk
                if self.persist:
                    self._remove_from_disk(key)
                
                return True
        return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        count = 0
        
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() 
                if tag in v.tags
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                if self.persist:
                    self._remove_from_disk(key)
                count += 1
            
            self._stats.total_entries = len(self._cache)
            self._stats.evictions += count
        
        return count
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats.total_entries = 0
            
            if self.persist:
                self._clear_disk_cache()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            self._stats.total_entries = len(self._cache)
            self._stats.total_size_bytes = sum(
                len(json.dumps(e.to_dict())) for e in self._cache.values()
            )
            return self._stats
    
    def _save_entry_to_disk(self, entry: CacheEntry):
        """Save a cache entry to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            filepath = os.path.join(self.cache_dir, f"{entry.key}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f)
        except Exception:
            pass  # Silently fail for persistence
    
    def _remove_from_disk(self, key: str):
        """Remove a cache entry from disk"""
        try:
            filepath = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
    
    def _load_from_disk(self):
        """Load cache entries from disk"""
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    entry = CacheEntry.from_dict(data)
                    
                    # Only load non-expired entries
                    if not entry.is_expired():
                        self._cache[entry.key] = entry
                except Exception:
                    continue
            
            self._stats.total_entries = len(self._cache)
        except Exception:
            pass
    
    def _clear_disk_cache(self):
        """Clear all cache files from disk"""
        try:
            if not os.path.exists(self.cache_dir):
                return
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
        except Exception:
            pass
    
    def get_report(self) -> str:
        """Generate cache report"""
        stats = self.get_stats()
        
        lines = ["# Query Cache Report\n"]
        lines.append(f"## Statistics")
        lines.append(f"- **Total Entries**: {stats.total_entries}")
        lines.append(f"- **Hits**: {stats.hits}")
        lines.append(f"- **Misses**: {stats.misses}")
        lines.append(f"- **Hit Rate**: {stats.hit_rate:.1%}")
        lines.append(f"- **Evictions**: {stats.evictions}")
        lines.append(f"- **Size**: {stats.total_size_bytes / 1024:.1f} KB")
        
        # Top accessed entries
        with self._lock:
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda e: e.access_count,
                reverse=True
            )[:10]
        
        if sorted_entries:
            lines.append(f"\n## Top 10 Accessed Entries")
            for entry in sorted_entries:
                lines.append(f"- {entry.key[:16]}...: {entry.access_count} accesses")
        
        return "\n".join(lines)


# Decorator for caching function results
def cached_knowledge_retrieval(ttl: float = 3600.0):
    """
    Decorator to cache knowledge retrieval results
    
    Usage:
        @cached_knowledge_retrieval(ttl=1800)
        async def retrieve_knowledge(query: str, ...):
            ...
    """
    def decorator(func: Callable):
        _cache = QueryCache(default_ttl=ttl, persist=False)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            query = args[0] if args else kwargs.get('query', '')
            context = {'args': str(args[1:]), 'kwargs': str(kwargs)}
            
            # Check cache
            cached_result = _cache.get(query, context)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            _cache.set(query, result, context)
            
            return result
        
        # Add cache access
        wrapper.cache = _cache
        
        return wrapper
    
    return decorator


# Global cache instance
_global_cache: Optional[QueryCache] = None


def get_global_cache() -> QueryCache:
    """Get or create the global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache()
    return _global_cache

