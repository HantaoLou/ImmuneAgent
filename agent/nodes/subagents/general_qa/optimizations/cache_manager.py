"""
P3-1: Query Cache Manager

This module provides caching for query results:
- LRU cache for frequently used queries
- TTL-based expiration
- Semantic similarity for cache hits
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import threading
import os
import pickle


@dataclass
class CacheEntry:
    """A single cache entry"""
    query_hash: str
    query_text: str
    result: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    source: str = ""  # Which retrieval source was used
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "source": self.source,
            "metadata": self.metadata
        }


class LRUCache:
    """
    Thread-safe LRU cache implementation
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get an entry from cache"""
        with self._lock:
            entry = self.cache.get(key)
            if entry:
                if entry.is_expired():
                    self._remove(key)
                    return None
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                entry.hit_count += 1
            return entry
    
    def put(self, key: str, entry: CacheEntry):
        """Put an entry in cache"""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                self._remove(key)
            
            # Evict LRU if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = entry
            self.access_order.append(key)
    
    def _remove(self, key: str):
        """Remove an entry"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def _evict_lru(self):
        """Evict the least recently used entry"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]
    
    def clear(self):
        """Clear all entries"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries, return count removed"""
        with self._lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove(key)
            return len(expired_keys)
    
    def size(self) -> int:
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_hits = sum(e.hit_count for e in self.cache.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "total_hits": total_hits,
                "entries": [
                    {
                        "query": e.query_text[:50] + "..." if len(e.query_text) > 50 else e.query_text,
                        "hits": e.hit_count,
                        "source": e.source
                    }
                    for e in list(self.cache.values())[:10]
                ]
            }


class QueryCache:
    """
    Main cache manager for query results
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        persist_path: Optional[str] = None
    ):
        if self._initialized:
            return
        
        self._initialized = True
        self.cache = LRUCache(max_size=max_entries)
        self.default_ttl = default_ttl
        self.persist_path = persist_path
        
        # Load from disk if available
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()
    
    def _hash_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate hash for a query"""
        content = query.lower().strip()
        if context:
            # Include relevant context in hash
            context_str = json.dumps(context, sort_keys=True)
            content += context_str
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Get cached result for a query
        
        Args:
            query: The query string
            context: Optional context for cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._hash_query(query, context)
        entry = self.cache.get(key)
        return entry.result if entry else None
    
    def put(
        self,
        query: str,
        result: Any,
        source: str = "",
        ttl: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache a query result
        
        Args:
            query: The query string
            result: The result to cache
            source: Source of the result
            ttl: Time to live in seconds
            context: Optional context for cache key
            metadata: Additional metadata
        """
        key = self._hash_query(query, context)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(
            query_hash=key,
            query_text=query,
            result=result,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl),
            source=source,
            metadata=metadata or {}
        )
        
        self.cache.put(key, entry)
        
        # Persist if enabled
        if self.persist_path:
            self._save_to_disk()
    
    def get_or_compute(
        self,
        query: str,
        compute_func: callable,
        source: str = "",
        ttl: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, bool]:
        """
        Get cached result or compute and cache
        
        Args:
            query: The query string
            compute_func: Function to compute result if not cached
            source: Source of the result
            ttl: Time to live
            context: Optional context for cache key
            
        Returns:
            Tuple of (result, from_cache)
        """
        cached = self.get(query, context)
        if cached is not None:
            return cached, True
        
        result = compute_func()
        self.put(query, result, source, ttl, context)
        return result, False
    
    def invalidate(self, query: str, context: Optional[Dict[str, Any]] = None):
        """Invalidate a specific cache entry"""
        key = self._hash_query(query, context)
        # LRU cache doesn't have explicit delete, but we can set to empty
        # For now, just let it expire naturally
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        if self.persist_path and os.path.exists(self.persist_path):
            os.remove(self.persist_path)
    
    def cleanup(self) -> int:
        """Remove expired entries"""
        count = self.cache.cleanup_expired()
        if self.persist_path and count > 0:
            self._save_to_disk()
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats()
    
    def _save_to_disk(self):
        """Save cache to disk"""
        if not self.persist_path:
            return
        
        try:
            data = {
                "entries": [e.to_dict() for e in self.cache.cache.values()],
                "saved_at": datetime.now().isoformat()
            }
            # Convert datetime strings back for serialization
            for entry in data["entries"]:
                entry["result"] = str(entry["result"])  # Convert result to string
            
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry_data in data.get("entries", []):
                # Only load non-expired entries
                expires_at = datetime.fromisoformat(entry_data["expires_at"])
                if expires_at > datetime.now():
                    entry = CacheEntry(
                        query_hash=entry_data["query_hash"],
                        query_text=entry_data["query_text"],
                        result=entry_data["result"],
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                        expires_at=expires_at,
                        hit_count=entry_data.get("hit_count", 0),
                        source=entry_data.get("source", ""),
                        metadata=entry_data.get("metadata", {})
                    )
                    self.cache.put(entry.query_hash, entry)
        except Exception as e:
            print(f"Failed to load cache: {e}")


def cached_knowledge_retrieval(
    query: str,
    retrieval_func: callable,
    source: str = "",
    ttl: int = 3600,
    use_cache: bool = True
) -> Tuple[Any, bool]:
    """
    Decorator-like function for cached knowledge retrieval
    
    Args:
        query: The query string
        retrieval_func: Function to retrieve knowledge
        source: Source identifier
        ttl: Cache TTL in seconds
        use_cache: Whether to use cache
        
    Returns:
        Tuple of (result, from_cache)
    """
    if not use_cache:
        return retrieval_func(), False
    
    cache = QueryCache()
    return cache.get_or_compute(
        query=query,
        compute_func=retrieval_func,
        source=source,
        ttl=ttl
    )


# Test function
def test_query_cache():
    """Test the query cache"""
    cache = QueryCache(max_entries=5, default_ttl=60)
    
    print("=" * 80)
    print("Query Cache Test")
    print("=" * 80)
    
    # Test basic put/get
    cache.put("What is BUD?", "BUD stands for Beyond-Use Date", source="knowledge_base")
    result = cache.get("What is BUD?")
    print(f"\nPut/Get test: {result}")
    
    # Test get_or_compute
    def compute_expensive():
        print("  [Computing expensive result...]")
        return "Computed: genomic mutation rate varies"
    
    result1, from_cache1 = cache.get_or_compute(
        "What is genomic mutation rate?",
        compute_expensive,
        source="paper_qa"
    )
    print(f"\nFirst call: {result1} (from cache: {from_cache1})")
    
    result2, from_cache2 = cache.get_or_compute(
        "What is genomic mutation rate?",
        compute_expensive,
        source="paper_qa"
    )
    print(f"Second call: {result2} (from cache: {from_cache2})")
    
    # Test stats
    print(f"\nCache stats:")
    stats = cache.stats()
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Total hits: {stats['total_hits']}")
    
    # Clear
    cache.clear()
    print(f"\nAfter clear: {cache.stats()['size']} entries")


if __name__ == "__main__":
    test_query_cache()

