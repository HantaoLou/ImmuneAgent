"""
Cache Manager for AIRR MCP Server

Caches frequently accessed metadata to improve performance.
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of AIRR metadata"""

    def __init__(
        self,
        cache_dir: str = "cache",
        ttl: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize cache manager

        Args:
            cache_dir: Directory for cache files
            ttl: Time to live in seconds (default: 1 hour)
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.enabled = enabled

        # Create cache directories
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "studies").mkdir(exist_ok=True)
            (self.cache_dir / "repertoires").mkdir(exist_ok=True)
            (self.cache_dir / "queries").mkdir(exist_ok=True)

    def _generate_key(self, prefix: str, identifier: str) -> str:
        """
        Generate cache key from prefix and identifier

        Args:
            prefix: Cache category (study, repertoire, query)
            identifier: Unique identifier

        Returns:
            Cache key
        """
        # Use hash for long identifiers
        if len(identifier) > 100:
            identifier = hashlib.md5(identifier.encode()).hexdigest()

        return f"{prefix}_{identifier}"

    def _get_cache_path(self, category: str, key: str) -> Path:
        """
        Get cache file path

        Args:
            category: Cache category (studies, repertoires, queries)
            key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / category / f"{key}.json"

    def get(
        self,
        category: str,
        identifier: str,
        prefix: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached data if not expired

        Args:
            category: Cache category
            identifier: Unique identifier
            prefix: Optional prefix

        Returns:
            Cached data or None if not found/expired
        """
        if not self.enabled:
            return None

        key = self._generate_key(prefix or category, identifier)
        cache_file = self._get_cache_path(category, key)

        try:
            if not cache_file.exists():
                logger.debug(f"Cache miss: {key}")
                return None

            # Check if expired
            stat = cache_file.stat()
            age = time.time() - stat.st_mtime

            if age > self.ttl:
                logger.debug(f"Cache expired: {key} (age: {age:.1f}s)")
                cache_file.unlink()  # Delete expired cache
                return None

            # Load cached data
            with open(cache_file) as f:
                data = json.load(f)

            logger.debug(f"Cache hit: {key} (age: {age:.1f}s)")
            return data

        except Exception as e:
            logger.error(f"Error reading cache {key}: {e}")
            return None

    def set(
        self,
        category: str,
        identifier: str,
        data: Dict[str, Any],
        prefix: str = ""
    ) -> bool:
        """
        Store data in cache

        Args:
            category: Cache category
            identifier: Unique identifier
            data: Data to cache
            prefix: Optional prefix

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        key = self._generate_key(prefix or category, identifier)
        cache_file = self._get_cache_path(category, key)

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Cached: {key}")
            return True

        except Exception as e:
            logger.error(f"Error writing cache {key}: {e}")
            return False

    def invalidate(
        self,
        category: str,
        identifier: Optional[str] = None,
        prefix: str = ""
    ) -> int:
        """
        Invalidate cache entries

        Args:
            category: Cache category
            identifier: Optional specific identifier (if None, invalidate all in category)
            prefix: Optional prefix

        Returns:
            Number of entries invalidated
        """
        if not self.enabled:
            return 0

        count = 0

        try:
            if identifier:
                # Invalidate specific entry
                key = self._generate_key(prefix or category, identifier)
                cache_file = self._get_cache_path(category, key)

                if cache_file.exists():
                    cache_file.unlink()
                    count = 1
                    logger.info(f"Invalidated cache: {key}")

            else:
                # Invalidate all entries in category
                category_dir = self.cache_dir / category

                if category_dir.exists():
                    for cache_file in category_dir.glob("*.json"):
                        cache_file.unlink()
                        count += 1

                    logger.info(f"Invalidated {count} cache entries in {category}")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

        return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired cache entries

        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0

        count = 0
        current_time = time.time()

        try:
            for category_dir in self.cache_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                for cache_file in category_dir.glob("*.json"):
                    age = current_time - cache_file.stat().st_mtime

                    if age > self.ttl:
                        cache_file.unlink()
                        count += 1

            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")

        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Statistics about cache usage
        """
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Caching is disabled"
            }

        stats = {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl,
            "categories": {}
        }

        try:
            total_size = 0
            total_entries = 0
            current_time = time.time()

            for category_dir in self.cache_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name
                category_stats = {
                    "entries": 0,
                    "size_bytes": 0,
                    "expired": 0
                }

                for cache_file in category_dir.glob("*.json"):
                    file_stat = cache_file.stat()
                    category_stats["entries"] += 1
                    category_stats["size_bytes"] += file_stat.st_size

                    age = current_time - file_stat.st_mtime
                    if age > self.ttl:
                        category_stats["expired"] += 1

                    total_size += file_stat.st_size
                    total_entries += 1

                stats["categories"][category_name] = category_stats

            stats["total_entries"] = total_entries
            stats["total_size_bytes"] = total_size
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            stats["error"] = str(e)

        return stats

    def clear_all(self) -> int:
        """
        Clear all cache entries

        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0

        count = 0

        try:
            for category_dir in self.cache_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                for cache_file in category_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1

            logger.info(f"Cleared {count} cache entries")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

        return count


class QueryCache:
    """Specialized cache for query results"""

    def __init__(self, cache_manager: CacheManager, ttl: int = 900):
        """
        Initialize query cache

        Args:
            cache_manager: Base cache manager
            ttl: Time to live for query results (default: 15 minutes)
        """
        self.cache = cache_manager
        self.ttl = ttl

    def _query_to_key(self, query: Dict[str, Any]) -> str:
        """
        Convert query dict to cache key

        Args:
            query: Query object

        Returns:
            Cache key
        """
        # Sort keys for consistent hashing
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

    def get(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached query results

        Args:
            query: Query object

        Returns:
            Cached results or None
        """
        key = self._query_to_key(query)
        return self.cache.get("queries", key)

    def set(self, query: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """
        Cache query results

        Args:
            query: Query object
            results: Query results

        Returns:
            True if successful
        """
        key = self._query_to_key(query)
        return self.cache.set("queries", key, results)
