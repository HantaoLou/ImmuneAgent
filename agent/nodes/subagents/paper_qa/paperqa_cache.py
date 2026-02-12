"""Caching layer for paper-qa Docs instances.

Avoids re-indexing papers when the same set of papers is queried
repeatedly (e.g., during benchmark runs with similar questions).
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "immuneagent" / "paperqa_docs"
CACHE_TTL_SECONDS = 3600


class DocsCache:
    """Two-tier cache: in-memory + pickle-on-disk."""

    def __init__(
        self, cache_dir: Path = DEFAULT_CACHE_DIR, ttl: int = CACHE_TTL_SECONDS
    ):
        self._memory: dict[str, tuple[object, float]] = {}
        self._cache_dir = cache_dir
        self._ttl = ttl
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, paper_ids: list[str]) -> str:
        return hashlib.md5(
            "|".join(sorted(set(paper_ids))).encode()
        ).hexdigest()

    def get(self, paper_ids: list[str]) -> Optional[object]:
        key = self._make_key(paper_ids)
        if key in self._memory:
            docs, ts = self._memory[key]
            if time.time() - ts < self._ttl:
                return docs
            del self._memory[key]

        disk_path = self._cache_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                if time.time() - disk_path.stat().st_mtime < self._ttl:
                    with open(disk_path, "rb") as f:
                        docs = pickle.load(f)
                    self._memory[key] = (docs, time.time())
                    return docs
                disk_path.unlink(missing_ok=True)
            except Exception:
                disk_path.unlink(missing_ok=True)
        return None

    def put(self, paper_ids: list[str], docs: object) -> None:
        key = self._make_key(paper_ids)
        self._memory[key] = (docs, time.time())
        try:
            with open(self._cache_dir / f"{key}.pkl", "wb") as f:
                pickle.dump(docs, f)
        except Exception as e:
            logger.warning(f"Failed to cache Docs to disk: {e}")
