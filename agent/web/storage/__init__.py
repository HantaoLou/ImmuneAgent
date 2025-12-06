"""
存储服务模块
"""
from web.storage.oss_service import OSSService
from web.storage.storage_factory import get_storage_service

__all__ = ["OSSService", "get_storage_service"]

