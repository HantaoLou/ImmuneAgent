"""
存储服务工厂
根据配置创建OSS或本地存储服务
"""
import logging
from web.settings import settings
from web.storage.oss_service import OSSService

logger = logging.getLogger(__name__)

_storage_service: OSSService = None


def get_storage_service() -> OSSService:
    """
    获取存储服务实例（单例模式）
    
    Returns:
        OSSService实例
    """
    global _storage_service
    
    if _storage_service is None:
        if settings.oss_enabled:
            if not all([
                settings.oss_access_key_id,
                settings.oss_access_key_secret,
                settings.oss_endpoint,
                settings.oss_bucket_name,
            ]):
                logger.warning("OSS配置不完整，将使用本地存储模式（请在.env中配置OSS相关参数）")
                _storage_service = OSSService(
                    access_key_id="",
                    access_key_secret="",
                    endpoint="",
                    bucket_name="",
                    use_oss=False,
                )
            else:
                _storage_service = OSSService(
                    access_key_id=settings.oss_access_key_id,
                    access_key_secret=settings.oss_access_key_secret,
                    endpoint=settings.oss_endpoint,
                    bucket_name=settings.oss_bucket_name,
                    use_oss=True,
                )
                logger.info("OSS已启用并初始化成功")
        else:
            logger.info("OSS已禁用，使用本地存储模式")
            _storage_service = OSSService(
                access_key_id="",
                access_key_secret="",
                endpoint="",
                bucket_name="",
                use_oss=False,
            )
    
    return _storage_service

