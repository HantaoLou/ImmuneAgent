"""
阿里云OSS存储服务
提供文件上传、下载、删除等操作
"""
import os
import logging
from typing import Optional, BinaryIO
from uuid import UUID
import oss2
from oss2.exceptions import OssError

logger = logging.getLogger(__name__)


class OSSService:
    """阿里云OSS服务类"""
    
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
        bucket_name: str,
        use_oss: bool = True,
    ):
        """
        初始化OSS服务
        
        Args:
            access_key_id: OSS AccessKey ID
            access_key_secret: OSS AccessKey Secret
            endpoint: OSS Endpoint (例如: https://oss-cn-hangzhou.aliyuncs.com)
            bucket_name: OSS Bucket名称
            use_oss: 是否使用OSS，如果为False则使用本地存储（用于兼容）
        """
        self.use_oss = use_oss
        self.bucket_name = bucket_name
        
        if use_oss:
            try:
                # 创建OSS认证对象
                auth = oss2.Auth(access_key_id, access_key_secret)
                # 创建Bucket对象
                self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
                # 测试连接
                self.bucket.get_bucket_info()
                logger.info(f"OSS服务初始化成功: bucket={bucket_name}, endpoint={endpoint}")
            except Exception as e:
                logger.error(f"OSS初始化失败: {e}")
                raise
        else:
            self.bucket = None
            logger.info("使用本地存储模式（OSS已禁用）")
    
    def get_object_key(self, session_id: UUID, file_name: str) -> str:
        """
        生成OSS对象键（Object Key）
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            
        Returns:
            OSS对象键，格式: artifacts/{session_id}/{file_name}
        """
        # 确保文件名安全（移除路径分隔符等）
        safe_filename = os.path.basename(file_name)
        return f"artifacts/{str(session_id)}/{safe_filename}"
    
    def upload_file(
        self,
        session_id: UUID,
        file_name: str,
        file_content: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """
        上传文件到OSS
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            file_content: 文件内容（字节）
            content_type: 文件MIME类型
            
        Returns:
            OSS对象键
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法上传文件")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            # 设置HTTP头
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            # 上传文件
            result = self.bucket.put_object(
                object_key,
                file_content,
                headers=headers
            )
            
            if result.status == 200:
                logger.info(f"文件上传成功: {object_key}")
                return object_key
            else:
                raise RuntimeError(f"文件上传失败，状态码: {result.status}")
                
        except OssError as e:
            logger.error(f"OSS上传错误: {e}")
            raise RuntimeError(f"文件上传失败: {str(e)}")
    
    def download_file(self, session_id: UUID, file_name: str) -> bytes:
        """
        从OSS下载文件
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            
        Returns:
            文件内容（字节）
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法下载文件")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            result = self.bucket.get_object(object_key)
            file_content = result.read()
            logger.info(f"文件下载成功: {object_key}")
            return file_content
        except oss2.exceptions.NoSuchKey:
            logger.error(f"文件不存在: {object_key}")
            raise FileNotFoundError(f"文件不存在: {object_key}")
        except OssError as e:
            logger.error(f"OSS下载错误: {e}")
            raise RuntimeError(f"文件下载失败: {str(e)}")
    
    def delete_file(self, session_id: UUID, file_name: str) -> bool:
        """
        从OSS删除文件
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            
        Returns:
            是否删除成功
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法删除文件")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            self.bucket.delete_object(object_key)
            logger.info(f"文件删除成功: {object_key}")
            return True
        except OssError as e:
            logger.error(f"OSS删除错误: {e}")
            return False
    
    def file_exists(self, session_id: UUID, file_name: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            
        Returns:
            文件是否存在
        """
        if not self.use_oss:
            return False
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            return self.bucket.object_exists(object_key)
        except OssError as e:
            logger.error(f"OSS检查文件存在性错误: {e}")
            return False
    
    def get_file_url(
        self,
        session_id: UUID,
        file_name: str,
        expires: int = 3600,
    ) -> str:
        """
        生成文件的临时访问URL（带签名）
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            expires: URL有效期（秒），默认1小时
            
        Returns:
            临时访问URL
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法生成URL")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            url = self.bucket.sign_url('GET', object_key, expires)
            return url
        except OssError as e:
            logger.error(f"OSS生成URL错误: {e}")
            raise RuntimeError(f"生成文件URL失败: {str(e)}")
    
    def get_download_url(
        self,
        session_id: UUID,
        file_name: str,
        expires: int = 8 * 3600,
    ) -> str:
        """
        生成文件的预签名下载URL
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            expires: URL有效期（秒），默认8小时
            
        Returns:
            预签名下载URL
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法生成下载URL")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            # 生成GET方法的预签名URL（用于下载）
            url = self.bucket.sign_url('GET', object_key, expires)
            logger.info(f"生成预签名下载URL成功: {object_key}, 有效期: {expires}秒")
            return url
        except OssError as e:
            logger.error(f"OSS生成下载URL错误: {e}")
            raise RuntimeError(f"生成下载URL失败: {str(e)}")
    
    def get_public_url(self, session_id: UUID, file_name: str) -> str:
        """
        获取文件的公共访问URL（如果Bucket是公共读）
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            
        Returns:
            公共访问URL
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法生成URL")
        
        object_key = self.get_object_key(session_id, file_name)
        # 构建公共URL（需要Bucket设置为公共读）
        endpoint = self.bucket.endpoint
        if endpoint.startswith('https://'):
            base_url = endpoint
        else:
            base_url = f"https://{endpoint}"
        
        return f"{base_url}/{self.bucket_name}/{object_key}"
    
    def get_upload_url(
        self,
        session_id: UUID,
        file_name: str,
        content_type: Optional[str] = None,
        expires: int = 3600,
    ) -> str:
        """
        生成文件的预签名上传URL
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            content_type: 文件MIME类型（可选）
            expires: URL有效期（秒），默认1小时
            
        Returns:
            预签名上传URL
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法生成上传URL")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            # 生成PUT方法的预签名URL
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            url = self.bucket.sign_url('PUT', object_key, expires, headers=headers)
            logger.info(f"生成上传URL成功: {object_key}")
            return url
        except OssError as e:
            logger.error(f"OSS生成上传URL错误: {e}")
            raise RuntimeError(f"生成上传URL失败: {str(e)}")
    
    def initiate_multipart_upload(
        self,
        session_id: UUID,
        file_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        初始化分片上传
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            content_type: 文件MIME类型（可选）
            
        Returns:
            上传ID (upload_id)
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法初始化分片上传")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            result = self.bucket.init_multipart_upload(object_key, headers=headers)
            upload_id = result.upload_id
            logger.info(f"初始化分片上传成功: {object_key}, upload_id={upload_id}")
            return upload_id
        except OssError as e:
            logger.error(f"OSS初始化分片上传错误: {e}")
            raise RuntimeError(f"初始化分片上传失败: {str(e)}")
    
    def upload_part(
        self,
        session_id: UUID,
        file_name: str,
        upload_id: str,
        part_number: int,
        part_data: bytes,
    ) -> str:
        """
        上传分片
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            upload_id: 上传ID
            part_number: 分片序号（从1开始）
            part_data: 分片数据
            
        Returns:
            分片的ETag
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法上传分片")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            result = self.bucket.upload_part(
                object_key,
                upload_id,
                part_number,
                part_data
            )
            etag = result.etag
            logger.info(f"分片上传成功: {object_key}, part_number={part_number}, etag={etag}")
            return etag
        except OssError as e:
            logger.error(f"OSS分片上传错误: {e}")
            raise RuntimeError(f"分片上传失败: {str(e)}")
    
    def complete_multipart_upload(
        self,
        session_id: UUID,
        file_name: str,
        upload_id: str,
        parts: list[tuple[int, str]],
    ) -> str:
        """
        完成分片上传
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            upload_id: 上传ID
            parts: 分片列表，格式为 [(part_number, etag), ...]
            
        Returns:
            OSS对象键
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法完成分片上传")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            from oss2.models import PartInfo
            part_list = [PartInfo(part_number, etag) for part_number, etag in parts]
            
            result = self.bucket.complete_multipart_upload(object_key, upload_id, part_list)
            logger.info(f"完成分片上传成功: {object_key}, upload_id={upload_id}")
            return object_key
        except OssError as e:
            logger.error(f"OSS完成分片上传错误: {e}")
            raise RuntimeError(f"完成分片上传失败: {str(e)}")
    
    def list_uploaded_parts(
        self,
        session_id: UUID,
        file_name: str,
        upload_id: str,
    ) -> list[dict]:
        """
        列出已上传的分片
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            upload_id: 上传ID
            
        Returns:
            已上传的分片列表，格式为 [{"part_number": 1, "etag": "...", "size": 1024}, ...]
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法列出分片")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            result = self.bucket.list_parts(object_key, upload_id)
            parts = []
            for part in result.parts:
                parts.append({
                    "part_number": part.part_number,
                    "etag": part.etag,
                    "size": part.size,
                })
            logger.info(f"列出分片成功: {object_key}, upload_id={upload_id}, 已上传分片数={len(parts)}")
            return parts
        except OssError as e:
            logger.error(f"OSS列出分片错误: {e}")
            raise RuntimeError(f"列出分片失败: {str(e)}")
    
    def abort_multipart_upload(
        self,
        session_id: UUID,
        file_name: str,
        upload_id: str,
    ) -> bool:
        """
        取消分片上传
        
        Args:
            session_id: 会话ID
            file_name: 文件名
            upload_id: 上传ID
            
        Returns:
            是否成功
        """
        if not self.use_oss:
            raise RuntimeError("OSS未启用，无法取消分片上传")
        
        object_key = self.get_object_key(session_id, file_name)
        
        try:
            self.bucket.abort_multipart_upload(object_key, upload_id)
            logger.info(f"取消分片上传成功: {object_key}, upload_id={upload_id}")
            return True
        except OssError as e:
            logger.error(f"OSS取消分片上传错误: {e}")
            return False

