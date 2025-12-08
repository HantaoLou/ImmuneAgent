from uuid import uuid4

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///C:/opt/antibody_gen/data.db"
    access_token: str = str(uuid4())
    frontend_path: str = "C:/opt/antibody_gen/dist"
    artifact_path: str = "C:/opt/antibody_gen/artifacts"
    port: int = 8000
    
    # OSS配置
    oss_enabled: bool = False  # 是否启用OSS（默认启用）
    oss_access_key_id: str = "your access key id"
    oss_access_key_secret: str = "your access key secret"
    oss_endpoint: str = "your oss endpoint"
    oss_bucket_name: str = "your oss bucket name"
    oss_use_public_url: bool = True  # 是否使用公共URL（需要Bucket设置为公共读）

    class Config:
        env_file = ".env"
        env_prefix = "ANTIBODY_GEN_"


settings = Settings()
