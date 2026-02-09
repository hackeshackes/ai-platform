"""
数据库和缓存配置 v2.0
"""
import os
from pydantic import BaseModel
from typing import Optional, List

# 应用设置
class Settings:
    """应用设置"""
    CORS_ORIGINS: List[str] = ["*"]
    SITE_NAME: str = "AI Platform"
    VERSION: str = "2.3.0-beta"

settings = Settings()

class DatabaseConfig(BaseModel):
    """数据库配置"""
    driver: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "ai_platform"
    user: str = "aiplatform"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def url(self) -> str:
        return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def async_url(self) -> str:
        return f"{self.driver}+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class RedisConfig(BaseModel):
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class CeleryConfig(BaseModel):
    """Celery配置"""
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"
    timezone: str = "Asia/Shanghai"

def load_config():
    """从环境变量加载配置"""
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        name=os.getenv("DB_NAME", "ai_platform"),
        user=os.getenv("DB_USER", "aiplatform"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    
    redis_config = RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", None),
    )
    
    celery_config = CeleryConfig(
        broker_url=os.getenv("CELERY_BROKER", "redis://localhost:6379/1"),
        result_backend=os.getenv("CELERY_RESULT", "redis://localhost:6379/2"),
    )
    
    return db_config, redis_config, celery_config
