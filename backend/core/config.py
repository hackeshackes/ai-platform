"""
Application Configuration - 应用配置

集中管理所有配置项
"""
import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel


class JWTSettings(BaseModel):
    """JWT配置"""
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7


class PasswordSettings(BaseModel):
    """密码策略配置"""
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_MAX_LENGTH: int = 128
    PASSWORD_REQUIRE_NUMBER: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True


class LoginSecuritySettings(BaseModel):
    """登录安全配置"""
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_DURATION: int = 900  # 15分钟


class OAuthSettings(BaseModel):
    """OAuth配置"""
    OAUTH_REDIRECT_URI: str = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/api/v1/auth/oauth/callback")
    
    # GitHub OAuth
    GITHUB_OAUTH_CLIENT_ID: Optional[str] = os.getenv("GITHUB_OAUTH_CLIENT_ID")
    GITHUB_OAUTH_CLIENT_SECRET: Optional[str] = os.getenv("GITHUB_OAUTH_CLIENT_SECRET")
    
    # Google OAuth
    GOOGLE_OAUTH_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    
    # Microsoft OAuth
    MICROSOFT_OAUTH_CLIENT_ID: Optional[str] = os.getenv("MICROSOFT_OAUTH_CLIENT_ID")
    MICROSOFT_OAUTH_CLIENT_SECRET: Optional[str] = os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET")
    MICROSOFT_OAUTH_TENANT_ID: Optional[str] = os.getenv("MICROSOFT_OAUTH_TENANT_ID")


class DatabaseSettings(BaseModel):
    """数据库配置"""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "ai_platform")


class RedisSettings(BaseModel):
    """Redis配置"""
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")


class Settings(BaseModel):
    """全局设置"""
    JWT: JWTSettings = JWTSettings()
    PASSWORD: PasswordSettings = PasswordSettings()
    LOGIN_SECURITY: LoginSecuritySettings = LoginSecuritySettings()
    OAUTH: OAuthSettings = OAuthSettings()
    DATABASE: DatabaseSettings = DatabaseSettings()
    REDIS: RedisSettings = RedisSettings()
    
    # 应用基本信息
    APP_NAME: str = "AI Platform"
    APP_VERSION: str = "5.0.0"
    API_PREFIX: str = "/api/v1"
    
    # CORS配置
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True


@lru_cache()
def get_settings() -> Settings:
    """获取全局设置实例"""
    return Settings()


# 创建全局设置实例
settings = get_settings()
