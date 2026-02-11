"""
Authentication Module - 认证模块

包含：
- JWT令牌处理
- OAuth 2.0支持
- 认证中间件
- 密码验证
"""
from .jwt_handler import jwt_handler, JWTHandler
from .oauth import OAuthHandler, OAuthProvider, OAuthUserInfo
from .middleware import (
    get_current_user,
    get_current_user_optional,
    require_role,
    LoginLockoutManager,
    PasswordValidator,
    AuthenticationMiddleware
)

__all__ = [
    "jwt_handler",
    "JWTHandler",
    "OAuthHandler",
    "OAuthProvider",
    "OAuthUserInfo",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "LoginLockoutManager",
    "PasswordValidator",
    "AuthenticationMiddleware"
]
