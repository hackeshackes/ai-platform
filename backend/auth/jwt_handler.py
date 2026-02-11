"""
JWT Token Handler - JWT令牌处理器

功能：
- 生成和验证JWT访问令牌
- 生成和验证刷新令牌
- 令牌黑名单管理
"""
import jwt
import uuid
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..core.exceptions import AuthenticationError, TokenExpiredError


class TokenType(Enum):
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class TokenPayload:
    """令牌载荷"""
    sub: str  # 用户ID
    type: TokenType
    exp: datetime
    iat: datetime
    jti: str  # 令牌ID
    email: Optional[str] = None
    role: Optional[str] = None


class JWTHandler:
    """JWT令牌处理器"""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建访问令牌
        
        Args:
            user_id: 用户ID
            email: 用户邮箱
            role: 用户角色
            expires_delta: 自定义过期时间
            
        Returns:
            JWT访问令牌字符串
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.access_token_expire_minutes
            )
        
        payload = {
            "sub": user_id,
            "type": TokenType.ACCESS.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4()),
            "email": email,
            "role": role
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        创建刷新令牌
        
        Args:
            user_id: 用户ID
            expires_delta: 自定义过期时间
            
        Returns:
            JWT刷新令牌字符串
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.refresh_token_expire_days
            )
        
        payload = {
            "sub": user_id,
            "type": TokenType.REFRESH.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(
        self,
        token: str,
        expected_type: TokenType = TokenType.ACCESS
    ) -> TokenPayload:
        """
        验证令牌
        
        Args:
            token: JWT令牌字符串
            expected_type: 期望的令牌类型
            
        Returns:
            TokenPayload: 令牌载荷
            
        Raises:
            TokenExpiredError: 令牌已过期
            AuthenticationError: 令牌验证失败
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # 检查令牌类型
            token_type = payload.get("type")
            if token_type != expected_type.value:
                raise AuthenticationError(f"Invalid token type: {token_type}")
            
            # 检查是否在黑名单中
            if self._is_token_blacklisted(payload.get("jti")):
                raise AuthenticationError("Token has been revoked")
            
            return TokenPayload(
                sub=payload["sub"],
                type=TokenType(token_type),
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                jti=payload["jti"],
                email=payload.get("email"),
                role=payload.get("role")
            )
            
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.PyJWTError as e:
            raise AuthenticationError(f"Token verification failed: {str(e)}")
    
    def _is_token_blacklisted(self, jti: str) -> bool:
        """
        检查令牌是否在黑名单中
        
        Args:
            jti: 令牌ID
            
        Returns:
            bool: 是否在黑名单中
        """
        # 实际实现应该连接Redis/数据库检查
        # 这里简化处理
        return False
    
    def revoke_token(self, jti: str) -> bool:
        """
        将令牌加入黑名单
        
        Args:
            jti: 令牌ID
            
        Returns:
            bool: 操作是否成功
        """
        # 实际实现应该将令牌ID存储到Redis/数据库
        # 并设置与过期时间相同的TTL
        return True
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        解码令牌（不验证签名）
        
        Args:
            token: JWT令牌字符串
            
        Returns:
            Dict[str, Any]: 令牌载荷
        """
        return jwt.decode(
            token,
            self.secret_key,
            algorithms=[self.algorithm],
            options={"verify_exp": False}
        )


# 全局JWT处理器实例
jwt_handler = JWTHandler()
