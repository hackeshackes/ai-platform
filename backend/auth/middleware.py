"""
Authentication Middleware - 认证中间件

功能：
- 请求认证验证
- 登录尝试限制
- 密码强度验证
- 登录锁定保护
"""
import time
import secrets
from typing import Optional, Callable
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.config import settings
from ..core.exceptions import AuthenticationError, TokenExpiredError
from .jwt_handler import jwt_handler, TokenType


# HTTP Bearer认证方案
security = HTTPBearer(auto_error=False)


class LoginLockoutManager:
    """登录锁定管理器"""
    
    _lockout_records: dict = {}  # {email: {"count": int, "last_attempt": float, "locked_until": Optional[float]}}
    
    @classmethod
    def check_lockout(cls, email: str) -> tuple[bool, int]:
        """
        检查账户是否被锁定
        
        Args:
            email: 用户邮箱
            
        Returns:
            tuple: (是否锁定, 剩余解锁秒数)
        """
        now = time.time()
        record = cls._lockout_records.get(email)
        
        if not record:
            return False, 0
        
        # 检查是否已解锁
        if record.get("locked_until") and now > record["locked_until"]:
            # 清除过期锁定
            del cls._lockout_records[email]
            return False, 0
        
        if record.get("locked_until"):
            remaining = int(record["locked_until"] - now)
            return True, max(0, remaining)
        
        return False, 0
    
    @classmethod
    def record_failed_attempt(cls, email: str) -> None:
        """
        记录失败的登录尝试
        
        Args:
            email: 用户邮箱
        """
        now = time.time()
        record = cls._lockout_records.get(email, {"count": 0, "last_attempt": now})
        
        record["count"] += 1
        record["last_attempt"] = now
        
        # 检查是否需要锁定
        if record["count"] >= settings.LOGIN_MAX_ATTEMPTS:
            lockout_duration = settings.LOGIN_LOCKOUT_DURATION
            record["locked_until"] = now + lockout_duration
        
        cls._lockout_records[email] = record
    
    @classmethod
    def reset_attempts(cls, email: str) -> None:
        """重置登录尝试次数"""
        cls._lockout_records.pop(email, None)
    
    @classmethod
    def clear_expired_locks(cls) -> None:
        """清除过期的锁定记录"""
        now = time.time()
        expired = [
            email for email, record in cls._lockout_records.items()
            if record.get("locked_until") and now > record["locked_until"]
        ]
        for email in expired:
            del cls._lockout_records[email]


class PasswordValidator:
    """密码强度验证器"""
    
    @staticmethod
    def validate_strength(password: str) -> tuple[bool, list[str]]:
        """
        验证密码强度
        
        Args:
            password: 密码明文
            
        Returns:
            tuple: (是否通过, 错误信息列表)
        """
        errors = []
        
        # 检查最小长度
        if len(password) < settings.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
        
        # 检查最大长度
        if len(password) > settings.PASSWORD_MAX_LENGTH:
            errors.append(f"Password must not exceed {settings.PASSWORD_MAX_LENGTH} characters")
        
        # 检查是否包含数字
        if settings.PASSWORD_REQUIRE_NUMBER and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        # 检查是否包含小写字母
        if settings.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        # 检查是否包含大写字母
        if settings.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        # 检查是否包含特殊字符
        if settings.PASSWORD_REQUIRE_SPECIAL:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def calculate_strength_score(password: str) -> int:
        """
        计算密码强度分数 (0-100)
        
        Args:
            password: 密码明文
            
        Returns:
            int: 强度分数
        """
        score = 0
        
        # 长度贡献 (0-40分)
        length = len(password)
        if length >= 16:
            score += 40
        elif length >= 12:
            score += 30
        elif length >= 8:
            score += 20
        else:
            score += 10
        
        # 字符类型贡献 (0-40分)
        if any(c.islower() for c in password):
            score += 10
        if any(c.isupper() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(not c.isalnum() for c in password):
            score += 10
        
        # 多样性贡献 (0-20分)
        unique_chars = len(set(password))
        if unique_chars >= 12:
            score += 20
        elif unique_chars >= 8:
            score += 10
        
        return min(100, score)
    
    @staticmethod
    def get_strength_label(score: int) -> str:
        """
        获取强度标签
        
        Args:
            score: 强度分数
            
        Returns:
            str: 强度标签
        """
        if score >= 80:
            return "Very Strong"
        elif score >= 60:
            return "Strong"
        elif score >= 40:
            return "Medium"
        elif score >= 20:
            return "Weak"
        else:
            return "Very Weak"


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    # 不需要认证的路径
    EXCLUDED_PATHS = [
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/auth/oauth/{provider}",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/"
    ]
    
    async def dispatch(self, request: Request, call_next):
        # 检查路径是否需要认证
        if self._should_skip_auth(request.url.path):
            response = await call_next(request)
            return response
        
        # 获取Authorization头
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing authorization header"}
            )
        
        # 验证令牌
        try:
            # Bearer <token>
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise AuthenticationError("Invalid authorization header format")
            
            token = parts[1]
            payload = jwt_handler.verify_token(token, TokenType.ACCESS)
            
            # 将用户信息注入请求状态
            request.state.user_id = payload.sub
            request.state.user_email = payload.email
            request.state.user_role = payload.role
            
            response = await call_next(request)
            return response
            
        except TokenExpiredError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token has expired", "code": "TOKEN_EXPIRED"}
            )
        except AuthenticationError as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": str(e)}
            )
    
    def _should_skip_auth(self, path: str) -> bool:
        """检查是否应该跳过认证"""
        from fnmatch import fnmatch
        
        for excluded in self.EXCLUDED_PATHS:
            if fnmatch(path, excluded):
                return True
        return False


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """
    获取当前登录用户
    
    Args:
        request: FastAPI请求对象
        credentials: HTTP Bearer凭证
        
    Returns:
        dict: 用户信息
        
    Raises:
        HTTPException: 未认证
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials"
        )
    
    try:
        payload = jwt_handler.verify_token(credentials.credentials, TokenType.ACCESS)
        
        return {
            "user_id": payload.sub,
            "email": payload.email,
            "role": payload.role
        }
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    可选获取当前登录用户（未登录时不抛出异常）
    
    Args:
        request: FastAPI请求对象
        credentials: HTTP Bearer凭证
        
    Returns:
        Optional[dict]: 用户信息，如果未登录则返回None
    """
    if not credentials:
        return None
    
    try:
        payload = jwt_handler.verify_token(credentials.credentials, TokenType.ACCESS)
        
        return {
            "user_id": payload.sub,
            "email": payload.email,
            "role": payload.role
        }
    except AuthenticationError:
        return None


def require_role(required_role: str) -> Callable:
    """
    角色依赖检查装饰器工厂
    
    Args:
        required_role: 所需角色
        
    Returns:
        Callable: 依赖函数
    """
    async def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' is required"
            )
        return user
    
    return role_checker
