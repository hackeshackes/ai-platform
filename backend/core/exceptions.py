"""
Custom Exceptions - 自定义异常

认证和授权相关的异常类
"""
from fastapi import HTTPException, status
from typing import Optional


class BaseAPIException(HTTPException):
    """API异常基类"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status_code,
            detail=detail,
            headers=headers
        )


class AuthenticationError(BaseAPIException):
    """认证错误"""
    
    def __init__(
        self,
        detail: str = "Authentication failed",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers or {"WWW-Authenticate": "Bearer"}
        )


class TokenExpiredError(AuthenticationError):
    """令牌过期错误"""
    
    def __init__(self, detail: str = "Token has expired"):
        super().__init__(
            detail=detail,
            headers={"WWW-Authenticate": "Bearer", "X-Token-Expired": "true"}
        )


class TokenInvalidError(AuthenticationError):
    """令牌无效错误"""
    
    def __init__(self, detail: str = "Invalid token"):
        super().__init__(detail=detail)


class AuthorizationError(BaseAPIException):
    """授权错误"""
    
    def __init__(
        self,
        detail: str = "Permission denied",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers=headers
        )


class OAuthError(BaseAPIException):
    """OAuth错误"""
    
    def __init__(self, detail: str = "OAuth authentication failed"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )


class ValidationError(BaseAPIException):
    """验证错误"""
    
    def __init__(
        self,
        detail: str = "Validation failed",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            headers=headers
        )


class NotFoundError(BaseAPIException):
    """资源不存在错误"""
    
    def __init__(
        self,
        detail: str = "Resource not found",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            headers=headers
        )


class ConflictError(BaseAPIException):
    """资源冲突错误"""
    
    def __init__(
        self,
        detail: str = "Resource conflict",
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            headers=headers
        )


class RateLimitError(BaseAPIException):
    """速率限制错误"""
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: int = 60,
        headers: Optional[dict] = None
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers or {"Retry-After": str(retry_after)}
        )
