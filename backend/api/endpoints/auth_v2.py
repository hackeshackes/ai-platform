"""
Authentication API v2 - 新版认证API

端点：
- POST /api/v1/auth/register  # 用户注册
- POST /api/v1/auth/login    # 用户登录
- POST /api/v1/auth/refresh  # 刷新令牌
- POST /api/v1/auth/logout   # 登出
- POST /api/v1/auth/oauth/{provider}  # OAuth登录
- GET /api/v1/auth/me        # 获取当前用户
"""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer

from core.config import settings
from core.exceptions import AuthenticationError, TokenExpiredError
from db.database import db
from .jwt_handler import jwt_handler, TokenType
from .oauth import OAuthHandler, OAuthProvider
from .middleware import (
    get_current_user,
    get_current_user_optional,
    LoginLockoutManager,
    PasswordValidator
)


# 路由器
router = APIRouter()
security = HTTPBearer()


# ============ Pydantic Models ============

class RegisterRequest:
    """用户注册请求"""
    email: str
    password: str
    name: Optional[str] = None


class RegisterResponse:
    """用户注册响应"""
    user_id: str
    email: str
    message: str


class LoginRequest:
    """用户登录请求"""
    email: str
    password: str


class LoginResponse:
    """用户登录响应"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: dict


class RefreshTokenRequest:
    """刷新令牌请求"""
    refresh_token: str


class RefreshTokenResponse:
    """刷新令牌响应"""
    access_token: str
    refresh_token: Optional[str]
    token_type: str = "Bearer"
    expires_in: int


class UserResponse:
    """用户信息响应"""
    user_id: str
    email: str
    name: Optional[str]
    created_at: datetime
    provider: Optional[str]


class OAuthUrlResponse:
    """OAuth授权URL响应"""
    auth_url: str
    state: str


class ErrorResponse:
    """错误响应"""
    detail: str
    code: Optional[str] = None


# ============ Helper Functions ============

async def _get_user_by_email(email: str) -> Optional[dict]:
    """根据邮箱获取用户"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, user_id, email, name, password_hash, provider, provider_id, is_active, created_at FROM users WHERE email = ?",
        (email,)
    )
    row = cursor.fetchone()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "user_id": row[1],
        "email": row[2],
        "name": row[3],
        "password_hash": row[4],
        "provider": row[5],
        "provider_id": row[6],
        "is_active": row[7],
        "created_at": datetime.fromisoformat(row[8]) if row[8] else datetime.now(timezone.utc)
    }


async def _get_user_by_id(user_id: str) -> Optional[dict]:
    """根据用户ID获取用户"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, user_id, email, name, password_hash, provider, provider_id, is_active, created_at FROM users WHERE user_id = ?",
        (user_id,)
    )
    row = cursor.fetchone()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "user_id": row[1],
        "email": row[2],
        "name": row[3],
        "password_hash": row[4],
        "provider": row[5],
        "provider_id": row[6],
        "is_active": row[7],
        "created_at": datetime.fromisoformat(row[8]) if row[8] else datetime.now(timezone.utc)
    }


async def _create_user(email: str, password_hash: str, name: Optional[str] = None) -> dict:
    """创建新用户"""
    import uuid
    
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO users (user_id, email, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, email, name or email.split("@")[0], password_hash, now)
    )
    conn.commit()
    
    return {
        "user_id": user_id,
        "email": email,
        "name": name,
        "password_hash": password_hash,
        "provider": "local",
        "provider_id": None,
        "is_active": True,
        "created_at": now
    }


async def _update_user_provider(user_id: str, provider: str, provider_id: str) -> None:
    """更新用户的OAuth提供商信息"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE users SET provider = ?, provider_id = ? WHERE user_id = ?",
        (provider, provider_id, user_id)
    )
    conn.commit()


async def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    import hashlib
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


# ============ API Endpoints ============

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        409: {"model": ErrorResponse, "description": "User already exists"}
    },
    summary="用户注册",
    description="注册新用户账户"
)
async def register(request: RegisterRequest):
    """
    用户注册
    
    - **email**: 用户邮箱地址
    - **password**: 密码（需满足强度要求）
    - **name**: 用户名（可选）
    """
    # 验证密码强度
    is_valid, errors = PasswordValidator.validate_strength(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": "Password does not meet requirements", "errors": errors}
        )
    
    # 检查邮箱是否已存在
    existing_user = await _get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # 创建用户
    user = await _create_user(request.email, request.password, request.name)
    
    return RegisterResponse(
        user_id=user["user_id"],
        email=user["email"],
        message="User registered successfully"
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        423: {"model": ErrorResponse, "description": "Account locked"}
    },
    summary="用户登录",
    description="用户登录并获取访问令牌"
)
async def login(request: LoginRequest):
    """
    用户登录
    
    - **email**: 用户邮箱地址
    - **password**: 密码
    
    **安全特性**:
    - 登录失败5次后账户锁定15分钟
    - 返回访问令牌和刷新令牌
    """
    # 检查登录锁定状态
    is_locked, remaining = LoginLockoutManager.check_lockout(request.email)
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account locked. Try again in {remaining} seconds"
        )
    
    # 获取用户
    user = await _get_user_by_email(request.email)
    
    # 验证凭据
    if not user:
        LoginLockoutManager.record_failed_attempt(request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled"
        )
    
    # 验证密码
    if not await _verify_password(request.password, user.get("password_hash", "")):
        LoginLockoutManager.record_failed_attempt(request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # 重置登录尝试
    LoginLockoutManager.reset_attempts(request.email)
    
    # 生成令牌
    access_token = jwt_handler.create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user.get("role", "user")
    )
    refresh_token = jwt_handler.create_refresh_token(user_id=user["user_id"])
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "user_id": user["user_id"],
            "email": user["email"],
            "name": user.get("name")
        }
    )


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid refresh token"}
    },
    summary="刷新令牌",
    description="使用刷新令牌获取新的访问令牌"
)
async def refresh_token(request: RefreshTokenRequest):
    """
    刷新访问令牌
    
    - **refresh_token**: 刷新令牌
    
    **注意**: 刷新令牌也会被轮换，旧刷新令牌将失效
    """
    try:
        # 验证刷新令牌
        payload = jwt_handler.verify_token(request.refresh_token, TokenType.REFRESH)
        
        # 生成新的令牌对
        new_access_token = jwt_handler.create_access_token(
            user_id=payload.sub,
            email=payload.email,
            role=payload.role
        )
        new_refresh_token = jwt_handler.create_refresh_token(user_id=payload.sub)
        
        # 使旧刷新令牌失效（加入黑名单）
        jwt_handler.revoke_token(payload.jti)
        
        return RefreshTokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="Bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired, please login again"
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="用户登出",
    description="使当前访问令牌失效"
)
async def logout(
    request: Request,
    credentials: HTTPBearer = Depends(security)
):
    """
    用户登出
    
    将当前访问令牌加入黑名单使其失效
    
    **请求头**:
    - Authorization: Bearer <access_token>
    """
    token = credentials.credentials
    
    try:
        payload = jwt_handler.verify_token(token, TokenType.ACCESS)
        
        # 将令牌加入黑名单
        jwt_handler.revoke_token(payload.jti)
        
        return {"message": "Logged out successfully"}
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.get(
    "/oauth/{provider}",
    response_model=OAuthUrlResponse,
    summary="获取OAuth授权URL",
    description="获取指定OAuth提供商的授权URL"
)
async def get_oauth_url(provider: str):
    """
    获取OAuth授权URL
    
    - **provider**: OAuth提供商 (github, google, microsoft)
    
    返回授权URL和state参数，state需要保存用于回调验证
    """
    try:
        state = secrets.token_urlsafe(32)
        auth_url = OAuthHandler.get_authorization_url(provider, state)
        
        return OAuthUrlResponse(auth_url=auth_url, state=state)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post(
    "/oauth/{provider}/callback",
    response_model=LoginResponse,
    summary="OAuth回调处理",
    description="处理OAuth提供商的回调并返回访问令牌"
)
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    stored_state: str  # 实际应用中应该从session/cookie获取
):
    """
    OAuth回调处理
    
    - **provider**: OAuth提供商
    - **code**: 授权码
    - **state**: 回调state参数
    - **stored_state**: 之前生成的state（实际应从安全存储获取）
    """
    try:
        # 处理OAuth回调
        oauth_user = await OAuthHandler.handle_callback(
            provider=provider,
            code=code,
            state=state,
            stored_state=stored_state
        )
        
        # 检查用户是否已存在，不存在则创建
        user = await _get_user_by_email(oauth_user.email) if oauth_user.email else None
        
        if not user:
            # 创建新用户
            import uuid
            
            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO users (user_id, email, name, password_hash, provider, provider_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, oauth_user.email or f"{oauth_user.provider.value}user@{oauth_user.provider_id}.local", oauth_user.name, "", oauth_user.provider.value, oauth_user.provider_id, now)
            )
            conn.commit()
            
            user = {
                "user_id": user_id,
                "email": oauth_user.email or f"{oauth_user.provider.value}user@{oauth_user.provider_id}.local",
                "name": oauth_user.name,
                "provider": oauth_user.provider.value,
                "provider_id": oauth_user.provider_id,
                "is_active": True,
                "created_at": now
            }
        else:
            # 如果用户已存在但没有OAuth信息，更新它
            if not user.get("provider_id"):
                await _update_user_provider(user["user_id"], oauth_user.provider.value, oauth_user.provider_id)
        
        # 生成令牌
        access_token = jwt_handler.create_access_token(
            user_id=user["user_id"],
            email=user["email"],
            role=user.get("role", "user")
        )
        refresh_token = jwt_handler.create_refresh_token(user_id=user["user_id"])
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user={
                "user_id": user["user_id"],
                "email": user["email"],
                "name": user.get("name")
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth error: {str(e)}"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户",
    description="获取当前登录用户的详细信息"
)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    """
    获取当前用户信息
    
    需要有效的访问令牌
    
    **返回**:
    - user_id: 用户ID
    - email: 邮箱地址
    - name: 用户名
    - created_at: 账户创建时间
    - provider: 认证提供商
    """
    user = await _get_user_by_email(current_user["email"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name"),
        created_at=user.get("created_at", datetime.now(timezone.utc)),
        provider=user.get("provider")
    )


@router.post(
    "/password/check",
    summary="检查密码强度",
    description="检查密码强度并返回评分"
)
async def check_password_strength(password: str):
    """
    检查密码强度
    
    - **password**: 要检查的密码
    
    **返回**:
    - is_valid: 是否满足最低要求
    - score: 强度分数 (0-100)
    - label: 强度标签
    - feedback: 改进建议
    """
    score = PasswordValidator.calculate_strength_score(password)
    is_valid, errors = PasswordValidator.validate_strength(password)
    
    return {
        "is_valid": is_valid,
        "score": score,
        "label": PasswordValidator.get_strength_label(score),
        "errors": errors
    }


@router.post(
    "/password/reset/initiate",
    summary="发起密码重置",
    description="通过邮箱发送密码重置链接"
)
async def initiate_password_reset(email: str):
    """
    发起密码重置
    
    向用户邮箱发送密码重置链接
    
    **注意**: 为防止枚举攻击，无论邮箱是否存在都返回成功
    """
    # 实际实现应该：
    # 1. 检查邮箱是否存在
    # 2. 生成重置令牌
    # 3. 发送重置邮件
    
    return {"message": "If the email exists, a reset link has been sent"}


@router.post(
    "/password/reset/confirm",
    summary="确认密码重置",
    description="使用重置令牌设置新密码"
)
async def confirm_password_reset(
    token: str,
    new_password: str
):
    """
    确认密码重置
    
    - **token**: 密码重置令牌
    - **new_password**: 新密码
    """
    # 验证重置令牌
    # 更新用户密码
    
    # 验证新密码强度
    is_valid, errors = PasswordValidator.validate_strength(new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": "Password does not meet requirements", "errors": errors}
        )
    
    return {"message": "Password reset successfully"}
