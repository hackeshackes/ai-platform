"""
SSO API - SSO认证接口

端点：
- POST /api/v1/auth/sso/okta       # Okta SSO登录
- POST /api/v1/auth/sso/azure     # Azure AD SSO登录
- GET /api/v1/auth/sso/url        # 获取SSO授权URL
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from backend.auth.sso import sso_handler, SSOProvider
from backend.auth.middleware import get_current_user


# 路由器
router = APIRouter()


# ============ Pydantic Models ============

class SSOProviderRequest(BaseModel):
    """SSO提供商请求"""
    code: str = Field(..., description="授权码")
    state: Optional[str] = Field(None, description="状态参数")
    redirect_uri: Optional[str] = Field(None, description="回调URL")


class SSOLoginResponse(BaseModel):
    """SSO登录响应"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: dict


class SSOUrlResponse(BaseModel):
    """SSO授权URL响应"""
    auth_url: str
    provider: str
    state: str


class SSOErrorResponse(BaseModel):
    """SSO错误响应"""
    detail: str
    provider: Optional[str] = None
    error_code: Optional[str] = None


# ============ SSO端点 ============

@router.post(
    "/sso/okta",
    response_model=SSOLoginResponse,
    responses={
        200: {"description": "Okta SSO登录成功"},
        400: {"model": SSOErrorResponse, "description": "登录失败"},
        403: {"model": SSOErrorResponse, "description": "访问被拒绝"}
    },
    summary="Okta SSO登录",
    description="使用Okta SSO进行身份认证"
)
async def okta_sso_login(
    request: SSOProviderRequest,
    tenant_id: Optional[str] = Query(None, description="租户ID（可选）")
):
    """
    Okta SSO登录
    
    使用Okta账户进行单点登录。
    
    - **code**: Okta授权回调返回的授权码
    - **state**: 状态参数，用于防止CSRF攻击
    - **tenant_id**: 指定的租户ID（可选，如果不指定将创建新租户）
    """
    try:
        result = await sso_handler.handle_callback(
            provider=SSOProvider.OKTA,
            code=request.code,
            tenant_id=tenant_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Okta SSO login failed: {str(e)}"
        )


@router.post(
    "/sso/azure",
    response_model=SSOLoginResponse,
    responses={
        200: {"description": "Azure AD SSO登录成功"},
        400: {"model": SSOErrorResponse, "description": "登录失败"},
        403: {"model": SSOErrorResponse, "description": "访问被拒绝"}
    },
    summary="Azure AD SSO登录",
    description="使用Azure Active Directory SSO进行身份认证"
)
async def azure_sso_login(
    request: SSOProviderRequest,
    tenant_id: Optional[str] = Query(None, description="租户ID（可选）")
):
    """
    Azure AD SSO登录
    
    使用Azure Active Directory账户进行单点登录。
    
    - **code**: Azure AD授权回调返回的授权码
    - **state**: 状态参数，用于防止CSRF攻击
    - **tenant_id**: 指定的租户ID（可选，如果不指定将创建新租户）
    """
    try:
        result = await sso_handler.handle_callback(
            provider=SSOProvider.AZURE_AD,
            code=request.code,
            tenant_id=tenant_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Azure AD SSO login failed: {str(e)}"
        )


@router.get(
    "/sso/url",
    response_model=SSOUrlResponse,
    responses={
        200: {"description": "获取SSO授权URL成功"},
        400: {"model": SSOErrorResponse, "description": "无效的提供商"}
    },
    summary="获取SSO授权URL",
    description="获取指定SSO提供商的授权URL"
)
async def get_sso_url(
    provider: SSOProvider = Query(..., description="SSO提供商 (okta 或 azure)")
):
    """
    获取SSO授权URL
    
    返回指定SSO提供商的授权URL，用户将被重定向到该URL进行认证。
    
    - **provider**: SSO提供商类型
    """
    import secrets
    
    state = secrets.token_urlsafe(32)
    auth_url = sso_handler.get_auth_url(provider, state)
    
    return {
        "auth_url": auth_url,
        "provider": provider.value,
        "state": state
    }


@router.get(
    "/sso/providers",
    summary="获取支持的SSO提供商",
    description="返回所有支持的SSO提供商列表"
)
async def list_sso_providers():
    """获取支持的SSO提供商列表"""
    return {
        "providers": [
            {
                "id": "okta",
                "name": "Okta",
                "description": "Okta Identity Management",
                "enabled": True
            },
            {
                "id": "azure",
                "name": "Azure Active Directory",
                "description": "Microsoft Azure AD",
                "enabled": True
            }
        ]
    }


@router.post(
    "/sso/link",
    response_model=dict,
    responses={
        200: {"description": "SSO账户关联成功"},
        400: {"model": SSOErrorResponse, "description": "关联失败"}
    },
    summary="关联SSO账户",
    description="将SSO账户与现有账户关联"
)
async def link_sso_account(
    request: SSOProviderRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    关联SSO账户
    
    将当前登录用户的账户与SSO账户关联。
    
    - **code**: SSO授权回调返回的授权码
    """
    # TODO: 实现SSO账户关联逻辑
    return {"message": "SSO account linking not yet implemented"}


@router.delete(
    "/sso/unlink/{provider}",
    response_model=dict,
    responses={
        200: {"description": "SSO账户取消关联成功"},
        400: {"model": SSOErrorResponse, "description": "取消关联失败"}
    },
    summary="取消关联SSO账户",
    description="取消当前用户与指定SSO提供商的关联"
)
async def unlink_sso_account(
    provider: SSOProvider,
    current_user: dict = Depends(get_current_user)
):
    """
    取消关联SSO账户
    
    取消当前用户与指定SSO提供商的账户关联。
    注意：如果用户没有设置密码，取消关联后可能无法登录。
    """
    # TODO: 实现SSO账户取消关联逻辑
    return {"message": f"SSO account unlinking for {provider.value} not yet implemented"}
