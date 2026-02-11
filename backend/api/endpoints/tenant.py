"""
Tenant API - 租户管理接口

端点：
- GET /api/v1/tenant/me              # 获取当前租户
- GET /api/v1/tenant/users           # 租户用户列表
- POST /api/v1/tenant/quota          # 设置配额
- PUT /api/v1/tenant/settings        # 更新租户设置
- POST /api/v1/tenant/members/{user_id}/role  # 更新成员角色
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field

from backend.tenant.manager import tenant_manager
from backend.tenant.middleware import (
    get_current_tenant,
    get_tenant_context,
    require_tenant_admin,
    TenantContext
)
from backend.auth.middleware import get_current_user


# 路由器
router = APIRouter()


# ============ Pydantic Models ============

class TenantResponse(BaseModel):
    """租户信息响应"""
    tenant_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    quota: dict
    usage: dict
    settings: dict


class TenantUserResponse(BaseModel):
    """租户用户信息响应"""
    user_id: str
    email: str
    name: Optional[str]
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime]


class TenantQuotaRequest(BaseModel):
    """配额设置请求"""
    max_users: Optional[int] = Field(None, ge=1, le=1000, description="最大用户数")
    max_projects: Optional[int] = Field(None, ge=1, le=100, description="最大项目数")
    max_storage_gb: Optional[int] = Field(None, ge=1, le=10000, description="最大存储空间(GB)")
    api_calls_per_month: Optional[int] = Field(None, ge=100, le=1000000, description="每月API调用次数")


class TenantQuotaResponse(BaseModel):
    """配额响应"""
    max_users: int
    max_projects: int
    max_storage_gb: int
    api_calls_per_month: int


class TenantSettingsRequest(BaseModel):
    """租户设置请求"""
    allow_custom_domains: Optional[bool] = None
    default_role: Optional[str] = Field(None, pattern="^(user|viewer)$")
    sso_provider: Optional[str] = None


class TenantSettingsResponse(BaseModel):
    """租户设置响应"""
    settings: dict


class UpdateMemberRoleRequest(BaseModel):
    """更新成员角色请求"""
    role: str = Field(..., pattern="^(admin|user|viewer)$")


class MemberResponse(BaseModel):
    """成员操作响应"""
    user_id: str
    message: str


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str
    code: Optional[str] = None


# ============ 租户端点 ============

@router.get(
    "/tenant/me",
    response_model=TenantResponse,
    responses={
        200: {"description": "获取当前租户成功"},
        400: {"model": ErrorResponse, "description": "用户不属于任何租户"},
        404: {"model": ErrorResponse, "description": "租户不存在"}
    },
    summary="获取当前租户",
    description="获取当前用户所属租户的信息"
)
async def get_current_tenant_info(
    tenant: dict = Depends(get_current_tenant)
):
    """
    获取当前租户信息
    
    返回当前用户所属租户的详细信息，包括配额和使用情况。
    """
    return {
        "tenant_id": str(tenant["_id"]),
        "name": tenant.get("name"),
        "created_at": tenant.get("created_at"),
        "updated_at": tenant.get("updated_at"),
        "is_active": tenant.get("is_active", True),
        "quota": tenant.get("quota", {}),
        "usage": tenant.get("usage", {}),
        "settings": tenant.get("settings", {})
    }


@router.get(
    "/tenant/users",
    response_model=List[TenantUserResponse],
    responses={
        200: {"description": "获取租户用户列表成功"},
        403: {"model": ErrorResponse, "description": "权限不足"}
    },
    summary="获取租户用户列表",
    description="获取当前租户的所有成员列表"
)
async def list_tenant_users(
    tenant: dict = Depends(get_current_tenant),
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(50, ge=1, le=100, description="返回数量")
):
    """
    获取租户用户列表
    
    返回当前租户的所有成员信息。
    """
    users = await tenant_manager.get_members(
        tenant_id=str(tenant["_id"]),
        skip=skip,
        limit=limit
    )
    
    return [
        {
            "user_id": str(user["_id"]),
            "email": user.get("email"),
            "name": user.get("name"),
            "roles": user.get("roles", []),
            "is_active": user.get("is_active", True),
            "created_at": user.get("created_at"),
            "last_login_at": user.get("last_login_at")
        }
        for user in users
    ]


@router.post(
    "/tenant/quota",
    response_model=TenantQuotaResponse,
    responses={
        200: {"description": "配额设置成功"},
        400: {"model": ErrorResponse, "description": "无效的配额值"},
        403: {"model": ErrorResponse, "description": "权限不足"}
    },
    summary="设置租户配额",
    description="设置当前租户的资源配额（仅管理员可操作）"
)
async def set_tenant_quota(
    quota_request: TenantQuotaRequest,
    context: TenantContext = Depends(require_tenant_admin())
):
    """
    设置租户配额
    
    配置租户可用的资源配额限制。
    
    - **max_users**: 最大用户数量
    - **max_projects**: 最大项目数量
    - **max_storage_gb**: 最大存储空间(GB)
    - **api_calls_per_month**: 每月API调用次数
    """
    quota = await tenant_manager.set_quota(
        tenant_id=context.tenant_id,
        max_users=quota_request.max_users,
        max_projects=quota_request.max_projects,
        max_storage_gb=quota_request.max_storage_gb,
        api_calls_per_month=quota_request.api_calls_per_month
    )
    
    return {
        "max_users": quota.get("max_users", 0),
        "max_projects": quota.get("max_projects", 0),
        "max_storage_gb": quota.get("max_storage_gb", 0),
        "api_calls_per_month": quota.get("api_calls_per_month", 0)
    }


@router.get(
    "/tenant/quota",
    response_model=TenantQuotaResponse,
    responses={
        200: {"description": "获取配额成功"},
        404: {"model": ErrorResponse, "description": "租户不存在"}
    },
    summary="获取租户配额",
    description="获取当前租户的资源配额"
)
async def get_tenant_quota(
    tenant: dict = Depends(get_current_tenant)
):
    """
    获取租户配额
    
    返回当前租户的资源配额限制和使用情况。
    """
    quota = tenant.get("quota", {})
    usage = tenant.get("usage", {})
    
    return {
        "max_users": quota.get("max_users", 0),
        "max_projects": quota.get("max_projects", 0),
        "max_storage_gb": quota.get("max_storage_gb", 0),
        "api_calls_per_month": quota.get("api_calls_per_month", 0),
        "current_users": usage.get("users_count", 0),
        "current_projects": usage.get("projects_count", 0),
        "current_storage_gb": usage.get("storage_used_gb", 0),
        "current_api_calls": usage.get("api_calls_this_month", 0)
    }


@router.put(
    "/tenant/settings",
    response_model=TenantSettingsResponse,
    responses={
        200: {"description": "设置更新成功"},
        403: {"model": ErrorResponse, "description": "权限不足"}
    },
    summary="更新租户设置",
    description="更新当前租户的设置（仅管理员可操作）"
)
async def update_tenant_settings(
    settings_request: TenantSettingsRequest,
    context: TenantContext = Depends(require_tenant_admin())
):
    """
    更新租户设置
    
    配置租户的行为设置。
    """
    updates = {}
    if settings_request.allow_custom_domains is not None:
        updates["settings.allow_custom_domains"] = settings_request.allow_custom_domains
    if settings_request.default_role:
        updates["settings.default_role"] = settings_request.default_role
    if settings_request.sso_provider:
        updates["settings.sso_provider"] = settings_request.sso_provider
    
    if updates:
        await tenant_manager.update_tenant(context.tenant_id, updates)
    
    # 获取更新后的设置
    tenant = await tenant_manager.get_tenant_by_id(context.tenant_id)
    
    return {
        "settings": tenant.get("settings", {})
    }


@router.get(
    "/tenant/settings",
    response_model=TenantSettingsResponse,
    summary="获取租户设置",
    description="获取当前租户的设置"
)
async def get_tenant_settings(
    tenant: dict = Depends(get_current_tenant)
):
    """
    获取租户设置
    
    返回当前租户的行为设置。
    """
    return {
        "settings": tenant.get("settings", {})
    }


@router.post(
    "/tenant/members/{user_id}/role",
    response_model=MemberResponse,
    responses={
        200: {"description": "角色更新成功"},
        400: {"model": ErrorResponse, "description": "无效的用户ID或角色"},
        403: {"model": ErrorResponse, "description": "权限不足"},
        404: {"model": ErrorResponse, "description": "用户不存在"}
    },
    summary="更新成员角色",
    description="更新租户成员的角色（仅管理员可操作）"
)
async def update_member_role(
    user_id: str,
    role_request: UpdateMemberRoleRequest,
    context: TenantContext = Depends(require_tenant_admin())
):
    """
    更新成员角色
    
    修改租户中指定成员的角色。
    """
    success = await tenant_manager.update_member_role(
        tenant_id=context.tenant_id,
        user_id=user_id,
        role=role_request.role
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not a member of this tenant"
        )
    
    return {
        "user_id": user_id,
        "message": f"Role updated to {role_request.role}"
    }


@router.delete(
    "/tenant/members/{user_id}",
    response_model=MemberResponse,
    responses={
        200: {"description": "成员移除成功"},
        400: {"model": ErrorResponse, "description": "无法移除自己"},
        403: {"model": ErrorResponse, "description": "权限不足"}
    },
    summary="移除租户成员",
    description="从当前租户中移除指定成员（仅管理员可操作）"
)
async def remove_tenant_member(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    context: TenantContext = Depends(require_tenant_admin())
):
    """
    移除租户成员
    
    将指定用户从当前租户中移除。
    """
    # 防止移除自己
    if str(current_user.get("_id")) == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself from the tenant"
        )
    
    success = await tenant_manager.remove_member(
        tenant_id=context.tenant_id,
        user_id=user_id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or not a member of this tenant"
        )
    
    return {
        "user_id": user_id,
        "message": "Member removed successfully"
    }


@router.post(
    "/tenant/members/{user_id}/invite",
    response_model=MemberResponse,
    responses={
        200: {"description": "邀请发送成功"},
        403: {"model": ErrorResponse, "description": "权限不足或配额已满"}
    },
    summary="邀请成员",
    description="邀请用户加入当前租户（仅管理员可操作）"
)
async def invite_tenant_member(
    user_id: str,
    role: str = Query("user", pattern="^(admin|user|viewer)$"),
    context: TenantContext = Depends(require_tenant_admin())
):
    """
    邀请成员加入租户
    
    将指定用户添加到当前租户。
    """
    success = await tenant_manager.add_member(
        tenant_id=context.tenant_id,
        user_id=user_id,
        role=role
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Failed to add member. User may not exist or quota exceeded."
        )
    
    return {
        "user_id": user_id,
        "message": f"User invited with role: {role}"
    }


@router.get(
    "/tenant/usage",
    summary="获取租户资源使用情况",
    description="获取当前租户的资源使用统计"
)
async def get_tenant_usage(
    tenant: dict = Depends(get_current_tenant)
):
    """
    获取租户资源使用情况
    
    返回当前租户各项资源的使用情况统计。
    """
    quota = tenant.get("quota", {})
    usage = tenant.get("usage", {})
    
    return {
        "quota": {
            "max_users": quota.get("max_users", 0),
            "max_projects": quota.get("max_projects", 0),
            "max_storage_gb": quota.get("max_storage_gb", 0),
            "api_calls_per_month": quota.get("api_calls_per_month", 0)
        },
        "usage": {
            "users_count": usage.get("users_count", 0),
            "projects_count": usage.get("projects_count", 0),
            "storage_used_gb": usage.get("storage_used_gb", 0),
            "api_calls_this_month": usage.get("api_calls_this_month", 0)
        },
        "utilization": {
            "users_percent": round(
                (usage.get("users_count", 0) / quota.get("max_users", 1)) * 100, 2
            ) if quota.get("max_users") else 0,
            "projects_percent": round(
                (usage.get("projects_count", 0) / quota.get("max_projects", 1)) * 100, 2
            ) if quota.get("max_projects") else 0,
            "storage_percent": round(
                (usage.get("storage_used_gb", 0) / quota.get("max_storage_gb", 1)) * 100, 2
            ) if quota.get("max_storage_gb") else 0,
            "api_calls_percent": round(
                (usage.get("api_calls_this_month", 0) / quota.get("api_calls_per_month", 1)) * 100, 2
            ) if quota.get("api_calls_per_month") else 0
        }
    }
