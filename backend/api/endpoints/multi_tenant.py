"""
多租户API端点 v2.2
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from backend.multi_tenant.manager import tenant_manager
from backend.core.auth import get_current_user

router = APIRouter()

class CreateTenantModel(BaseModel):
    name: str
    plan: str = "free"

class AddMemberModel(BaseModel):
    user_id: str
    role_name: str = "member"

class CreateRoleModel(BaseModel):
    name: str
    permissions: List[str]

@router.post("/tenants")
async def create_tenant(
    request: CreateTenantModel,
    current_user = Depends(get_current_user)
):
    """
    创建租户
    
    v2.2: 多租户
    """
    tenant = await tenant_manager.create_tenant(
        name=request.name,
        owner_id=str(current_user.id),
        plan=request.plan
    )
    
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "plan": tenant.plan,
        "quota": tenant.quota,
        "created_at": tenant.created_at.isoformat()
    }

@router.get("/tenants")
async def list_tenants(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    列出租户
    
    v2.2: 多租户
    """
    tenants = tenant_manager.list_tenants(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit
    )
    
    return {
        "total": len(tenants),
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "plan": t.plan,
                "members_count": len(t.members),
                "created_at": t.created_at.isoformat()
            }
            for t in tenants
        ]
    }

@router.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str):
    """
    获取租户详情
    
    v2.2: 多租户
    """
    tenant = tenant_manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "plan": tenant.plan,
        "quota": tenant.quota,
        "members": tenant.members,
        "settings": tenant.settings,
        "created_at": tenant.created_at.isoformat()
    }

@router.post("/tenants/{tenant_id}/members")
async def add_member(
    tenant_id: str,
    request: AddMemberModel,
    current_user = Depends(get_current_user)
):
    """
    添加成员
    
    v2.2: 多租户
    """
    try:
        membership = await tenant_manager.add_member(
            tenant_id=tenant_id,
            user_id=request.user_id,
            role_name=request.role_name
        )
        
        return {
            "membership_id": membership.membership_id,
            "user_id": membership.user_id,
            "role_id": membership.role_id,
            "joined_at": membership.joined_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/tenants/{tenant_id}/members/{user_id}")
async def remove_member(
    tenant_id: str,
    user_id: str
):
    """
    移除成员
    
    v2.2: 多租户
    """
    result = tenant_manager.remove_member(tenant_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Member not found")
    
    return {"message": "Member removed"}

@router.get("/tenants/{tenant_id}/roles")
async def get_tenant_roles(tenant_id: str):
    """
    获取租户角色
    
    v2.2: 多租户
    """
    roles = tenant_manager.get_tenant_roles(tenant_id)
    
    return {
        "total": len(roles),
        "roles": [
            {
                "role_id": r.role_id,
                "name": r.name,
                "permissions": r.permissions
            }
            for r in roles
        ]
    }

@router.post("/tenants/{tenant_id}/roles")
async def create_role(
    tenant_id: str,
    request: CreateRoleModel
):
    """
    创建角色
    
    v2.2: 多租户
    """
    role = await tenant_manager.create_role(
        tenant_id=tenant_id,
        name=request.name,
        permissions=request.permissions
    )
    
    return {
        "role_id": role.role_id,
        "name": role.name,
        "permissions": role.permissions
    }

@router.get("/tenants/{tenant_id}/usage")
async def get_tenant_usage(tenant_id: str):
    """
    获取租户资源使用量
    
    v2.2: 多租户
    """
    usage = tenant_manager.get_usage(tenant_id)
    tenant = tenant_manager.get_tenant(tenant_id)
    
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "usage": usage,
        "quota": tenant.quota,
        "available": {
            k: (v - usage.get(k, 0) if v > 0 else -1)
            for k, v in tenant.quota.items()
        }
    }

@router.get("/my-tenants")
async def get_my_tenants(current_user = Depends(get_current_user)):
    """
    获取我的租户
    
    v2.2: 多租户
    """
    tenants = tenant_manager.get_user_tenants(str(current_user.id))
    
    return {
        "total": len(tenants),
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "plan": t.plan
            }
            for t in tenants
        ]
    }

@router.post("/check-permission")
async def check_permission(
    tenant_id: str,
    permission: str,
    current_user = Depends(get_current_user)
):
    """
    检查权限
    
    v2.2: 多租户
    """
    has_permission = tenant_manager.check_permission(
        tenant_id=tenant_id,
        user_id=str(current_user.id),
        permission=permission
    )
    
    return {
        "tenant_id": tenant_id,
        "permission": permission,
        "allowed": has_permission
    }
