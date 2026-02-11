"""
Tenant Middleware - 租户中间件

功能：
- 租户上下文注入
- 租户资源隔离
- 配额检查中间件
"""

from typing import Optional, Callable, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException, status, Depends

from auth.middleware import get_current_user
from tenant.manager import tenant_manager


# ============ 租户上下文 ============

class TenantContext:
    """租户上下文"""
    
    def __init__(
        self,
        tenant_id: str,
        tenant_name: str = None,
        roles: list = None,
        quota: dict = None
    ):
        self.tenant_id = tenant_id
        self.tenant_name = tenant_name
        self.roles = roles or []
        self.quota = quota or {}
    
    def is_admin(self) -> bool:
        """检查是否为租户管理员"""
        return "admin" in self.roles or "owner" in self.roles
    
    def has_role(self, role: str) -> bool:
        """检查是否具有指定角色"""
        return role in self.roles


# =========中间=== 租户件 ============

class TenantMiddleware(BaseHTTPMiddleware):
    """租户中间件 - 自动注入租户上下文"""
    
    # 不需要租户上下文的路径
    EXCLUDED_PATHS = [
        "/api/v1/auth",
        "/api/v1/health",
        "/docs",
        "/openapi",
        "/static"
    ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """处理请求"""
        # 检查是否需要租户上下文
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.EXCLUDED_PATHS):
            return await call_next(request)
        
        # 从请求头获取租户ID
        tenant_id = request.headers.get("X-Tenant-ID")
        
        if tenant_id:
            # 将租户ID存储在请求状态中
            request.state.tenant_id = tenant_id
        
        response = await call_next(request)
        return response


# ============ 依赖注入 ============

async def get_current_tenant(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取当前用户所属租户"""
    tenant_id = current_user.get("tenant_id")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User does not belong to any tenant"
        )
    
    tenant = await tenant_manager.get_tenant_by_id(str(tenant_id))
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    if not tenant.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is inactive"
        )
    
    return tenant


async def get_tenant_context(
    tenant: Dict[str, Any] = Depends(get_current_tenant),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TenantContext:
    """获取租户上下文"""
    return TenantContext(
        tenant_id=str(tenant["_id"]),
        tenant_name=tenant.get("name"),
        roles=current_user.get("roles", []),
        quota=tenant.get("quota", {})
    )


def require_tenant_role(*allowed_roles: str):
    """租户角色依赖 - 检查用户是否具有指定角色"""
    async def role_checker(
        context: TenantContext = Depends(get_tenant_context)
    ) -> TenantContext:
        if not any(context.has_role(role) for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of these roles: {', '.join(allowed_roles)}"
            )
        return context
    
    return role_checker


def require_tenant_admin():
    """租户管理员依赖"""
    async def admin_checker(
        context: TenantContext = Depends(get_tenant_context)
    ) -> TenantContext:
        if not context.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Requires tenant admin role"
            )
        return context
    
    return admin_checker


# ============ 配额检查中间件 ============

class QuotaCheckMiddleware(BaseHTTPMiddleware):
    """配额检查中间件"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """检查配额"""
        # 从依赖注入中获取租户上下文
        tenant_id = getattr(request.state, "tenant_id", None)
        
        if tenant_id:
            # 检查API调用配额
            current, limit = await tenant_manager.check_quota(tenant_id, "api_calls")
            if current >= limit:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "API quota exceeded",
                        "current_usage": current,
                        "quota_limit": limit
                    }
                )
            
            # 增加API调用计数
            await tenant_manager.increment_usage(tenant_id, "api_calls", 1)
        
        return await call_next(request)


# ============ 资源隔离过滤器 ============

class TenantResourceFilter:
    """租户资源过滤器 - 用于查询时自动过滤"""
    
    @staticmethod
    def filter_by_tenant(query: dict, tenant_id: str) -> dict:
        """为查询添加租户过滤条件"""
        if not tenant_id:
            return query
        
        # 确保租户隔离
        return {
            **query,
            "tenant_id": tenant_id
        }
    
    @staticmethod
    async def ensure_tenant_access(
        resource_id: str,
        resource_collection: str,
        tenant_id: str
    ) -> bool:
        """确保用户有权访问指定资源"""
        from bson import ObjectId
        
        if not ObjectId.is_valid(resource_id):
            return False
        
        resource = await db[resource_collection].find_one({
            "_id": ObjectId(resource_id),
            "tenant_id": ObjectId(tenant_id)
        })
        
        return resource is not None


# ============ 租户设置中间件 ============

class TenantSettingsMiddleware(BaseHTTPMiddleware):
    """租户设置中间件 - 应用租户特定的设置"""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """应用租户设置"""
        tenant_id = getattr(request.state, "tenant_id", None)
        
        if tenant_id:
            tenant = await tenant_manager.get_tenant_by_id(tenant_id)
            if tenant:
                # 将租户设置存储在请求状态中
                request.state.tenant_settings = tenant.get("settings", {})
        
        response = await call_next(request)
        return response


# ============ 导出 ============

__all__ = [
    "TenantContext",
    "TenantMiddleware",
    "QuotaCheckMiddleware",
    "TenantResourceFilter",
    "TenantSettingsMiddleware",
    "get_current_tenant",
    "get_tenant_context",
    "require_tenant_role",
    "require_tenant_admin"
]
