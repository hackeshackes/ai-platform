"""
Tenant Module - 租户管理模块

包含：
- 租户管理器
- 租户中间件
- 租户资源隔离
"""

from tenant.manager import tenant_manager, TenantManager
from tenant.middleware import (
    TenantContext,
    TenantMiddleware,
    TenantResourceFilter,
    TenantSettingsMiddleware,
    QuotaCheckMiddleware,
    get_current_tenant,
    get_tenant_context,
    require_tenant_role,
    require_tenant_admin
)

__all__ = [
    "tenant_manager",
    "TenantManager",
    "TenantContext",
    "TenantMiddleware",
    "TenantResourceFilter",
    "TenantSettingsMiddleware",
    "QuotaCheckMiddleware",
    "get_current_tenant",
    "get_tenant_context",
    "require_tenant_role",
    "require_tenant_admin"
]
