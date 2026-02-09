"""
多租户模块 v2.2
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

@dataclass
class Tenant:
    """租户"""
    tenant_id: str
    name: str
    owner_id: str
    plan: str = "free"  # free, pro, enterprise
    quota: Dict = field(default_factory=dict)  # 资源配额
    settings: Dict = field(default_factory=dict)
    members: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Role:
    """角色"""
    role_id: str
    tenant_id: str
    name: str
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Membership:
    """成员关系"""
    membership_id: str
    tenant_id: str
    user_id: str
    role_id: str
    joined_at: datetime = field(default_factory=datetime.utcnow)

class MultiTenantManager:
    """多租户管理器"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.roles: Dict[str, Role] = {}
        self.memberships: Dict[str, Membership] = {}
        self.user_tenants: Dict[str, List[str]] = {}  # user_id -> [tenant_id]
    
    # 租户管理
    async def create_tenant(
        self,
        name: str,
        owner_id: str,
        plan: str = "free"
    ) -> Tenant:
        """创建租户"""
        tenant = Tenant(
            tenant_id=str(uuid4()),
            name=name,
            owner_id=owner_id,
            plan=plan,
            quota=self._get_default_quota(plan)
        )
        
        self.tenants[tenant.tenant_id] = tenant
        
        # 创建默认角色
        await self.create_role(tenant.tenant_id, "owner", ["*"])
        await self.create_role(tenant.tenant_id, "admin", ["read", "write", "delete"])
        await self.create_role(tenant.tenant_id, "member", ["read", "write"])
        await self.create_role(tenant.tenant_id, "viewer", ["read"])
        
        return tenant
    
    def _get_default_quota(self, plan: str) -> Dict:
        """获取默认配额"""
        quotas = {
            "free": {
                "projects": 5,
                "storage_gb": 10,
                "models": 10,
                "members": 3
            },
            "pro": {
                "projects": 50,
                "storage_gb": 100,
                "models": 100,
                "members": 10
            },
            "enterprise": {
                "projects": -1,  # 无限制
                "storage_gb": 1000,
                "models": -1,
                "members": -1
            }
        }
        return quotas.get(plan, quotas["free"])
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """获取租户"""
        return self.tenants.get(tenant_id)
    
    def list_tenants(
        self,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Tenant]:
        """列出租户"""
        tenants = list(self.tenants.values())
        
        if user_id:
            tenants = [t for t in tenants if user_id in t.members or t.owner_id == user_id]
        
        return tenants[skip:skip+limit]
    
    # 角色管理
    async def create_role(
        self,
        tenant_id: str,
        name: str,
        permissions: List[str]
    ) -> Role:
        """创建角色"""
        role = Role(
            role_id=str(uuid4()),
            tenant_id=tenant_id,
            name=name,
            permissions=permissions
        )
        
        self.roles[role.role_id] = role
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        return self.roles.get(role_id)
    
    def get_tenant_roles(self, tenant_id: str) -> List[Role]:
        """获取租户所有角色"""
        return [r for r in self.roles.values() if r.tenant_id == tenant_id]
    
    # 成员管理
    async def add_member(
        self,
        tenant_id: str,
        user_id: str,
        role_name: str = "member"
    ) -> Membership:
        """添加成员"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        # 查找角色
        role = next(
            (r for r in self.roles.values() if r.tenant_id == tenant_id and r.name == role_name),
            None
        )
        if not role:
            raise ValueError(f"Role {role_name} not found")
        
        membership = Membership(
            membership_id=str(uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            role_id=role.role_id
        )
        
        self.memberships[membership.membership_id] = membership
        
        # 更新租户成员列表
        if user_id not in tenant.members:
            tenant.members.append(user_id)
        
        # 更新用户租户映射
        if user_id not in self.user_tenants:
            self.user_tenants[user_id] = []
        if tenant_id not in self.user_tenants[user_id]:
            self.user_tenants[user_id].append(tenant_id)
        
        return membership
    
    def remove_member(self, tenant_id: str, user_id: str) -> bool:
        """移除成员"""
        for mid, m in list(self.memberships.items()):
            if m.tenant_id == tenant_id and m.user_id == user_id:
                del self.memberships[mid]
                
                # 从租户成员列表移除
                tenant = self.tenants.get(tenant_id)
                if tenant and user_id in tenant.members:
                    tenant.members.remove(user_id)
                
                return True
        return False
    
    def get_user_tenants(self, user_id: str) -> List[Tenant]:
        """获取用户所属所有租户"""
        tenant_ids = self.user_tenants.get(user_id, [])
        return [self.tenants[tid] for tid in tenant_ids if tid in self.tenants]
    
    def check_permission(
        self,
        tenant_id: str,
        user_id: str,
        permission: str
    ) -> bool:
        """检查权限"""
        for m in self.memberships.values():
            if m.tenant_id == tenant_id and m.user_id == user_id:
                role = self.roles.get(m.role_id)
                if role and ("*" in role.permissions or permission in role.permissions):
                    return True
        return False
    
    def get_usage(self, tenant_id: str) -> Dict[str, int]:
        """获取资源使用量"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}
        
        # 简化实现
        return {
            "projects": len(tenant.members),  # 模拟值
            "storage_gb": 1,
            "models": 3,
            "members": len(tenant.members)
        }
    
    def check_quota(self, tenant_id: str, resource: str, count: int = 1) -> bool:
        """检查配额"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        quota = tenant.quota.get(resource, -1)
        if quota == -1:  # 无限制
            return True
        
        usage = self.get_usage(tenant_id).get(resource, 0)
        return (usage + count) <= quota

# MultiTenantManager实例
tenant_manager = MultiTenantManager()
