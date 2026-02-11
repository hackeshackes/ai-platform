"""
访问控制 (Access Control)
基于RBAC的细粒度权限管理，支持资源级和操作级控制
"""

from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class Permission(Enum):
    """基础权限定义"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(Enum):
    """资源类型"""
    USER = "user"
    DATA = "data"
    REPORT = "report"
    CONFIG = "config"
    AUDIT = "audit"
    API = "api"
    DOCUMENT = "document"


@dataclass
class Role:
    """角色定义"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    resource_access: Dict[str, Set[Permission]] = field(default_factory=dict)  # resource_type -> permissions
    is_system: bool = False


@dataclass
class User:
    """用户定义"""
    id: str
    username: str
    roles: List[str] = field(default_factory=list)
    department: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class AccessRequest:
    """访问请求"""
    user_id: str
    resource_type: ResourceType
    permission: Permission
    resource_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessPolicy:
    """访问策略"""
    id: str
    name: str
    description: str
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    effect: str = "allow"  # allow/deny
    roles: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)


class AccessControlEngine:
    """访问控制引擎"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.policies: List[AccessPolicy] = []
        self.role_hierarchy: Dict[str, List[str]] = {}  # role -> inherited roles
        
        # 初始化默认角色
        self._init_default_roles()
    
    def _init_default_roles(self):
        """初始化默认角色"""
        admin_role = Role(
            name="admin",
            description="系统管理员",
            permissions=set(Permission),
            is_system=True
        )
        self.roles["admin"] = admin_role
        
        auditor_role = Role(
            name="auditor",
            description="审计员",
            permissions={Permission.READ},
            resource_access={
                ResourceType.AUDIT.value: {Permission.READ}
            },
            is_system=True
        )
        self.roles["auditor"] = auditor_role
        
        user_role = Role(
            name="user",
            description="普通用户",
            permissions={Permission.READ},
            resource_access={
                ResourceType.DATA.value: {Permission.READ, Permission.WRITE}
            },
            is_system=True
        )
        self.roles["user"] = user_role
    
    def create_role(self, role: Role) -> str:
        """创建角色"""
        self.roles[role.name] = role
        return role.name
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """为用户分配角色"""
        if role_name not in self.roles:
            return False
        if user_id not in self.users:
            self.users[user_id] = User(id=user_id, username=user_id)
        if role_name not in self.users[user_id].roles:
            self.users[user_id].roles.append(role_name)
        return True
    
    def remove_role(self, user_id: str, role_name: str) -> bool:
        """移除用户角色"""
        if user_id in self.users and role_name in self.users[user_id].roles:
            self.users[user_id].roles.remove(role_name)
            return True
        return False
    
    def add_policy(self, policy: AccessPolicy) -> None:
        """添加访问策略"""
        self.policies.append(policy)
        # 按优先级排序
        self.policies.sort(key=lambda p: p.priority, reverse=True)
    
    def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        permission: Permission,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str]:
        """
        检查用户是否有权限访问资源
        
        返回: (是否允许, 原因说明)
        """
        # 获取用户
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False, "用户不存在或已禁用"
        
        # 检查角色权限
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if not role:
                continue
            
            # 检查基础权限
            if permission in role.permissions:
                return True, f"通过角色 {role_name} 授权"
            
            # 检查资源级权限
            if resource_type.value in role.resource_access:
                if permission in role.resource_access[resource_type.value]:
                    return True, f"通过角色 {role_name} 资源授权"
        
        # 检查策略
        for policy in self.policies:
            if self._match_policy(policy, user, resource_type, permission, resource_id, context):
                return policy.effect == "allow", f"策略匹配: {policy.name}"
        
        return False, "无匹配权限"
    
    def _match_policy(
        self,
        policy: AccessPolicy,
        user: User,
        resource_type: ResourceType,
        permission: Permission,
        resource_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """检查策略是否匹配"""
        # 检查角色
        if policy.roles and user.roles and not any(r in policy.roles for r in user.roles):
            return False
        
        # 检查资源
        if policy.resources and resource_type.value not in policy.resources:
            return False
        
        # 检查权限
        if policy.permissions and permission.value not in policy.permissions:
            return False
        
        # 检查条件
        if policy.conditions:
            conditions = policy.conditions
            if 'department' in conditions:
                if user.department != conditions['department']:
                    return False
            if 'time_range' in conditions:
                # 时间范围检查
                pass
        
        return True
    
    def get_user_permissions(self, user_id: str) -> Dict[str, Set[str]]:
        """获取用户所有权限"""
        user = self.users.get(user_id)
        if not user:
            return {}
        
        all_permissions: Dict[str, Set[str]] = {}
        
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if not role:
                continue
            
            # 基础权限
            for perm in role.permissions:
                all_permissions.setdefault("global", set()).add(perm.value)
            
            # 资源级权限
            for resource, perms in role.resource_access.items():
                all_permissions.setdefault(resource, set()).update(p.value for p in perms)
        
        return all_permissions
    
    def create_access_policy(
        self,
        name: str,
        description: str = "",
        roles: Optional[List[str]] = None,
        resources: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        effect: str = "allow",
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> str:
        """创建访问策略"""
        policy = AccessPolicy(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            roles=roles or [],
            resources=resources or [],
            permissions=permissions or [],
            effect=effect,
            conditions=conditions or {},
            priority=priority
        )
        self.add_policy(policy)
        return policy.id
    
    def audit_access(self, request: AccessRequest, result: bool, reason: str) -> Dict[str, Any]:
        """记录访问审计"""
        return {
            "request_id": str(uuid.uuid4()),
            "user_id": request.user_id,
            "resource_type": request.resource_type.value,
            "resource_id": request.resource_id,
            "permission": request.permission.value,
            "result": result,
            "reason": reason,
            "timestamp": request.timestamp.isoformat(),
            "context": request.context
        }


# 默认实例
access_control = AccessControlEngine()


def get_access_control() -> AccessControlEngine:
    """获取全局访问控制实例"""
    return access_control
