"""
权限控制模块
实现RBAC（基于角色的访问控制）
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PermissionLevel(str, Enum):
    """权限级别枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    OWNER = "owner"


class Role:
    """角色定义"""

    def __init__(
        self,
        name: str,
        permissions: Set[PermissionLevel],
        inherit_from: Optional[List[str]] = None
    ):
        self.name = name
        self.permissions = permissions
        self.inherit_from = inherit_from or []
        self.resource_permissions: Dict[str, Set[PermissionLevel]] = {}

    def has_permission(self, permission: PermissionLevel) -> bool:
        """检查是否拥有指定权限"""
        return permission in self.permissions

    def add_permission(self, permission: PermissionLevel) -> None:
        """添加权限"""
        self.permissions.add(permission)

    def remove_permission(self, permission: PermissionLevel) -> None:
        """移除权限"""
        self.permissions.discard(permission)

    def grant_resource_permission(
        self,
        resource: str,
        permission: PermissionLevel
    ) -> None:
        """授予资源特定权限"""
        if resource not in self.resource_permissions:
            self.resource_permissions[resource] = set()
        self.resource_permissions[resource].add(permission)

    def get_resource_permissions(self, resource: str) -> Set[PermissionLevel]:
        """获取资源特定权限"""
        return self.resource_permissions.get(resource, set())


class User:
    """用户定义"""

    def __init__(
        self,
        user_id: str,
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.user_id = user_id
        self.roles = roles or []
        self.attributes = attributes or {}
        self.resource_access: Dict[str, Set[PermissionLevel]] = {}

    def assign_role(self, role: str) -> None:
        """分配角色"""
        if role not in self.roles:
            self.roles.append(role)

    def revoke_role(self, role: str) -> None:
        """撤销角色"""
        if role in self.roles:
            self.roles.remove(role)

    def grant_resource_access(
        self,
        resource: str,
        permission: PermissionLevel
    ) -> None:
        """授予资源直接访问权限"""
        if resource not in self.resource_access:
            self.resource_access[resource] = set()
        self.resource_access[resource].add(permission)

    def can_access(
        self,
        resource: str,
        permission: PermissionLevel,
        role_permissions: Dict[str, Role]
    ) -> bool:
        """检查是否可以访问资源"""
        # 检查直接授权
        if resource in self.resource_access:
            if permission in self.resource_access[resource]:
                return True

        # 检查角色权限
        for role_name in self.roles:
            if role_name in role_permissions:
                role = role_permissions[role_name]
                # 检查资源特定权限
                if permission in role.get_resource_permissions(resource):
                    return True
                # 检查通用权限
                if permission in role.permissions:
                    return True

        return False


class RBACManager:
    """RBAC管理器"""

    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.resource_hierarchy: Dict[str, List[str]] = {}

    # ============= 角色管理 =============

    def create_role(
        self,
        name: str,
        permissions: Optional[List[PermissionLevel]] = None,
        inherit_from: Optional[List[str]] = None
    ) -> Role:
        """创建角色"""
        role = Role(
            name=name,
            permissions=set(permissions or []),
            inherit_from=inherit_from or []
        )
        self.roles[name] = role
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """获取角色"""
        return self.roles.get(name)

    def delete_role(self, name: str) -> bool:
        """删除角色"""
        if name in self.roles:
            del self.roles[name]
            return True
        return False

    def update_role_permissions(
        self,
        name: str,
        permissions: List[PermissionLevel],
        mode: str = "set"  # set, add, remove
    ) -> bool:
        """更新角色权限"""
        role = self.roles.get(name)
        if not role:
            return False

        if mode == "set":
            role.permissions = set(permissions)
        elif mode == "add":
            role.permissions.update(permissions)
        elif mode == "remove":
            role.permissions.difference_update(permissions)
        return True

    # ============= 用户管理 =============

    def register_user(
        self,
        user_id: str,
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> User:
        """注册用户"""
        user = User(user_id=user_id, roles=roles, attributes=attributes)
        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)

    def assign_user_role(self, user_id: str, role: str) -> bool:
        """为用户分配角色"""
        user = self.users.get(user_id)
        if not user:
            return False
        user.assign_role(role)
        return True

    def revoke_user_role(self, user_id: str, role: str) -> bool:
        """撤销用户角色"""
        user = self.users.get(user_id)
        if not user:
            return False
        user.revoke_role(role)
        return True

    def set_user_attributes(
        self,
        user_id: str,
        attributes: Dict[str, Any]
    ) -> bool:
        """设置用户属性"""
        user = self.users.get(user_id)
        if not user:
            return False
        user.attributes.update(attributes)
        return True

    # ============= 权限检查 =============

    def check_permission(
        self,
        user_id: str,
        resource: str,
        permission: PermissionLevel
    ) -> bool:
        """检查用户对资源的权限"""
        user = self.users.get(user_id)
        if not user:
            return False

        return user.can_access(resource, permission, self.roles)

    def check_any_permission(
        self,
        user_id: str,
        resource: str,
        permissions: List[PermissionLevel]
    ) -> bool:
        """检查用户是否拥有任一权限"""
        return any(
            self.check_permission(user_id, resource, p)
            for p in permissions
        )

    def check_all_permissions(
        self,
        user_id: str,
        resource: str,
        permissions: List[PermissionLevel]
    ) -> bool:
        """检查用户是否拥有所有权限"""
        return all(
            self.check_permission(user_id, resource, p)
            for p in permissions
        )

    def grant_resource_permission_to_role(
        self,
        role_name: str,
        resource: str,
        permission: PermissionLevel
    ) -> bool:
        """为角色授予资源特定权限"""
        role = self.roles.get(role_name)
        if not role:
            return False
        role.grant_resource_permission(resource, permission)
        return True

    def grant_direct_access(
        self,
        user_id: str,
        resource: str,
        permission: PermissionLevel
    ) -> bool:
        """授予用户直接访问权限"""
        user = self.users.get(user_id)
        if not user:
            return False
        user.grant_resource_access(resource, permission)
        return True

    # ============= 资源层级 =============

    def add_resource_hierarchy(
        self,
        resource: str,
        parent_resources: List[str]
    ) -> None:
        """添加资源层级关系"""
        self.resource_hierarchy[resource] = parent_resources

    def get_accessible_resources(
        self,
        user_id: str,
        permission: PermissionLevel
    ) -> List[str]:
        """获取用户可访问的资源列表"""
        user = self.users.get(user_id)
        if not user:
            return []

        resources = set()

        for role_name in user.roles:
            role = self.roles.get(role_name)
            if not role:
                continue

            # 添加资源特定权限的资源
            for resource, perms in role.resource_permissions.items():
                if permission in perms:
                    resources.add(resource)

            # 添加通用权限的资源
            if permission in role.permissions:
                resources.update(role.resource_permissions.keys())

        # 添加直接授权的资源
        for resource, perms in user.resource_access.items():
            if permission in perms:
                resources.add(resource)

        return list(resources)

    def to_dict(self) -> Dict[str, Any]:
        """导出配置为字典"""
        return {
            "roles": {
                name: {
                    "permissions": list(role.permissions),
                    "inherit_from": role.inherit_from,
                    "resource_permissions": {
                        r: list(p) for r, p in role.resource_permissions.items()
                    }
                }
                for name, role in self.roles.items()
            },
            "users": {
                uid: {
                    "roles": user.roles,
                    "attributes": user.attributes,
                    "resource_access": {
                        r: list(p) for r, p in user.resource_access.items()
                    }
                }
                for uid, user in self.users.items()
            },
            "resource_hierarchy": self.resource_hierarchy
        }
