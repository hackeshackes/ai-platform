"""
Tenant Manager - 租户管理器

功能：
- 租户CRUD操作
- 租户资源隔离
- 租户配额管理
- 租户成员管理
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from bson import ObjectId
from fastapi import HTTPException, status
from pymongo import ASCENDING, DESCENDING

from db.database import db


class TenantManager:
    """租户管理器"""
    
    @staticmethod
    async def create_tenant(
        name: str,
        owner_id: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建新租户"""
        tenant = {
            "name": name,
            "owner_id": owner_id,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "is_active": True,
            "quota": {
                "max_users": 10,
                "max_projects": 5,
                "max_storage_gb": 100,
                "api_calls_per_month": 10000
            },
            "usage": {
                "users_count": 1,
                "projects_count": 0,
                "storage_used_gb": 0,
                "api_calls_this_month": 0
            },
            "settings": settings or {
                "allow_custom_domains": False,
                "default_role": "user",
                "sso_provider": None
            },
            "metadata": {}
        }
        
        result = await db.tenants.insert_one(tenant)
        tenant["_id"] = result.inserted_id
        
        return tenant
    
    @staticmethod
    async def get_tenant_by_id(tenant_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取租户"""
        if not ObjectId.is_valid(tenant_id):
            return None
        return await db.tenants.find_one({"_id": ObjectId(tenant_id)})
    
    @staticmethod
    async def get_tenant_by_name(name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取租户"""
        return await db.tenants.find_one({"name": name})
    
    @staticmethod
    async def list_tenants(
        skip: int = 0,
        limit: int = 20,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """列出租户"""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        cursor = db.tenants.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)
    
    @staticmethod
    async def update_tenant(
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """更新租户"""
        if not ObjectId.is_valid(tenant_id):
            return None
        
        updates["updated_at"] = datetime.now(timezone.utc)
        
        result = await db.tenants.update_one(
            {"_id": ObjectId(tenant_id)},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            return await TenantManager.get_tenant_by_id(tenant_id)
        return None
    
    @staticmethod
    async def delete_tenant(tenant_id: str) -> bool:
        """删除租户（软删除）"""
        if not ObjectId.is_valid(tenant_id):
            return False
        
        result = await db.tenants.update_one(
            {"_id": ObjectId(tenant_id)},
            {
                "$set": {
                    "is_active": False,
                    "deleted_at": datetime.now(timezone.utc)
                }
            }
        )
        
        return result.modified_count > 0
    
    # ============ 配额管理 ============
    
    @staticmethod
    async def get_quota(tenant_id: str) -> Dict[str, Any]:
        """获取租户配额"""
        tenant = await TenantManager.get_tenant_by_id(tenant_id)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        return tenant.get("quota", {})
    
    @staticmethod
    async def set_quota(
        tenant_id: str,
        max_users: Optional[int] = None,
        max_projects: Optional[int] = None,
        max_storage_gb: Optional[int] = None,
        api_calls_per_month: Optional[int] = None
    ) -> Dict[str, Any]:
        """设置租户配额"""
        if not ObjectId.is_valid(tenant_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant ID"
            )
        
        quota_updates = {}
        if max_users is not None:
            quota_updates["quota.max_users"] = max_users
        if max_projects is not None:
            quota_updates["quota.max_projects"] = max_projects
        if max_storage_gb is not None:
            quota_updates["quota.max_storage_gb"] = max_storage_gb
        if api_calls_per_month is not None:
            quota_updates["quota.api_calls_per_month"] = api_calls_per_month
        
        quota_updates["updated_at"] = datetime.now(timezone.utc)
        
        result = await db.tenants.update_one(
            {"_id": ObjectId(tenant_id)},
            {"$set": quota_updates}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found or no changes made"
            )
        
        return await TenantManager.get_quota(tenant_id)
    
    @staticmethod
    async def check_quota(tenant_id: str, resource_type: str) -> tuple:
        """检查租户配额使用情况"""
        tenant = await TenantManager.get_tenant_by_id(tenant_id)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        quota = tenant.get("quota", {})
        usage = tenant.get("usage", {})
        
        if resource_type == "users":
            limit = quota.get("max_users", 0)
            current = usage.get("users_count", 0)
        elif resource_type == "projects":
            limit = quota.get("max_projects", 0)
            current = usage.get("projects_count", 0)
        elif resource_type == "storage":
            limit = quota.get("max_storage_gb", 0)
            current = usage.get("storage_used_gb", 0)
        elif resource_type == "api_calls":
            limit = quota.get("api_calls_per_month", 0)
            current = usage.get("api_calls_this_month", 0)
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        return current, limit
    
    @staticmethod
    async def increment_usage(
        tenant_id: str,
        resource_type: str,
        amount: int = 1
    ) -> None:
        """增加资源使用量"""
        if not ObjectId.is_valid(tenant_id):
            return
        
        field_map = {
            "users": "usage.users_count",
            "projects": "usage.projects_count",
            "storage": "usage.storage_used_gb",
            "api_calls": "usage.api_calls_this_month"
        }
        
        if resource_type in field_map:
            await db.tenants.update_one(
                {"_id": ObjectId(tenant_id)},
                {"$inc": {field_map[resource_type]: amount}}
            )
    
    # ============ 成员管理 ============
    
    @staticmethod
    async def add_member(
        tenant_id: str,
        user_id: str,
        role: str = "user"
    ) -> bool:
        """添加租户成员"""
        if not ObjectId.is_valid(tenant_id) or not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant or user ID"
            )
        
        # 检查配额
        current, limit = await TenantManager.check_quota(tenant_id, "users")
        if current >= limit:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User quota exceeded"
            )
        
        # 更新用户租户信息
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "tenant_id": ObjectId(tenant_id),
                    "updated_at": datetime.now(timezone.utc)
                },
                "$addToSet": {"roles": role}
            }
        )
        
        if result.modified_count > 0:
            # 增加使用计数
            await TenantManager.increment_usage(tenant_id, "users")
            return True
        
        return False
    
    @staticmethod
    async def remove_member(tenant_id: str, user_id: str) -> bool:
        """移除租户成员"""
        if not ObjectId.is_valid(tenant_id) or not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant or user ID"
            )
        
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$unset": {"tenant_id": ""},
                "$pull": {"roles": "admin"}  # 移除管理员角色
            }
        )
        
        if result.modified_count > 0:
            # 减少使用计数
            await db.tenants.update_one(
                {"_id": ObjectId(tenant_id)},
                {"$inc": {"usage.users_count": -1}}
            )
            return True
        
        return False
    
    @staticmethod
    async def get_members(
        tenant_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取租户成员列表"""
        if not ObjectId.is_valid(tenant_id):
            return []
        
        cursor = db.users.find({"tenant_id": ObjectId(tenant_id)})
        users = await cursor.to_list(length=limit)
        
        # 脱敏处理
        for user in users:
            user.pop("password_hash", None)
            user.pop("sso_tokens", None)
        
        return users
    
    @staticmethod
    async def update_member_role(
        tenant_id: str,
        user_id: str,
        role: str
    ) -> bool:
        """更新成员角色"""
        if not ObjectId.is_valid(tenant_id) or not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant or user ID"
            )
        
        result = await db.users.update_one(
            {
                "_id": ObjectId(user_id),
                "tenant_id": ObjectId(tenant_id)
            },
            {
                "$set": {
                    f"roles.$[role]": role if role in ["admin", "user", "viewer"] else "user"
                }
            }
        )
        
        return result.modified_count > 0
    
    # ============ 资源隔离 ============
    
    @staticmethod
    async def get_tenant_resources(tenant_id: str) -> Dict[str, Any]:
        """获取租户所有资源"""
        if not ObjectId.is_valid(tenant_id):
            return {}
        
        tenant_id_obj = ObjectId(tenant_id)
        
        # 获取项目
        projects = await db.projects.find({"tenant_id": tenant_id_obj}).to_list(length=100)
        
        # 获取数据集
        datasets = await db.datasets.find({"tenant_id": tenant_id_obj}).to_list(length=100)
        
        # 获取模型
        models = await db.models.find({"tenant_id": tenant_id_obj}).to_list(length=100)
        
        return {
            "projects": projects,
            "datasets": datasets,
            "models": models
        }


# 全局租户管理器实例
tenant_manager = TenantManager()
