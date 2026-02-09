"""User management endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter()

# 模拟数据库
fake_users_db = {
    1: {
        "id": 1,
        "username": "admin",
        "email": "admin@ai-platform.com",
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow()
    }
}

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None

@router.get("")
async def list_users(skip: int = 0, limit: int = 100):
    """获取用户列表"""
    users = list(fake_users_db.values())[skip:skip+limit]
    return {
        "total": len(fake_users_db),
        "users": users
    }

@router.get("/{user_id}")
async def get_user(user_id: int):
    """获取单个用户"""
    if user_id not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return fake_users_db[user_id]

@router.put("/{user_id}")
async def update_user(user_id: int, user_data: UserUpdate):
    """更新用户"""
    if user_id not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    for field, value in user_data.dict(exclude_unset=True).items():
        fake_users_db[user_id][field] = value
    
    return fake_users_db[user_id]

@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: int):
    """删除用户"""
    if user_id not in fake_users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    del fake_users_db[user_id]
    return None
