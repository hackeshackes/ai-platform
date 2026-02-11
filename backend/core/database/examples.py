"""
PostgreSQL使用示例
"""
from fastapi import APIRouter
from typing import List

router = APIRouter(prefix="/db", tags=["Database"])

@router.get("/users")
async def list_users():
    """列出用户"""
    from .postgres import execute_query
    users = execute_query("SELECT id, username, email FROM users LIMIT 10")
    return {"users": users}

@router.get("/projects")
async def list_projects():
    """列出项目"""
    from .postgres import execute_query
    projects = execute_query("SELECT id, name, status FROM projects LIMIT 10")
    return {"projects": projects}

@router.get("/agents")
async def list_agents():
    """列出Agent"""
    from .postgres import execute_query
    agents = execute_query("SELECT id, name, type FROM agents LIMIT 10")
    return {"agents": agents}
