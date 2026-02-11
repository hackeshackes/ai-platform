"""
缓存使用示例
"""
from fastapi import APIRouter
from typing import List, Optional
from .decorators import cached, cache_invalidate

router = APIRouter(prefix="/examples", tags=["Cache Examples"])

# 示例1: 基本缓存
@router.get("/projects")
@cached(ttl=300, prefix="projects")
async def list_projects():
    """获取项目列表 (缓存5分钟)"""
    # 模拟数据库查询
    await asyncio.sleep(0.1)
    return [{"id": 1, "name": "项目A"}, {"id": 2, "name": "项目B"}]

# 示例2: 带参数的缓存
def project_key_builder(project_id: str):
    return f"aip:projects:detail:{project_id}"

@router.get("/projects/{project_id}")
@cached(ttl=600, key_builder=project_key_builder)
async def get_project(project_id: str):
    """获取单个项目 (缓存10分钟)"""
    await asyncio.sleep(0.05)
    return {"id": project_id, "name": f"项目{project_id}"}

# 示例3: 自动失效
@router.put("/projects/{project_id}")
@cache_invalidate(pattern="projects")
async def update_project(project_id: str):
    """更新项目 (自动失效相关缓存)"""
    await asyncio.sleep(0.1)
    return {"id": project_id, "updated": True}

# 示例4: 批量缓存
@router.get("/projects/batch")
async def get_projects_batch(ids: str):
    """批量获取项目"""
    id_list = ids.split(",")
    results = []
    for pid in id_list:
        key = f"aip:projects:batch:{pid}"
        cached = get_cache(key)
        if cached:
            results.append(cached)
        else:
            result = {"id": pid, "name": f"项目{pid}"}
            set_cache(key, result, 600)
            results.append(result)
    return results

import asyncio
