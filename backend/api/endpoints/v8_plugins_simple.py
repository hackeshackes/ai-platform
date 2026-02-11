"""V8 Plugin市场 - 简化稳定版"""
from fastapi import APIRouter
from typing import Dict, List
import json

router = APIRouter(prefix="/v8/plugins", tags=["V8 Plugins"])

PLUGINS_DB = [
    {"id": "search-tool", "name": "search-tool", "display_name": "搜索工具", "description": "提供强大的搜索功能", "version": "1.0.0", "category": "tool"},
    {"id": "coding-agent", "name": "coding-agent", "display_name": "编程助手", "description": "专业的代码生成Agent", "version": "1.0.0", "category": "agent"},
    {"id": "data-connector", "name": "data-connector", "display_name": "数据连接器", "description": "连接各种数据源", "version": "1.0.0", "category": "integration"}
]

@router.get("/")
async def list_plugins() -> List[Dict]:
    return PLUGINS_DB

@router.get("/installed")
async def installed() -> List[Dict]:
    return []

@router.get("/categories")
async def categories() -> Dict:
    return {"tool": 1, "agent": 1, "integration": 1}

@router.get("/search")
async def search(q: str = "") -> List[Dict]:
    if not q:
        return PLUGINS_DB
    q = q.lower()
    return [p for p in PLUGINS_DB if q in p.get('name', '').lower() or q in p.get('description', '').lower()]

@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> Dict:
    for p in PLUGINS_DB:
        if p.get('id') == plugin_id:
            return p
    return {"error": "not found"}
