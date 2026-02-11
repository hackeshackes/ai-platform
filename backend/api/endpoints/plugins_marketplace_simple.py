"""
Plugin Marketplace Simple API - 简化版
"""
import json
import os
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/plugins/marketplace", tags=["Plugin Marketplace"])

# 模拟数据
INSTALLED_PLUGINS_FILE = "./data/plugins/installed.json"
PLUGINS_DIR = "./data/plugins/marketplace"

def get_plugins() -> List[Dict]:
    """获取所有Plugin"""
    plugins = []
    if os.path.exists(PLUGINS_DIR):
        for f in os.listdir(PLUGINS_DIR):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(PLUGINS_DIR, f), 'r') as file:
                        plugins.append(json.load(file))
                except:
                    pass
    return plugins

def get_categories() -> Dict[str, Dict]:
    """获取分类"""
    cats = {
        "tool": {"name": "tool", "count": 0, "description": "工具类"},
        "agent": {"name": "agent", "count": 0, "description": "Agent类"},
        "integration": {"name": "integration", "count": 0, "description": "集成类"},
        "ui": {"name": "ui", "count": 0, "description": "UI组件"},
        "visualization": {"name": "visualization", "count": 0, "description": "可视化"},
        "data_source": {"name": "data_source", "count": 0, "description": "数据源"}
    }
    for cat in cats:
        cats[cat]["count"] = len([p for p in get_plugins() if p.get('category') == cat])
    return cats

@router.get("/")
async def list_plugins() -> List[Dict]:
    return get_plugins()

@router.get("/installed")
async def list_installed() -> List[Dict]:
    if os.path.exists(INSTALLED_PLUGINS_FILE):
        try:
            with open(INSTALLED_PLUGINS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

@router.get("/categories")
async def categories() -> Dict:
    return get_categories()

@router.get("/search")
async def search_plugins(q: str = "") -> List[Dict]:
    if not q:
        return get_plugins()
    q = q.lower()
    return [p for p in get_plugins() if q in p.get('name', '').lower() or q in p.get('description', '').lower()]

@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> Dict:
    for p in get_plugins():
        if p.get('id') == plugin_id or p.get('name') == plugin_id:
            return p
    raise HTTPException(status_code=404, detail="Plugin not found")
