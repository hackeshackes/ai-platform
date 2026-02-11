"""V8 Plugin市场 - 简化版"""
from fastapi import APIRouter
from typing import Dict, List
import json
import os

router = APIRouter(prefix="/v8/plugins", tags=["V8 Plugins"])

PLUGINS_FILE = "./data/plugins/v8_plugins.json"

def load_plugins():
    if os.path.exists(PLUGINS_FILE):
        try:
            with open(PLUGINS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

@router.get("/")
async def list_plugins() -> List[Dict]:
    return load_plugins()

@router.get("/installed")
async def installed() -> List[Dict]:
    return []

@router.get("/categories")
async def categories() -> Dict:
    return {"tool": 1, "agent": 1, "integration": 1}

@router.get("/search")
async def search(q: str = "") -> List[Dict]:
    plugins = load_plugins()
    if q:
        q = q.lower()
        return [p for p in plugins if q in p.get('name', '').lower() or q in p.get('description', '').lower()]
    return plugins

@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str) -> Dict:
    for p in load_plugins():
        if p.get('id') == plugin_id:
            return p
    return {"error": "not found"}
