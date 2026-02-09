"""
插件API端点 v2.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from plugins.manager import plugin_manager

router = APIRouter()

class RegisterPluginModel(BaseModel):
    name: str
    version: str
    description: str
    plugin_type: str  # metric, storage, integration, ui
    entry_point: str
    settings: Optional[Dict] = None

class UpdateSettingsModel(BaseModel):
    settings: Dict

@router.get("")
async def list_plugins(
    plugin_type: Optional[str] = None,
    enabled: Optional[bool] = None
):
    """
    列出插件
    
    v2.2: 插件系统
    """
    plugins = plugin_manager.list_plugins(
        plugin_type=plugin_type,
        enabled=enabled
    )
    
    return {
        "total": len(plugins),
        "plugins": [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "type": p.plugin_type,
                "enabled": p.enabled,
                "settings": p.settings,
                "loaded_at": p.loaded_at.isoformat() if p.loaded_at else None
            }
            for p in plugins
        ]
    }

@router.post("")
async def register_plugin(request: RegisterPluginModel):
    """
    注册插件
    
    v2.2: 插件系统
    """
    try:
        plugin = plugin_manager.register_plugin(
            name=request.name,
            version=request.version,
            description=request.description,
            plugin_type=request.plugin_type,
            entry_point=request.entry_point,
            settings=request.settings
        )
        
        return {
            "plugin_id": plugin.plugin_id,
            "name": plugin.name,
            "version": plugin.version,
            "message": "Plugin registered"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/marketplace")
async def get_marketplace():
    """
    获取插件市场
    
    v2.2: 插件系统
    """
    plugins = plugin_manager.get_marketplace_plugins()
    
    return {
        "total": len(plugins),
        "plugins": plugins
    }

@router.post("/marketplace/{name}/install")
async def install_marketplace_plugin(name: str):
    """
    安装市场插件
    
    v2.2: 插件系统
    """
    try:
        plugin = plugin_manager.install_marketplace_plugin(name)
        return {
            "plugin_id": plugin.plugin_id,
            "name": plugin.name,
            "message": "Plugin installed"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{plugin_id}")
async def get_plugin(plugin_id: str):
    """
    获取插件详情
    
    v2.2: 插件系统
    """
    plugin = plugin_manager.get_plugin(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {
        "plugin_id": plugin.plugin_id,
        "name": plugin.name,
        "version": plugin.version,
        "description": plugin.description,
        "type": plugin.plugin_type,
        "enabled": plugin.enabled,
        "settings": plugin.settings,
        "loaded_at": plugin.loaded_at.isoformat() if plugin.loaded_at else None
    }

@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str):
    """
    启用插件
    
    v2.2: 插件系统
    """
    result = plugin_manager.enable_plugin(plugin_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {"message": "Plugin enabled"}

@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str):
    """
    禁用插件
    
    v2.2: 插件系统
    """
    result = plugin_manager.disable_plugin(plugin_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {"message": "Plugin disabled"}

@router.put("/{plugin_id}/settings")
async def update_settings(plugin_id: str, request: UpdateSettingsModel):
    """
    更新插件设置
    
    v2.2: 插件系统
    """
    result = plugin_manager.update_settings(plugin_id, request.settings)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {"message": "Settings updated"}

@router.get("/{plugin_id}/load")
async def load_plugin(plugin_id: str):
    """
    加载插件模块
    
    v2.2: 插件系统
    """
    try:
        module = plugin_manager.load_plugin_module(plugin_id)
        return {
            "plugin_id": plugin_id,
            "module": str(module),
            "message": "Plugin loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/hooks/{hook_name}")
async def get_hooks(hook_name: str):
    """
    获取钩子列表
    
    v2.2: 插件系统
    """
    hooks = plugin_manager.hooks.get(hook_name, [])
    
    return {
        "hook_name": hook_name,
        "total": len(hooks),
        "hooks": [
            {
                "hook_id": h.hook_id,
                "plugin_id": h.plugin_id,
                "priority": h.priority
            }
            for h in hooks
        ]
    }

@router.post("/hooks/{hook_name}/trigger")
async def trigger_hook(hook_name: str, data: Dict):
    """
    手动触发钩子
    
    v2.2: 插件系统
    """
    results = plugin_manager.call_hooks(hook_name, data)
    
    return {
        "hook_name": hook_name,
        "results_count": len(results),
        "results": results
    }
