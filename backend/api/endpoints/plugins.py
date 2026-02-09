"""
Plugin Marketplace API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'plugins/engine.py')

spec = importlib.util.spec_from_file_location("plugins_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    plugin_engine = module.plugin_marketplace_engine
    PluginCategory = module.PluginCategory
    PluginStatus = module.PluginStatus
except Exception as e:
    print(f"Failed to import plugins module: {e}")
    plugin_engine = None
    PluginCategory = None
    PluginStatus = None

router = APIRouter()

from pydantic import BaseModel

class RegisterPluginModel(BaseModel):
    name: str
    description: str
    category: str
    version: str
    author: str
    repository_url: Optional[str] = None
    tags: Optional[List[str]] = None

class AddVersionModel(BaseModel):
    version: str
    changelog: str = ""
    download_url: Optional[str] = None

class InstallPluginModel(BaseModel):
    version: str = "latest"
    config: Optional[Dict] = None

class UpdateConfigModel(BaseModel):
    config: Dict

class AddReviewModel(BaseModel):
    user_id: str
    rating: int
    title: str
    content: str

# ==================== 插件管理 ====================

@router.get("/plugins")
async def list_plugins(
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None
):
    """列出插件"""
    pcategory = PluginCategory(category) if category else None
    pstatus = PluginStatus(status) if status else None
    
    plugins = plugin_engine.list_plugins(
        category=pcategory,
        status=pstatus,
        search=search
    )
    
    return {
        "total": len(plugins),
        "plugins": [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "description": p.description,
                "category": p.category.value,
                "version": p.version,
                "status": p.status.value,
                "author": p.author,
                "rating": p.metrics.get("rating", 0),
                "downloads": p.metrics.get("downloads", 0)
            }
            for p in plugins
        ]
    }

@router.post("/plugins")
async def register_plugin(request: RegisterPluginModel):
    """注册插件"""
    try:
        category = PluginCategory(request.category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {request.category}")
    
    plugin = plugin_engine.register_plugin(
        name=request.name,
        description=request.description,
        category=category,
        version=request.version,
        author=request.author,
        repository_url=request.repository_url,
        tags=request.tags
    )
    
    return {
        "plugin_id": plugin.plugin_id,
        "name": plugin.name,
        "message": "Plugin registered"
    }

@router.get("/plugins/{plugin_id}")
async def get_plugin(plugin_id: str):
    """获取插件详情"""
    plugin = plugin_engine.get_plugin(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    versions = plugin_engine.get_versions(plugin_id)
    
    return {
        "plugin_id": plugin.plugin_id,
        "name": plugin.name,
        "description": plugin.description,
        "category": plugin.category.value,
        "version": plugin.version,
        "status": plugin.status.value,
        "author": plugin.author,
        "repository_url": plugin.repository_url,
        "tags": plugin.tags,
        "dependencies": plugin.dependencies,
        "config_schema": plugin.config_schema,
        "metrics": plugin.metrics,
        "versions_count": len(versions)
    }

@router.put("/plugins/{plugin_id}")
async def update_plugin(
    plugin_id: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """更新插件"""
    result = plugin_engine.update_plugin(
        plugin_id=plugin_id,
        description=description,
        tags=tags
    )
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return {"message": "Plugin updated"}

@router.post("/plugins/{plugin_id}/submit")
async def submit_for_review(plugin_id: str):
    """提交审核"""
    result = plugin_engine.submit_for_review(plugin_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return {"message": "Plugin submitted for review"}

@router.post("/plugins/{plugin_id}/approve")
async def approve_plugin(plugin_id: str):
    """批准插件"""
    result = plugin_engine.approve_plugin(plugin_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return {"message": "Plugin approved"}

@router.post("/plugins/{plugin_id}/reject")
async def reject_plugin(plugin_id: str, reason: str):
    """拒绝插件"""
    result = plugin_engine.reject_plugin(plugin_id, reason)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not found")
    return {"message": "Plugin rejected"}

# ==================== 版本管理 ====================

@router.get("/plugins/{plugin_id}/versions")
async def get_versions(plugin_id: str):
    """获取版本列表"""
    versions = plugin_engine.get_versions(plugin_id)
    
    return {
        "total": len(versions),
        "versions": [
            {
                "version_id": v.version_id,
                "version": v.version,
                "changelog": v.changelog,
                "file_size": v.file_size,
                "created_at": v.created_at.isoformat()
            }
            for v in versions
        ]
    }

@router.post("/plugins/{plugin_id}/versions")
async def add_version(plugin_id: str, request: AddVersionModel):
    """添加版本"""
    try:
        version = plugin_engine.add_version(
            plugin_id=plugin_id,
            version=request.version,
            changelog=request.changelog,
            download_url=request.download_url
        )
        return {
            "version_id": version.version_id,
            "version": version.version,
            "message": "Version added"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ==================== 评论管理 ====================

@router.get("/plugins/{plugin_id}/reviews")
async def get_reviews(plugin_id: str):
    """获取评论"""
    reviews = plugin_engine.get_reviews(plugin_id)
    
    return {
        "total": len(reviews),
        "reviews": [
            {
                "review_id": r.review_id,
                "user_id": r.user_id,
                "rating": r.rating,
                "title": r.title,
                "content": r.content,
                "created_at": r.created_at.isoformat()
            }
            for r in reviews
        ]
    }

@router.post("/plugins/{plugin_id}/reviews")
async def add_review(plugin_id: str, request: AddReviewModel):
    """添加评论"""
    if not 1 <= request.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    review = plugin_engine.add_review(
        plugin_id=plugin_id,
        user_id=request.user_id,
        rating=request.rating,
        title=request.title,
        content=request.content
    )
    
    return {
        "review_id": review.review_id,
        "rating": review.rating,
        "message": "Review added"
    }

# ==================== 插件安装 ====================

@router.get("/installed")
async def get_installed_plugins():
    """获取已安装插件"""
    installations = plugin_engine.get_installed_plugins()
    
    return {
        "total": len(installations),
        "plugins": [
            {
                "installation_id": i.installation_id,
                "plugin_id": i.plugin_id,
                "version": i.version,
                "status": i.status.value,
                "enabled": i.enabled
            }
            for i in installations
        ]
    }

@router.post("/plugins/{plugin_id}/install")
async def install_plugin(plugin_id: str, request: InstallPluginModel):
    """安装插件"""
    try:
        installation = plugin_engine.install_plugin(
            plugin_id=plugin_id,
            version=request.version,
            config=request.config
        )
        return {
            "installation_id": installation.installation_id,
            "status": installation.status.value,
            "message": "Plugin installed"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/plugins/{plugin_id}/uninstall")
async def uninstall_plugin(plugin_id: str):
    """卸载插件"""
    result = plugin_engine.uninstall_plugin(plugin_id)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not installed")
    return {"message": "Plugin uninstalled"}

@router.put("/plugins/{plugin_id}/config")
async def update_plugin_config(plugin_id: str, request: UpdateConfigModel):
    """更新插件配置"""
    result = plugin_engine.update_plugin_config(plugin_id, request.config)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not installed")
    return {"message": "Config updated"}

@router.put("/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str, enabled: bool):
    """启用/禁用插件"""
    result = plugin_engine.enable_plugin(plugin_id, enabled)
    if not result:
        raise HTTPException(status_code=404, detail="Plugin not installed")
    return {"message": f"Plugin {'enabled' if enabled else 'disabled'}"}

# ==================== 统计信息 ====================

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return plugin_engine.get_summary()

@router.get("/health")
async def plugins_health():
    """健康检查"""
    return {
        "status": "healthy",
        "plugins": len(plugin_engine.plugins),
        "installed": len(plugin_engine.installations)
    }
