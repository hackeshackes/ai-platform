"""
Plugin Marketplace API Endpoints - v1.0

Plugin市场API端点

提供Plugin发布、安装、卸载、搜索等功能
"""
import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from loguru import logger

from plugins.marketplace import (
    PluginCategory,
    PluginManifest,
    PluginRegistry,
    PluginInstaller,
    PluginPublisher
)

# 全局实例
registry = PluginRegistry()
installer = PluginInstaller(registry=registry)
publisher = PluginPublisher(registry=registry)


router = APIRouter(prefix="/plugins/marketplace", tags=["Plugin Marketplace"])


# ============ 数据模型 ============

class PluginInfo(BaseModel):
    """Plugin基本信息"""
    id: str
    name: str
    display_name: str
    description: str
    version: str
    author: str
    category: str
    tags: List[str] = []
    downloads: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    installed: bool = False
    created_at: str
    updated_at: str


class PluginDetail(PluginInfo):
    """Plugin详细信息"""
    permissions: List[str] = []
    dependencies: Dict[str, str] = {}
    homepage: Optional[str] = None
    repository: Optional[str] = None
    readme: Optional[str] = None


class PluginSearchResult(BaseModel):
    """Plugin搜索结果"""
    plugins: List[PluginInfo]
    total: int
    page: int
    page_size: int


class PublishRequest(BaseModel):
    """发布Plugin请求"""
    name: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')
    category: PluginCategory
    tags: List[str] = []
    permissions: List[str] = []
    dependencies: Dict[str, str] = {}
    homepage: Optional[str] = None
    repository: Optional[str] = None
    content: str = Field(..., description="Plugin代码")


class InstallRequest(BaseModel):
    """安装Plugin请求"""
    plugin_id: str = Field(..., description="Plugin ID")
    version: Optional[str] = Field(None, description="指定版本，默认最新")


class RateRequest(BaseModel):
    """评分请求"""
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = Field(None, max_length=1000)


# ============ API端点 ============

@router.get("/", response_model=List[PluginInfo])
async def list_plugins(
    category: Optional[PluginCategory] = Query(None, description="按分类筛选"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("downloads", regex="^(downloads|rating|created_at|updated_at)$")
) -> List[PluginInfo]:
    """
    获取Plugin列表
    
    - **category**: 按分类筛选 (tool, agent, integration, ui, visualization, data_source)
    - **page**: 页码
    - **page_size**: 每页数量
    """
    result = registry.list_all(
        category=category.value if category else None,
        page=page,
        page_size=page_size
    )
    
    plugins_data = result.get('plugins', [])
    
    # 获取已安装的plugin IDs
    installed_ids = set(installer.list_installed())
    
    return [
        PluginInfo(
            id=p.get('plugin_id', p.get('id')),
            name=p.get('name', ''),
            display_name=p.get('display_name', p.get('name', '')),
            description=p.get('description', ''),
            version=p.get('version', '1.0.0'),
            author=p.get('author', ''),
            category=p.get('category', 'tool'),
            tags=p.get('tags', []),
            downloads=p.get('downloads', 0),
            rating=p.get('rating', 0.0),
            reviews_count=p.get('reviews_count', 0),
            installed=p.get('plugin_id', p.get('id')) in installed_ids,
            created_at=p.get('created_at', ''),
            updated_at=p.get('updated_at', '')
        )
        for p in plugins_data
    ]


@router.get("/installed", response_model=List[PluginInfo])
async def list_installed_plugins() -> List[PluginInfo]:
    """获取已安装的Plugin列表"""
    plugins_data = installer.list_installed()
    return [
        PluginInfo(
            id=p.get('plugin_id', p.get('id')),
            name=p.get('name', ''),
            display_name=p.get('display_name', p.get('name', '')),
            description=p.get('description', ''),
            version=p.get('version', '1.0.0'),
            author=p.get('author', ''),
            category=p.get('category', 'tool'),
            tags=p.get('tags', []),
            downloads=p.get('downloads', 0),
            rating=p.get('rating', 0.0),
            reviews_count=p.get('reviews_count', 0),
            installed=True,
            created_at=p.get('created_at', ''),
            updated_at=p.get('updated_at', '')
        )
        for p in plugins_data
    ]


@router.get("/categories")
async def get_categories() -> Dict[str, Dict[str, Any]]:
    """获取Plugin分类列表"""
    categories = {}
    for cat in PluginCategory:
        plugins = registry.list_all(category=cat.value)
        categories[cat.value] = {
            "name": cat.value,
            "count": len(plugins),
            "description": get_category_description(cat)
        }
    return categories


@router.get("/search", response_model=PluginSearchResult)
async def search_plugins(
    q: str = Query(..., min_length=1, max_length=200, description="搜索关键词"),
    category: Optional[PluginCategory] = Query(None),
    tags: Optional[str] = Query(None, description="标签筛选，逗号分隔"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
) -> PluginSearchResult:
    """
    搜索Plugin
    
    - **q**: 搜索关键词 (支持名称、描述、标签)
    - **category**: 按分类筛选
    - **tags**: 按标签筛选 (逗号分隔)
    - **page**: 页码
    - **page_size**: 每页数量
    """
    # 解析标签
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    result = registry.search(
        query=q,
        category=category.value if category else None,
        tags=tag_list
    )
    
    plugins_data = result.get('plugins', [])
    
    # 统计总数
    total = len(plugins_data)
    
    # 切片分页
    start = (page - 1) * page_size
    end = start + page_size
    paginated = plugins_data[start:end]
    
    installed_ids = set(installer.list_installed())
    
    return PluginSearchResult(
        plugins=[
            PluginInfo(
                id=p.get('plugin_id', p.get('id')),
                name=p.get('name', ''),
                display_name=p.get('display_name', p.get('name', '')),
                description=p.get('description', ''),
                version=p.get('version', '1.0.0'),
                author=p.get('author', ''),
                category=p.get('category', 'tool'),
                tags=p.get('tags', []),
                downloads=p.get('downloads', 0),
                rating=p.get('rating', 0.0),
                reviews_count=p.get('reviews_count', 0),
                installed=p.get('plugin_id', p.get('id')) in installed_ids,
                created_at=p.get('created_at', ''),
                updated_at=p.get('updated_at', '')
            )
            for p in paginated
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{plugin_id}", response_model=PluginDetail)
async def get_plugin_detail(plugin_id: str) -> PluginDetail:
    """获取Plugin详情"""
    plugin = registry.get_plugin(plugin_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")
    
    installed = plugin_id in installer.list_installed()
    
    return PluginDetail(
        id=plugin.id,
        name=plugin.name,
        display_name=plugin.display_name,
        description=plugin.description,
        version=plugin.version,
        author=plugin.author,
        category=plugin.category.value,
        tags=plugin.tags,
        downloads=plugin.downloads,
        rating=plugin.rating,
        reviews_count=plugin.reviews_count,
        installed=installed,
        permissions=plugin.permissions,
        dependencies=plugin.dependencies,
        homepage=plugin.homepage,
        repository=plugin.repository,
        readme=plugin.readme,
        created_at=plugin.created_at.isoformat(),
        updated_at=plugin.updated_at.isoformat()
    )


@router.get("/{plugin_id}/versions")
async def get_plugin_versions(plugin_id: str) -> List[Dict[str, Any]]:
    """获取Plugin版本历史"""
    versions = registry.get_version_history(plugin_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")
    return versions


@router.post("/publish")
async def publish_plugin(request: PublishRequest) -> Dict[str, Any]:
    """
    发布新Plugin
    
    提供Plugin代码包进行发布。
    """
    try:
        result = publisher.publish(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            version=request.version,
            category=request.category,
            tags=request.tags,
            permissions=request.permissions,
            dependencies=request.dependencies,
            homepage=request.homepage,
            repository=request.repository,
            content=request.content
        )
        
        logger.info(f"Plugin published: {request.name} v{request.version}")
        
        return {
            "success": True,
            "plugin_id": result.id,
            "message": f"Plugin '{request.display_name}' published successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to publish plugin: {e}")
        raise HTTPException(status_code=500, detail="Failed to publish plugin")


@router.post("/install")
async def install_plugin(request: InstallRequest) -> Dict[str, Any]:
    """
    安装Plugin
    
    - **plugin_id**: Plugin ID
    - **version**: 指定版本，默认安装最新版本
    """
    try:
        result = installer.install(
            plugin_id=request.plugin_id,
            version=request.version
        )
        
        logger.info(f"Plugin installed: {request.plugin_id} v{request.version or 'latest'}")
        
        return {
            "success": True,
            "plugin_id": request.plugin_id,
            "version": result.version,
            "message": f"Plugin '{request.plugin_id}' installed successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to install plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to install plugin: {str(e)}")


@router.post("/uninstall")
async def uninstall_plugin(plugin_id: str) -> Dict[str, Any]:
    """
    卸载Plugin
    
    - **plugin_id**: 要卸载的Plugin ID
    """
    try:
        installer.uninstall(plugin_id)
        
        logger.info(f"Plugin uninstalled: {plugin_id}")
        
        return {
            "success": True,
            "plugin_id": plugin_id,
            "message": f"Plugin '{plugin_id}' uninstalled successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to uninstall plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to uninstall plugin: {str(e)}")


@router.post("/{plugin_id}/rate")
async def rate_plugin(
    plugin_id: str,
    request: RateRequest
) -> Dict[str, Any]:
    """
    评分Plugin
    
    - **rating**: 评分 (1-5)
    - **review**: 评价文字 (可选)
    """
    try:
        registry.add_rating(
            plugin_id=plugin_id,
            rating=request.rating,
            review=request.review
        )
        
        return {
            "success": True,
            "message": "Rating submitted successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ 辅助函数 ============

def get_category_description(category: PluginCategory) -> str:
    """获取分类描述"""
    descriptions = {
        PluginCategory.TOOL: "工具类Plugin，提供各种实用功能",
        PluginCategory.AGENT: "Agent类Plugin，增强AI能力",
        PluginCategory.INTEGRATION: "集成类Plugin，连接外部服务",
        PluginCategory.UI: "UI组件Plugin，扩展用户界面",
        PluginCategory.VISUALIZATION: "可视化Plugin，提供图表和仪表盘",
        PluginCategory.DATA_SOURCE: "数据源Plugin，连接各类数据"
    }
    return descriptions.get(category, "未知分类")
