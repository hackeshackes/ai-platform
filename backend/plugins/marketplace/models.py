"""
Plugin Models - Plugin模型定义
定义Plugin市场的数据模型
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .manifest import PluginCategory

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    """Plugin安装状态"""
    INSTALLED = "installed"
    AVAILABLE = "available"
    UPDATE_AVAILABLE = "update_available"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginInfo:
    """Plugin信息"""
    plugin_id: str
    name: str
    display_name: str
    version: str
    description: str
    author: str
    category: str
    tags: List[str] = field(default_factory=list)
    status: PluginStatus = PluginStatus.AVAILABLE
    installed_version: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    entry_point: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    icon: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'display_name': self.display_name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'category': self.category,
            'tags': self.tags,
            'status': self.status.value if isinstance(self.status, PluginStatus) else self.status,
            'installed_version': self.installed_version,
            'permissions': self.permissions,
            'dependencies': self.dependencies,
            'entry_point': self.entry_point,
            'homepage': self.homepage,
            'repository': self.repository,
            'license': self.license,
            'icon': self.icon,
            'screenshots': self.screenshots,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
    
    @classmethod
    def from_manifest(cls, manifest: 'PluginManifest', plugin_id: str = None) -> 'PluginInfo':
        """从Manifest创建PluginInfo"""
        if plugin_id is None:
            plugin_id = manifest.name
        
        return cls(
            plugin_id=plugin_id,
            name=manifest.name,
            display_name=manifest.display_name,
            version=manifest.version,
            description=manifest.description,
            author=manifest.author,
            category=manifest.category.value if isinstance(manifest.category, PluginCategory) else manifest.category,
            tags=manifest.tags,
            permissions=manifest.permissions,
            dependencies=manifest.dependencies,
            entry_point=manifest.entry_point,
            homepage=manifest.homepage,
            repository=manifest.repository,
            license=manifest.license,
            icon=manifest.icon,
            screenshots=manifest.screenshots,
        )


@dataclass
class PluginRating:
    """Plugin评分"""
    plugin_id: str
    user_id: str
    rating: int
    review: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'plugin_id': self.plugin_id,
            'user_id': self.user_id,
            'rating': self.rating,
            'review': self.review,
            'created_at': self.created_at,
        }


@dataclass
class PluginStats:
    """Plugin统计信息"""
    plugin_id: str
    total_downloads: int = 0
    total_ratings: int = 0
    average_rating: float = 0.0
    total_reviews: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'plugin_id': self.plugin_id,
            'total_downloads': self.total_downloads,
            'total_ratings': self.total_ratings,
            'average_rating': self.average_rating,
            'total_reviews': self.total_reviews,
            'last_updated': self.last_updated,
        }


@dataclass
class InstallationInfo:
    """Plugin安装信息"""
    plugin_id: str
    version: str
    installed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    install_path: str = ""
    is_active: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'plugin_id': self.plugin_id,
            'version': self.version,
            'installed_at': self.installed_at,
            'install_path': self.install_path,
            'is_active': self.is_active,
            'config': self.config,
        }


@dataclass
class SearchFilters:
    """搜索过滤器"""
    query: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    min_rating: Optional[float] = None
    author: Optional[str] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 20
    
    def to_dict(self) -> dict:
        return {
            'query': self.query,
            'category': self.category,
            'tags': self.tags,
            'min_rating': self.min_rating,
            'author': self.author,
            'sort_by': self.sort_by,
            'sort_order': self.sort_order,
            'page': self.page,
            'page_size': self.page_size,
        }


@dataclass
class SearchResult:
    """搜索结果"""
    plugins: List[PluginInfo]
    total: int
    page: int
    page_size: int
    total_pages: int
    facets: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'plugins': [p.to_dict() for p in self.plugins],
            'total': self.total,
            'page': self.page,
            'page_size': self.page_size,
            'total_pages': self.total_pages,
            'facets': self.facets,
        }


@dataclass
class CategoryInfo:
    """分类信息"""
    category: str
    count: int
    display_name: str = ""
    description: str = ""
    icon: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'category': self.category,
            'count': self.count,
            'display_name': self.display_name,
            'description': self.description,
            'icon': self.icon,
        }


@dataclass
class PublishResult:
    """发布结果"""
    success: bool
    plugin_id: Optional[str] = None
    version: Optional[str] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'plugin_id': self.plugin_id,
            'version': self.version,
            'message': self.message,
            'errors': self.errors,
        }


@dataclass
class InstallResult:
    """安装结果"""
    success: bool
    plugin_id: Optional[str] = None
    version: Optional[str] = None
    message: str = ""
    dependencies: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'plugin_id': self.plugin_id,
            'version': self.version,
            'message': self.message,
            'dependencies': self.dependencies,
            'errors': self.errors,
        }


@dataclass
class UninstallResult:
    """卸载结果"""
    success: bool
    plugin_id: str
    message: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'plugin_id': self.plugin_id,
            'message': self.message,
            'errors': self.errors,
        }
