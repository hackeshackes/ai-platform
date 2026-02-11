"""
Plugin Marketplace - Plugin市场模块

提供完整的Plugin市场功能:
- Plugin清单解析 (manifest.py)
- Plugin注册表 (registry.py)
- Plugin安装/卸载 (installer.py)
- Plugin发布 (publisher.py)
- Plugin市场核心 (marketplace.py)

核心类:
- PluginMarketplace: Plugin市场主类
- PluginRegistry: Plugin注册表
- PluginInstaller: Plugin安装器
- PluginPublisher: Plugin发布器
- PluginManifest: Plugin清单
"""

from .manifest import (
    PluginManifest,
    ManifestParser,
    ManifestValidator,
    PluginCategory,
    HookConfig,
)

from .models import (
    PluginInfo,
    PluginStatus,
    PluginStats,
    PluginRating,
    InstallationInfo,
    SearchFilters,
    SearchResult,
    CategoryInfo,
    PublishResult,
    InstallResult,
    UninstallResult,
)

from .registry import PluginRegistry

from .installer import PluginInstaller, DependencyManager

from .publisher import PluginPublisher, VersionManager

from .marketplace import PluginMarketplace

__version__ = "1.0.0"

__all__ = [
    # Manifest
    'PluginManifest',
    'ManifestParser',
    'ManifestValidator',
    'PluginCategory',
    'HookConfig',
    
    # Models
    'PluginInfo',
    'PluginStatus',
    'PluginStats',
    'PluginRating',
    'InstallationInfo',
    'SearchFilters',
    'SearchResult',
    'CategoryInfo',
    'PublishResult',
    'InstallResult',
    'UninstallResult',
    
    # Core
    'PluginRegistry',
    'PluginInstaller',
    'DependencyManager',
    'PluginPublisher',
    'VersionManager',
    'PluginMarketplace',
]
