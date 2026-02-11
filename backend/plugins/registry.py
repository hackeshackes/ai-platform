"""
Plugin Registry - Plugin注册表
负责Plugin的注册、发现和元数据管理
"""

import json
import hashlib
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin元数据"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    category: str
    tags: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    entry_point: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginMetadata':
        return cls(**data)


class PluginRegistry:
    """Plugin注册表"""
    
    def __init__(self, registry_path: str = "./data/plugins"):
        self.registry_path = registry_path
        self.plugins: Dict[str, PluginMetadata] = {}
        self.index_file = os.path.join(registry_path, "registry.json")
        os.makedirs(registry_path, exist_ok=True)
        self._load()
    
    def _load(self) -> None:
        """加载注册表"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for plugin_data in data.get('plugins', []):
                        metadata = PluginMetadata.from_dict(plugin_data)
                        self.plugins[metadata.plugin_id] = metadata
                logger.info(f"Loaded {len(self.plugins)} plugins from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save(self) -> None:
        """保存注册表"""
        data = {
            'version': '1.0.0',
            'updated_at': datetime.now().isoformat(),
            'plugins': [p.to_dict() for p in self.plugins.values()]
        }
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register(self, metadata: PluginMetadata) -> bool:
        """注册Plugin"""
        if metadata.plugin_id in self.plugins:
            logger.warning(f"Plugin {metadata.plugin_id} already exists")
            return False
        
        self.plugins[metadata.plugin_id] = metadata
        self._save()
        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
        return True
    
    def unregister(self, plugin_id: str) -> bool:
        """注销Plugin"""
        if plugin_id not in self.plugins:
            return False
        
        del self.plugins[plugin_id]
        self._save()
        logger.info(f"Unregistered plugin: {plugin_id}")
        return True
    
    def get(self, plugin_id: str) -> Optional[PluginMetadata]:
        """获取Plugin元数据"""
        return self.plugins.get(plugin_id)
    
    def list_all(self, category: Optional[str] = None, 
                 search_query: Optional[str] = None) -> List[PluginMetadata]:
        """列出所有Plugin"""
        plugins = list(self.plugins.values())
        
        if category:
            plugins = [p for p in plugins if p.category == category]
        
        if search_query:
            query = search_query.lower()
            plugins = [p for p in plugins if 
                       query in p.name.lower() or 
                       query in p.description.lower() or
                       query in p.tags]
        
        return sorted(plugins, key=lambda x: x.name)
    
    def search(self, query: str, limit: int = 20) -> List[PluginMetadata]:
        """搜索Plugin"""
        results = []
        query = query.lower()
        
        for plugin in self.plugins.values():
            score = 0
            if query in plugin.name.lower():
                score += 10
            if query in plugin.description.lower():
                score += 5
            if query in plugin.tags:
                score += 3
            if query in plugin.category.lower():
                score += 2
            
            if score > 0:
                results.append((plugin, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results[:limit]]
    
    def get_categories(self) -> Dict[str, int]:
        """获取Plugin分类统计"""
        categories = {}
        for plugin in self.plugins.values():
            categories[plugin.category] = categories.get(plugin.category, 0) + 1
        return categories
    
    def generate_plugin_id(self, name: str) -> str:
        """生成唯一Plugin ID"""
        base_id = name.lower().replace(' ', '-').replace('_', '-')
        timestamp = datetime.now().strftime('%Y%m%d')
        return f"{base_id}-{timestamp}"
    
    def validate_metadata(self, metadata: PluginMetadata) -> List[str]:
        """验证Plugin元数据"""
        errors = []
        
        if not metadata.name:
            errors.append("Plugin name is required")
        if not metadata.version:
            errors.append("Plugin version is required")
        if not metadata.author:
            errors.append("Plugin author is required")
        if not metadata.entry_point:
            errors.append("Plugin entry point is required")
        
        return errors


# 全局注册表实例
registry = PluginRegistry()
