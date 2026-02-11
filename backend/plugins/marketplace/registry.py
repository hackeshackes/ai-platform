"""
Plugin Registry - Plugin注册表
负责Plugin的注册、发现和元数据管理
"""

import json
import hashlib
import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .models import PluginInfo, PluginStatus, PluginStats
from .manifest import PluginManifest, ManifestParser

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Plugin注册表"""
    
    def __init__(self, registry_dir: str = "./data/plugins/registry"):
        self.registry_dir = registry_dir
        self.plugins: Dict[str, PluginInfo] = {}
        self.stats: Dict[str, PluginStats] = {}
        self.plugins_file = os.path.join(registry_dir, "plugins.json")
        self.stats_file = os.path.join(registry_dir, "stats.json")
        self.index_file = os.path.join(registry_dir, "registry.json")
        
        os.makedirs(registry_dir, exist_ok=True)
        self._load()
    
    def _load(self) -> None:
        """加载注册表数据"""
        # 加载Plugin信息
        if os.path.exists(self.plugins_file):
            try:
                with open(self.plugins_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for plugin_data in data.get('plugins', []):
                        plugin = PluginInfo(**plugin_data)
                        self.plugins[plugin.plugin_id] = plugin
                logger.info(f"Loaded {len(self.plugins)} plugins from registry")
            except Exception as e:
                logger.error(f"Failed to load plugins registry: {e}")
        
        # 加载统计数据
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for plugin_id, stat_data in data.items():
                        self.stats[plugin_id] = PluginStats(**stat_data)
            except Exception as e:
                logger.error(f"Failed to load stats registry: {e}")
    
    def _save(self) -> None:
        """保存注册表数据"""
        # 保存Plugin信息
        plugins_data = {
            'version': '1.0.0',
            'updated_at': datetime.now().isoformat(),
            'plugins': [p.to_dict() for p in self.plugins.values()]
        }
        with open(self.plugins_file, 'w', encoding='utf-8') as f:
            json.dump(plugins_data, f, indent=2, ensure_ascii=False)
        
        # 保存统计数据
        stats_data = {
            plugin_id: stat.to_dict()
            for plugin_id, stat in self.stats.items()
        }
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    def generate_plugin_id(self, name: str) -> str:
        """
        生成唯一Plugin ID
        
        Args:
            name: Plugin名称
        
        Returns:
            唯一Plugin ID
        """
        # 转换为小写，替换特殊字符
        clean_name = name.lower().strip()
        clean_name = re.sub(r'[^a-z0-9]+', '-', clean_name)
        clean_name = clean_name.strip('-')
        
        if not clean_name:
            clean_name = f"plugin-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 添加时间戳确保唯一性
        timestamp = datetime.now().strftime('%Y%m%d')
        plugin_id = f"{clean_name}-{timestamp}"
        
        # 如果已存在，添加随机后缀
        if plugin_id in self.plugins:
            import random
            plugin_id = f"{clean_name}-{timestamp}-{random.randint(1000, 9999)}"
        
        return plugin_id
    
    def register(self, manifest: PluginManifest, plugin_id: str = None) -> bool:
        """
        注册Plugin
        
        Args:
            manifest: Plugin清单
            plugin_id: 可选的Plugin ID，如果不提供则自动生成
        
        Returns:
            是否注册成功
        """
        if plugin_id is None:
            plugin_id = self.generate_plugin_id(manifest.name)
        
        if plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin_id} already exists")
            return False
        
        plugin_info = PluginInfo.from_manifest(manifest, plugin_id)
        self.plugins[plugin_id] = plugin_info
        
        if plugin_id not in self.stats:
            self.stats[plugin_id] = PluginStats(plugin_id=plugin_id)
        
        self._save()
        logger.info(f"Registered plugin: {plugin_info.display_name} v{plugin_info.version} ({plugin_id})")
        return True
    
    def unregister(self, plugin_id: str) -> bool:
        """
        注销Plugin
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            是否注销成功
        """
        if plugin_id not in self.plugins:
            return False
        
        del self.plugins[plugin_id]
        self._save()
        logger.info(f"Unregistered plugin: {plugin_id}")
        return True
    
    def get(self, plugin_id: str) -> Optional[PluginInfo]:
        """
        获取Plugin信息
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            PluginInfo对象，未找到返回None
        """
        return self.plugins.get(plugin_id)
    
    def update(self, plugin_id: str, **kwargs) -> bool:
        """
        更新Plugin信息
        
        Args:
            plugin_id: Plugin ID
            **kwargs: 更新的字段
        
        Returns:
            是否更新成功
        """
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        for key, value in kwargs.items():
            if hasattr(plugin, key):
                setattr(plugin, key, value)
        
        plugin.updated_at = datetime.now().isoformat()
        self._save()
        return True
    
    def list_all(self,
                category: Optional[str] = None,
                status: Optional[PluginStatus] = None,
                tags: Optional[List[str]] = None,
                author: Optional[str] = None,
                page: int = 1,
                page_size: int = 20) -> Dict:
        """
        列出所有Plugin
        
        Args:
            category: 分类过滤
            status: 状态过滤
            tags: 标签过滤
            author: 作者过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            Plugin列表和分页信息
        """
        plugins = list(self.plugins.values())
        
        # 应用过滤器
        if category:
            plugins = [p for p in plugins if p.category == category]
        
        if status:
            status_val = status.value if isinstance(status, PluginStatus) else status
            plugins = [p for p in plugins if p.status == status_val]
        
        if tags:
            plugins = [p for p in plugins if any(tag in p.tags for tag in tags)]
        
        if author:
            plugins = [p for p in plugins if p.author == author]
        
        # 统计分类
        category_counts = {}
        for plugin in self.plugins.values():
            category_counts[plugin.category] = category_counts.get(plugin.category, 0) + 1
        
        # 分页
        total = len(plugins)
        start = (page - 1) * page_size
        end = start + page_size
        
        plugins = plugins[start:end]
        
        return {
            'plugins': [p.to_dict() for p in plugins],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            },
            'categories': category_counts
        }
    
    def search(self,
              query: str,
              category: Optional[str] = None,
              tags: Optional[List[str]] = None,
              min_rating: Optional[float] = None,
              sort_by: str = "relevance",
              page: int = 1,
              page_size: int = 20) -> Dict:
        """
        搜索Plugin
        
        Args:
            query: 搜索关键词
            category: 分类过滤
            tags: 标签过滤
            min_rating: 最低评分
            sort_by: 排序方式
            page: 页码
            page_size: 每页数量
        
        Returns:
            搜索结果
        """
        query = query.lower().strip()
        results = []
        
        for plugin in self.plugins.values():
            score = 0
            
            # 关键词匹配
            if query in plugin.name.lower():
                score += 10
            if query in plugin.display_name.lower():
                score += 8
            if query in plugin.description.lower():
                score += 5
            if query in plugin.author.lower():
                score += 3
            
            # 标签匹配
            for tag in plugin.tags:
                if query in tag.lower():
                    score += 4
            
            # 分类匹配
            if query in plugin.category.lower():
                score += 2
            
            if score > 0:
                stats = self.stats.get(plugin.plugin_id, PluginStats(plugin_id=plugin.plugin_id))
                results.append({
                    'plugin': plugin,
                    'score': score,
                    'rating': stats.average_rating,
                    'downloads': stats.total_downloads
                })
        
        # 应用过滤器
        filtered = []
        for item in results:
            plugin = item['plugin']
            
            if category and plugin.category != category:
                continue
            
            if tags and not any(tag in plugin.tags for tag in tags):
                continue
            
            if min_rating and item['rating'] < min_rating:
                continue
            
            filtered.append(item)
        
        # 排序
        if sort_by == "relevance":
            filtered.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == "rating":
            filtered.sort(key=lambda x: (x['rating'], x['downloads']), reverse=True)
        elif sort_by == "downloads":
            filtered.sort(key=lambda x: x['downloads'], reverse=True)
        elif sort_by == "recent":
            filtered.sort(key=lambda x: plugin.updated_at, reverse=True)
        elif sort_by == "name":
            filtered.sort(key=lambda x: x['plugin'].name)
        
        # 分页
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        
        paginated_results = filtered[start:end]
        
        # 统计
        facets = {
            'categories': {},
            'tags': {},
        }
        for plugin in self.plugins.values():
            facets['categories'][plugin.category] = facets['categories'].get(plugin.category, 0) + 1
            for tag in plugin.tags:
                facets['tags'][tag] = facets['tags'].get(tag, 0) + 1
        
        return {
            'plugins': [item['plugin'].to_dict() for item in paginated_results],
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
            'query': query,
            'facets': facets
        }
    
    def get_categories(self) -> List[Dict]:
        """
        获取所有分类及数量
        
        Returns:
            分类列表
        """
        category_counts = {}
        for plugin in self.plugins.values():
            category_counts[plugin.category] = category_counts.get(plugin.category, 0) + 1
        
        category_info = []
        for category, count in category_counts.items():
            category_info.append({
                'category': category,
                'count': count,
                'display_name': category.replace('_', ' ').title()
            })
        
        return sorted(category_info, key=lambda x: (-x['count'], x['category']))
    
    def increment_downloads(self, plugin_id: str) -> bool:
        """
        增加下载计数
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            是否成功
        """
        if plugin_id not in self.plugins:
            return False
        
        if plugin_id not in self.stats:
            self.stats[plugin_id] = PluginStats(plugin_id=plugin_id)
        
        self.stats[plugin_id].total_downloads += 1
        self.stats[plugin_id].last_updated = datetime.now().isoformat()
        self._save()
        return True
    
    def validate_plugin(self, plugin_dir: str) -> tuple:
        """
        验证Plugin目录
        
        Args:
            plugin_dir: Plugin目录路径
        
        Returns:
            (成功标志, PluginInfo或错误信息)
        """
        # 查找manifest文件
        manifest_path = ManifestParser.find_manifest(plugin_dir)
        if not manifest_path:
            return (False, "No plugin.yaml found in plugin directory")
        
        # 解析manifest
        try:
            manifest = ManifestParser.parse(manifest_path)
        except Exception as e:
            return (False, f"Failed to parse manifest: {e}")
        
        # 创建plugin info
        plugin_id = self.generate_plugin_id(manifest.name)
        plugin_info = PluginInfo.from_manifest(manifest, plugin_id)
        
        return (True, plugin_info)
