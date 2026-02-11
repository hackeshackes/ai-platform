"""
Plugin Marketplace - Plugin市场核心
整合所有Plugin市场功能
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .models import (
    PluginInfo, PluginStatus, PluginStats, PluginRating,
    SearchFilters, SearchResult, CategoryInfo,
    PublishResult, InstallResult, UninstallResult
)
from .manifest import PluginManifest, ManifestParser
from .registry import PluginRegistry
from .installer import PluginInstaller
from .publisher import PluginPublisher

logger = logging.getLogger(__name__)


class PluginMarketplace:
    """Plugin市场"""
    
    def __init__(self,
                 registry_dir: str = "./data/plugins/registry",
                 installed_dir: str = "./data/plugins/installed",
                 published_dir: str = "./data/plugins/published",
                 packages_dir: str = "./data/plugins/packages"):
        self.registry = PluginRegistry(registry_dir=registry_dir)
        self.installer = PluginInstaller(
            self.registry,
            plugins_dir=installed_dir
        )
        self.publisher = PluginPublisher(
            self.registry,
            publish_dir=published_dir,
            package_dir=packages_dir
        )
        
        # 评分数据
        self.ratings: Dict[str, List[PluginRating]] = {}
        self.ratings_dir = os.path.join(registry_dir, "ratings.json")
        self._load_ratings()
    
    def _load_ratings(self) -> None:
        """加载评分数据"""
        if os.path.exists(self.ratings_dir):
            try:
                with open(self.ratings_dir, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for plugin_id, rating_list in data.items():
                        self.ratings[plugin_id] = [
                            PluginRating(**r) for r in rating_list
                        ]
            except Exception as e:
                logger.error(f"Failed to load ratings: {e}")
    
    def _save_ratings(self) -> None:
        """保存评分数据"""
        data = {}
        for plugin_id, rating_list in self.ratings.items():
            data[plugin_id] = [r.to_dict() for r in rating_list]
        
        with open(self.ratings_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ==================== 浏览功能 ====================
    
    def browse_plugins(self,
                      category: str = None,
                      status: str = None,
                      page: int = 1,
                      page_size: int = 20,
                      sort_by: str = "popularity") -> Dict:
        """
        浏览Plugin市场
        
        Args:
            category: 分类过滤
            status: 状态过滤
            page: 页码
            page_size: 每页数量
            sort_by: 排序方式
        
        Returns:
            Plugin列表和分页信息
        """
        result = self.registry.list_all(
            category=category,
            page=page,
            page_size=page_size
        )
        
        plugins = result['plugins']
        
        # 添加安装状态和统计信息
        for plugin in plugins:
            plugin_id = plugin['plugin_id']
            plugin['is_installed'] = self.installer.is_installed(plugin_id)
            
            stats = self._get_plugin_stats(plugin_id)
            plugin['stats'] = stats.to_dict() if stats else None
        
        # 排序
        if sort_by == "popularity":
            plugins.sort(key=lambda x: x['stats']['total_downloads'] if x.get('stats') else 0, reverse=True)
        elif sort_by == "rating":
            plugins.sort(key=lambda x: x['stats']['average_rating'] if x.get('stats') else 0, reverse=True)
        elif sort_by == "recent":
            plugins.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        elif sort_by == "name":
            plugins.sort(key=lambda x: x.get('display_name', ''))
        
        return {
            'plugins': plugins,
            'pagination': result['pagination'],
            'categories': result['categories'],
            'sort_by': sort_by
        }
    
    def get_plugin(self, plugin_id: str) -> Optional[Dict]:
        """
        获取Plugin详情
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            Plugin详情
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            return None
        
        result = plugin.to_dict()
        
        # 添加安装状态
        result['is_installed'] = self.installer.is_installed(plugin_id)
        
        # 添加安装信息
        if result['is_installed']:
            installation = self.installer.get_installation(plugin_id)
            if installation:
                result['installation'] = installation.to_dict()
        
        # 添加统计信息
        stats = self._get_plugin_stats(plugin_id)
        result['stats'] = stats.to_dict() if stats else None
        
        # 添加评分
        ratings = self.ratings.get(plugin_id, [])
        result['ratings'] = [r.to_dict() for r in ratings[-10:]]
        result['total_reviews'] = len(ratings)
        
        return result
    
    def get_categories(self) -> List[Dict]:
        """
        获取所有分类
        
        Returns:
            分类列表
        """
        categories = self.registry.get_categories()
        
        # 添加显示名称
        category_names = {
            'tool': '工具类',
            'agent': 'Agent类',
            'integration': '集成类',
            'ui': 'UI组件',
            'visualization': '可视化',
            'data_source': '数据源',
        }
        
        result = []
        for cat in categories:
            result.append({
                'category': cat['category'],
                'count': cat['count'],
                'display_name': category_names.get(cat['category'], cat['category'].replace('_', ' ').title())
            })
        
        return result
    
    # ==================== 搜索功能 ====================
    
    def search_plugins(self,
                      query: str = None,
                      category: str = None,
                      tags: List[str] = None,
                      min_rating: float = None,
                      author: str = None,
                      page: int = 1,
                      page_size: int = 20,
                      sort_by: str = "relevance") -> Dict:
        """
        搜索Plugin
        
        Args:
            query: 搜索关键词
            category: 分类过滤
            tags: 标签过滤
            min_rating: 最低评分
            author: 作者过滤
            page: 页码
            page_size: 每页数量
            sort_by: 排序方式
        
        Returns:
            搜索结果
        """
        result = self.registry.search(
            query=query or "",
            category=category,
            tags=tags,
            min_rating=min_rating,
            sort_by=sort_by,
            page=page,
            page_size=page_size
        )
        
        # 添加安装状态
        for plugin in result['plugins']:
            plugin_id = plugin['plugin_id']
            plugin['is_installed'] = self.installer.is_installed(plugin_id)
            
            stats = self._get_plugin_stats(plugin_id)
            plugin['stats'] = stats.to_dict() if stats else None
        
        return result
    
    def search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """
        获取搜索建议
        
        Args:
            query: 搜索关键词
            limit: 返回数量
        
        Returns:
            建议列表
        """
        suggestions = set()
        query = query.lower()
        
        for plugin in self.registry.plugins.values():
            if query in plugin.name.lower():
                suggestions.add(plugin.name)
            if query in plugin.display_name.lower():
                suggestions.add(plugin.display_name)
            for tag in plugin.tags:
                if query in tag.lower():
                    suggestions.add(tag)
        
        return list(suggestions)[:limit]
    
    # ==================== 安装功能 ====================
    
    def install_plugin(self,
                      plugin_id: str,
                      source: str = None) -> InstallResult:
        """
        安装Plugin
        
        Args:
            plugin_id: Plugin ID
            source: 源路径
        
        Returns:
            InstallResult
        """
        result = self.installer.install(plugin_id, source=source)
        
        if result.success:
            # 增加下载计数
            self.registry.increment_downloads(plugin_id)
        
        return result
    
    def uninstall_plugin(self,
                        plugin_id: str,
                        remove_dependencies: bool = True) -> UninstallResult:
        """
        卸载Plugin
        
        Args:
            plugin_id: Plugin ID
            remove_dependencies: 是否移除依赖
        
        Returns:
            UninstallResult
        """
        return self.installer.uninstall(plugin_id, remove_dependencies=remove_dependencies)
    
    def update_plugin(self,
                     plugin_id: str,
                     source: str = None) -> InstallResult:
        """
        更新Plugin
        
        Args:
            plugin_id: Plugin ID
            source: 新版本源路径
        
        Returns:
            InstallResult
        """
        return self.installer.update(plugin_id, source=source)
    
    def activate_plugin(self, plugin_id: str) -> bool:
        """激活Plugin"""
        return self.installer.activate(plugin_id)
    
    def deactivate_plugin(self, plugin_id: str) -> bool:
        """停用Plugin"""
        return self.installer.deactivate(plugin_id)
    
    def list_installed(self) -> List[Dict]:
        """
        列出已安装的Plugin
        
        Returns:
            已安装Plugin列表
        """
        installations = self.installer.list_installed()
        
        result = []
        for installation in installations:
            plugin = self.registry.get(installation.plugin_id)
            if plugin:
                info = plugin.to_dict()
                info['installation'] = installation.to_dict()
                result.append(info)
        
        return result
    
    # ==================== 发布功能 ====================
    
    def publish_plugin(self,
                      plugin_dir: str,
                      version: str = None,
                      publish_notes: str = None) -> PublishResult:
        """
        发布Plugin
        
        Args:
            plugin_dir: Plugin目录
            version: 版本号
            publish_notes: 发布说明
        
        Returns:
            PublishResult
        """
        return self.publisher.publish(
            plugin_dir=plugin_dir,
            version=version,
            publish_notes=publish_notes
        )
    
    def unpublish_plugin(self,
                        plugin_id: str,
                        version: str = None) -> PublishResult:
        """
        取消发布Plugin
        
        Args:
            plugin_id: Plugin ID
            version: 版本号
        
        Returns:
            PublishResult
        """
        return self.publisher.unpublish(plugin_id, version=version)
    
    def list_published(self,
                      category: str = None,
                      author: str = None,
                      page: int = 1,
                      page_size: int = 20) -> Dict:
        """
        列出已发布的Plugin
        
        Args:
            category: 分类过滤
            author: 作者过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            Plugin列表
        """
        return self.publisher.list_published(
            category=category,
            author=author,
            page=page,
            page_size=page_size
        )
    
    def get_plugin_versions(self, plugin_id: str) -> Optional[Dict]:
        """
        获取Plugin版本历史
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            版本信息
        """
        return self.publisher.get_versions(plugin_id)
    
    # ==================== 评分功能 ====================
    
    def rate_plugin(self,
                   plugin_id: str,
                   user_id: str,
                   rating: int,
                   review: str = None) -> Dict:
        """
        评分Plugin
        
        Args:
            plugin_id: Plugin ID
            user_id: 用户ID
            rating: 评分 (1-5)
            review: 评论
        
        Returns:
            评分结果
        """
        if not 1 <= rating <= 5:
            return {'success': False, 'error': 'Rating must be between 1 and 5'}
        
        # 检查plugin是否存在
        if not self.registry.get(plugin_id):
            return {'success': False, 'error': 'Plugin not found'}
        
        plugin_rating = PluginRating(
            plugin_id=plugin_id,
            user_id=user_id,
            rating=rating,
            review=review
        )
        
        if plugin_id not in self.ratings:
            self.ratings[plugin_id] = []
        
        # 移除用户之前的评分
        self.ratings[plugin_id] = [
            r for r in self.ratings[plugin_id] if r.user_id != user_id
        ]
        
        # 添加新评分
        self.ratings[plugin_id].append(plugin_rating)
        self._save_ratings()
        
        # 更新统计
        self._update_plugin_stats(plugin_id)
        
        return {'success': True, 'rating': rating}
    
    def get_plugin_ratings(self,
                          plugin_id: str,
                          page: int = 1,
                          page_size: int = 20) -> Dict:
        """
        获取Plugin评分
        
        Args:
            plugin_id: Plugin ID
            page: 页码
            page_size: 每页数量
        
        Returns:
            评分列表
        """
        ratings = self.ratings.get(plugin_id, [])
        
        total = len(ratings)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'ratings': [r.to_dict() for r in ratings[start:end]],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            }
        }
    
    # ==================== 统计功能 ====================
    
    def _get_plugin_stats(self, plugin_id: str) -> Optional[PluginStats]:
        """获取Plugin统计"""
        plugin = self.registry.plugins.get(plugin_id)
        if not plugin:
            return None
        
        ratings = self.ratings.get(plugin_id, [])
        avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else 0.0
        
        return PluginStats(
            plugin_id=plugin_id,
            total_downloads=len(ratings),
            total_ratings=len(ratings),
            average_rating=round(avg_rating, 2),
            total_reviews=sum(1 for r in ratings if r.review)
        )
    
    def _update_plugin_stats(self, plugin_id: str) -> None:
        """更新Plugin统计"""
        pass  # 统计信息已在上方计算
    
    def get_marketplace_stats(self) -> Dict:
        """
        获取市场统计
        
        Returns:
            市场统计数据
        """
        total_plugins = len(self.registry.plugins)
        installed_count = len(self.installer.installations)
        published_count = len(self.publisher.versions)
        
        # 计算总评分
        total_ratings = sum(len(r) for r in self.ratings.values())
        avg_rating = 0
        if total_ratings > 0:
            all_ratings = [r.rating for ratings in self.ratings.values() for r in ratings]
            avg_rating = sum(all_ratings) / total_ratings
        
        return {
            'total_plugins': total_plugins,
            'installed_plugins': installed_count,
            'published_plugins': published_count,
            'total_ratings': total_ratings,
            'average_rating': round(avg_rating, 2),
            'categories_count': len(self.registry.get_categories())
        }
    
    def get_featured_plugins(self, limit: int = 5) -> List[Dict]:
        """
        获取推荐Plugin
        
        Args:
            limit: 返回数量
        
        Returns:
            推荐Plugin列表
        """
        # 按评分和下载量排序
        featured = []
        
        for plugin_id in self.registry.plugins:
            ratings = self.ratings.get(plugin_id, [])
            avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else 0
            
            featured.append({
                'plugin_id': plugin_id,
                'score': avg_rating * 10 + min(len(ratings), 100)
            })
        
        featured.sort(key=lambda x: x['score'], reverse=True)
        
        return [
            self.get_plugin(item['plugin_id'])
            for item in featured[:limit]
            if self.get_plugin(item['plugin_id'])
        ]
