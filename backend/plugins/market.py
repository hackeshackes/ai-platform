"""
Plugin Market - Plugin市场
提供Plugin浏览、搜索、评分和评论功能
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
import logging

from .registry import PluginRegistry, PluginMetadata
from .manager import PluginManager

logger = logging.getLogger(__name__)


@dataclass
class PluginRating:
    """Plugin评分"""
    plugin_id: str
    user_id: str
    rating: int  # 1-5
    review: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PluginStats:
    """Plugin统计信息"""
    plugin_id: str
    total_downloads: int = 0
    total_ratings: int = 0
    average_rating: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class PluginMarket:
    """Plugin市场"""
    
    def __init__(self, 
                 registry: PluginRegistry,
                 manager: PluginManager,
                 market_data_dir: str = "./data/plugins/market"):
        self.registry = registry
        self.manager = manager
        self.market_data_dir = market_data_dir
        self.ratings_file = os.path.join(market_data_dir, "ratings.json")
        self.stats_file = os.path.join(market_data_dir, "stats.json")
        self.featured_file = os.path.join(market_data_dir, "featured.json")
        
        os.makedirs(market_data_dir, exist_ok=True)
        self._load_ratings()
        self._load_stats()
    
    def _load_ratings(self) -> None:
        """加载评分数据"""
        self.ratings: Dict[str, List[PluginRating]] = {}
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, 'r') as f:
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
            data[plugin_id] = [asdict(r) for r in rating_list]
        with open(self.ratings_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_stats(self) -> None:
        """加载统计数据"""
        self.stats: Dict[str, PluginStats] = {}
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    for plugin_id, stat_data in data.items():
                        self.stats[plugin_id] = PluginStats(**stat_data)
            except Exception as e:
                logger.error(f"Failed to load stats: {e}")
    
    def _save_stats(self) -> None:
        """保存统计数据"""
        data = {}
        for plugin_id, stat in self.stats.items():
            data[plugin_id] = asdict(stat)
        with open(self.stats_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def browse_market(self,
                     category: Optional[str] = None,
                     sort_by: str = "popularity",
                     page: int = 1,
                     page_size: int = 20) -> Dict:
        """浏览市场"""
        plugins = self.registry.list_all(category=category)
        
        market_plugins = []
        for plugin in plugins:
            stats = self.stats.get(plugin.plugin_id, PluginStats(plugin_id=plugin.plugin_id))
            is_installed = self.manager.is_installed(plugin.plugin_id)
            
            market_plugins.append({
                'metadata': plugin.to_dict(),
                'stats': asdict(stats),
                'is_installed': is_installed
            })
        
        if sort_by == "popularity":
            market_plugins.sort(key=lambda x: x['stats']['total_downloads'], reverse=True)
        elif sort_by == "rating":
            market_plugins.sort(key=lambda x: x['stats']['average_rating'], reverse=True)
        elif sort_by == "recent":
            market_plugins.sort(key=lambda x: x['metadata']['updated_at'], reverse=True)
        elif sort_by == "name":
            market_plugins.sort(key=lambda x: x['metadata']['name'])
        
        total = len(market_plugins)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'plugins': market_plugins[start:end],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            },
            'categories': self.registry.get_categories()
        }
    
    def search_market(self, 
                     query: str,
                     category: Optional[str] = None,
                     min_rating: Optional[float] = None,
                     page: int = 1,
                     page_size: int = 20) -> Dict:
        """搜索市场"""
        results = self.registry.search(query)
        
        filtered = []
        for plugin in results:
            if category and plugin.category != category:
                continue
            
            stats = self.stats.get(plugin.plugin_id)
            if min_rating and stats and stats.average_rating < min_rating:
                continue
            
            is_installed = self.manager.is_installed(plugin.plugin_id)
            filtered.append({
                'metadata': plugin.to_dict(),
                'stats': asdict(stats) if stats else None,
                'is_installed': is_installed
            })
        
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'plugins': filtered[start:end],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total
            },
            'query': query
        }
    
    def get_plugin_details(self, plugin_id: str) -> Optional[Dict]:
        """获取Plugin详情"""
        metadata = self.registry.get(plugin_id)
        if not metadata:
            return None
        
        stats = self.stats.get(plugin_id, PluginStats(plugin_id=plugin_id))
        ratings = self.ratings.get(plugin_id, [])
        is_installed = self.manager.is_installed(plugin_id)
        
        return {
            'metadata': metadata.to_dict(),
            'stats': asdict(stats),
            'ratings': [asdict(r) for r in ratings[-10:]],
            'total_reviews': len(ratings),
            'is_installed': is_installed
        }
    
    def rate_plugin(self, 
                   plugin_id: str, 
                   user_id: str, 
                   rating: int,
                   review: Optional[str] = None) -> Dict:
        """评分Plugin"""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        metadata = self.registry.get(plugin_id)
        if not metadata:
            raise ValueError(f"Plugin not found: {plugin_id}")
        
        plugin_rating = PluginRating(
            plugin_id=plugin_id,
            user_id=user_id,
            rating=rating,
            review=review
        )
        
        if plugin_id not in self.ratings:
            self.ratings[plugin_id] = []
        
        self.ratings[plugin_id] = [
            r for r in self.ratings[plugin_id] if r.user_id != user_id
        ]
        
        self.ratings[plugin_id].append(plugin_rating)
        self._save_ratings()
        
        self._update_stats(plugin_id)
        
        return {'status': 'success', 'rating': rating}
    
    def _update_stats(self, plugin_id: str) -> None:
        """更新Plugin统计"""
        ratings = self.ratings.get(plugin_id, [])
        
        if not ratings:
            return
        
        avg_rating = sum(r.rating for r in ratings) / len(ratings)
        
        stats = self.stats.get(plugin_id, PluginStats(plugin_id=plugin_id))
        stats.total_ratings = len(ratings)
        stats.average_rating = round(avg_rating, 2)
        stats.last_updated = datetime.now().isoformat()
        
        self.stats[plugin_id] = stats
        self._save_stats()
    
    def download_plugin(self, plugin_id: str) -> Dict:
        """下载Plugin"""
        metadata = self.registry.get(plugin_id)
        if not metadata:
            raise ValueError(f"Plugin not found: {plugin_id}")
        
        if plugin_id not in self.stats:
            self.stats[plugin_id] = PluginStats(plugin_id=plugin_id)
        
        self.stats[plugin_id].total_downloads += 1
        self._save_stats()
        
        return {
            'status': 'success',
            'plugin_id': plugin_id,
            'download_url': f"/api/v1/plugins/download/{plugin_id}",
            'metadata': metadata.to_dict()
        }
    
    def get_featured_plugins(self) -> List[Dict]:
        """获取推荐Plugin"""
        if os.path.exists(self.featured_file):
            try:
                with open(self.featured_file, 'r') as f:
                    featured_ids = json.load(f)
                    return [
                        self.get_plugin_details(pid) 
                        for pid in featured_ids 
                        if self.registry.get(pid)
                    ]
            except Exception as e:
                logger.error(f"Failed to load featured: {e}")
        
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: (x[1].average_rating, x[1].total_downloads),
            reverse=True
        )
        
        return [
            self.get_plugin_details(plugin_id)
            for plugin_id, _ in sorted_stats[:5]
        ]
    
    def set_featured_plugins(self, plugin_ids: List[str]) -> Dict:
        """设置推荐Plugin"""
        valid_ids = [pid for pid in plugin_ids if self.registry.get(pid)]
        with open(self.featured_file, 'w') as f:
            json.dump(valid_ids, f)
        
        return {'status': 'success', 'featured': valid_ids}
