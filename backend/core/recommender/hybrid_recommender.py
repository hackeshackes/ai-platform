"""
混合推荐模块 - hybrid_recommender.py

多路召回、特征融合和排序优化
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class RecommendationRequest:
    """推荐请求"""
    user_id: str
    context: Dict = field(default_factory=dict)
    item_type: str = None  # agent, pipeline, template, tutorial
    top_k: int = 10
    exclude_items: List[str] = field(default_factory=list)
    diversity_weight: float = 0.1
    freshness_weight: float = 0.1


@dataclass
class RecommendationResult:
    """推荐结果"""
    items: List[Dict]
    total_count: int
    response_time: float
    metadata: Dict = field(default_factory=dict)


class HybridRecommender:
    """混合推荐引擎"""
    
    def __init__(self):
        """初始化混合推荐器"""
        # 各推荐组件
        self.cf_recommender = None
        self.content_recommender = None
        self.user_profile = None
        self.item_features = None
        
        # 召回通道配置
        self.recall_channels = {
            'collaborative': {'weight': 0.3, 'enabled': True},
            'content_based': {'weight': 0.3, 'enabled': True},
            'popular': {'weight': 0.2, 'enabled': True},
            'recent': {'weight': 0.1, 'enabled': True},
            'similar_users': {'weight': 0.1, 'enabled': True}
        }
        
        # 缓存
        self.recall_cache: Dict[str, Tuple[float, List[tuple]]] = {}
        self.cache_ttl = 300  # 5分钟
        
        # 排序模型
        self.ranker_weights = {
            'cf_score': 0.25,
            'content_score': 0.25,
            'popularity': 0.20,
            'freshness': 0.15,
            'diversity': 0.15
        }
        
    def set_components(self, cf_recommender=None, content_recommender=None,
                       user_profile=None, item_features=None):
        """设置推荐组件"""
        self.cf_recommender = cf_recommender
        self.content_recommender = content_recommender
        self.user_profile = user_profile
        self.item_features = item_features
    
    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """
        执行混合推荐
        
        Args:
            request: 推荐请求
            
        Returns:
            推荐结果
        """
        start_time = time.time()
        
        # 多路召回
        candidates = self.multi_channel_recall(request)
        
        if not candidates:
            return RecommendationResult(
                items=[],
                total_count=0,
                response_time=time.time() - start_time
            )
        
        # 特征融合
        fused_scores = self.fuse_features(candidates, request)
        
        # 排序优化
        ranked_items = self.rank_items(fused_scores, request)
        
        # 多样性优化
        final_items = self.optimize_diversity(ranked_items, request)
        
        # 限制返回数量
        final_items = final_items[:request.top_k]
        
        # 格式化结果
        result_items = self._format_results(final_items)
        
        response_time = time.time() - start_time
        
        return RecommendationResult(
            items=result_items,
            total_count=len(candidates),
            response_time=response_time,
            metadata={
                'channels_used': [ch for ch, cfg in self.recall_channels.items() if cfg['enabled']],
                'candidate_count': len(candidates),
                'diversity_score': self._calculate_diversity_score(final_items)
            }
        )
    
    def multi_channel_recall(self, request: RecommendationRequest) -> List[tuple]:
        """
        多路召回
        
        Args:
            request: 推荐请求
            
        Returns:
            [(item_id, score, source), ...]
        """
        candidates = []
        
        # 并行执行各召回通道
        for channel, config in self.recall_channels.items():
            if not config['enabled']:
                continue
                
            result = self._recall_channel(channel, request)
            weight = config['weight']
            
            for item_id, score in result:
                candidates.append((item_id, score * weight, channel))
        
        return candidates
    
    def _recall_channel(self, channel: str, request: RecommendationRequest) -> List[tuple]:
        """执行单个召回通道"""
        cache_key = f"{request.user_id}:{channel}:{request.item_type or 'all'}"
        
        # 检查缓存
        if cache_key in self.recall_cache:
            cached_time, cached_result = self.recall_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        result = []
        
        if channel == 'collaborative' and self.cf_recommender:
            result = self.cf_recommender.recommend_user_based(
                request.user_id, 
                top_k=50,
                exclude_items=request.exclude_items
            )
            
        elif channel == 'content_based' and self.content_recommender:
            if self.user_profile:
                preferences = self._get_user_preferences(request.user_id)
                result = self.content_recommender.recommend_by_preferences(preferences, top_k=50)
            else:
                result = self._get_popular_items(request)
                
        elif channel == 'popular':
            result = self._get_popular_items(request)
            
        elif channel == 'recent':
            result = self._get_recent_items(request)
            
        elif channel == 'similar_users' and self.cf_recommender:
            similar_users = self.cf_recommender.get_similar_users(request.user_id, top_k=10)
            for user_id, similarity in similar_users:
                user_recs = self.cf_recommender.recommend_user_based(user_id, top_k=10)
                for item_id, score in user_recs:
                    if item_id not in request.exclude_items:
                        result.append((item_id, score * similarity))
        
        # 缓存结果
        self.recall_cache[cache_key] = (time.time(), result)
        
        return result
    
    def fuse_features(self, candidates: List[tuple], 
                      request: RecommendationRequest) -> Dict[str, Dict]:
        """
        特征融合
        
        Args:
            candidates: 候选物品列表
            request: 推荐请求
            
        Returns:
            {item_id: {'scores': {...}, 'features': {...}}}
        """
        fused = {}
        
        for item_id, base_score, source in candidates:
            if item_id not in fused:
                fused[item_id] = {
                    'scores': defaultdict(float),
                    'features': {},
                    'total_score': 0.0
                }
            
            fused[item_id]['scores'][source] = base_score
        
        # 计算综合分数
        for item_id, data in fused.items():
            scores = data['scores']
            
            cf_score = scores.get('collaborative', 0) + scores.get('similar_users', 0)
            content_score = scores.get('content_based', 0)
            popularity = scores.get('popular', 0)
            freshness = scores.get('recent', 0)
            
            total_score = (
                self.ranker_weights['cf_score'] * cf_score +
                self.ranker_weights['content_score'] * content_score +
                self.ranker_weights['popularity'] * popularity +
                self.ranker_weights['freshness'] * freshness
            )
            
            fused[item_id]['total_score'] = total_score
            fused[item_id]['features'] = {
                'cf_score': cf_score,
                'content_score': content_score,
                'popularity': popularity,
                'freshness': freshness
            }
        
        return fused
    
    def rank_items(self, fused_scores: Dict[str, Dict],
                   request: RecommendationRequest) -> List[tuple]:
        """
        排序优化
        
        Args:
            fused_scores: 融合后的分数
            request: 推荐请求
            
        Returns:
            [(item_id, final_score), ...]
        """
        ranked = []
        
        for item_id, data in fused_scores.items():
            diversity_boost = self._calculate_diversity_boost(item_id, request)
            
            final_score = (
                data['total_score'] * (1 - request.diversity_weight) +
                diversity_boost * request.diversity_weight
            )
            
            ranked.append((item_id, final_score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def optimize_diversity(self, ranked_items: List[tuple],
                          request: RecommendationRequest) -> List[tuple]:
        """
        多样性优化
        
        Args:
            ranked_items: 排序后的物品
            request: 推荐请求
            
        Returns:
            多样化后的物品列表
        """
        if not request.item_type:
            return ranked_items
            
        if len(ranked_items) <= 5:
            return ranked_items
        
        selected = []
        selected_categories = set()
        
        for item_id, score in ranked_items:
            category = self._get_item_category(item_id)
            
            # 限制同类物品数量
            if category not in selected_categories or len(selected_categories) < 3:
                selected.append((item_id, score))
                selected_categories.add(category)
            
            if len(selected) >= request.top_k:
                break
        
        # 如果多样性选择后数量不足，补充剩余高分项
        if len(selected) < request.top_k:
            selected_ids = set(item[0] for item in selected)
            for item_id, score in ranked_items:
                if item_id not in selected_ids:
                    selected.append((item_id, score))
                    if len(selected) >= request.top_k:
                        break
        
        return selected
    
    def _calculate_diversity_boost(self, item_id: str, request: RecommendationRequest) -> float:
        """计算多样性提升"""
        return 0.5
    
    def _get_item_category(self, item_id: str) -> str:
        """获取物品类别"""
        if self.item_features and item_id in self.item_features:
            item = self.item_features.items.get(item_id)
            if item:
                return getattr(item, 'industry', 'general') or 'general'
        return 'general'
    
    def _calculate_diversity_score(self, items: List[tuple]) -> float:
        """计算多样性分数"""
        if len(items) <= 1:
            return 1.0
        
        categories = []
        for item_id, _ in items:
            cat = self._get_item_category(item_id)
            categories.append(cat)
        
        unique = len(set(categories))
        return unique / len(categories)
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """获取用户偏好"""
        if self.user_profile and hasattr(self.user_profile, 'preferences'):
            prefs = self.user_profile.preferences
            return {
                **prefs.agent_preferences,
                **prefs.pipeline_preferences,
                **prefs.template_preferences
            }
        return {}
    
    def _get_popular_items(self, request: RecommendationRequest) -> List[tuple]:
        """获取热门物品"""
        return []
    
    def _get_recent_items(self, request: RecommendationRequest) -> List[tuple]:
        """获取最近物品"""
        return []
    
    def _format_results(self, items: List[tuple]) -> List[Dict]:
        """格式化结果"""
        results = []
        for rank, (item_id, score) in enumerate(items, 1):
            results.append({
                'item_id': item_id,
                'score': round(score, 4),
                'rank': rank
            })
        return results
    
    def update_channel_weight(self, channel: str, weight: float):
        """更新召回通道权重"""
        if channel in self.recall_channels:
            self.recall_channels[channel]['weight'] = weight
    
    def enable_channel(self, channel: str):
        """启用召回通道"""
        if channel in self.recall_channels:
            self.recall_channels[channel]['enabled'] = True
    
    def disable_channel(self, channel: str):
        """禁用召回通道"""
        if channel in self.recall_channels:
            self.recall_channels[channel]['enabled'] = False
    
    def clear_cache(self):
        """清空缓存"""
        self.recall_cache.clear()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'channels': self.recall_channels,
            'cache_size': len(self.recall_cache),
            'cache_ttl': self.cache_ttl
        }
