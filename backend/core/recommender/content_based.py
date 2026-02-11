"""
内容推荐模块 - content_based.py

基于内容的推荐算法，包括相似度计算和排序
"""

import numpy as np
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class ContentFeature:
    """内容特征"""
    item_id: str
    text_features: Dict[str, float] = None
    category_features: Dict[str, float] = None
    tag_features: Dict[str, float] = None
    metadata_features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.text_features is None:
            self.text_features = defaultdict(float)
        if self.category_features is None:
            self.category_features = defaultdict(float)
        if self.tag_features is None:
            self.tag_features = defaultdict(float)
        if self.metadata_features is None:
            self.metadata_features = defaultdict(float)


class ContentBasedRecommender:
    """基于内容的推荐类"""
    
    def __init__(self):
        """初始化内容推荐器"""
        self.item_features: Dict[str, ContentFeature] = {}
        self.vocabulary: Set[str] = set()
        self.category_weights: Dict[str, float] = defaultdict(float)
        self.tag_weights: Dict[str, float] = defaultdict(float)
        
        # IDF缓存
        self.idf_cache: Dict[str, float] = {}
        
    def add_item(self, item_id: str, name: str, description: str, 
                 categories: List[str] = None, tags: List[str] = None,
                 metadata: Dict = None):
        """
        添加物品及其内容特征
        
        Args:
            item_id: 物品ID
            name: 物品名称
            description: 物品描述
            categories: 类别列表
            tags: 标签列表
            metadata: 元数据字典
        """
        feature = ContentFeature(item_id=item_id)
        
        # 提取文本特征
        self._extract_text_features(name + " " + description, feature.text_features)
        
        # 提取类别特征
        if categories:
            for cat in categories:
                feature.category_features[cat] = 1.0
                self.vocabulary.add(f"cat_{cat}")
        
        # 提取标签特征
        if tags:
            for tag in tags:
                feature.tag_features[tag] = 1.0
                self.vocabulary.add(f"tag_{tag}")
        
        # 提取元数据特征
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    feature.metadata_features[key] = float(value)
                    self.vocabulary.add(f"meta_{key}")
        
        self.item_features[item_id] = feature
        
        # 更新IDF
        self._update_idf(feature)
    
    def add_items_batch(self, items: List[Dict]):
        """批量添加物品"""
        for item in items:
            self.add_item(
                item_id=item['id'],
                name=item.get('name', ''),
                description=item.get('description', ''),
                categories=item.get('categories', []),
                tags=item.get('tags', []),
                metadata=item.get('metadata', {})
            )
    
    def compute_similarity(self, item_id1: str, item_id2: str) -> float:
        """
        计算两物品的内容相似度
        
        Args:
            item_id1: 物品1 ID
            item_id2: 物品2 ID
            
        Returns:
            相似度分数 (0-1)
        """
        if item_id1 not in self.item_features or item_id2 not in self.item_features:
            return 0.0
        
        f1 = self.item_features[item_id1]
        f2 = self.item_features[item_id2]
        
        # 文本相似度
        text_sim = self._cosine_similarity(
            list(f1.text_features.keys()),
            list(f2.text_features.keys()),
            f1.text_features,
            f2.text_features
        )
        
        # 类别相似度
        category_sim = self._cosine_similarity(
            list(f1.category_features.keys()),
            list(f2.category_features.keys()),
            f1.category_features,
            f2.category_features
        )
        
        # 标签相似度
        tag_sim = self._cosine_similarity(
            list(f1.tag_features.keys()),
            list(f2.tag_features.keys()),
            f1.tag_features,
            f2.tag_features
        )
        
        # 元数据相似度
        meta_sim = self._jaccard_similarity(
            set(f1.metadata_features.keys()),
            set(f2.metadata_features.keys())
        )
        
        # 加权融合
        weights = {
            'text': 0.35,
            'category': 0.30,
            'tag': 0.25,
            'metadata': 0.10
        }
        
        total_sim = (
            weights['text'] * text_sim +
            weights['category'] * category_sim +
            weights['tag'] * tag_sim +
            weights['metadata'] * meta_sim
        )
        
        return min(1.0, total_sim)
    
    def find_similar_items(self, item_id: str, top_k: int = 10,
                           exclude_items: List[str] = None) -> List[tuple]:
        """
        查找相似物品
        
        Args:
            item_id: 种子物品ID
            top_k: 返回数量
            exclude_items: 排除列表
            
        Returns:
            [(item_id, similarity), ...]
        """
        if item_id not in self.item_features:
            return []
        
        exclude_set = set(exclude_items or [])
        exclude_set.add(item_id)
        
        similarities = []
        for other_id in self.item_features:
            if other_id not in exclude_set:
                sim = self.compute_similarity(item_id, other_id)
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def recommend_by_query(self, query: str, item_type: str = None,
                           top_k: int = 10) -> List[tuple]:
        """
        基于查询文本推荐
        
        Args:
            query: 查询文本
            item_type: 物品类型过滤
            top_k: 返回数量
            
        Returns:
            [(item_id, score), ...]
        """
        # 构建查询特征
        query_features = ContentFeature(item_id="query")
        self._extract_text_features(query, query_features.text_features)
        
        # 计算与所有物品的相似度
        scores = []
        for item_id, features in self.item_features.items():
            if item_type and not self._check_item_type(item_id, item_type):
                continue
            
            # 综合相似度
            text_sim = self._cosine_similarity(
                list(query_features.text_features.keys()),
                list(features.text_features.keys()),
                query_features.text_features,
                features.text_features
            )
            
            # 类别匹配
            category_sim = self._cosine_similarity(
                list(query_features.text_features.keys()),
                list(features.category_features.keys()),
                query_features.text_features,
                features.category_features
            )
            
            total_score = 0.7 * text_sim + 0.3 * category_sim
            scores.append((item_id, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def recommend_by_preferences(self, preferences: Dict[str, float],
                                 top_k: int = 10) -> List[tuple]:
        """
        基于用户偏好推荐
        
        Args:
            preferences: 偏好字典 {特征名: 权重}
            top_k: 返回数量
            
        Returns:
            [(item_id, score), ...]
        """
        scores = []
        
        for item_id, features in self.item_features.items():
            score = 0.0
            
            # 匹配文本特征
            for pref_key, pref_weight in preferences.items():
                if pref_key in features.text_features:
                    score += pref_weight * features.text_features[pref_key]
                elif pref_key in features.category_features:
                    score += pref_weight * features.category_features[pref_key]
                elif pref_key in features.tag_features:
                    score += pref_weight * features.tag_features[pref_key]
            
            scores.append((item_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def compute_item_vector(self, item_id: str) -> np.ndarray:
        """计算物品特征向量"""
        if item_id not in self.item_features:
            return np.zeros(128)
        
        feature = self.item_features[item_id]
        vector = np.zeros(128)
        
        # 文本特征 (0-63)
        for i, (word, weight) in enumerate(feature.text_features.items()):
            if i < 32:
                vector[i] = weight
                vector[i + 32] = self.idf_cache.get(word, 0.5)
        
        # 类别特征 (64-95)
        for i, (cat, weight) in enumerate(feature.category_features.items()):
            if i < 16:
                vector[64 + i] = weight
        
        # 标签特征 (96-127)
        for i, (tag, weight) in enumerate(feature.tag_features.items()):
            if i < 16:
                vector[96 + i] = weight
        
        return vector
    
    def rank_items(self, item_ids: List[str], query: str = None,
                   weights: Dict = None) -> List[tuple]:
        """
        对物品列表进行排序
        
        Args:
            item_ids: 物品ID列表
            query: 查询文本
            weights: 排序权重
            
        Returns:
            [(item_id, rank_score), ...]
        """
        weights = weights or {
            'relevance': 0.4,
            'popularity': 0.3,
            'freshness': 0.2,
            'diversity': 0.1
        }
        
        scores = []
        for item_id in item_ids:
            if item_id not in self.item_features:
                continue
            
            relevance = 0.0
            if query:
                relevance = self._compute_query_relevance(item_id, query)
            
            popularity = self._get_popularity(item_id)
            freshness = self._get_freshness(item_id)
            diversity = self._get_diversity(item_id)
            
            total_score = (
                weights['relevance'] * relevance +
                weights['popularity'] * popularity +
                weights['freshness'] * freshness +
                weights['diversity'] * diversity
            )
            
            scores.append((item_id, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    # 内部辅助方法
    def _extract_text_features(self, text: str, features: Dict[str, float]):
        """提取文本特征"""
        # 分词
        words = self._tokenize(text)
        
        # 词频统计
        for word in words:
            if len(word) > 1:
                features[word] += 1.0
        
        # TF-IDF归一化
        total = sum(features.values())
        if total > 0:
            for word in features:
                features[word] /= total
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        # 移除标点
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        words = text.split()
        # 移除停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'and', 'or', 'but', 'if', 'else', 'when', 'at', 'by', 'for',
                    'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'to', 'from',
                    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'}
        words = [w for w in words if w not in stopwords and len(w) > 1]
        return words
    
    def _cosine_similarity(self, keys1: List[str], keys2: List[str],
                           features1: Dict[str, float],
                           features2: Dict[str, float]) -> float:
        """计算余弦相似度"""
        common_keys = set(keys1) & set(keys2)
        if not common_keys:
            return 0.0
        
        dot_product = sum(features1[k] * features2[k] for k in common_keys)
        norm1 = np.sqrt(sum(f*f for f in features1.values()))
        norm2 = np.sqrt(sum(f*f for f in features2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """计算Jaccard相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _update_idf(self, feature: ContentFeature):
        """更新IDF值"""
        for word in feature.text_features.keys():
            current = self.idf_cache.get(word, 0)
            self.idf_cache[word] = current + 1
    
    def _check_item_type(self, item_id: str, item_type: str) -> bool:
        """检查物品类型"""
        # 需要外部注入物品类型信息
        return True
    
    def _compute_query_relevance(self, item_id: str, query: str) -> float:
        """计算查询相关性"""
        if item_id not in self.item_features:
            return 0.0
        
        feature = self.item_features[item_id]
        query_features = ContentFeature(item_id="query")
        self._extract_text_features(query, query_features.text_features)
        
        return self._cosine_similarity(
            list(query_features.text_features.keys()),
            list(feature.text_features.keys()),
            query_features.text_features,
            feature.text_features
        )
    
    def _get_popularity(self, item_id: str) -> float:
        """获取物品流行度 (需要外部数据)"""
        return 0.5
    
    def _get_freshness(self, item_id: str) -> float:
        """获取物品新鲜度 (需要外部数据)"""
        return 0.5
    
    def _get_diversity(self, item_id: str) -> float:
        """获取物品多样性 (需要外部数据)"""
        return 0.5
    
    def get_vocabulary_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocabulary)
    
    def get_item_count(self) -> int:
        """获取物品数量"""
        return len(self.item_features)
