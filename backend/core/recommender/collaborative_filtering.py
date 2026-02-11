"""
协同过滤模块 - collaborative_filtering.py

实现用户协同和物品协同过滤，以及矩阵分解
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random


@dataclass
class UserItemInteraction:
    """用户-物品交互"""
    user_id: str
    item_id: str
    rating: float  # 评分 0-1
    timestamp: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class CollaborativeFiltering:
    """协同过滤推荐类"""
    
    def __init__(self, k: int = 20, min_similarity: float = 0.1):
        """
        初始化协同过滤
        
        Args:
            k: 近邻数量
            min_similarity: 最小相似度阈值
        """
        self.k = k
        self.min_similarity = min_similarity
        
        # 交互数据
        self.interactions: List[UserItemInteraction] = []
        self.user_item_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.item_user_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 相似度矩阵
        self.user_similarity: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.item_similarity: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 矩阵分解模型
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_factor_matrix: Dict[str, np.ndarray] = {}
        self.item_factor_matrix: Dict[str, np.ndarray] = {}
        
        # 配置
        self.latent_dim = 32
        self.learning_rate = 0.01
        self.regularization = 0.02
        self.iterations = 100
        
    def add_interaction(self, interaction: UserItemInteraction):
        """添加交互数据"""
        self.interactions.append(interaction)
        self.user_item_matrix[interaction.user_id][interaction.item_id] = interaction.rating
        self.item_user_matrix[interaction.item_id][interaction.user_id] = interaction.rating
    
    def add_interactions_batch(self, interactions: List[UserItemInteraction]):
        """批量添加交互数据"""
        for interaction in interactions:
            self.add_interaction(interaction)
    
    def compute_user_similarity(self) -> Dict[str, Dict[str, float]]:
        """计算用户相似度 (基于物品的协同过滤)"""
        users = list(self.user_item_matrix.keys())
        
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users):
                if i < j:
                    sim = self._compute_user_similarity(user1, user2)
                    self.user_similarity[user1][user2] = sim
                    self.user_similarity[user2][user1] = sim
                    
        return self.user_similarity
    
    def compute_item_similarity(self) -> Dict[str, Dict[str, float]]:
        """计算物品相似度 (基于用户的协同过滤)"""
        items = list(self.item_user_matrix.keys())
        
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i < j:
                    sim = self._compute_item_similarity(item1, item2)
                    self.item_similarity[item1][item2] = sim
                    self.item_similarity[item2][item1] = sim
                    
        return self.item_similarity
    
    def recommend_user_based(self, user_id: str, top_k: int = 10,
                             exclude_items: List[str] = None) -> List[tuple]:
        """
        基于用户的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 返回数量
            exclude_items: 排除的物品列表
            
        Returns:
            [(item_id, score), ...]
        """
        if user_id not in self.user_item_matrix:
            return []
            
        if not self.user_similarity:
            self.compute_user_similarity()
            
        # 获取用户的近邻
        neighbors = self._get_top_neighbors(user_id, self.k)
        
        # 计算推荐分数
        scores = defaultdict(float)
        weights = defaultdict(float)
        
        for neighbor_id, similarity in neighbors:
            if neighbor_id == user_id:
                continue
                
            for item_id, rating in self.user_item_matrix[neighbor_id].items():
                if item_id in self.user_item_matrix[user_id]:
                    continue
                if exclude_items and item_id in exclude_items:
                    continue
                    
                scores[item_id] += similarity * rating
                weights[item_id] += abs(similarity)
        
        # 归一化
        recommendations = []
        for item_id in scores:
            if weights[item_id] > 0:
                score = scores[item_id] / weights[item_id]
                recommendations.append((item_id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def recommend_item_based(self, user_id: str, item_id: str, top_k: int = 10) -> List[tuple]:
        """
        基于物品的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            item_id: 种子物品ID
            top_k: 返回数量
            
        Returns:
            [(item_id, score), ...]
        """
        if not self.item_similarity:
            self.compute_item_similarity()
            
        if item_id not in self.item_similarity:
            return []
            
        # 获取相似物品
        similar_items = sorted(
            self.item_similarity[item_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 获取用户已交互的物品
        user_items = set(self.user_item_matrix.get(user_id, {}).keys())
        
        # 过滤并返回
        recommendations = []
        for sim_item_id, similarity in similar_items:
            if sim_item_id not in user_items:
                recommendations.append((sim_item_id, similarity))
                
        return recommendations[:top_k]
    
    def recommend_matrix_factorization(self, user_id: str, top_k: int = 10) -> List[tuple]:
        """
        基于矩阵分解的推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 返回数量
            
        Returns:
            [(item_id, score), ...]
        """
        if self.user_factors is None or self.item_factors is None:
            self.train_matrix_factorization()
            
        users = list(self.user_item_matrix.keys())
        items = list(self.item_user_matrix.keys())
        
        if user_id not in users or not items:
            return []
        
        user_idx = {user: idx for idx, user in enumerate(users)}
        item_idx = {item: idx for idx, item in enumerate(items)}
        
        ui = user_idx[user_id]
        user_vector = self.user_factors[ui]
        
        # 计算所有物品的预测评分
        scores = []
        for item_id in items:
            if item_id not in self.user_item_matrix[user_id]:
                ii = item_idx[item_id]
                pred = np.dot(user_vector, self.item_factors[ii])
                scores.append((item_id, pred))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_similar_users(self, user_id: str, top_k: int = 10) -> List[tuple]:
        """获取相似用户"""
        if not self.user_similarity:
            self.compute_user_similarity()
            
        if user_id not in self.user_similarity:
            return []
            
        similarities = sorted(
            self.user_similarity[user_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return similarities
    
    def get_similar_items(self, item_id: str, top_k: int = 10) -> List[tuple]:
        """获取相似物品"""
        if not self.item_similarity:
            self.compute_item_similarity()
            
        if item_id not in self.item_similarity:
            return []
            
        similarities = sorted(
            self.item_similarity[item_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return similarities
    
    # 内部辅助方法
    def _compute_user_similarity(self, user1: str, user2: str) -> float:
        """计算两用户的相似度"""
        items1 = set(self.user_item_matrix.get(user1, {}).keys())
        items2 = set(self.user_item_matrix.get(user2, {}).keys())
        
        if not items1 or not items2:
            return 0.0
            
        common_items = items1 & items2
        if not common_items:
            return 0.0
        
        # Jaccard相似度
        union = items1 | items2
        return len(common_items) / len(union)
    
    def _compute_item_similarity(self, item1: str, item2: str) -> float:
        """计算两物品的相似度"""
        users1 = set(self.item_user_matrix.get(item1, {}).keys())
        users2 = set(self.item_user_matrix.get(item2, {}).keys())
        
        if not users1 or not users2:
            return 0.0
            
        common_users = users1 & users2
        if not common_users:
            return 0.0
        
        # Jaccard相似度
        union = users1 | users2
        return len(common_users) / len(union)
    
    def _get_top_neighbors(self, user_id: str, k: int) -> List[tuple]:
        """获取top-k近邻"""
        if user_id not in self.user_similarity:
            return []
            
        similarities = sorted(
            self.user_similarity[user_id].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [sim for sim in similarities if sim[1] >= self.min_similarity][:k]
    
    def train_matrix_factorization(self) -> Tuple[np.ndarray, np.ndarray]:
        """训练矩阵分解模型"""
        users = list(self.user_item_matrix.keys())
        items = list(self.item_user_matrix.keys())
        
        n_users = len(users)
        n_items = len(items)
        
        if n_users == 0 or n_items == 0:
            return None, None
        
        # 创建索引映射
        user_idx = {user: idx for idx, user in enumerate(users)}
        item_idx = {item: idx for idx, item in enumerate(items)}
        
        # 初始化因子矩阵
        np.random.seed(42)
        user_factors = np.random.randn(n_users, self.latent_dim) * 0.1
        item_factors = np.random.randn(n_items, self.latent_dim) * 0.1
        
        # 收集所有交互
        interactions = []
        for user, items_dict in self.user_item_matrix.items():
            for item, rating in items_dict.items():
                if item in item_idx:
                    interactions.append((user_idx[user], item_idx[item], rating))
        
        # SGD训练
        for iteration in range(self.iterations):
            random.shuffle(interactions)
            
            for ui, ii, rating in interactions:
                pred = np.dot(user_factors[ui], item_factors[ii])
                error = rating - pred
                
                # 更新因子
                user_factors[ui] += self.learning_rate * (
                    error * item_factors[ii] - self.regularization * user_factors[ui]
                )
                item_factors[ii] += self.learning_rate * (
                    error * user_factors[ui] - self.regularization * item_factors[ii]
                )
        
        self.user_factors = user_factors
        self.item_factors = item_factors
        
        # 保存用户因子
        for user, idx in user_idx.items():
            self.user_factor_matrix[user] = user_factors[idx]
        
        # 保存物品因子
        for item, idx in item_idx.items():
            self.item_factor_matrix[item] = item_factors[idx]
        
        return user_factors, item_factors
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_interactions': len(self.interactions),
            'total_users': len(self.user_item_matrix),
            'total_items': len(self.item_user_matrix),
            'avg_interactions_per_user': (
                len(self.interactions) / max(1, len(self.user_item_matrix))
            ),
            'avg_interactions_per_item': (
                len(self.interactions) / max(1, len(self.item_user_matrix))
            ),
            'sparsity': (
                1 - len(self.interactions) / 
                max(1, len(self.user_item_matrix) * len(self.item_user_matrix))
            )
        }
    
    def clear(self):
        """清空数据"""
        self.interactions.clear()
        self.user_item_matrix.clear()
        self.item_user_matrix.clear()
        self.user_similarity.clear()
        self.item_similarity.clear()
        self.user_factors = None
        self.item_factors = None
        self.user_factor_matrix.clear()
        self.item_factor_matrix.clear()
