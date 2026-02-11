"""
推荐指标模块 - metrics.py

推荐系统评估指标：点击率、转化率、满意度等
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class InteractionRecord:
    """交互记录"""
    user_id: str
    item_id: str
    recommended: bool
    timestamp: int
    action: str  # impression, click, use, like, share
    reward: float = 0.0


@dataclass
class EvaluationResult:
    """评估结果"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    diversity: float
    coverage: float
    response_time_avg: float
    satisfaction_score: float
    metadata: Dict = field(default_factory=dict)


class RecommenderMetrics:
    """推荐系统指标类"""
    
    def __init__(self):
        """初始化指标计算器"""
        self.interactions: List[InteractionRecord] = []
        self.item_counts: Dict[str, int] = defaultdict(int)
        self.user_item_sets: Dict[str, set] = defaultdict(set)
        
        # 推荐历史
        self.recommendation_logs: List[Dict] = []
        
        # 时序数据
        self.daily_metrics: Dict[str, Dict] = defaultdict(dict)
        
    def log_interaction(self, interaction: InteractionRecord):
        """记录交互"""
        self.interactions.append(interaction)
        self.item_counts[interaction.item_id] += 1
        
        if interaction.recommended:
            key = f"{interaction.user_id}:{interaction.item_id}"
            self.user_item_sets[interaction.user_id].add(interaction.item_id)
        
        # 记录日指标
        date_key = str(interaction.timestamp)[:10]
        self._update_daily_metrics(date_key, interaction)
    
    def log_recommendation(self, user_id: str, item_ids: List[str], 
                          context: Dict = None):
        """记录推荐"""
        self.recommendation_logs.append({
            'user_id': user_id,
            'item_ids': item_ids,
            'context': context or {},
            'timestamp': self._get_timestamp()
        })
    
    def calculate_accuracy(self) -> float:
        """
        计算准确率 (Accuracy)
        
        准确率 = 正确预测数 / 总预测数
        """
        if not self.interactions:
            return 0.0
        
        recommended_interactions = [i for i in self.interactions if i.recommended]
        
        if not recommended_interactions:
            return 0.0
        
        positive_interactions = [i for i in recommended_interactions 
                                if i.action in ['click', 'use', 'like', 'share']]
        
        return len(positive_interactions) / len(recommended_interactions)
    
    def calculate_precision_at_k(self, k: int = 10) -> float:
        """
        计算Precision@K
        
        Precision@K = 相关物品数 / 推荐物品数
        """
        if not self.recommendation_logs:
            return 0.0
        
        precisions = []
        for log in self.recommendation_logs:
            recommended_items = set(log['item_ids'][:k])
            user_id = log['user_id']
            
            # 获取用户实际交互的物品
            user_interactions = self._get_user_positive_items(user_id)
            
            if recommended_items:
                relevant = len(recommended_items & user_interactions)
                precisions.append(relevant / len(recommended_items))
        
        return np.mean(precisions) if precisions else 0.0
    
    def calculate_recall_at_k(self, k: int = 10) -> float:
        """
        计算Recall@K
        
        Recall@K = 相关物品数 / 用户相关物品总数
        """
        if not self.recommendation_logs:
            return 0.0
        
        recalls = []
        for log in self.recommendation_logs:
            recommended_items = set(log['item_ids'][:k])
            user_id = log['user_id']
            
            user_interactions = self._get_user_positive_items(user_id)
            
            if user_interactions:
                relevant = len(recommended_items & user_interactions)
                recalls.append(relevant / len(user_interactions))
        
        return np.mean(recalls) if recalls else 0.0
    
    def calculate_f1_at_k(self, k: int = 10) -> float:
        """计算F1@K"""
        precision = self.calculate_precision_at_k(k)
        recall = self.calculate_recall_at_k(k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_diversity(self) -> float:
        """
        计算推荐多样性 (Diversity)
        
        使用物品间的平均距离或ILS (Intra-List Similarity)
        """
        if not self.recommendation_logs or len(self.recommendation_logs) < 2:
            return 0.0
        
        diversities = []
        for log in self.recommendation_logs:
            items = log['item_ids']
            if len(items) < 2:
                continue
            
            # 简化的多样性计算：基于类别的多样性
            categories = [self._get_item_category(item_id) for item_id in items]
            unique_ratio = len(set(categories)) / len(categories)
            diversities.append(unique_ratio)
        
        return np.mean(diversities) if diversities else 0.0
    
    def calculate_coverage(self) -> float:
        """
        计算覆盖率 (Coverage)
        
        Coverage = 被推荐过的物品数 / 总物品数
        """
        if not self.recommendation_logs or not self.item_counts:
            return 0.0
        
        recommended_items = set()
        for log in self.recommendation_logs:
            recommended_items.update(log['item_ids'])
        
        return len(recommended_items) / len(self.item_counts)
    
    def calculate_click_through_rate(self) -> float:
        """
        计算点击率 (CTR)
        
        CTR = 点击数 / 展示数
        """
        impressions = len([i for i in self.interactions if i.action == 'impression'])
        clicks = len([i for i in self.interactions if i.action == 'click'])
        
        return clicks / impressions if impressions > 0 else 0.0
    
    def calculate_conversion_rate(self) -> float:
        """
        计算转化率 (CVR)
        
        CVR = 转化数 / 点击数
        """
        clicks = len([i for i in self.interactions if i.action == 'click'])
        conversions = len([i for i in self.interactions if i.action in ['use', 'like', 'share']])
        
        return conversions / clicks if clicks > 0 else 0.0
    
    def calculate_satisfaction_score(self) -> float:
        """
        计算满意度评分
        
        基于用户反馈和交互深度
        """
        positive_actions = ['like', 'share']
        neutral_actions = ['use']
        negative_actions = []
        
        if not self.interactions:
            return 0.0
        
        scores = []
        for interaction in self.interactions:
            if interaction.recommended:
                if interaction.action in positive_actions:
                    scores.append(1.0)
                elif interaction.action in neutral_actions:
                    scores.append(0.5)
                elif interaction.action in negative_actions:
                    scores.append(0.0)
        
        return np.mean(scores) if scores else 0.5
    
    def calculate_response_time_avg(self) -> float:
        """计算平均响应时间"""
        if not self.recommendation_logs:
            return 0.0
        
        # 从上下文中获取响应时间
        times = []
        for log in self.recommendation_logs:
            if 'response_time' in log.get('context', {}):
                times.append(log['context']['response_time'])
        
        return np.mean(times) if times else 0.0
    
    def calculate_mean_reciprocal_rank(self, k: int = 10) -> float:
        """
        计算MRR@K (Mean Reciprocal Rank)
        
        MRR = (1/n) * Σ(1/rank_i)
        """
        if not self.recommendation_logs:
            return 0.0
        
        reciprocal_ranks = []
        for log in self.recommendation_logs:
            user_id = log['user_id']
            user_interactions = self._get_user_positive_items(user_id)
            
            rank = k + 1  # 默认未命中
            for i, item_id in enumerate(log['item_ids'][:k]):
                if item_id in user_interactions:
                    rank = i + 1
                    break
            
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_hit_rate_at_k(self, k: int = 10) -> float:
        """
        计算Hit Rate@K
        
        Hit Rate = 至少命中一次的用户数 / 总用户数
        """
        if not self.recommendation_logs:
            return 0.0
        
        users_with_hits = set()
        for log in self.recommendation_logs:
            user_id = log['user_id']
            user_interactions = self._get_user_positive_items(user_id)
            
            for i, item_id in enumerate(log['item_ids'][:k]):
                if item_id in user_interactions:
                    users_with_hits.add(user_id)
                    break
        
        unique_users = len(set(log['user_id'] for log in self.recommendation_logs))
        return len(users_with_hits) / unique_users if unique_users > 0 else 0.0
    
    def calculate_ndcg_at_k(self, k: int = 10) -> float:
        """
        计算NDCG@K (Normalized Discounted Cumulative Gain)
        
        NDCG = DCG / IDCG
        """
        if not self.recommendation_logs:
            return 0.0
        
        ndcgs = []
        for log in self.recommendation_logs:
            user_id = log['user_id']
            user_interactions = self._get_user_positive_items(user_id)
            
            dcg = 0.0
            for i, item_id in enumerate(log['item_ids'][:k]):
                if item_id in user_interactions:
                    dcg += 1.0 / math.log2(i + 2)
            
            # IDCG: 理想情况下的DCG
            ideal_count = min(len(user_interactions), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def evaluate_full(self) -> EvaluationResult:
        """
        全面评估
        
        Returns:
            包含所有指标的评估结果
        """
        return EvaluationResult(
            accuracy=self.calculate_accuracy(),
            precision=self.calculate_precision_at_k(10),
            recall=self.calculate_recall_at_k(10),
            f1_score=self.calculate_f1_at_k(10),
            diversity=self.calculate_diversity(),
            coverage=self.calculate_coverage(),
            response_time_avg=self.calculate_response_time_avg(),
            satisfaction_score=self.calculate_satisfaction_score(),
            metadata={
                'total_interactions': len(self.interactions),
                'total_recommendations': len(self.recommendation_logs),
                'total_items': len(self.item_counts),
                'click_through_rate': self.calculate_click_through_rate(),
                'conversion_rate': self.calculate_conversion_rate(),
                'mrr': self.calculate_mean_reciprocal_rank(10),
                'ndcg': self.calculate_ndcg_at_k(10),
                'hit_rate': self.calculate_hit_rate_at_k(10)
            }
        )
    
    def get_ab_test_metrics(self, group_a: List[str], group_b: List[str]) -> Dict:
        """
        A/B测试指标对比
        
        Args:
            group_a: A组用户ID列表
            group_b: B组用户ID列表
            
        Returns:
            对比指标
        """
        return {
            'group_a': {
                'precision': self._calculate_group_precision(group_a),
                'recall': self._calculate_group_recall(group_a),
                'diversity': self._calculate_group_diversity(group_a)
            },
            'group_b': {
                'precision': self._calculate_group_precision(group_b),
                'recall': self._calculate_group_recall(group_b),
                'diversity': self._calculate_group_diversity(group_b)
            }
        }
    
    def export_metrics(self) -> Dict:
        """导出所有指标"""
        eval_result = self.evaluate_full()
        
        return {
            'evaluation': {
                'accuracy': eval_result.accuracy,
                'precision': eval_result.precision,
                'recall': eval_result.recall,
                'f1_score': eval_result.f1_score,
                'diversity': eval_result.diversity,
                'coverage': eval_result.coverage,
                'response_time_avg': eval_result.response_time_avg,
                'satisfaction_score': eval_result.satisfaction_score
            },
            'metadata': eval_result.metadata
        }
    
    # 内部辅助方法
    def _get_user_positive_items(self, user_id: str) -> set:
        """获取用户正向交互的物品"""
        positive_actions = ['click', 'use', 'like', 'share']
        
        items = set()
        for interaction in self.interactions:
            if interaction.user_id == user_id and interaction.action in positive_actions:
                items.add(interaction.item_id)
        
        return items
    
    def _get_item_category(self, item_id: str) -> str:
        """获取物品类别"""
        # 这里应该从物品特征中获取
        return 'general'
    
    def _get_timestamp(self) -> int:
        """获取当前时间戳"""
        import time
        return int(time.time())
    
    def _update_daily_metrics(self, date_key: str, interaction: InteractionRecord):
        """更新日指标"""
        if 'total' not in self.daily_metrics[date_key]:
            self.daily_metrics[date_key]['total'] = 0
        self.daily_metrics[date_key]['total'] += 1
    
    def _calculate_group_precision(self, user_ids: List[str]) -> float:
        """计算用户组的精确率"""
        return 0.0
    
    def _calculate_group_recall(self, user_ids: List[str]) -> float:
        """计算用户组的召回率"""
        return 0.0
    
    def _calculate_group_diversity(self, user_ids: List[str]) -> float:
        """计算用户组的多样性"""
        return 0.0
    
    def clear(self):
        """清空数据"""
        self.interactions.clear()
        self.item_counts.clear()
        self.user_item_sets.clear()
        self.recommendation_logs.clear()
        self.daily_metrics.clear()
