"""
Evaluator - Learning Effectiveness Assessment
效果评估器 - 学习效果评估
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import math

from .models import EvaluationResult, Pattern
from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class Evaluator:
    """
    学习效果评估器
    
    评估Agent的学习效果，包括成功率、提升率、一致性等指标
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        初始化评估器
        
        Args:
            knowledge_base: 知识库实例
        """
        self.knowledge_base = knowledge_base
        
        # 评估历史
        self._evaluation_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # 性能指标缓存
        self._metrics_cache: Dict[str, Dict] = {}
        
        logger.info("Evaluator initialized")
    
    async def evaluate(self, agent_id: str) -> EvaluationResult:
        """
        评估Agent学习效果
        
        Args:
            agent_id: Agent ID
            
        Returns:
            评估结果
        """
        # 获取学习历史
        history = await self.get_learning_history(agent_id)
        
        # 计算各项指标
        success_rate = self.calc_success_rate(history)
        improvement = self.calc_improvement(history)
        consistency = self.calc_consistency(history)
        
        # 构建结果
        result = EvaluationResult(
            agent_id=agent_id,
            success_rate=success_rate,
            improvement_rate=improvement,
            consistency=consistency,
            total_interactions=len(history),
            details={
                "recent_trend": self.calc_recent_trend(history),
                "strength_areas": self.identify_strength_areas(history),
                "weak_areas": self.identify_weak_areas(history)
            }
        )
        
        # 缓存评估结果
        self._metrics_cache[agent_id] = result.to_dict()
        
        # 记录评估历史
        self._evaluation_history[agent_id].append({
            "timestamp": datetime.now().isoformat(),
            "result": result.to_dict()
        })
        
        logger.info(
            f"Evaluation completed for agent {agent_id}: "
            f"success_rate={success_rate:.2%}, improvement={improvement:.2%}"
        )
        
        return result
    
    async def get_learning_history(
        self,
        agent_id: str,
        limit: int = 1000
    ) -> List[Dict]:
        """
        获取学习历史
        
        Args:
            agent_id: Agent ID
            limit: 返回数量限制
            
        Returns:
            学习历史列表
        """
        if self.knowledge_base:
            # 从知识库获取
            patterns = await self.knowledge_base.list(limit=limit)
            history = [
                {
                    "pattern_id": p.id,
                    "success": p.success_count > 0,
                    "reward": p.avg_reward,
                    "frequency": p.frequency,
                    "timestamp": p.last_seen.isoformat()
                }
                for p in patterns
            ]
        else:
            # 使用本地历史
            history = [
                entry for entry in self._evaluation_history[agent_id][-limit:]
                if "result" in entry
            ]
        
        return history
    
    def calc_success_rate(self, history: List[Dict]) -> float:
        """
        计算成功率
        
        Args:
            history: 学习历史
            
        Returns:
            成功率 (0.0 - 1.0)
        """
        if not history:
            return 0.0
        
        success_count = sum(1 for entry in history if entry.get("success", True))
        return success_count / len(history)
    
    def calc_improvement(self, history: List[Dict]) -> float:
        """
        计算提升率
        
        通过比较最近和早期的表现来计算学习提升
        
        Args:
            history: 学习历史
            
        Returns:
            提升率 (-1.0 - 1.0)
        """
        if len(history) < 10:
            return 0.0
        
        # 分割为早期和最近
        early_period = history[:len(history) // 3]
        recent_period = history[-len(history) // 3:]
        
        # 计算各时期的平均奖励
        early_avg = self._calc_avg_reward(early_period)
        recent_avg = self._calc_avg_reward(recent_period)
        
        # 计算提升率
        if early_avg == 0:
            if recent_avg > 0:
                return 1.0
            return 0.0
        
        improvement = (recent_avg - early_avg) / max(abs(early_avg), abs(recent_avg))
        
        return max(-1.0, min(1.0, improvement))
    
    def calc_consistency(self, history: List[Dict]) -> float:
        """
        计算一致性
        
        通过计算奖励的标准差来衡量表现的稳定性
        
        Args:
            history: 学习历史
            
        Returns:
            一致性分数 (0.0 - 1.0, 越高越一致)
        """
        if len(history) < 2:
            return 1.0
        
        rewards = [entry.get("reward", 0.0) for entry in history]
        
        # 计算标准差
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance)
        
        # 将标准差转换为一致性分数 (0-1)
        # 标准差为0时一致性最高(1.0)
        # 标准差较大时一致性降低
        consistency = 1.0 / (1.0 + std)
        
        return consistency
    
    def calc_recent_trend(self, history: List[Dict]) -> str:
        """
        计算近期趋势
        
        Args:
            history: 学习历史
            
        Returns:
            趋势描述: "improving", "declining", "stable"
        """
        if len(history) < 10:
            return "stable"
        
        # 分为5个时间段
        segment_size = len(history) // 5
        segments = [
            history[i * segment_size:(i + 1) * segment_size]
            for i in range(5)
            if i * segment_size < len(history)
        ]
        
        # 计算每个时段的平均成功率
        segment_rates = [self.calc_success_rate(s) for s in segments]
        
        # 判断趋势
        if len(segment_rates) < 2:
            return "stable"
        
        # 线性回归斜率
        n = len(segment_rates)
        x_mean = (n - 1) / 2
        y_mean = sum(segment_rates) / n
        
        numerator = sum((i - x_mean) * (rate - y_mean) for i, rate in enumerate(segment_rates))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def identify_strength_areas(self, history: List[Dict]) -> List[str]:
        """
        识别优势领域
        
        Args:
            history: 学习历史
            
        Returns:
            优势领域列表
        """
        strengths = []
        
        # 按类型分析
        type_performance = defaultdict(list)
        for entry in history:
            pattern_type = entry.get("pattern_type", "unknown")
            reward = entry.get("reward", 0.0)
            type_performance[pattern_type].append(reward)
        
        # 找出表现好的类型
        avg_performance = {
            p_type: sum(rewards) / len(rewards)
            for p_type, rewards in type_performance.items()
            if rewards
        }
        
        threshold = sum(avg_performance.values()) / len(avg_performance) if avg_performance else 0
        
        for p_type, avg in avg_performance.items():
            if avg > threshold:
                strengths.append(p_type)
        
        return strengths
    
    def identify_weak_areas(self, history: List[Dict]) -> List[str]:
        """
        识别薄弱领域
        
        Args:
            history: 学习历史
            
        Returns:
            薄弱领域列表
        """
        weaknesses = []
        
        # 按类型分析
        type_performance = defaultdict(list)
        for entry in history:
            pattern_type = entry.get("pattern_type", "unknown")
            reward = entry.get("reward", 0.0)
            type_performance[pattern_type].append(reward)
        
        # 找出表现差的类型
        avg_performance = {
            p_type: sum(rewards) / len(rewards)
            for p_type, rewards in type_performance.items()
            if rewards
        }
        
        if not avg_performance:
            return weaknesses
        
        threshold = sum(avg_performance.values()) / len(avg_performance)
        
        for p_type, avg in avg_performance.items():
            if avg < threshold * 0.8:  # 低于平均值80%
                weaknesses.append(p_type)
        
        return weaknesses
    
    async def compare_agents(
        self,
        agent_ids: List[str]
    ) -> Dict[str, EvaluationResult]:
        """
        比较多个Agent的学习效果
        
        Args:
            agent_ids: Agent ID列表
            
        Returns:
            各Agent的评估结果
        """
        results = {}
        
        for agent_id in agent_ids:
            results[agent_id] = await self.evaluate(agent_id)
        
        return results
    
    def get_cached_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的指标
        
        Args:
            agent_id: Agent ID
            
        Returns:
            缓存的指标字典或None
        """
        return self._metrics_cache.get(agent_id)
    
    def _calc_avg_reward(self, history: List[Dict]) -> float:
        """计算平均奖励"""
        if not history:
            return 0.0
        
        rewards = [entry.get("reward", 0.0) for entry in history]
        return sum(rewards) / len(rewards)
    
    async def get_evaluation_history(
        self,
        agent_id: str,
        days: int = 7
    ) -> List[Dict]:
        """
        获取评估历史
        
        Args:
            agent_id: Agent ID
            days: 获取最近几天的记录
            
        Returns:
            评估历史记录列表
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        history = self._evaluation_history.get(agent_id, [])
        
        return [
            entry for entry in history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
    
    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            agent_id: 指定的Agent ID，不指定则清除所有
        """
        if agent_id:
            self._metrics_cache.pop(agent_id, None)
        else:
            self._metrics_cache.clear()
        
        logger.info(f"Cache cleared for agent: {agent_id or 'all'}")
