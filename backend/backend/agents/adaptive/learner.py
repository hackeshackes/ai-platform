"""
Adaptive Learner - Core Learning Engine
自适应学习引擎 - 核心学习引擎
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .models import (
    Interaction,
    Pattern,
    LearningResult,
)
from .pattern_extractor import PatternExtractor
from .knowledge_base import KnowledgeBase
from .strategy_optimizer import StrategyOptimizer
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """
    自适应学习引擎
    
    整合模式提取、知识库、策略优化和效果评估
    """
    
    def __init__(
        self,
        agent_id: str,
        knowledge_base: Optional[KnowledgeBase] = None,
        extractor: Optional[PatternExtractor] = None,
        optimizer: Optional[StrategyOptimizer] = None,
        evaluator: Optional[Evaluator] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化自适应学习引擎
        
        Args:
            agent_id: Agent ID
            knowledge_base: 知识库实例
            extractor: 模式提取器实例
            optimizer: 策略优化器实例
            evaluator: 效果评估器实例
            config: 配置选项
        """
        self.agent_id = agent_id
        self.config = config or {}
        
        # 初始化组件
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.extractor = extractor or PatternExtractor()
        self.optimizer = optimizer or StrategyOptimizer()
        self.evaluator = evaluator or Evaluator(self.knowledge_base)
        
        # 学习统计
        self._learning_count = 0
        self._start_time = datetime.now()
        
        logger.info(f"AdaptiveLearner initialized for agent: {agent_id}")
    
    async def _record_pattern(self, pattern: Pattern) -> None:
        """记录模式数据"""
        # 评估器记录功能已整合到 evaluate 中
        pass
    
    async def learn_from_interaction(
        self,
        interaction: Interaction
    ) -> LearningResult:
        """
        从交互中学习
        
        这是核心学习方法，整合所有组件
        
        Args:
            interaction: 交互对象
            
        Returns:
            学习结果
        """
        try:
            # 1. 提取交互模式
            pattern = await self.extractor.extract(interaction)
            
            # 2. 更新知识库
            await self.knowledge_base.update(pattern)
            
            # 3. 计算奖励并优化策略
            reward = self._calculate_reward(pattern)
            optimized_strategy = await self.optimizer.optimize(pattern, reward)
            
            # 4. 记录评估数据
            await self._record_pattern(pattern)
            
            # 更新统计
            self._learning_count += 1
            
            logger.info(
                f"Learning completed: pattern_id={pattern.id}, "
                f"strategy={optimized_strategy.action}, "
                f"q_value={optimized_strategy.q_value:.4f}"
            )
            
            return LearningResult(
                success=True,
                pattern_id=pattern.id,
                message="Learning completed successfully",
                details={
                    "strategy": optimized_strategy.action,
                    "q_value": optimized_strategy.q_value,
                    "exploration": optimized_strategy.exploration,
                    "reward": reward
                }
            )
            
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return LearningResult(
                success=False,
                pattern_id="",
                message=f"Learning failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def learn_from_request(
        self,
        request: InteractionRequest
    ) -> LearningResult:
        """
        从API请求中学习
        
        Args:
            request: 交互请求
            
        Returns:
            学习结果
        """
        # 转换为Interaction对象
        interaction = Interaction(
            text=request.text,
            context=request.context or {},
            actions=request.actions or [],
            session_id=request.session_id or "",
            user_id=request.user_id
        )
        
        return await self.learn_from_interaction(interaction)
    
    def _calculate_reward(self, pattern: Pattern) -> float:
        """
        计算奖励值
        
        Args:
            pattern: 交互模式
            
        Returns:
            奖励值
        """
        reward = 0.0
        
        # 成功奖励
        if pattern.success:
            reward += 1.0
        
        # 执行路径效率奖励
        path_length = len(pattern.execution_path)
        if path_length > 0:
            efficiency = 1.0 / path_length
            reward += efficiency * 0.5
        
        # 实体识别奖励
        entity_count = len(pattern.entities)
        reward += min(entity_count * 0.1, 0.5)
        
        # 意图置信度奖励
        reward += pattern.intent_confidence * 0.3
        
        # 意图准确性奖励
        if pattern.intent.value != "unknown":
            reward += 0.5
        
        return reward
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """
        获取学习状态
        
        Returns:
            学习状态字典
        """
        # 获取评估结果
        evaluation = await self.evaluator.evaluate(self.agent_id)
        
        # 获取优化器统计
        optimizer_stats = self.optimizer.get_stats()
        
        # 获取知识库统计
        kb_stats = await self.knowledge_base.get_stats()
        
        return {
            "agent_id": self.agent_id,
            "learning_count": self._learning_count,
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "evaluation": evaluation.to_dict(),
            "optimizer_stats": optimizer_stats,
            "knowledge_base_stats": kb_stats
        }
    
    async def get_recommended_strategy(self, interaction: Interaction) -> Dict[str, Any]:
        """
        获取推荐策略
        
        Args:
            interaction: 交互对象
            
        Returns:
            推荐策略信息
        """
        # 提取模式
        pattern = await self.extractor.extract(interaction)
        
        # 获取策略
        strategy = await self.optimizer.get_policy(pattern)
        
        # 查找相似模式
        similar_patterns = await self.knowledge_base.find_similar(
            pattern.intent,
            [e.name for e in pattern.entities],
            limit=3
        )
        
        return {
            "recommended_strategy": strategy,
            "confidence": self.optimizer._calculate_confidence(
                self.optimizer.encode_state(pattern),
                0,  # 默认
                0.5
            ),
            "similar_patterns": [
                {
                    "pattern_id": p.id,
                    "success_rate": p.metadata.get("success_rate", 0)
                }
                for p in similar_patterns
            ]
        }
    
    async def batch_learn(
        self,
        interactions: List[Interaction]
    ) -> List[LearningResult]:
        """
        批量学习
        
        Args:
            interactions: 交互列表
            
        Returns:
            学习结果列表
        """
        results = []
        
        for interaction in interactions:
            result = await self.learn_from_interaction(interaction)
            results.append(result)
        
        logger.info(f"Batch learning completed: {len(results)} interactions")
        return results
    
    async def reset(self) -> None:
        """重置学习引擎"""
        self._learning_count = 0
        self._start_time = datetime.now()
        
        # 清空组件
        await self.knowledge_base.clear()
        self.optimizer = StrategyOptimizer()
        self.evaluator = Evaluator(self.knowledge_base)
        
        logger.info(f"AdaptiveLearner reset for agent: {self.agent_id}")


# 简化的请求类，用于API
class InteractionRequest:
    """交互请求类"""
    
    def __init__(
        self,
        text: str,
        context: Optional[Dict] = None,
        actions: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.text = text
        self.context = context or {}
        self.actions = actions or []
        self.session_id = session_id
        self.user_id = user_id
