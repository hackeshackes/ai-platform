"""
Strategy Optimizer - Q-Learning Based Policy Optimization
策略优化器 - 基于Q-Learning的策略优化
"""

import random
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import logging

from .models import Pattern, OptimizedStrategy, IntentType
from .q_table import QTable

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    基于Q-Learning的策略优化器
    
    使用强化学习方法优化Agent行为策略
    """
    
    # 常用动作定义
    DEFAULT_ACTIONS = [
        "analyze_intent",
        "extract_entities",
        "retrieve_knowledge",
        "execute_tool",
        "generate_response",
        "ask_clarification",
        "defer_to_human"
    ]
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 64,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01
    ):
        """
        初始化策略优化器
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            exploration_rate: 探索率 (epsilon)
            exploration_decay: 探索率衰减
            min_exploration_rate: 最小探索率
        """
        self.q_table = QTable(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # 注册默认动作
        for action in self.DEFAULT_ACTIONS:
            self.q_table.register_action(action)
        
        # 学习历史
        self._learning_history: List[Dict] = []
        
        logger.info("Strategy optimizer initialized")
    
    async def optimize(self, pattern: Pattern, reward: float) -> OptimizedStrategy:
        """
        优化策略
        
        Args:
            pattern: 交互模式
            reward: 奖励值
            
        Returns:
            优化后的策略
        """
        # 编码状态
        state = self.encode_state(pattern)
        next_state = self.encode_next_state(pattern)
        
        # ε-greedy探索
        exploration = random.random() < self.exploration_rate
        
        if exploration:
            action, action_name = await self.explore()
        else:
            action, action_name = await self.exploit(state)
        
        # Q-Learning更新
        q_value = self.q_table.get(state, action)
        next_q = self.q_table.max(next_state)
        
        # 计算新的Q值
        new_q = q_value + self.q_table.learning_rate * (
            reward + self.q_table.discount_factor * next_q - q_value
        )
        
        # 更新Q表
        self.q_table.update(state, action, new_q)
        
        # 衰减探索率
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        # 记录学习历史
        self._learning_history.append({
            "state": state,
            "action": action,
            "action_name": action_name,
            "reward": reward,
            "q_value": new_q,
            "exploration": exploration,
            "timestamp": datetime.now().isoformat()
        })
        
        # 获取备选动作
        alternatives = self._get_alternatives(state, action)
        
        # 计算置信度
        confidence = self._calculate_confidence(state, action, new_q)
        
        logger.info(
            f"Strategy optimized: state={state[:8]}, action={action_name}, "
            f"reward={reward:.2f}, q_value={new_q:.4f}"
        )
        
        return OptimizedStrategy(
            action=action_name,
            q_value=new_q,
            exploration=exploration,
            confidence=confidence,
            alternatives=alternatives
        )
    
    async def explore(self) -> Tuple[int, str]:
        """
        探索：随机选择动作
        
        Returns:
            (动作ID, 动作名称)
        """
        action_id = random.randint(0, len(self.DEFAULT_ACTIONS) - 1)
        action_name = self.DEFAULT_ACTIONS[action_id]
        return action_id, action_name
    
    async def exploit(self, state: str) -> Tuple[int, str]:
        """
        利用：选择Q值最大的动作
        
        Args:
            state: 状态
            
        Returns:
            (动作ID, 动作名称)
        """
        action_id = self.q_table.argmax(state)
        
        # 获取动作名称
        action_name = self.q_table._action_names.get(
            action_id,
            f"action_{action_id}"
        )
        
        return action_id, action_name
    
    def encode_state(self, pattern: Pattern) -> str:
        """
        编码状态
        
        Args:
            pattern: 交互模式
            
        Returns:
            状态字符串
        """
        # 构建特征向量
        features = []
        
        # 意图特征 (one-hot编码)
        intent_features = [0.0] * 6  # 6种意图类型
        intent_idx = list([
            IntentType.QUERY, IntentType.ACTION, IntentType.CREATION,
            IntentType.ANALYSIS, IntentType.LEARNING, IntentType.UNKNOWN
        ]).index(pattern.intent)
        intent_features[intent_idx] = 1.0
        features.extend(intent_features)
        
        # 实体特征
        entity_count = len(pattern.entities)
        features.append(min(entity_count / 10.0, 1.0))  # 归一化
        
        # 路径特征
        path_length = len(pattern.execution_path)
        features.append(min(path_length / 20.0, 1.0))  # 归一化
        
        # 成功率特征
        features.append(1.0 if pattern.success else 0.0)
        
        # 奖励特征
        features.append(max(min(pattern.reward / 10.0, 1.0), -1.0))  # 归一化到[-1, 1]
        
        # 文本长度特征
        text_length = len(pattern.metadata.get("text", ""))
        features.append(min(text_length / 500.0, 1.0))
        
        # 填充到固定维度
        while len(features) < self.q_table.state_dim:
            features.append(0.0)
        
        return self.q_table._encode_state(features)
    
    def encode_next_state(self, pattern: Pattern) -> str:
        """
        编码下一状态
        
        Args:
            pattern: 交互模式
            
        Returns:
            下一状态字符串
        """
        # 基于当前模式的变体生成下一状态
        next_features = []
        
        # 意图特征 - 保持不变
        intent_features = [0.0] * 6
        intent_idx = list([
            IntentType.QUERY, IntentType.ACTION, IntentType.CREATION,
            IntentType.ANALYSIS, IntentType.LEARNING, IntentType.UNKNOWN
        ]).index(pattern.intent)
        intent_features[intent_idx] = 1.0
        next_features.extend(intent_features)
        
        # 增加实体计数（假设学习后实体理解提升）
        entity_count = len(pattern.entities)
        next_features.append(min((entity_count + 1) / 10.0, 1.0))
        
        # 路径特征 - 假设执行更高效
        path_length = len(pattern.execution_path)
        next_features.append(max(min(path_length / 20.0, 1.0) * 0.9, 0.0))
        
        # 成功率特征
        next_features.append(1.0 if pattern.success else 0.0)
        
        # 奖励特征 - 假设奖励增加
        reward = max(min((pattern.reward + 1) / 10.0, 1.0), -1.0)
        next_features.append(reward)
        
        # 文本长度特征
        text_length = len(pattern.metadata.get("text", ""))
        next_features.append(min(text_length / 500.0, 1.0))
        
        # 填充到固定维度
        while len(next_features) < self.q_table.state_dim:
            next_features.append(0.0)
        
        return self.q_table._encode_state(next_features)
    
    def _get_alternatives(self, state: str, best_action: int) -> List[str]:
        """
        获取备选动作
        
        Args:
            state: 状态
            best_action: 最佳动作ID
            
        Returns:
            备选动作名称列表
        """
        alternatives = []
        
        # 获取Q值排序
        state_actions = self.q_table._get_state_actions(state)
        sorted_actions = sorted(
            state_actions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 取前3个（排除最佳动作）
        for action_id, q_value in sorted_actions[1:4]:
            action_name = self.q_table._action_names.get(
                action_id,
                f"action_{action_id}"
            )
            alternatives.append(action_name)
        
        return alternatives
    
    def _calculate_confidence(self, state: str, action: int, q_value: float) -> float:
        """
        计算置信度
        
        Args:
            state: 状态
            action: 动作ID
            q_value: Q值
            
        Returns:
            置信度分数
        """
        # 基于Q值差异计算
        state_actions = self.q_table._get_state_actions(state)
        
        if not state_actions:
            return 0.5
        
        max_q = max(state_actions.values())
        min_q = min(state_actions.values())
        
        if max_q == min_q:
            return 0.5
        
        # 归一化
        confidence = (q_value - min_q) / (max_q - min_q)
        
        return max(0.0, min(1.0, confidence))
    
    async def get_policy(self, pattern: Pattern) -> str:
        """
        获取当前策略
        
        Args:
            pattern: 交互模式
            
        Returns:
            建议的动作名称
        """
        state = self.encode_state(pattern)
        _, action_name = await self.exploit(state)
        return action_name
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        q_stats = self.q_table.get_stats()
        
        return {
            **q_stats,
            "exploration_rate": self.exploration_rate,
            "learning_history_size": len(self._learning_history),
            "avg_q_value": self._calculate_avg_q_value()
        }
    
    def _calculate_avg_q_value(self) -> float:
        """计算平均Q值"""
        if not self._learning_history:
            return 0.0
        
        q_values = [h["q_value"] for h in self._learning_history]
        return sum(q_values) / len(q_values)
    
    def save(self, filepath: str) -> None:
        """
        保存优化器状态
        
        Args:
            filepath: 文件路径
        """
        # 保存Q表
        self.q_table.save(filepath)
        
        # 保存额外状态
        state_data = {
            "exploration_rate": self.exploration_rate,
            "exploration_decay": self.exploration_decay,
            "min_exploration_rate": self.min_exploration_rate,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath + ".state", 'w') as f:
            import json
            json.dump(state_data, f, indent=2)
        
        logger.info(f"Strategy optimizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        加载优化器状态
        
        Args:
            filepath: 文件路径
        """
        # 加载Q表
        self.q_table = QTable.load(filepath)
        
        # 加载额外状态
        try:
            with open(filepath + ".state", 'r') as f:
                import json
                state_data = json.load(f)
            
            self.exploration_rate = state_data["exploration_rate"]
            self.exploration_decay = state_data["exploration_decay"]
            self.min_exploration_rate = state_data["min_exploration_rate"]
        except FileNotFoundError:
            logger.warning("No state file found, using default values")
        
        logger.info(f"Strategy optimizer loaded from {filepath}")
