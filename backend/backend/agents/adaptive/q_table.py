"""
Q-Table Implementation - Q-Learning
基于Q-Learning的Q表实现
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import hashlib
import logging

logger = logging.getLogger(__name__)


class QTable:
    """
    Q-Table实现
    
    使用字典存储Q值，支持状态-动作对的Q-Learning更新
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 64,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        initial_value: float = 0.0
    ):
        """
        初始化Q表
        
        Args:
            state_dim: 状态空间维度（用于生成状态编码）
            action_dim: 动作空间维度
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            initial_value: Q值初始值
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_value = initial_value
        
        # Q表存储: {state_hash: {action: q_value}}
        self._q_table: Dict[str, Dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: initial_value)
        )
        
        # 访问统计
        self._visit_count: Dict[str, int] = defaultdict(int)
        
        # 动作名称映射
        self._action_names: Dict[int, str] = {}
        self._next_action_id = 0
    
    def _encode_state(self, state_features: List[float]) -> str:
        """
        将状态特征编码为字符串键
        
        Args:
            state_features: 状态特征向量
            
        Returns:
            状态的哈希字符串
        """
        # 截断或填充到固定维度
        features = state_features[:self.state_dim]
        while len(features) < self.state_dim:
            features.append(0.0)
        
        # 转换为字节并哈希
        features_array = np.array(features, dtype=np.float32)
        state_bytes = features_array.tobytes()
        return hashlib.sha256(state_bytes).hexdigest()[:16]
    
    def _get_state_actions(self, state: str) -> Dict[int, float]:
        """获取状态的Q值字典"""
        return self._q_table[state]
    
    def register_action(self, action_name: str) -> int:
        """
        注册动作并返回动作ID
        
        Args:
            action_name: 动作名称
            
        Returns:
            动作ID
        """
        # 检查是否已存在
        for action_id, name in self._action_names.items():
            if name == action_name:
                return action_id
        
        # 创建新动作
        action_id = self._next_action_id
        self._action_names[action_id] = action_name
        self._next_action_id += 1
        return action_id
    
    def get(self, state: str, action: int) -> float:
        """
        获取Q值
        
        Args:
            state: 状态
            action: 动作ID
            
        Returns:
            Q值
        """
        return self._q_table[state].get(action, self.initial_value)
    
    def update(self, state: str, action: int, q_value: float) -> None:
        """
        更新Q值
        
        Args:
            state: 状态
            action: 动作ID
            q_value: 新的Q值
        """
        self._q_table[state][action] = q_value
        self._visit_count[state] += 1
    
    def max(self, state: str) -> float:
        """
        获取状态的最大Q值
        
        Args:
            state: 状态
            
        Returns:
            最大Q值
        """
        if not self._q_table[state]:
            return self.initial_value
        return max(self._q_table[state].values())
    
    def argmax(self, state: str) -> int:
        """
        获取具有最大Q值的动作
        
        Args:
            state: 状态
            
        Returns:
            动作ID
        """
        state_actions = self._q_table[state]
        if not state_actions:
            return 0
        
        max_q = float('-inf')
        best_action = 0
        
        for action, q_value in state_actions.items():
            if q_value > max_q:
                max_q = q_value
                best_action = action
        
        return best_action
    
    def get_policy(self, state: str) -> int:
        """
        获取策略（贪婪选择最大Q值的动作）
        
        Args:
            state: 状态
            
        Returns:
            选择的动作ID
        """
        return self.argmax(state)
    
    def batch_update(
        self,
        states: List[str],
        actions: List[int],
        rewards: List[float],
        next_states: List[str],
        dones: List[bool]
    ) -> None:
        """
        批量更新Q值（用于经验回放）
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            next_states: 下一状态列表
            dones: 终止状态标志列表
        """
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * self.max(next_state)
            
            current_q = self.get(state, action)
            new_q = current_q + self.learning_rate * (target - current_q)
            self.update(state, action, new_q)
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        total_states = len(self._q_table)
        total_updates = sum(self._visit_count.values())
        return {
            "total_states": total_states,
            "total_updates": total_updates,
            "total_actions": len(self._action_names)
        }
    
    def save(self, filepath: str) -> None:
        """
        保存Q表到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        
        data = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "q_table": {
                state: {str(action): q_value for action, q_value in actions.items()}
                for state, actions in self._q_table.items()
            },
            "action_names": self._action_names,
            "next_action_id": self._next_action_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Q-table saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "QTable":
        """
        从文件加载Q表
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的QTable实例
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        q_table = cls(
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
            learning_rate=data["learning_rate"],
            discount_factor=data["discount_factor"]
        )
        
        # 恢复Q表数据
        q_table._q_table = defaultdict(
            lambda: defaultdict(lambda: q_table.initial_value),
            {
                state: {int(action): q_value for action, q_value in actions.items()}
                for state, actions in data["q_table"].items()
            }
        )
        
        q_table._action_names = data["action_names"]
        q_table._next_action_id = data["next_action_id"]
        
        logger.info(f"Q-table loaded from {filepath}")
        return q_table
