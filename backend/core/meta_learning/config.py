"""
元学习配置模块
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class MetaLearningConfig:
    """
    元学习配置类
    """
    
    # 任务配置
    n_way: int = 5
    k_shot: int = 1
    query_size: int = 15
    
    # 模型配置
    embedding_dim: int = 64
    hidden_dim: int = 128
    encoder_type: str = "convnet"
    
    # 训练配置
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    meta_batch_size: int = 4
    meta_epochs: int = 1000
    
    # 适配配置
    adaptation_steps: int = 5
    adaptation_lr: float = 0.1
    
    # 算法配置
    algorithm: str = "maml"
    use_contrastive: bool = True
    use_metric: bool = True
    temperature: float = 0.07
    
    # 评估配置
    eval_episodes: int = 100
    eval_support_size: int = 5
    eval_query_size: int = 15
    
    # 设备配置
    device: str = "auto"
    
    # 早停配置
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "query_size": self.query_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "encoder_type": self.encoder_type,
            "meta_lr": self.meta_lr,
            "inner_lr": self.inner_lr,
            "outer_lr": self.outer_lr,
            "meta_batch_size": self.meta_batch_size,
            "meta_epochs": self.meta_epochs,
            "adaptation_steps": self.adaptation_steps,
            "adaptation_lr": self.adaptation_lr,
            "algorithm": self.algorithm,
            "use_contrastive": self.use_contrastive,
            "use_metric": self.use_metric,
            "temperature": self.temperature,
            "eval_episodes": self.eval_episodes,
            "eval_support_size": self.eval_support_size,
            "eval_query_size": self.eval_query_size,
            "device": self.device,
            "use_early_stopping": self.use_early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta
        }
    
    def save(self, path: str):
        """保存配置到JSON文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "MetaLearningConfig":
        """从JSON文件加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
