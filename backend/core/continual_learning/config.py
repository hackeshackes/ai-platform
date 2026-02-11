"""
持续学习系统配置 (Continual Learning Configuration)

提供默认配置和配置管理
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class ReplayStrategy(Enum):
    """回放策略"""
    EXPERIENCE = "experience"
    IMPORTANCE = "importance"
    GENERATIVE = "generative"
    COMPRESSED = "compressed"


class ConsolidationMethod(Enum):
    """知识巩固方法"""
    EWC = "ewc"
    SI = "si"
    DISTILLATION = "distillation"
    LWF = "lwf"
    REGULARIZATION = "regularization"
    MAS = "mas"
    META = "meta"


class CurriculumStrategy(Enum):
    """课程学习策略"""
    DIFFICULTY = "difficulty"
    PROGRESSIVE = "progressive"
    ACTIVE = "active"
    ADAPTIVE = "adaptive"


@dataclass
class ReplayConfig:
    """记忆回放配置"""
    strategy: ReplayStrategy = ReplayStrategy.EXPERIENCE
    capacity: int = 10000
    device: Optional[str] = None
    alpha: float = 0.6  # PER参数
    beta: float = 0.4  # PER参数
    beta_annealing: float = 0.001
    compression_ratio: float = 0.1
    sample_interval: int = 100
    
    
@dataclass
class ConsolidationConfig:
    """知识巩固配置"""
    method: ConsolidationMethod = ConsolidationMethod.EWC
    importance_weight: float = 3000
    fisher_update_period: int = 100
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    regularization_weight: float = 0.01
    use_omega: bool = False
    use_lambda: bool = True
    meta_lr: float = 0.001
    device: Optional[str] = None
    
    
@dataclass
class CurriculumConfig:
    """课程学习配置"""
    strategy: CurriculumStrategy = CurriculumStrategy.DIFFICULTY
    sort_ascending: bool = True
    warmup_tasks: int = 1
    base_difficulty: float = 0.3
    max_difficulty: float = 2.0
    difficulty_increment: float = 0.1
    selection_strategy: str = "uncertainty"
    window_size: int = 5
    adaptation_rate: float = 0.1
    
    
@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    use_replay: bool = True
    use_consolidation: bool = True
    replay_weight: float = 0.5
    consolidation_weight: float = 0.1
    validation_ratio: float = 0.1
    early_stopping_patience: int = 5
    device: Optional[str] = None
    
    
@dataclass
class ContinualLearningConfig:
    """持续学习主配置"""
    # 基础配置
    experiment_name: str = "continual_learning"
    random_seed: int = 42
    verbose: bool = True
    
    # 子系统配置
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 模型配置
    model_type: str = "mlp"
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 10
    dropout: float = 0.3
    
    # 任务配置
    max_tasks: int = 100
    task_sequence: List[str] = field(default_factory=list)
    
    # 评估配置
    evaluate_after_each_task: bool = True
    compute_forgetting: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 5
    
    # 性能配置
    max_memory_mb: Optional[float] = None
    use_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "replay": {
                "strategy": self.replay.strategy.value,
                "capacity": self.replay.capacity,
                "alpha": self.replay.alpha,
                "beta": self.replay.beta,
                "compression_ratio": self.replay.compression_ratio
            },
            "consolidation": {
                "method": self.consolidation.method.value,
                "importance_weight": self.consolidation.importance_weight,
                "distillation_temperature": self.consolidation.distillation_temperature,
                "regularization_weight": self.consolidation.regularization_weight
            },
            "curriculum": {
                "strategy": self.curriculum.strategy.value,
                "sort_ascending": self.curriculum.sort_ascending,
                "base_difficulty": self.curriculum.base_difficulty,
                "max_difficulty": self.curriculum.max_difficulty
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "use_replay": self.training.use_replay,
                "use_consolidation": self.training.use_consolidation
            },
            "model": {
                "type": self.model_type,
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "output_dim": self.output_dim,
                "dropout": self.dropout
            },
            "max_tasks": self.max_tasks,
            "evaluate_after_each_task": self.evaluate_after_each_task,
            "compute_forgetting": self.compute_forgetting
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ContinualLearningConfig':
        """从字典创建配置"""
        config = cls()
        
        if "experiment_name" in config_dict:
            config.experiment_name = config_dict["experiment_name"]
        if "random_seed" in config_dict:
            config.random_seed = config_dict["random_seed"]
        if "verbose" in config_dict:
            config.verbose = config_dict["verbose"]
            
        # 更新子配置
        if "replay" in config_dict:
            replay_dict = config_dict["replay"]
            if "strategy" in replay_dict:
                config.replay.strategy = ReplayStrategy(replay_dict["strategy"])
            if "capacity" in replay_dict:
                config.replay.capacity = replay_dict["capacity"]
                
        if "consolidation" in config_dict:
            cons_dict = config_dict["consolidation"]
            if "method" in cons_dict:
                config.consolidation.method = ConsolidationMethod(cons_dict["method"])
            if "importance_weight" in cons_dict:
                config.consolidation.importance_weight = cons_dict["importance_weight"]
                
        if "training" in config_dict:
            train_dict = config_dict["training"]
            if "epochs" in train_dict:
                config.training.epochs = train_dict["epochs"]
            if "batch_size" in train_dict:
                config.training.batch_size = train_dict["batch_size"]
            if "learning_rate" in train_dict:
                config.training.learning_rate = train_dict["learning_rate"]
                
        return config
    
    def save(self, path: str):
        """保存配置到文件"""
        import json
        
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ContinualLearningConfig':
        """从文件加载配置"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)


# 默认配置
DEFAULT_CONFIG = ContinualLearningConfig()

# 预定义配置
CONFIGS = {
    "ewc_benchmark": ContinualLearningConfig(
        replay=ReplayConfig(strategy=ReplayStrategy.EXPERIENCE, capacity=5000),
        consolidation=ConsolidationConfig(
            method=ConsolidationMethod.EWC,
            importance_weight=5000
        ),
        curriculum=CurriculumConfig(strategy=CurriculumStrategy.DIFFICULTY),
        training=TrainingConfig(
            epochs=5,
            batch_size=32,
            use_replay=True,
            use_consolidation=True
        )
    ),
    
    "derpp_benchmark": ContinualLearningConfig(
        replay=ReplayConfig(strategy=ReplayStrategy.IMPORTANCE, capacity=5000),
        consolidation=ConsolidationConfig(
            method=ConsolidationMethod.EWC,
            importance_weight=5000
        ),
        curriculum=CurriculumConfig(strategy=CurriculumStrategy.DIFFICULTY),
        training=TrainingConfig(
            epochs=5,
            batch_size=32,
            use_replay=True,
            use_consolidation=True,
            replay_weight=1.0,
            consolidation_weight=0.5
        )
    ),
    
    "lwf_benchmark": ContinualLearningConfig(
        replay=ReplayConfig(strategy=ReplayStrategy.EXPERIENCE),
        consolidation=ConsolidationConfig(
            method=ConsolidationMethod.LWF,
            distillation_alpha=0.5
        ),
        curriculum=CurriculumConfig(strategy=CurriculumStrategy.DIFFICULTY),
        training=TrainingConfig(
            epochs=5,
            batch_size=32,
            use_replay=False,
            use_consolidation=True
        )
    ),
    
    "mem_benchmark": ContinualLearningConfig(
        replay=ReplayConfig(strategy=ReplayStrategy.EXPERIENCE, capacity=50000),
        consolidation=ConsolidationConfig(
            method=ConsolidationMethod.REGULARIZATION,
            regularization_weight=0.001
        ),
        curriculum=CurriculumConfig(strategy=CurriculumStrategy.DIFFICULTY),
        training=TrainingConfig(
            epochs=5,
            batch_size=128,
            use_replay=True,
            use_consolidation=False
        )
    )
}


def get_config(name: str) -> ContinualLearningConfig:
    """获取预定义配置"""
    if name in CONFIGS:
        return CONFIGS[name]
    else:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
