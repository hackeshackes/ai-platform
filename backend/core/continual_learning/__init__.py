"""
持续学习系统 (Continual Learning System)

核心功能：
1. 增量学习 - 顺序学习新任务而不遗忘旧知识
2. 记忆回放 - 通过经验回放防止灾难性遗忘
3. 知识巩固 - 使用EWC、知识蒸馏等方法巩固知识
4. 课程学习 - 按难度渐进式学习

模块：
- IncrementalLearner: 增量学习器
- MemoryReplay: 记忆回放系统
- KnowledgeConsolidation: 知识巩固机制
- CurriculumLearning: 课程学习策略
"""

from .incremental_learner import IncrementalLearner
from .memory_replay import MemoryReplay, ExperienceReplay, ImportanceSamplingReplay, GenerativeReplay, CompressedReplay
from .knowledge_consolidation import KnowledgeConsolidation, EWC, KnowledgeDistillation, MetaLearningConsolidation
from .curriculum_learning import CurriculumLearning, DifficultyScheduler, ActiveLearning, AdaptiveCurriculum
from .config import ContinualLearningConfig
from .api import ContinualLearningAPI

__version__ = "1.0.0"
__all__ = [
    "IncrementalLearner",
    "MemoryReplay",
    "ExperienceReplay",
    "ImportanceSamplingReplay",
    "GenerativeReplay", 
    "CompressedReplay",
    "KnowledgeConsolidation",
    "EWC",
    "KnowledgeDistillation",
    "MetaLearningConsolidation",
    "CurriculumLearning",
    "DifficultyScheduler",
    "ActiveLearning",
    "AdaptiveCurriculum",
    "ContinualLearningConfig",
    "ContinualLearningAPI"
]
