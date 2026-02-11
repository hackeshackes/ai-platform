"""
元学习框架 - Meta Learning Framework
"学习如何学习"框架，新任务适应<1小时
"""

from .learner import MetaLearner
from .few_shot_learner import FewShotLearner
from .task_generator import TaskGenerator
from .adaptation_engine import AdaptationEngine
from .config import MetaLearningConfig

__version__ = "1.0.0"
__all__ = [
    "MetaLearner",
    "FewShotLearner", 
    "TaskGenerator",
    "AdaptationEngine",
    "MetaLearningConfig"
]
