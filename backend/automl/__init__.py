"""
AutoML Module - 自动化机器学习

提供完整的AutoML功能:
- 超参数优化 (Hyperparameter Tuning)
- 模型自动选择 (AutoML)
- 神经架构搜索 (NAS)
- 自动化特征工程 (Feature Engineering)
"""

from automl.tuner import (
    HyperparameterTuner,
    TuneParam,
    TuneMethod,
    TuneObjective,
    ParamType,
    TuneTrial,
    TuneResult,
    default_tuner
)

from automl.selector import (
    ModelSelector,
    TaskType,
    ModelCategory,
    ModelCandidate,
    ModelScore,
    SelectionResult,
    default_selector
)

from automl.nas import (
    NeuralArchitectureSearcher,
    NASTask,
    LayerType,
    LayerSpec,
    ArchitectureGenome,
    NASSearchSpace,
    NASResult,
    default_nas
)

from automl.feature import (
    FeatureEngineer,
    FeatureType,
    TransformationType,
    FeatureInfo,
    GeneratedFeature,
    FeatureEngineeringResult,
    default_engineer
)

__all__ = [
    # Tuner
    "HyperparameterTuner",
    "TuneParam",
    "TuneMethod",
    "TuneObjective",
    "ParamType",
    "TuneTrial",
    "TuneResult",
    "default_tuner",
    # Selector
    "ModelSelector",
    "TaskType",
    "ModelCategory",
    "ModelCandidate",
    "ModelScore",
    "SelectionResult",
    "default_selector",
    # NAS
    "NeuralArchitectureSearcher",
    "NASTask",
    "LayerType",
    "LayerSpec",
    "ArchitectureGenome",
    "NASSearchSpace",
    "NASResult",
    "default_nas",
    # Feature
    "FeatureEngineer",
    "FeatureType",
    "TransformationType",
    "FeatureInfo",
    "GeneratedFeature",
    "FeatureEngineeringResult",
    "default_engineer"
]
