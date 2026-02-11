"""
Visualization Module - AI Platform v5

训练可视化模块 - 提供Loss曲线、GPU监控、评估指标等可视化功能
"""

from .charts import (
    ChartGenerator,
    LossChartConfig,
    GPUChartConfig,
    MetricsChartConfig,
    LearningRateChartConfig,
    get_chart_generator,
)

from .realtime import (
    TrainingDataStore,
    RealtimeDataHandler,
    SSEPublisher,
    get_training_store,
    get_realtime_handler,
)

__all__ = [
    # Charts
    "ChartGenerator",
    "LossChartConfig",
    "GPUChartConfig",
    "MetricsChartConfig",
    "LearningRateChartConfig",
    "get_chart_generator",
    
    # Realtime
    "TrainingDataStore",
    "RealtimeDataHandler",
    "SSEPublisher",
    "get_training_store",
    "get_realtime_handler",
]
