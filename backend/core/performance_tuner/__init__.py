"""
Performance Tuner - v12 性能自动优化模块

提供系统自动调优功能，性能提升30%目标。

核心模块:
- PerformanceAnalyzer: 性能分析
- AutoTuner: 自动调优
- BenchmarkSuite: 基准测试
- OptimizationRecommender: 优化建议
"""

from .performance_analyzer import PerformanceAnalyzer
from .auto_tuner import AutoTuner
from .benchmark_suite import BenchmarkSuite
from .optimization_recommender import OptimizationRecommender
from .api import create_api_router

__version__ = "1.0.0"
__all__ = [
    "PerformanceAnalyzer",
    "AutoTuner",
    "BenchmarkSuite",
    "OptimizationRecommender",
    "create_api_router",
]
