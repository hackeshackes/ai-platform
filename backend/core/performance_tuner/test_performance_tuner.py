"""
测试用例 - Performance Tuner v12
"""

import pytest
import asyncio
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_analyzer import (
    PerformanceAnalyzer,
    MetricType,
    BottleneckSeverity,
    PerformanceReport
)
from auto_tuner import (
    AutoTuner,
    OptimizationStrategy,
    OptimizationType
)
from benchmark_suite import (
    BenchmarkSuite,
    BenchmarkType,
    LoadTestConfig
)
from optimization_recommender import (
    OptimizationRecommender,
    RecommendationPriority
)
from config import PerformanceConfig


class TestPerformanceAnalyzer:
    """性能分析器测试"""

    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        return PerformanceAnalyzer(
            sample_interval=0.1,
            max_samples=100
        )

    @pytest.mark.asyncio
    async def test_analyze_basic(self, analyzer):
        """测试基本分析功能"""
        # 执行分析
        report = await analyzer.analyze(
            target="test_target",
            metrics=["cpu", "memory"]
        )

        # 验证报告
        assert report is not None
        assert isinstance(report, PerformanceReport)
        assert report.target == "test_target"
        assert report.overall_score >= 0
        assert report.overall_score <= 100
        assert report.health_status in ["healthy", "warning", "degraded", "critical"]

    @pytest.mark.asyncio
    async def test_monitoring(self, analyzer):
        """测试监控功能"""
        # 开始监控
        await analyzer.start_monitoring("test_target")

        # 等待收集指标
        await asyncio.sleep(0.5)

        # 停止监控
        await analyzer.stop_monitoring()

        # 验证指标已收集
        assert len(analyzer._samples) > 0

    @pytest.mark.asyncio
    async def test_detect_bottlenecks(self, analyzer):
        """测试瓶颈检测"""
        # 添加模拟的高CPU指标
        from performance_analyzer import MetricSample
        analyzer._add_sample(MetricSample(
            metric_type=MetricType.CPU,
            value=90.0,
            timestamp=asyncio.get_event_loop().time(),
            unit="percent"
        ))

        # 检测瓶颈
        bottlenecks = analyzer._detect_bottlenecks()

        # 应该检测到高CPU瓶颈
        assert len(bottlenecks) > 0
        assert any(b.metric_type == MetricType.CPU for b in bottlenecks)


class TestAutoTuner:
    """自动调优器测试"""

    @pytest.fixture
    def tuner(self):
        """创建调优器实例"""
        analyzer = PerformanceAnalyzer()
        return AutoTuner(
            config=PerformanceConfig(),
            analyzer=analyzer
        )

    @pytest.mark.asyncio
    async def test_tune_basic(self, tuner):
        """测试基本调优功能"""
        # 执行调优
        result = await tuner.tune(
            target="test_database",
            strategy=OptimizationStrategy.MODERATE
        )

        # 验证结果
        assert result is not None
        assert result.target == "test_database"
        assert result.strategy == OptimizationStrategy.MODERATE
        assert result.start_time > 0
        assert result.end_time > result.start_time
        assert isinstance(result.improvements, dict)

    @pytest.mark.asyncio
    async def test_tune_different_strategies(self, tuner):
        """测试不同优化策略"""
        strategies = [
            OptimizationStrategy.CONSERVATIVE,
            OptimizationStrategy.MODERATE,
            OptimizationStrategy.AGGRESSIVE
        ]

        for strategy in strategies:
            result = await tuner.tune(
                target="test_target",
                strategy=strategy
            )
            assert result.strategy == strategy

    def test_optimization_history(self, tuner):
        """测试优化历史"""
        # 验证历史记录
        history = tuner.get_optimization_history()
        assert isinstance(history, list)


class TestBenchmarkSuite:
    """基准测试套件测试"""

    @pytest.fixture
    def suite(self):
        """创建基准测试套件实例"""
        return BenchmarkSuite()

    @pytest.mark.asyncio
    async def test_performance_benchmark(self, suite):
        """测试性能基准测试"""
        # 模拟请求函数
        async def mock_request():
            await asyncio.sleep(0.01)

        # 运行基准测试
        result = await suite.run_performance_benchmark(
            target="test_endpoint",
            warmup_requests=5,
            benchmark_requests=20,
            concurrent=5,
            request_func=mock_request
        )

        # 验证结果
        assert result is not None
        assert result.type == BenchmarkType.PERFORMANCE
        assert result.total_requests >= 0
        assert result.latency_avg >= 0
        assert result.throughput >= 0

    @pytest.mark.asyncio
    async def test_load_test(self, suite):
        """测试负载测试"""
        async def mock_request():
            await asyncio.sleep(0.01)

        config = LoadTestConfig(
            target="test_endpoint",
            concurrent_users=5,
            ramp_up_time=5,
            test_duration=10
        )

        result = await suite.run_load_test(config, mock_request)

        assert result is not None
        assert result.type == BenchmarkType.LOAD
        assert result.total_requests > 0

    @pytest.mark.asyncio
    async def test_stress_test(self, suite):
        """测试压力测试"""
        async def mock_request():
            await asyncio.sleep(0.01)

        from benchmark_suite import StressTestConfig
        config = StressTestConfig(
            target="test_endpoint",
            initial_users=5,
            max_users=20,
            ramp_up_increment=5,
            ramp_up_interval=2,
            test_duration=15
        )

        result = await suite.run_stress_test(config, mock_request)

        assert result is not None
        assert result.type == BenchmarkType.STRESS

    def test_results_management(self, suite):
        """测试结果管理"""
        # 清除结果
        suite.clear_results()
        assert len(suite.get_results()) == 0


class TestOptimizationRecommender:
    """优化建议器测试"""

    @pytest.fixture
    def recommender(self):
        """创建建议器实例"""
        return OptimizationRecommender(target_improvement=0.3)

    def test_get_suggestions(self, recommender):
        """测试获取建议"""
        # 模拟当前指标
        current_metrics = {
            "cpu": {"avg": 85.0, "min": 70.0, "max": 95.0},
            "memory": {"avg": 75.0, "min": 60.0, "max": 90.0},
            "latency": {"avg": 200.0, "min": 50.0, "max": 500.0},
            "throughput": {"avg": 500.0, "min": 300.0, "max": 800.0}
        }

        # 获取建议
        suggestions = recommender.get_suggestions(current_metrics)

        # 验证
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # 验证建议结构
        for suggestion in suggestions:
            assert suggestion.id.startswith("rec_")
            assert suggestion.title is not None
            assert suggestion.priority in list(RecommendationPriority)

    def test_priority_sorting(self, recommender):
        """测试优先级排序"""
        current_metrics = {
            "cpu": {"avg": 90.0},
            "memory": {"avg": 95.0}
        }

        suggestions = recommender.get_suggestions(current_metrics)

        # 验证排序
        for i in range(len(suggestions) - 1):
            assert suggestions[i].priority.value <= suggestions[i + 1].priority.value

    def test_track_implementation(self, recommender):
        """测试跟踪实施"""
        current_metrics = {"cpu": {"avg": 85.0}}
        suggestions = recommender.get_suggestions(current_metrics)

        if suggestions:
            suggestion_id = suggestions[0].id

            # 跟踪实施
            success = recommender.track_implementation(
                recommendation_id=suggestion_id,
                status="completed",
                results={"improvement": 0.25}
            )

            assert success is True

            # 验证更新
            updated = recommender.get_recommendations(
                implemented=True
            )
            assert any(s.id == suggestion_id for s in updated)

    def test_get_implementation_summary(self, recommender):
        """测试获取实施摘要"""
        # 生成建议
        current_metrics = {"cpu": {"avg": 85.0}}
        recommender.get_suggestions(current_metrics)

        # 获取摘要
        summary = recommender.get_implementation_summary()

        assert "total_recommendations" in summary
        assert "implemented" in summary
        assert "pending" in summary
        assert "progress" in summary


class TestConfig:
    """配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = PerformanceConfig()

        assert config.analysis_interval == 60
        assert config.strategy == OptimizationStrategy.MODERATE
        assert config.max_downtime == 300

    def test_optimization_strategy_enum(self):
        """测试优化策略枚举"""
        assert OptimizationStrategy.CONSERVATIVE.value == "conservative"
        assert OptimizationStrategy.MODERATE.value == "moderate"
        assert OptimizationStrategy.AGGRESSIVE.value == "aggressive"


# 测试运行器
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
