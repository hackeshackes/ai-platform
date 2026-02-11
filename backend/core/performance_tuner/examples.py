"""
示例代码 - Performance Tuner v12

提供使用示例。
"""

import asyncio
import logging
from typing import Dict, Any

from .performance_analyzer import PerformanceAnalyzer
from .auto_tuner import AutoTuner, OptimizationStrategy
from .benchmark_suite import BenchmarkSuite, LoadTestConfig, StabilityTestConfig
from .optimization_recommender import OptimizationRecommender, RecommendationPriority
from .config import PerformanceConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_analysis():
    """示例: 基本性能分析"""
    logger.info("=" * 50)
    logger.info("示例: 基本性能分析")
    logger.info("=" * 50)

    # 创建分析器
    analyzer = PerformanceAnalyzer(
        sample_interval=1.0,
        max_samples=100
    )

    # 开始监控
    await analyzer.start_monitoring("api_server")

    # 等待收集指标
    await asyncio.sleep(5)

    # 执行分析
    report = await analyzer.analyze(
        target="api_server",
        metrics=["cpu", "memory", "latency", "throughput"]
    )

    # 打印报告
    logger.info(f"目标: {report.target}")
    logger.info(f"整体评分: {report.overall_score:.1f}")
    logger.info(f"健康状态: {report.health_status}")
    logger.info(f"瓶颈数量: {len(report.bottlenecks)}")

    # 停止监控
    await analyzer.stop_monitoring()

    return report


async def example_auto_tuning():
    """示例: 自动调优"""
    logger.info("=" * 50)
    logger.info("示例: 自动调优")
    logger.info("=" * 50)

    # 创建分析器和调优器
    analyzer = PerformanceAnalyzer()
    tuner = AutoTuner(config=PerformanceConfig(), analyzer=analyzer)

    # 执行调优
    result = await tuner.tune(
        target="database",
        strategy=OptimizationStrategy.MODERATE,
        constraints={"max_downtime": "5m"}
    )

    # 打印结果
    logger.info(f"优化ID: {result.id}")
    logger.info(f"目标: {result.target}")
    logger.info(f"状态: {result.status}")
    logger.info(f"改进摘要: {result.summary}")
    logger.info(f"操作数量: {len(result.actions)}")

    for action in result.actions:
        logger.info(f"  - {action.type.value}: {action.status.value}")

    return result


async def example_benchmark_testing():
    """示例: 基准测试"""
    logger.info("=" * 50)
    logger.info("示例: 基准测试")
    logger.info("=" * 50)

    # 创建基准测试套件
    suite = BenchmarkSuite()

    # 模拟请求函数
    async def mock_request():
        # 模拟一些处理
        await asyncio.sleep(0.05)

    # 运行性能基准测试
    perf_result = await suite.run_performance_benchmark(
        target="api_endpoint",
        warmup_requests=10,
        benchmark_requests=100,
        concurrent=5,
        request_func=mock_request
    )

    logger.info(f"性能基准测试结果:")
    logger.info(f"  总请求数: {perf_result.total_requests}")
    logger.info(f"  成功率: {perf_result.successful_requests / max(1, perf_result.total_requests):.1%}")
    logger.info(f"  平均延迟: {perf_result.latency_avg:.2f}ms")
    logger.info(f"  P95延迟: {perf_result.latency_p95:.2f}ms")
    logger.info(f"  吞吐量: {perf_result.throughput:.2f}req/s")

    # 运行负载测试
    load_config = LoadTestConfig(
        target="api_endpoint",
        concurrent_users=20,
        ramp_up_time=30,
        test_duration=60
    )

    load_result = await suite.run_load_test(load_config, mock_request)

    logger.info(f"负载测试结果:")
    logger.info(f"  总请求数: {load_result.total_requests}")
    logger.info(f"  吞吐量: {load_result.throughput:.2f}req/s")

    return perf_result, load_result


async def example_optimization_recommendations():
    """示例: 优化建议"""
    logger.info("=" * 50)
    logger.info("示例: 优化建议")
    logger.info("=" * 50)

    # 创建建议器
    recommender = OptimizationRecommender(target_improvement=0.3)

    # 模拟当前指标
    current_metrics = {
        "cpu": {"avg": 85.0, "min": 70.0, "max": 95.0},
        "memory": {"avg": 75.0, "min": 60.0, "max": 90.0},
        "latency": {"avg": 200.0, "min": 50.0, "max": 500.0},
        "throughput": {"avg": 500.0, "min": 300.0, "max": 800.0}
    }

    # 获取建议
    suggestions = recommender.get_suggestions(
        current_metrics=current_metrics,
        target_improvement=0.3
    )

    logger.info(f"生成建议数量: {len(suggestions)}")

    for i, suggestion in enumerate(suggestions[:5], 1):
        logger.info(f"\n建议 {i}:")
        logger.info(f"  标题: {suggestion.title}")
        logger.info(f"  优先级: {suggestion.priority.name}")
        logger.info(f"  预期改进: {suggestion.estimated_improvement:.1%}")
        logger.info(f"  实现难度: {suggestion.implementation_ease}")
        logger.info(f"  预计时间: {suggestion.estimated_time}")

    # 获取实施摘要
    summary = recommender.get_implementation_summary()
    logger.info(f"\n实施摘要:")
    logger.info(f"  总建议数: {summary['total_recommendations']}")
    logger.info(f"  已实施: {summary['implemented']}")
    logger.info(f"  目标改进: {summary['target_improvement']:.1%}")

    return suggestions, summary


async def example_stability_test():
    """示例: 稳定性测试"""
    logger.info("=" * 50)
    logger.info("示例: 稳定性测试")
    logger.info("=" * 50)

    # 创建基准测试套件
    suite = BenchmarkSuite()

    # 模拟请求函数
    async def mock_request():
        await asyncio.sleep(0.02)

    # 配置稳定性测试
    stability_config = StabilityTestConfig(
        target="api_endpoint",
        duration=300,  # 5分钟
        concurrent_users=10,
        error_rate_threshold=0.02
    )

    # 运行稳定性测试
    result = await suite.run_stability_test(stability_config, mock_request)

    logger.info(f"稳定性测试结果:")
    logger.info(f"  总请求数: {result.total_requests}")
    logger.info(f"  错误率: {result.error_rate:.2%}")
    logger.info(f"  平均延迟: {result.latency_avg:.2f}ms")
    logger.info(f"  状态: {result.status.value}")

    return result


async def example_comprehensive_optimization():
    """示例: 综合优化流程"""
    logger.info("=" * 50)
    logger.info("示例: 综合优化流程")
    logger.info("=" * 50)

    # 1. 创建组件
    analyzer = PerformanceAnalyzer()
    tuner = AutoTuner(analyzer=analyzer)
    suite = BenchmarkSuite()
    recommender = OptimizationRecommender()

    # 2. 收集基准指标
    logger.info("收集基准指标...")
    await asyncio.sleep(3)
    baseline = await analyzer.analyze("production_system")

    logger.info(f"基准评分: {baseline.overall_score:.1f}")
    logger.info(f"基准健康状态: {baseline.health_status}")

    # 3. 运行基准测试
    logger.info("运行基准测试...")
    await asyncio.sleep(2)
    benchmark_result = await suite.run_performance_benchmark(
        target="production_system",
        benchmark_requests=200,
        concurrent=10
    )

    logger.info(f"基准吞吐量: {benchmark_result.throughput:.2f}req/s")
    logger.info(f"基准P95延迟: {benchmark_result.latency_p95:.2f}ms")

    # 4. 执行自动调优
    logger.info("执行自动调优...")
    tune_result = await tuner.tune(
        target="production_system",
        strategy=OptimizationStrategy.MODERATE
    )

    logger.info(f"调优后状态: {tune_result.status}")
    logger.info(f"改进摘要: {tune_result.summary}")

    # 5. 生成优化建议
    logger.info("生成优化建议...")
    metrics = analyzer.get_all_metrics()
    current_metrics = {
        metric_type.value: {"avg": sum(s.value for s in samples) / len(samples)}
        for metric_type, samples in metrics.items()
        if samples
    }

    suggestions = recommender.get_suggestions(current_metrics)

    logger.info(f"生成建议: {len(suggestions)}条")

    # 6. 收集优化后指标
    logger.info("收集优化后指标...")
    await asyncio.sleep(3)
    optimized = await analyzer.analyze("production_system")

    logger.info(f"优化后评分: {optimized.overall_score:.1f}")
    logger.info(f"改进: {optimized.overall_score - baseline.overall_score:.1f}")

    return {
        "baseline": baseline,
        "benchmark": benchmark_result,
        "tuning": tune_result,
        "suggestions": suggestions,
        "optimized": optimized
    }


async def main():
    """主函数 - 运行所有示例"""
    examples = [
        ("基本分析", example_basic_analysis),
        ("自动调优", example_auto_tuning),
        ("基准测试", example_benchmark_testing),
        ("优化建议", example_optimization_recommendations),
        ("稳定性测试", example_stability_test),
        ("综合优化", example_comprehensive_optimization)
    ]

    for name, func in examples:
        try:
            await func()
            logger.info(f"\n{'=' * 50}")
            logger.info(f"示例 '{name}' 完成")
            logger.info(f"{'=' * 50}\n")
        except Exception as e:
            logger.error(f"示例 '{name}' 失败: {e}")

    logger.info("所有示例运行完成!")


if __name__ == "__main__":
    asyncio.run(main())
