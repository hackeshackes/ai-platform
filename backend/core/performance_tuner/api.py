"""
API接口 - Performance Tuner API v12

提供REST API接口用于性能调优。
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from .performance_analyzer import PerformanceAnalyzer, PerformanceReport
from .auto_tuner import AutoTuner, OptimizationStrategy
from .benchmark_suite import BenchmarkSuite, LoadTestConfig, StressTestConfig, StabilityTestConfig
from .optimization_recommender import OptimizationRecommender, RecommendationPriority

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/performance-tuner", tags=["Performance Tuner"])

# 全局实例
_analyzer: Optional[PerformanceAnalyzer] = None
_tuner: Optional[AutoTuner] = None
_benchmark_suite: Optional[BenchmarkSuite] = None
_recommender: Optional[OptimizationRecommender] = None


def get_analyzer() -> PerformanceAnalyzer:
    """获取分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PerformanceAnalyzer()
    return _analyzer


def get_tuner() -> AutoTuner:
    """获取调优器实例"""
    global _tuner
    if _tuner is None:
        _tuner = AutoTuner(analyzer=get_analyzer())
    return _tuner


def get_benchmark_suite() -> BenchmarkSuite:
    """获取基准测试套件"""
    global _benchmark_suite
    if _benchmark_suite is None:
        _benchmark_suite = BenchmarkSuite()
    return _benchmark_suite


def get_recommender() -> OptimizationRecommender:
    """获取建议器实例"""
    global _recommender
    if _recommender is None:
        _recommender = OptimizationRecommender()
    return _recommender


# ========== 请求/响应模型 ==========

class AnalyzeRequest(BaseModel):
    """分析请求"""
    target: str = Field(..., description="分析目标")
    metrics: Optional[List[str]] = Field(
        default=["cpu", "memory", "latency", "throughput"],
        description="指标列表"
    )
    duration: Optional[int] = Field(default=60, description="分析持续时间(秒)")


class TuneRequest(BaseModel):
    """调优请求"""
    target: str = Field(..., description="调优目标")
    strategy: str = Field(default="moderate", description="优化策略")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="约束条件")


class BenchmarkRequest(BaseModel):
    """基准测试请求"""
    target: str = Field(..., description="测试目标")
    benchmark_type: str = Field(..., description="基准测试类型")
    concurrent_users: Optional[int] = Field(default=10, description="并发用户数")
    duration: Optional[int] = Field(default=60, description="测试持续时间(秒)")


class PerformanceReportResponse(BaseModel):
    """性能报告响应"""
    id: str
    target: str
    generated_at: float
    duration: float
    overall_score: float
    health_status: str
    metrics_summary: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    resource_usage: Dict[str, Any]
    recommendations: List[str]


class OptimizationResultResponse(BaseModel):
    """优化结果响应"""
    id: str
    target: str
    strategy: str
    start_time: float
    end_time: float
    status: str
    improvements: Dict[str, float]
    summary: str
    recommendations: List[str]


class BenchmarkResultResponse(BaseModel):
    """基准测试结果响应"""
    id: str
    type: str
    target: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    latency_avg: float
    latency_p95: float
    throughput: float


# ========== API 端点 ==========

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "performance-tuner"}


@router.post("/analyze", response_model=PerformanceReportResponse)
async def analyze(request: AnalyzeRequest):
    """
    执行性能分析

    分析指定目标的性能指标，识别瓶颈并生成报告。
    """
    try:
        analyzer = get_analyzer()
        report = await analyzer.analyze(
            target=request.target,
            metrics=request.metrics
        )

        return {
            "id": report.id,
            "target": report.target,
            "generated_at": report.generated_at,
            "duration": report.duration,
            "overall_score": report.overall_score,
            "health_status": report.health_status,
            "metrics_summary": report.metrics_summary,
            "bottlenecks": [
                {
                    "id": b.id,
                    "name": b.name,
                    "severity": b.severity.value,
                    "description": b.description,
                    "impact_score": b.impact_score
                }
                for b in report.bottlenecks
            ],
            "resource_usage": report.resource_usage,
            "recommendations": report.recommendations
        }
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tune", response_model=OptimizationResultResponse)
async def tune(request: TuneRequest):
    """
    执行自动调优

    根据分析结果自动执行优化操作。
    """
    try:
        # 解析策略
        strategy_map = {
            "conservative": OptimizationStrategy.CONSERVATIVE,
            "moderate": OptimizationStrategy.MODERATE,
            "aggressive": OptimizationStrategy.AGGRESSIVE
        }
        strategy = strategy_map.get(request.strategy, OptimizationStrategy.MODERATE)

        tuner = get_tuner()
        result = await tuner.tune(
            target=request.target,
            strategy=strategy,
            constraints=request.constraints
        )

        return {
            "id": result.id,
            "target": result.target,
            "strategy": result.strategy.value,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "status": result.status,
            "improvements": result.improvements,
            "summary": result.summary,
            "recommendations": result.recommendations
        }
    except Exception as e:
        logger.error(f"调优失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", response_model=BenchmarkResultResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """
    执行基准测试

    运行性能基准、负载测试、压力测试或稳定性测试。
    """
    try:
        benchmark_suite = get_benchmark_suite()

        if request.benchmark_type == "performance":
            result = await benchmark_suite.run_performance_benchmark(
                target=request.target,
                concurrent=request.concurrent_users or 10
            )
        elif request.benchmark_type == "load":
            config = LoadTestConfig(
                target=request.target,
                concurrent_users=request.concurrent_users or 10,
                ramp_up_time=60,
                test_duration=request.duration or 60
            )
            result = await benchmark_suite.run_load_test(config)
        elif request.benchmark_type == "stress":
            config = StressTestConfig(
                target=request.target,
                initial_users=request.concurrent_users or 10,
                max_users=request.concurrent_users * 5 if request.concurrent_users else 50,
                ramp_up_increment=10,
                ramp_up_interval=30,
                test_duration=request.duration or 120
            )
            result = await benchmark_suite.run_stress_test(config)
        elif request.benchmark_type == "stability":
            config = StabilityTestConfig(
                target=request.target,
                duration=request.duration or 3600,
                concurrent_users=request.concurrent_users or 5
            )
            result = await benchmark_suite.run_stability_test(config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的测试类型: {request.benchmark_type}"
            )

        return {
            "id": result.id,
            "type": result.type.value,
            "target": result.target,
            "duration": result.duration,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "error_rate": result.error_rate,
            "latency_avg": result.latency_avg,
            "latency_p95": result.latency_p95,
            "throughput": result.throughput
        }
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    implemented: Optional[bool] = None
):
    """
    获取优化建议

    基于当前性能指标获取优化建议列表。
    """
    try:
        recommender = get_recommender()

        # 获取当前指标
        analyzer = get_analyzer()
        metrics = analyzer.get_all_metrics()

        # 转换指标格式
        current_metrics = {}
        for metric_type, samples in metrics.items():
            if samples:
                values = [s.value for s in samples]
                current_metrics[metric_type.value] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }

        # 生成建议
        priority_enum = None
        if priority:
            try:
                priority_enum = RecommendationPriority[priority.upper()]
            except KeyError:
                pass

        suggestions = recommender.get_recommendations(
            category=category,
            priority=priority_enum,
            implemented=implemented
        )

        return {
            "suggestions": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "category": s.category,
                    "priority": s.priority.name,
                    "impact": s.impact.value,
                    "estimated_improvement": s.estimated_improvement,
                    "implementation_ease": s.implementation_ease,
                    "estimated_time": s.estimated_time,
                    "implemented": s.implemented
                }
                for s in suggestions
            ],
            "total": len(suggestions)
        }
    except Exception as e:
        logger.error(f"获取建议失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/summary")
async def get_recommendations_summary():
    """获取优化建议摘要"""
    try:
        recommender = get_recommender()
        return recommender.get_implementation_summary()
    except Exception as e:
        logger.error(f"获取摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations/{recommendation_id}/track")
async def track_implementation(
    recommendation_id: str,
    status: str,
    results: Optional[Dict[str, Any]] = None
):
    """跟踪建议实施进度"""
    try:
        recommender = get_recommender()
        success = recommender.track_implementation(
            recommendation_id=recommendation_id,
            status=status,
            results=results
        )

        if success:
            return {"success": True, "message": "实施进度已更新"}
        else:
            raise HTTPException(status_code=404, detail="建议不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"跟踪失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/results")
async def get_benchmark_results(test_id: Optional[str] = None):
    """获取基准测试结果"""
    try:
        benchmark_suite = get_benchmark_suite()
        results = benchmark_suite.get_results(test_id)

        return {
            "results": [
                {
                    "id": r.id,
                    "type": r.type.value,
                    "target": r.target,
                    "duration": r.duration,
                    "total_requests": r.total_requests,
                    "error_rate": r.error_rate,
                    "latency_avg": r.latency_avg,
                    "throughput": r.throughput,
                    "status": r.status.value
                }
                for r in results
            ],
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    metric_type: Optional[str] = None
):
    """获取性能指标"""
    try:
        analyzer = get_analyzer()

        if metric_type:
            from .performance_analyzer import MetricType
            try:
                mt = MetricType(metric_type)
                samples = analyzer.get_metrics(mt)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的指标类型: {metric_type}"
                )
        else:
            all_metrics = analyzer.get_all_metrics()
            samples = []
            for metric_type, metric_samples in all_metrics.items():
                samples.extend(metric_samples)

        return {
            "samples": [
                {
                    "metric_type": s.metric_type.value,
                    "value": s.value,
                    "timestamp": s.timestamp,
                    "unit": s.unit
                }
                for s in samples[-100:]  # 限制返回数量
            ],
            "total": len(samples)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/start")
async def start_monitoring(target: str = "system"):
    """开始监控"""
    try:
        analyzer = get_analyzer()
        await analyzer.start_monitoring(target)
        return {"success": True, "message": f"开始监控: {target}"}
    except Exception as e:
        logger.error(f"启动监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop")
async def stop_monitoring():
    """停止监控"""
    try:
        analyzer = get_analyzer()
        await analyzer.stop_monitoring()
        return {"success": True, "message": "监控已停止"}
    except Exception as e:
        logger.error(f"停止监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_api_router() -> APIRouter:
    """创建API路由器"""
    return router
