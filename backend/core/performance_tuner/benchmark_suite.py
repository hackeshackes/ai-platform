"""
基准测试套件 - Benchmark Suite v12

功能:
- 性能基准
- 负载测试
- 压力测试
- 稳定性测试
"""

import asyncio
import time
import random
import statistics
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """基准测试类型"""
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    STABILITY = "stability"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    id: str
    type: BenchmarkType
    target: str
    start_time: float
    end_time: float
    duration: float
    status: TestStatus

    # 请求统计
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float

    # 延迟统计 (ms)
    latency_min: float
    latency_max: float
    latency_avg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

    # 吞吐量
    throughput: float  # req/s

    # 资源使用
    cpu_usage: float
    memory_usage: float

    # 详细指标
    metrics: Dict[str, Any]

    # 错误列表
    errors: List[Dict[str, Any]]


@dataclass
class LoadTestConfig:
    """负载测试配置"""
    target: str
    concurrent_users: int
    ramp_up_time: int  # 秒
    test_duration: int  # 秒
    request_interval: float = 0.1  # 请求间隔(秒)
    think_time: float = 0  # 用户思考时间(秒)


@dataclass
class StressTestConfig:
    """压力测试配置"""
    target: str
    initial_users: int
    max_users: int
    ramp_up_increment: int
    ramp_up_interval: int  # 秒
    test_duration: int  # 秒
    max_error_rate: float = 0.1  # 最大错误率


@dataclass
class StabilityTestConfig:
    """稳定性测试配置"""
    target: str
    duration: int  # 秒
    concurrent_users: int
    error_rate_threshold: float = 0.01
    check_interval: int = 60  # 检查间隔(秒)


class BenchmarkSuite:
    """
    基准测试套件

    提供全面的基准测试功能，包括性能基准、负载测试、压力测试和稳定性测试。
    """

    def __init__(self):
        """初始化基准测试套件"""
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running_tests: Dict[str, asyncio.Task] = {}
        self._results: List[BenchmarkResult] = []

    async def run_performance_benchmark(
        self,
        target: str,
        warmup_requests: int = 10,
        benchmark_requests: int = 1000,
        concurrent: int = 10,
        request_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        运行性能基准测试

        Args:
            target: 测试目标
            warmup_requests: 预热请求数
            benchmark_requests: 基准测试请求数
            concurrent: 并发数
            request_func: 请求函数(可选)

        Returns:
            BenchmarkResult: 测试结果
        """
        start_time = time.time()
        logger.info(f"开始性能基准测试: target={target}")

        # 预热阶段
        await self._warmup(warmup_requests, request_func)

        # 基准测试阶段
        latencies = []
        successful = 0
        failed = 0
        errors = []

        async def execute_request():
            nonlocal successful, failed
            req_start = time.time()
            try:
                if request_func:
                    await request_func()
                else:
                    await self._simulate_request(target)

                latency = (time.time() - req_start) * 1000
                latencies.append(latency)
                successful += 1
            except Exception as e:
                failed += 1
                errors.append({
                    "time": time.time(),
                    "error": str(e)
                })

        # 并发执行
        tasks = []
        for i in range(benchmark_requests):
            tasks.append(execute_request())
            if len(tasks) >= concurrent:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

        end_time = time.time()

        # 计算统计
        result = self._calculate_results(
            id=f"perf_{int(start_time)}",
            type=BenchmarkType.PERFORMANCE,
            target=target,
            start_time=start_time,
            end_time=end_time,
            latencies=latencies,
            successful=successful,
            failed=failed,
            errors=errors
        )

        self._results.append(result)
        return result

    async def run_load_test(
        self,
        config: LoadTestConfig,
        request_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        运行负载测试

        Args:
            config: 负载测试配置
            request_func: 请求函数(可选)

        Returns:
            BenchmarkResult: 测试结果
        """
        start_time = time.time()
        logger.info(f"开始负载测试: target={config.target}")

        latencies = []
        successful = 0
        failed = 0
        errors = []

        async def worker(user_id: int):
            nonlocal successful, failed
            try:
                while time.time() - start_time < config.test_duration:
                    req_start = time.time()
                    try:
                        if request_func:
                            await request_func()
                        else:
                            await self._simulate_request(config.target)

                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                        successful += 1
                    except Exception as e:
                        failed += 1
                        errors.append({
                            "time": time.time(),
                            "error": str(e)
                        })

                    await asyncio.sleep(config.think_time)
            except asyncio.CancelledError:
                pass

        # 爬坡阶段
        workers = []
        for i in range(config.concurrent_users):
            task = asyncio.create_task(worker(i))
            workers.append(task)

        # 等待测试完成
        await asyncio.sleep(config.test_duration)

        # 取消所有worker
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        end_time = time.time()

        # 计算资源使用
        cpu_usage = await self._measure_cpu_usage()
        memory_usage = await self._measure_memory_usage()

        result = self._calculate_results(
            id=f"load_{int(start_time)}",
            type=BenchmarkType.LOAD,
            target=config.target,
            start_time=start_time,
            end_time=end_time,
            latencies=latencies,
            successful=successful,
            failed=failed,
            errors=errors,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage
        )

        self._results.append(result)
        return result

    async def run_stress_test(
        self,
        config: StressTestConfig,
        request_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        运行压力测试

        Args:
            config: 压力测试配置
            request_func: 请求函数(可选)

        Returns:
            BenchmarkResult: 测试结果
        """
        start_time = time.time()
        logger.info(f"开始压力测试: target={config.target}")

        latencies = []
        successful = 0
        failed = 0
        errors = []
        current_users = config.initial_users

        async def worker(user_id: int):
            nonlocal successful, failed
            try:
                while time.time() - start_time < config.test_duration:
                    req_start = time.time()
                    try:
                        if request_func:
                            await request_func()
                        else:
                            await self._simulate_request(config.target)

                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                        successful += 1
                    except Exception as e:
                        failed += 1
                        errors.append({
                            "time": time.time(),
                            "error": str(e)
                        })
                    await asyncio.sleep(0.01)  # 短间隔
            except asyncio.CancelledError:
                pass

        workers = []

        # 逐步增加负载
        while time.time() - start_time < config.test_duration:
            # 检查错误率
            total = successful + failed
            if total > 0 and failed / total > config.max_error_rate:
                logger.warning("错误率超过阈值，停止增加负载")
                break

            # 增加用户
            if current_users < config.max_users:
                new_users = min(config.ramp_up_increment, config.max_users - current_users)
                current_users += new_users

                # 添加新worker
                for i in range(new_users):
                    task = asyncio.create_task(worker(i))
                    workers.append(task)

            await asyncio.sleep(config.ramp_up_interval)

        # 等待一段时间
        await asyncio.sleep(10)

        # 清理
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        end_time = time.time()

        result = self._calculate_results(
            id=f"stress_{int(start_time)}",
            type=BenchmarkType.STRESS,
            target=config.target,
            start_time=start_time,
            end_time=end_time,
            latencies=latencies,
            successful=successful,
            failed=failed,
            errors=errors,
            metrics={
                "max_concurrent_users": current_users,
                "error_rate_threshold": config.max_error_rate
            }
        )

        self._results.append(result)
        return result

    async def run_stability_test(
        self,
        config: StabilityTestConfig,
        request_func: Optional[Callable] = None
    ) -> BenchmarkResult:
        """
        运行稳定性测试

        Args:
            config: 稳定性测试配置
            request_func: 请求函数(可选)

        Returns:
            BenchmarkResult: 测试结果
        """
        start_time = time.time()
        logger.info(f"开始稳定性测试: target={config.target}")

        latencies = []
        successful = 0
        failed = 0
        errors = []
        checkpoints = []

        async def worker(user_id: int):
            nonlocal successful, failed
            try:
                while time.time() - start_time < config.duration:
                    req_start = time.time()
                    try:
                        if request_func:
                            await request_func()
                        else:
                            await self._simulate_request(config.target)

                        latency = (time.time() - req_start) * 1000
                        latencies.append(latency)
                        successful += 1
                    except Exception as e:
                        failed += 1
                        errors.append({
                            "time": time.time(),
                            "error": str(e)
                        })
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

        # 启动worker
        workers = []
        for i in range(config.concurrent_users):
            task = asyncio.create_task(worker(i))
            workers.append(task)

        # 定期检查点
        elapsed = 0
        while elapsed < config.duration:
            await asyncio.sleep(config.check_interval)
            elapsed = time.time() - start_time

            total = successful + failed
            error_rate = failed / total if total > 0 else 0

            checkpoint = {
                "elapsed": elapsed,
                "total_requests": total,
                "error_rate": error_rate,
                "latency_avg": statistics.mean(latencies) if latencies else 0,
                "stable": error_rate <= config.error_rate_threshold
            }
            checkpoints.append(checkpoint)

            if error_rate > config.error_rate_threshold:
                logger.warning(f"稳定性测试失败: 错误率 {error_rate:.2%}")
                break

        # 清理
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        end_time = time.time()

        result = self._calculate_results(
            id=f"stability_{int(start_time)}",
            type=BenchmarkType.STABILITY,
            target=config.target,
            start_time=start_time,
            end_time=end_time,
            latencies=latencies,
            successful=successful,
            failed=failed,
            errors=errors,
            metrics={
                "checkpoints": checkpoints,
                "duration": config.duration,
                "error_rate_threshold": config.error_rate_threshold
            }
        )

        self._results.append(result)
        return result

    async def _warmup(self, count: int, request_func: Optional[Callable]):
        """预热"""
        logger.info(f"预热: {count} 请求")
        for _ in range(count):
            try:
                if request_func:
                    await request_func()
                else:
                    await self._simulate_request("warmup")
            except Exception:
                pass
            await asyncio.sleep(0.01)

    async def _simulate_request(self, target: str) -> Dict[str, Any]:
        """模拟请求"""
        # 模拟网络延迟
        latency = random.uniform(10, 100)
        await asyncio.sleep(latency / 1000)

        # 模拟偶尔的失败
        if random.random() < 0.01:
            raise Exception("模拟请求失败")

        return {"status": "ok", "latency": latency}

    def _calculate_results(
        self,
        id: str,
        type: BenchmarkType,
        target: str,
        start_time: float,
        end_time: float,
        latencies: List[float],
        successful: int,
        failed: int,
        errors: List[Dict[str, Any]],
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0
    ) -> BenchmarkResult:
        """计算测试结果"""
        duration = end_time - start_time
        total_requests = successful + failed

        # 计算延迟统计
        if latencies:
            latencies.sort()
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
        else:
            p50 = p95 = p99 = 0

        # 计算错误率
        error_rate = failed / total_requests if total_requests > 0 else 0

        # 计算吞吐量
        throughput = successful / duration if duration > 0 else 0

        return BenchmarkResult(
            id=id,
            type=type,
            target=target,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status=TestStatus.COMPLETED if error_rate < 0.1 else TestStatus.FAILED,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            error_rate=error_rate,
            latency_min=min(latencies) if latencies else 0,
            latency_max=max(latencies) if latencies else 0,
            latency_avg=statistics.mean(latencies) if latencies else 0,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput=throughput,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            metrics={},
            errors=errors[:10]
        )

    async def _measure_cpu_usage(self) -> float:
        """测量CPU使用率"""
        import psutil
        return psutil.cpu_percent()

    async def _measure_memory_usage(self) -> float:
        """测量内存使用率"""
        import psutil
        return psutil.virtual_memory().percent

    def get_results(self, test_id: Optional[str] = None) -> List[BenchmarkResult]:
        """获取测试结果"""
        if test_id:
            return [r for r in self._results if r.id == test_id]
        return self._results

    def clear_results(self):
        """清除结果"""
        self._results.clear()
