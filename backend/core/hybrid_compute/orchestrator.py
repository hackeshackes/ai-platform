"""
编排器 - 负责混合任务的调度和执行
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import logging
from datetime import datetime

from .task_decomposer import TaskGraph, SubTask, TaskType, DependencyType

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    task_id: str
    status: ExecutionStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """计算执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class ExecutionContext:
    """执行上下文"""
    task_graph: TaskGraph
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    current_phase: int = 0
    
    def set_result(self, task_id: str, result: ExecutionResult) -> None:
        """设置任务结果"""
        self.results[task_id] = result
    
    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取任务结果"""
        return self.results.get(task_id)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """设置共享数据"""
        self.shared_data[key] = value
    
    def get_shared_data(self, key: str) -> Optional[Any]:
        """获取共享数据"""
        return self.shared_data.get(key)


class HybridOrchestrator:
    """
    混合编排器核心类
    
    负责任务调度、资源分配、量子-经典桥接和结果融合。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化编排器"""
        self.config = config or {}
        self.quantum_backend = self.config.get('quantum_backend', 'simulator')
        self.classical_backend = self.config.get('classical_backend', 'python')
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout = self.config.get('timeout', 3600)  # 默认1小时
        
        # 量子执行器（可配置）
        self.quantum_executor = None
        self.classical_executor = None
        
        # 任务钩子函数
        self.pre_hooks: List[Callable] = []
        self.post_hooks: List[Callable] = []
        
        # 资源管理器
        self.resource_manager = None
    
    def run(self, workflow: TaskGraph, backend: str = "quantum_simulator") -> ExecutionContext:
        """
        执行混合工作流
        
        Args:
            workflow: 任务图
            backend: 量子后端选择
            
        Returns:
            ExecutionContext: 执行上下文，包含所有结果
        """
        # 创建执行上下文
        context = ExecutionContext(task_graph=workflow)
        
        # 设置量子后端
        self.quantum_backend = backend
        
        logger.info(f"开始执行混合工作流，共 {len(workflow.tasks)} 个任务")
        
        try:
            # 根据执行顺序调度任务
            execution_order = workflow.get_execution_order()
            
            for level_idx, level_tasks in enumerate(execution_order):
                logger.info(f"执行层级 {level_idx}: {len(level_tasks)} 个任务")
                
                # 并行执行当前层级的任务
                level_results = self._execute_level(context, level_tasks)
                
                # 合并结果到上下文
                for task_id, result in level_results.items():
                    context.set_result(task_id, result)
                
                # 检查是否有失败
                failed = [r for r in level_results.values() if r.status == ExecutionStatus.FAILED]
                if failed and self.config.get('fail_fast', False):
                    logger.error(f"层级 {level_idx} 执行失败，终止工作流")
                    break
            
            # 融合结果
            final_result = self._fuse_results(context)
            context.set_shared_data('final_result', final_result)
            
            logger.info("工作流执行完成")
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            context.set_shared_data('error', str(e))
        
        return context
    
    async def run_async(self, workflow: TaskGraph, backend: str = "quantum_simulator") -> ExecutionContext:
        """异步执行混合工作流"""
        return await asyncio.to_thread(self.run, workflow, backend)
    
    def _execute_level(self, context: ExecutionContext, task_ids: List[str]) -> Dict[str, ExecutionResult]:
        """执行一层任务"""
        results = {}
        
        # 并行执行可用任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for task_id in task_ids:
                task = context.task_graph.get_task(task_id)
                if task is None:
                    continue
                
                # 检查依赖是否都已完成
                if self._check_dependencies(context, task):
                    future = executor.submit(self._execute_task, context, task)
                    futures[future] = task_id
            
            # 收集结果
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results[task_id] = result
                except Exception as e:
                    logger.error(f"任务 {task_id} 执行异常: {str(e)}")
                    results[task_id] = ExecutionResult(
                        task_id=task_id,
                        status=ExecutionStatus.FAILED,
                        error=str(e)
                    )
        
        return results
    
    def _check_dependencies(self, context: ExecutionContext, task: SubTask) -> bool:
        """检查任务依赖是否都已完成"""
        for dep_id in task.dependencies:
            result = context.get_result(dep_id)
            if result is None or result.status != ExecutionStatus.COMPLETED:
                return False
        return True
    
    def _execute_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行单个任务"""
        start_time = datetime.now()
        
        logger.info(f"开始执行任务: {task.name} ({task.task_id})")
        
        # 执行前置钩子
        for hook in self.pre_hooks:
            try:
                hook(context, task)
            except Exception as e:
                logger.warning(f"前置钩子执行失败: {str(e)}")
        
        try:
            # 根据任务类型执行
            if task.task_type == TaskType.QUANTUM:
                result = self._execute_quantum_task(context, task)
            elif task.task_type == TaskType.CLASSICAL:
                result = self._execute_classical_task(context, task)
            elif task.task_type == TaskType.DATA_PREPROCESSING:
                result = self._execute_preprocessing_task(context, task)
            elif task.task_type == TaskType.RESULT_POSTPROCESSING:
                result = self._execute_postprocessing_task(context, task)
            else:
                result = self._execute_hybrid_task(context, task)
            
            end_time = datetime.now()
            result.start_time = start_time
            result.end_time = end_time
            result.status = ExecutionStatus.COMPLETED
            
            # 执行后置钩子
            for hook in self.post_hooks:
                try:
                    hook(context, task, result)
                except Exception as e:
                    logger.warning(f"后置钩子执行失败: {str(e)}")
            
            logger.info(f"任务 {task.name} 完成，耗时 {result.duration}s")
            
        except Exception as e:
            end_time = datetime.now()
            result = ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=end_time
            )
            logger.error(f"任务 {task.name} 失败: {str(e)}")
        
        return result
    
    def _execute_quantum_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行量子任务"""
        # 准备量子计算输入
        quantum_config = task.quantum_config or {}
        
        # 从上下文获取输入数据
        input_data = self._prepare_input(context, task)
        
        # 模拟量子计算（实际可接入真实量子设备）
        try:
            # 量子采样或演化
            result_data = self._simulate_quantum_computation(quantum_config, input_data)
            
            # 更新共享数据
            context.set_shared_data(f"quantum_result_{task.task_id}", result_data)
            
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.RUNNING,
                output=result_data,
                metrics={
                    'qubits_used': quantum_config.get('qubits', 4),
                    'depth': quantum_config.get('depth', 10),
                    'shots': quantum_config.get('shots', 1024)
                }
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    def _execute_classical_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行经典任务"""
        classical_config = task.classical_config or {}
        input_data = self._prepare_input(context, task)
        
        try:
            # 执行经典计算
            algorithm = classical_config.get('algorithm', 'default')
            result_data = self._run_classical_algorithm(algorithm, input_data, classical_config)
            
            context.set_shared_data(f"classical_result_{task.task_id}", result_data)
            
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.RUNNING,
                output=result_data,
                metrics={
                    'algorithm': algorithm,
                    'iterations': classical_config.get('max_iterations', 1000)
                }
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    def _execute_preprocessing_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行数据预处理任务"""
        input_data = self._prepare_input(context, task)
        
        try:
            processed_data = self._preprocess_data(input_data)
            context.set_shared_data('preprocessed_data', processed_data)
            
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.RUNNING,
                output=processed_data,
                metrics={'samples_processed': len(processed_data) if isinstance(processed_data, list) else 1}
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    def _execute_postprocessing_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行后处理任务"""
        all_results = {
            k: v.output for k, v in context.results.items() 
            if v.status == ExecutionStatus.COMPLETED and v.output is not None
        }
        
        try:
            final_result = self._postprocess_results(all_results)
            
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.RUNNING,
                output=final_result,
                metrics={'input_results_count': len(all_results)}
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    def _execute_hybrid_task(self, context: ExecutionContext, task: SubTask) -> ExecutionResult:
        """执行混合任务"""
        result = self._execute_quantum_task(context, task)
        
        if result.status == ExecutionStatus.FAILED:
            return result
        
        output = result.output
        if task.classical_config:
            classical_config = task.classical_config.copy()
            classical_config['input'] = output
            output = self._run_classical_algorithm(
                classical_config.get('algorithm', 'postprocess'),
                output,
                classical_config
            )
        
        result.output = output
        return result
    
    def _prepare_input(self, context: ExecutionContext, task: SubTask) -> Dict[str, Any]:
        """准备任务输入数据"""
        input_data = {}
        
        for dep_id in task.dependencies:
            dep_result = context.get_result(dep_id)
            if dep_result and dep_result.output is not None:
                input_data[f"from_{dep_id}"] = dep_result.output
        
        for key, spec in task.input_specs.items():
            value = context.get_shared_data(key)
            if value is not None:
                input_data[key] = value
        
        return input_data
    
    def _simulate_quantum_computation(self, config: Dict, input_data: Dict) -> Dict[str, Any]:
        """模拟量子计算"""
        import random
        
        qubits = config.get('qubits', 4)
        depth = config.get('depth', 10)
        shots = config.get('shots', 1024)
        
        measurements = {}
        for _ in range(min(shots, 100)):
            outcome = ''.join(random.choice('01') for _ in range(qubits))
            measurements[outcome] = measurements.get(outcome, 0) + 1
        
        total = sum(measurements.values())
        probabilities = {k: v / total for k, v in measurements.items()}
        
        expectation_value = sum(
            int(bit) * prob for bits in probabilities 
            for i, bit in enumerate(reversed(bits)) 
            for prob in [probabilities[bits]]
            if i == 0
        )
        
        return {
            'measurements': measurements,
            'probabilities': probabilities,
            'expectation_value': expectation_value,
            'circuit_depth': depth,
            'qubits': qubits
        }
    
    def _run_classical_algorithm(self, algorithm: str, input_data: Dict, config: Dict) -> Dict[str, Any]:
        """运行经典算法"""
        import math
        
        if algorithm == 'objective_eval':
            x = input_data.get('x', 0)
            return {'objective_value': x**2 - 4*x + 4}
        
        elif algorithm == 'gradient_compute':
            return {'gradient': [2*x - 4 for x in input_data.get('x', [0])]}
        
        elif algorithm == 'convergence_check':
            return {'converged': True, 'tolerance': 1e-6}
        
        elif algorithm == 'postprocess':
            return {'processed': input_data}
        
        else:
            return {'result': input_data, 'algorithm': algorithm}
    
    def _preprocess_data(self, data: Dict) -> Any:
        """数据预处理"""
        raw = data.get('raw_data', [])
        if isinstance(raw, list):
            if raw:
                mean = sum(raw) / len(raw)
                std = (sum((x - mean)**2 for x in raw) / len(raw)) ** 0.5
                normalized = [(x - mean) / std if std > 0 else 0 for x in raw]
                return normalized
        return data
    
    def _postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """后处理结果"""
        outputs = list(results.values())
        
        fused = {
            'total_tasks': len(outputs),
            'outputs': outputs,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            numeric_values = [v for o in outputs for v in (o.values() if isinstance(o, dict) else [o]) 
                            if isinstance(v, (int, float))]
            if numeric_values:
                fused['statistics'] = {
                    'sum': sum(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values)
                }
        except:
            pass
        
        return fused
    
    def _fuse_results(self, context: ExecutionContext) -> Dict[str, Any]:
        """融合所有任务结果"""
        return {
            'total_tasks': len(context.results),
            'completed_tasks': sum(1 for r in context.results.values() if r.status == ExecutionStatus.COMPLETED),
            'failed_tasks': sum(1 for r in context.results.values() if r.status == ExecutionStatus.FAILED),
            'shared_data_keys': list(context.shared_data.keys()),
            'final_output': context.get_shared_data('final_result')
        }
    
    def add_pre_hook(self, hook: Callable) -> None:
        """添加前置钩子"""
        self.pre_hooks.append(hook)
    
    def add_post_hook(self, hook: Callable) -> None:
        """添加后置钩子"""
        self.post_hooks.append(hook)
