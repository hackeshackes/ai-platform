"""
混合计算API接口
"""

from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uuid
import logging
from datetime import datetime

from .task_decomposer import TaskDecomposer, TaskGraph
from .orchestrator import HybridOrchestrator, ExecutionContext
from .hybrid_circuits import HybridCircuit, VQEHybridCircuit, QAOAHybridCircuit
from .resource_manager import ResourceManager, ComputeJob, BackendType, JobPriority
from .config import HybridComputeConfig, get_preset

logger = logging.getLogger(__name__)


# FastAPI 应用
app = FastAPI(
    title="量子-经典混合计算平台",
    description="提供量子经典混合计算、任务分解、编排和资源管理API",
    version="1.0.0"
)


# 全局管理器实例
orchestrator: Optional[HybridOrchestrator] = None
resource_manager: Optional[ResourceManager] = None
task_graphs: Dict[str, TaskGraph] = {}
execution_contexts: Dict[str, ExecutionContext] = {}


def get_orchestrator() -> HybridOrchestrator:
    """获取编排器实例"""
    global orchestrator
    if orchestrator is None:
        orchestrator = HybridOrchestrator()
    return orchestrator


def get_resource_manager() -> ResourceManager:
    """获取资源管理器实例"""
    global resource_manager
    if resource_manager is None:
        resource_manager = ResourceManager()
    return resource_manager


# ============ 请求/响应模型 ============

class ProblemDecomposeRequest(BaseModel):
    """问题分解请求"""
    problem_type: str = Field(..., description="问题类型: optimization, quantum_chemistry, machine_learning, cryptography")
    data: Dict[str, Any] = Field(default_factory=dict, description="问题数据")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")
    strategy: str = Field(default="auto", description="分解策略: auto, quantum_heavy, classical_heavy, balanced")


class TaskExecuteRequest(BaseModel):
    """任务执行请求"""
    task_graph_id: str = Field(..., description="任务图ID")
    backend: str = Field(default="quantum_simulator", description="量子后端")


class CircuitExecuteRequest(BaseModel):
    """电路执行请求"""
    circuit_type: str = Field(..., description="电路类型: vqe, qaoa")
    qubits: int = Field(default=4, description="量子比特数")
    parameters: List[float] = Field(default_factory=list, description="电路参数")
    shots: int = Field(default=1024, description="测量次数")


class JobSubmitRequest(BaseModel):
    """作业提交请求"""
    job_type: str = Field(..., description="作业类型")
    qubits_required: int = Field(default=4, description="所需量子比特")
    shots: int = Field(default=1024, description="测量次数")
    circuit_depth: int = Field(default=10, description="电路深度")
    priority: str = Field(default="normal", description="优先级: low, normal, high, urgent")


# ============ API 端点 ============

@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "量子-经典混合计算平台",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============ 任务分解 API ============

@app.post("/api/v1/decompose")
async def decompose_problem(request: ProblemDecomposeRequest) -> Dict[str, Any]:
    """
    分解问题为子任务
    
    Args:
        request: 问题分解请求
        
    Returns:
        Dict: 包含任务图和估算加速比
    """
    try:
        decomposer = TaskDecomposer()
        
        problem = {
            'type': request.problem_type,
            'data': request.data,
            'constraints': request.constraints
        }
        
        task_graph = decomposer.decompose(problem, strategy=request.strategy)
        
        # 保存任务图
        graph_id = str(uuid.uuid4())[:8]
        task_graphs[graph_id] = task_graph
        
        # 估算加速比
        speedup = decomposer.estimate_speedup(task_graph)
        
        # 优化划分
        optimized = decomposer.optimize_partition(task_graph)
        
        return {
            'success': True,
            'graph_id': graph_id,
            'total_tasks': len(task_graph.tasks),
            'quantum_tasks': len(task_graph.get_quantum_tasks()),
            'classical_tasks': len(task_graph.get_classical_tasks()),
            'estimated_speedup': speedup,
            'execution_order': [list(level) for level in task_graph.get_execution_order()]
        }
        
    except Exception as e:
        logger.error(f"问题分解失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/task-graph/{graph_id}")
async def get_task_graph(graph_id: str) -> Dict[str, Any]:
    """获取任务图详情"""
    if graph_id not in task_graphs:
        raise HTTPException(status_code=404, detail="任务图不存在")
    
    task_graph = task_graphs[graph_id]
    
    return {
        'graph_id': graph_id,
        'tasks': {
            tid: {
                'name': t.name,
                'type': t.task_type.value,
                'description': t.description,
                'dependencies': t.dependencies,
                'estimated_cost': t.estimated_cost,
                'estimated_time': t.estimated_time
            }
            for tid, t in task_graph.tasks.items()
        },
        'edges': [
            {'from': e[0], 'to': e[1], 'type': e[2].value}
            for e in task_graph.edges
        ]
    }


# ============ 任务执行 API ============

@app.post("/api/v1/execute")
async def execute_workflow(request: TaskExecuteRequest) -> Dict[str, Any]:
    """
    执行混合工作流
    
    Args:
        request: 任务执行请求
        
    Returns:
        Dict: 执行结果
    """
    if request.task_graph_id not in task_graphs:
        raise HTTPException(status_code=404, detail="任务图不存在")
    
    try:
        orchestrator = get_orchestrator()
        task_graph = task_graphs[request.task_graph_id]
        
        context = orchestrator.run(task_graph, backend=request.backend)
        
        # 保存执行上下文
        execution_id = str(uuid.uuid4())[:8]
        execution_contexts[execution_id] = context
        
        # 统计结果
        completed = sum(1 for r in context.results.values() if r.status.value == 'completed')
        failed = sum(1 for r in context.results.values() if r.status.value == 'failed')
        
        return {
            'success': failed == 0,
            'execution_id': execution_id,
            'total_tasks': len(context.results),
            'completed_tasks': completed,
            'failed_tasks': failed,
            'final_result': context.get_shared_data('final_result')
        }
        
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/execution/{execution_id}/status")
async def get_execution_status(execution_id: str) -> Dict[str, Any]:
    """获取执行状态"""
    if execution_id not in execution_contexts:
        raise HTTPException(status_code=404, detail="执行记录不存在")
    
    context = execution_contexts[execution_id]
    
    return {
        'execution_id': execution_id,
        'total_tasks': len(context.results),
        'results': {
            tid: {
                'status': r.status.value,
                'duration': r.duration,
                'error': r.error
            }
            for tid, r in context.results.items()
        }
    }


# ============ 电路执行 API ============

@app.post("/api/v1/circuit/execute")
async def execute_circuit(request: CircuitExecuteRequest) -> Dict[str, Any]:
    """
    执行混合电路
    
    Args:
        request: 电路执行请求
        
    Returns:
        Dict: 执行结果
    """
    try:
        if request.circuit_type == 'vqe':
            circuit = VQEHybridCircuit(qubits=request.qubits, layers=2)
            circuit.set_parameters(request.parameters)
        elif request.circuit_type == 'qaoa':
            circuit = QAOAHybridCircuit(qubits=request.qubits, p=1)
            n_params = request.qubits
            gammas = request.parameters[:n_params] if request.parameters else [0.5]
            betas = request.parameters[n_params:] if len(request.parameters) > n_params else [0.5]
            circuit.set_parameters(gammas, betas)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的电路类型: {request.circuit_type}")
        
        result = circuit.execute({'shots': request.shots})
        
        return {
            'success': True,
            'circuit_type': request.circuit_type,
            'qubits': request.qubits,
            'result': result
        }
        
    except Exception as e:
        logger.error(f"电路执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 资源管理 API ============

@app.get("/api/v1/devices")
async def list_devices() -> Dict[str, Any]:
    """列出可用设备"""
    rm = get_resource_manager()
    stats = rm.get_device_stats()
    return stats


@app.post("/api/v1/jobs/submit")
async def submit_job(request: JobSubmitRequest) -> Dict[str, Any]:
    """
    提交计算作业
    
    Args:
        request: 作业提交请求
        
    Returns:
        Dict: 作业ID
    """
    try:
        rm = get_resource_manager()
        
        # 转换作业类型
        job_type_map = {
            'quantum_simulator': BackendType.QUANTUM_SIMULATOR,
            'quantum_device': BackendType.QUANTUM_DEVICE,
            'classical_cpu': BackendType.CLASSICAL_CPU,
            'classical_gpu': BackendType.CLASSICAL_GPU
        }
        job_type = job_type_map.get(request.job_type, BackendType.QUANTUM_SIMULATOR)
        
        # 转换优先级
        priority_map = {
            'low': JobPriority.LOW,
            'normal': JobPriority.NORMAL,
            'high': JobPriority.HIGH,
            'urgent': JobPriority.URGENT
        }
        priority = priority_map.get(request.priority, JobPriority.NORMAL)
        
        # 创建作业
        job = ComputeJob(
            job_id=str(uuid.uuid4())[:8],
            job_type=job_type,
            priority=priority,
            qubits_required=request.qubits_required,
            shots=request.shots,
            circuit_depth=request.circuit_depth
        )
        
        job_id = rm.submit_job(job)
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'queued'
        }
        
    except Exception as e:
        logger.error(f"作业提交失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """获取作业状态"""
    rm = get_resource_manager()
    status = rm.get_job_status(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="作业不存在")
    
    return status


@app.get("/api/v1/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    """获取作业结果"""
    rm = get_resource_manager()
    result = rm.get_job_result(job_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="作业不存在或未完成")
    
    return {'job_id': job_id, 'result': result}


@app.get("/api/v1/queues/status")
async def get_queue_status() -> Dict[str, Any]:
    """获取队列状态"""
    rm = get_resource_manager()
    
    return {
        queue_type.value: rm.get_queue_status(queue_type)
        for queue_type in BackendType
    }


# ============ 配置 API ============

@app.get("/api/v1/config/presets")
async def list_config_presets() -> Dict[str, Any]:
    """列出配置预设"""
    return {
        preset_name: config.to_dict()
        for preset_name, config in PRESETS.items()
    }


@app.post("/api/v1/config/apply/{preset_name}")
async def apply_config_preset(preset_name: str) -> Dict[str, Any]:
    """应用配置预设"""
    config = get_preset(preset_name)
    
    if config is None:
        raise HTTPException(status_code=404, detail=f"预设配置不存在: {preset_name}")
    
    # 更新全局编排器配置
    global orchestrator
    if orchestrator is not None:
        orchestrator.config = config.to_dict()
        orchestrator.quantum_backend = config.quantum_backend
        orchestrator.max_workers = config.max_workers
        orchestrator.timeout = config.timeout
    
    return {
        'success': True,
        'preset': preset_name,
        'config': config.to_dict()
    }
