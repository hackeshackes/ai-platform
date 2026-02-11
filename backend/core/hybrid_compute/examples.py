"""
混合计算示例

展示各种量子-经典混合计算场景的使用方法。
"""

from typing import Dict, Any
from .task_decomposer import TaskDecomposer
from .orchestrator import HybridOrchestrator
from .hybrid_circuits import VQEHybridCircuit, QAOAHybridCircuit
from .resource_manager import ResourceManager, ComputeJob, BackendType, JobPriority


def example_optimization_problem():
    """
    示例：优化问题
    
    使用量子采样加速优化问题的求解。
    """
    print("=" * 60)
    print("示例：优化问题")
    print("=" * 60)
    
    # 1. 任务分解
    decomposer = TaskDecomposer()
    
    problem = {
        'type': 'optimization',
        'data': {
            'n_qubits': 4,
            'shots': 1024,
            'circuit_depth': 8,
            'objective_function': 'max_cut'
        },
        'constraints': {
            'max_iterations': 100,
            'tolerance': 1e-6
        }
    }
    
    task_graph = decomposer.decompose(problem, strategy="balanced")
    
    print(f"任务分解完成:")
    print(f"  - 总任务数: {len(task_graph.tasks)}")
    print(f"  - 量子任务: {len(task_graph.get_quantum_tasks())}")
    print(f"  - 经典任务: {len(task_graph.get_classical_tasks())}")
    print(f"  - 估算加速比: {decomposer.estimate_speedup(task_graph):.2f}x")
    
    # 2. 执行工作流
    orchestrator = HybridOrchestrator()
    context = orchestrator.run(task_graph, backend="quantum_simulator")
    
    # 3. 输出结果
    final_result = context.get_shared_data('final_result')
    print(f"\n执行完成:")
    print(f"  - 完成任务: {sum(1 for r in context.results.values() if r.status.value == 'completed')}")
    print(f"  - 失败任务: {sum(1 for r in context.results.values() if r.status.value == 'failed')}")
    
    return context


def example_quantum_chemistry():
    """
    示例：量子化学
    
    使用VQE算法计算分子基态能量。
    """
    print("\n" + "=" * 60)
    print("示例：量子化学")
    print("=" * 60)
    
    # 1. 任务分解
    decomposer = TaskDecomposer()
    
    problem = {
        'type': 'quantum_chemistry',
        'data': {
            'molecule': 'H2',
            'basis': 'sto-3g',
            'n_qubits': 4,
            'ansatz': 'UCCSD',
            'shots': 2048
        }
    }
    
    task_graph = decomposer.decompose(problem, strategy="quantum_heavy")
    
    print(f"任务分解完成:")
    print(f"  - 总任务数: {len(task_graph.tasks)}")
    print(f"  - 量子任务: {len(task_graph.get_quantum_tasks())}")
    print(f"  - 经典任务: {len(task_graph.get_classical_tasks())}")
    print(f"  - 估算加速比: {decomposer.estimate_speedup(task_graph):.2f}x")
    
    # 2. 使用VQE电路
    print("\n使用VQE混合电路:")
    vqe = VQEHybridCircuit(name="H2_VQE", qubits=4, layers=2)
    
    # 设置变分参数
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    energy = vqe.compute_expectation(params)
    print(f"  - 变分参数: {params}")
    print(f"  - 计算能量: {energy}")
    
    # 3. 执行工作流
    orchestrator = HybridOrchestrator()
    context = orchestrator.run(task_graph, backend="quantum_simulator")
    
    print(f"\n执行完成:")
    print(f"  - 完成任务: {sum(1 for r in context.results.values() if r.status.value == 'completed')}")
    
    return context


def example_machine_learning():
    """
    示例：机器学习
    
    使用量子核函数加速SVM训练。
    """
    print("\n" + "=" * 60)
    print("示例：机器学习")
    print("=" * 60)
    
    # 1. 任务分解
    decomposer = TaskDecomposer()
    
    problem = {
        'type': 'machine_learning',
        'data': {
            'dataset': 'iris',
            'feature_dim': 4,
            'n_qubits': 4,
            'kernel_type': 'quantum',
            'train_size': 0.8
        }
    }
    
    task_graph = decomposer.decompose(problem, strategy="balanced")
    
    print(f"任务分解完成:")
    print(f"  - 总任务数: {len(task_graph.tasks)}")
    print(f"  - 量子任务: {len(task_graph.get_quantum_tasks())}")
    print(f"  - 经典任务: {len(task_graph.get_classical_tasks())}")
    print(f"  - 估算加速比: {decomposer.estimate_speedup(task_graph):.2f}x")
    
    # 2. 使用QAOA电路模拟特征映射
    print("\n使用QAOA混合电路:")
    qaoa = QAOAHybridCircuit(name="ML_QAOA", qubits=4, p=1)
    
    # 设置参数
    gammas = [0.5, 0.6]
    betas = [0.3, 0.4]
    qaoa.set_parameters(gammas, betas)
    
    result = qaoa.execute({'shots': 1024})
    print(f"  - QAOA结果概率分布: {result['probabilities']}")
    
    # 3. 执行工作流
    orchestrator = HybridOrchestrator()
    context = orchestrator.run(task_graph, backend="quantum_simulator")
    
    print(f"\n执行完成:")
    print(f"  - 完成任务: {sum(1 for r in context.results.values() if r.status.value == 'completed')}")
    
    return context


def example_resource_management():
    """
    示例：资源管理
    
    展示如何管理量子设备和作业队列。
    """
    print("\n" + "=" * 60)
    print("示例：资源管理")
    print("=" * 60)
    
    # 1. 初始化资源管理器
    rm = ResourceManager()
    
    # 2. 查看可用设备
    devices = rm.get_available_devices()
    print(f"\n可用设备: {len(devices)}")
    for device in devices:
        print(f"  - {device.name} ({device.device_id})")
        print(f"    类型: {device.backend_type.value}")
        print(f"    量子比特: {device.qubits}")
        print(f"    准确度: {device.accuracy}")
    
    # 3. 查看设备统计
    stats = rm.get_device_stats()
    print(f"\n设备统计:")
    print(f"  - 总设备数: {stats['total_devices']}")
    print(f"  - 可用设备: {stats['available_devices']}")
    print(f"  - 活跃作业: {stats['active_jobs']}")
    print(f"  - 队列作业: {stats['queued_jobs']}")
    print(f"  - 已完成作业: {stats['completed_jobs']}")
    
    # 4. 提交作业
    print("\n提交计算作业:")
    job = ComputeJob(
        job_id="job_001",
        job_type=BackendType.QUANTUM_SIMULATOR,
        priority=JobPriority.NORMAL,
        qubits_required=4,
        shots=1024,
        circuit_depth=8
    )
    
    job_id = rm.submit_job(job)
    print(f"  - 作业ID: {job_id}")
    print(f"  - 状态: {job.status}")
    
    # 5. 队列状态
    queue_status = rm.get_queue_status(BackendType.QUANTUM_SIMULATOR)
    print(f"\n量子模拟器队列状态:")
    print(f"  - 队列长度: {queue_status['queue_length']}")
    
    return rm


def example_api_usage():
    """
    示例：API使用
    
    展示如何通过API使用混合计算平台。
    """
    print("\n" + "=" * 60)
    print("示例：API使用")
    print("=" * 60)
    
    # 注意：此示例需要运行FastAPI服务器
    # 启动方式: uvicorn api:app --host 0.0.0.0 --port 8000
    
    endpoints = [
        ("GET", "/", "API根路径"),
        ("GET", "/health", "健康检查"),
        ("POST", "/api/v1/decompose", "问题分解"),
        ("POST", "/api/v1/execute", "执行工作流"),
        ("POST", "/api/v1/circuit/execute", "执行电路"),
        ("GET", "/api/v1/devices", "列出设备"),
        ("POST", "/api/v1/jobs/submit", "提交作业"),
        ("GET", "/api/v1/queues/status", "队列状态"),
    ]
    
    print("\n可用API端点:")
    for method, path, description in endpoints:
        print(f"  [{method}] {path}")
        print(f"    {description}")
    
    print("\n使用示例:")
    print("```bash")
    print("# 健康检查")
    print("curl http://localhost:8000/health")
    print()
    print("# 问题分解")
    print('curl -X POST http://localhost:8000/api/v1/decompose \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"problem_type": "optimization", "data": {"n_qubits": 4}}\'')
    print()
    print("# 提交作业")
    print('curl -X POST http://localhost:8000/api/v1/jobs/submit \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"job_type": "quantum_simulator", "qubits_required": 4}\'')
    print("```")


def run_all_examples():
    """运行所有示例"""
    print("量子-经典混合计算平台 - 示例演示")
    print("=" * 60)
    
    # 运行各个示例
    example_optimization_problem()
    example_quantum_chemistry()
    example_machine_learning()
    example_resource_management()
    example_api_usage()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
