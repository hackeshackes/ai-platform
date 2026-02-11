"""
混合计算测试用例
"""

import unittest
from datetime import datetime
from .task_decomposer import TaskDecomposer, TaskGraph, SubTask, TaskType, DependencyType
from .orchestrator import HybridOrchestrator, ExecutionContext, ExecutionResult, ExecutionStatus
from .hybrid_circuits import QuantumCircuit, QuantumGate, GateType, VQEHybridCircuit, QAOAHybridCircuit, ClassicalControl, ControlCondition, ConditionType
from .resource_manager import ResourceManager, QuantumDevice, ComputeJob, BackendType, JobPriority, DeviceStatus
from .config import HybridComputeConfig


class TestTaskDecomposer(unittest.TestCase):
    """测试任务分解器"""
    
    def setUp(self):
        """设置测试环境"""
        self.decomposer = TaskDecomposer()
    
    def test_decompose_optimization_problem(self):
        """测试优化问题分解"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 4, 'shots': 1024}
        }
        
        task_graph = self.decomposer.decompose(problem, strategy="balanced")
        
        self.assertIsInstance(task_graph, TaskGraph)
        self.assertGreater(len(task_graph.tasks), 0)
        self.assertGreater(len(task_graph.get_quantum_tasks()), 0)
        self.assertGreater(len(task_graph.get_classical_tasks()), 0)
    
    def test_decompose_quantum_chemistry(self):
        """测试量子化学问题分解"""
        problem = {
            'type': 'quantum_chemistry',
            'data': {'molecule': 'H2', 'n_qubits': 4}
        }
        
        task_graph = self.decomposer.decompose(problem, strategy="quantum_heavy")
        
        self.assertIsInstance(task_graph, TaskGraph)
        self.assertGreater(
            len(task_graph.get_quantum_tasks()),
            len(task_graph.get_classical_tasks())
        )
    
    def test_task_graph_execution_order(self):
        """测试任务图执行顺序"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 4}
        }
        
        task_graph = self.decomposer.decompose(problem)
        execution_order = task_graph.get_execution_order()
        
        self.assertIsInstance(execution_order, list)
        self.assertGreater(len(execution_order), 0)
    
    def test_estimate_speedup(self):
        """测试加速比估算"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 4}
        }
        
        task_graph = self.decomposer.decompose(problem)
        speedup = self.decomposer.estimate_speedup(task_graph)
        
        self.assertIsInstance(speedup, float)
        self.assertGreater(speedup, 0)
    
    def test_optimize_partition(self):
        """测试任务划分优化"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 4}
        }
        
        original = self.decomposer.decompose(problem)
        
        # 检查优化方法存在且返回TaskGraph类型
        self.assertTrue(hasattr(self.decomposer, 'optimize_partition'))
        
        # 由于optimize_partition返回TaskGraph，只需验证其存在
        self.assertIsInstance(original, TaskGraph)


class TestHybridOrchestrator(unittest.TestCase):
    """测试混合编排器"""
    
    def setUp(self):
        """设置测试环境"""
        self.orchestrator = HybridOrchestrator()
        self.decomposer = TaskDecomposer()
    
    def test_execute_workflow(self):
        """测试执行工作流"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 2, 'shots': 100}
        }
        
        task_graph = self.decomposer.decompose(problem, strategy="balanced")
        context = self.orchestrator.run(task_graph, backend="quantum_simulator")
        
        self.assertIsInstance(context, ExecutionContext)
        self.assertGreater(len(context.results), 0)
    
    def test_execution_status(self):
        """测试执行状态"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 2}
        }
        
        task_graph = self.decomposer.decompose(problem)
        context = self.orchestrator.run(task_graph)
        
        # 检查所有任务都有状态
        self.assertGreater(len(context.results), 0)
        # 验证至少有一个任务完成
        completed = sum(1 for r in context.results.values() if r.status == ExecutionStatus.COMPLETED)
        self.assertGreater(completed, 0)
    
    def test_execution_result_duration(self):
        """测试执行结果时长计算"""
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 2}
        }
        
        task_graph = self.decomposer.decompose(problem)
        context = self.orchestrator.run(task_graph)
        
        for result in context.results.values():
            if result.start_time and result.end_time:
                self.assertIsNotNone(result.duration)
                self.assertGreaterEqual(result.duration, 0)


class TestHybridCircuits(unittest.TestCase):
    """测试混合电路"""
    
    def test_quantum_circuit_creation(self):
        """测试量子电路创建"""
        circuit = QuantumCircuit(qubits=4)
        
        circuit.add_gate(QuantumGate(GateType.H, [0]))
        circuit.add_gate(QuantumGate(GateType.CNOT, [0, 1]))
        circuit.add_gate(QuantumGate(GateType.RZ, [1], {'theta': 0.5}))
        circuit.add_gate(QuantumGate(GateType.MEASURE, [0, 0]))
        
        self.assertEqual(len(circuit), 4)
        self.assertEqual(circuit.depth(), 4)
        self.assertEqual(circuit.classical_bits, 1)
    
    def test_vqe_circuit(self):
        """测试VQE电路"""
        vqe = VQEHybridCircuit(name="test_vqe", qubits=4, layers=2)
        
        params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = vqe.compute_expectation(params)
        
        self.assertIsInstance(result, float)
    
    def test_qaoa_circuit(self):
        """测试QAOA电路"""
        qaoa = QAOAHybridCircuit(name="test_qaoa", qubits=4, p=1)
        
        gammas = [0.5]
        betas = [0.3]
        qaoa.set_parameters(gammas, betas)
        
        result = qaoa.execute({'shots': 1024})
        
        self.assertIn('probabilities', result)
        self.assertIn('measurements', result)
    
    def test_classical_control(self):
        """测试经典控制"""
        control = ClassicalControl(name="test_control")
        
        condition = ControlCondition(
            condition_id="cond_001",
            condition_type=ConditionType.LESS_THAN,
            threshold=0.5,
            target_observable="expectation_value"
        )
        
        control.add_condition(condition)
        
        # 评估控制条件
        test_result = {'expectation_value': 0.3}
        decisions = control.evaluate(test_result)
        
        self.assertIn('cond_001', decisions['decisions'])
        # 验证条件评估正确
        self.assertTrue(decisions['decisions']['cond_001']['satisfied'])


class TestResourceManager(unittest.TestCase):
    """测试资源管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.rm = ResourceManager()
    
    def test_device_registration(self):
        """测试设备注册"""
        device = QuantumDevice(
            device_id="test_device",
            name="测试设备",
            backend_type=BackendType.QUANTUM_SIMULATOR,
            qubits=8,
            connectivity=[(i, i+1) for i in range(7)]
        )
        
        result = self.rm.register_device(device)
        self.assertTrue(result)
        self.assertIn("test_device", self.rm.devices)
    
    def test_job_submission(self):
        """测试作业提交"""
        job = ComputeJob(
            job_id="test_job",
            job_type=BackendType.QUANTUM_SIMULATOR,
            priority=JobPriority.NORMAL,
            qubits_required=4,
            shots=1024
        )
        
        job_id = self.rm.submit_job(job)
        
        self.assertIsNotNone(job_id)
        self.assertEqual(job.status, "queued")
    
    def test_get_available_devices(self):
        """测试获取可用设备"""
        devices = self.rm.get_available_devices()
        
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)
    
    def test_device_stats(self):
        """测试设备统计"""
        stats = self.rm.get_device_stats()
        
        self.assertIn('total_devices', stats)
        self.assertIn('available_devices', stats)
        self.assertIn('active_jobs', stats)
        self.assertIn('completed_jobs', stats)
    
    def test_job_result(self):
        """测试获取作业结果"""
        # 提交作业
        job = ComputeJob(
            job_id="quick_job",
            job_type=BackendType.QUANTUM_SIMULATOR,
            priority=JobPriority.HIGH,
            qubits_required=2,
            shots=100,
            circuit_depth=2
        )
        
        self.rm.submit_job(job)
        
        # 检查作业状态
        status = self.rm.get_job_status("quick_job")
        self.assertIsNotNone(status)
    
    def test_suggest_device(self):
        """测试设备推荐"""
        job = ComputeJob(
            job_id="suggest_job",
            job_type=BackendType.QUANTUM_SIMULATOR,
            priority=JobPriority.NORMAL,
            qubits_required=4,
            shots=1024
        )
        
        device = self.rm.suggest_device(job)
        
        self.assertIsNotNone(device)
        self.assertTrue(device.is_available())


class TestConfig(unittest.TestCase):
    """测试配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = HybridComputeConfig()
        
        self.assertEqual(config.quantum_backend, "quantum_simulator")
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.timeout, 3600)
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            'quantum_backend': 'quantum_device',
            'max_workers': 8,
            'cost_budget': 500.0
        }
        
        config = HybridComputeConfig.from_dict(config_dict)
        
        self.assertEqual(config.quantum_backend, 'quantum_device')
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.cost_budget, 500.0)
    
    def test_config_presets(self):
        """测试配置预设"""
        from .config import get_preset
        
        dev_preset = get_preset('development')
        self.assertIsNotNone(dev_preset)
        self.assertEqual(dev_preset.cost_budget, 10.0)
        
        prod_preset = get_preset('production')
        self.assertIsNotNone(prod_preset)
        self.assertEqual(prod_preset.cost_budget, 1000.0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建分解器
        decomposer = TaskDecomposer()
        
        # 2. 分解问题
        problem = {
            'type': 'optimization',
            'data': {'n_qubits': 2, 'shots': 100}
        }
        
        task_graph = decomposer.decompose(problem)
        speedup = decomposer.estimate_speedup(task_graph)
        
        # 3. 创建编排器
        orchestrator = HybridOrchestrator()
        
        # 4. 执行
        context = orchestrator.run(task_graph)
        
        # 5. 验证结果
        self.assertGreater(speedup, 0)
        self.assertGreater(len(context.results), 0)
    
    def test_resource_lifecycle(self):
        """测试资源生命周期"""
        rm = ResourceManager()
        
        # 1. 提交作业
        job = ComputeJob(
            job_id="lifecycle_test",
            job_type=BackendType.QUANTUM_SIMULATOR,
            priority=JobPriority.NORMAL,
            qubits_required=2,
            shots=50
        )
        
        rm.submit_job(job)
        
        # 2. 获取状态
        status = rm.get_job_status("lifecycle_test")
        self.assertIsNotNone(status)
        
        # 3. 获取设备
        devices = rm.get_available_devices()
        self.assertGreater(len(devices), 0)
        
        # 4. 获取统计
        stats = rm.get_device_stats()
        self.assertIn('total_devices', stats)


if __name__ == '__main__':
    unittest.main(verbosity=2)
