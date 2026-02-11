"""
测试模块
Test Module

测试量子优化算法的正确性和性能
"""

import unittest
import numpy as np
import time
from typing import Dict, Any, List

from .qaoa import QAOA
from .vqe import VQE, MoleculeData, MoleculeBuilder
from .variational_forms import (
    HardwareEfficientAnsatz,
    UCCSD,
    QAOAAnsatz,
    VariationalFormFactory
)
from .optimizers import (
    COBYLA,
    SPSA,
    GradientDescent,
    NaturalGradient,
    OptimizerFactory
)
from .config import QuantumOptimizerConfig, PRESETS


class TestQAOA(unittest.TestCase):
    """QAOA测试类"""
    
    def test_qaoa_initialization(self):
        """测试QAOA初始化"""
        qaoa = QAOA(optimizer="cobyla", p_layers=3)
        
        self.assertEqual(qaoa.p_layers, 3)
        self.assertEqual(qaoa.num_params, 6)
        self.assertIsNotNone(qaoa.optimizer)
    
    def test_qaoa_max_cut_triangle(self):
        """测试三角形图最大割"""
        graph = [(0, 1), (1, 2), (2, 0)]
        
        qaoa = QAOA(optimizer="cobyla", p_layers=2)
        result = qaoa.max_cut(graph, num_nodes=3)
        
        # 验证结果
        self.assertIn('cut_value', result)
        self.assertIn('approximation_ratio', result)
        self.assertIn('optimal_params', result)
        self.assertTrue(result['success'] or result['num_iterations'] > 0)
    
    def test_qaoa_max_cut_square(self):
        """测试正方形图最大割"""
        graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        qaoa = QAOA(optimizer="spsa", p_layers=3)
        result = qaoa.max_cut(graph, num_nodes=4)
        
        # 割值应该在边的数量范围内
        self.assertGreaterEqual(result['cut_value'], 0)
        self.assertLessEqual(result['cut_value'], len(graph))
    
    def test_qaoa_sparse_graph(self):
        """测试稀疏图"""
        edges = [(0, 1), (2, 3), (4, 5)]
        weights = [1.0, 1.5, 2.0]
        
        qaoa = QAOA(optimizer="spsa", p_layers=2)
        result = qaoa.max_cut_sparse(edges, weights, num_nodes=6)
        
        self.assertIn('final_energy', result)
        self.assertIn('num_iterations', result)
    
    def test_qaoa_parameter_initialization(self):
        """测试参数初始化"""
        qaoa = QAOA(optimizer="cobyla", p_layers=3)
        initial_params = qaoa._initialize_parameters()
        
        self.assertEqual(len(initial_params), qaoa.num_params)
        self.assertTrue(np.all(initial_params >= 0))
        self.assertTrue(np.all(initial_params <= 2*np.pi))
    
    def test_qaoa_custom_problem(self):
        """测试自定义问题"""
        def simple_cost(params):
            return np.sum(params**2)
        
        qaoa = QAOA(optimizer="cobyla", p_layers=2)
        result = qaoa.solve_custom(simple_cost, num_params=4)
        
        self.assertIn('optimal_params', result)
        self.assertIn('optimal_value', result)
    
    def test_qaoa_history(self):
        """测试历史记录"""
        graph = [(0, 1), (1, 2)]
        
        qaoa = QAOA(optimizer="cobyla", p_layers=1)
        result = qaoa.max_cut(graph, num_nodes=3)
        
        history = qaoa.get_optimization_history()
        self.assertIsInstance(history, list)


class TestVQE(unittest.TestCase):
    """VQE测试类"""
    
    def test_vqe_initialization(self):
        """测试VQE初始化"""
        vqe = VQE(optimizer="spsa")
        
        self.assertIsNotNone(vqe.optimizer)
        self.assertIsNotNone(vqe.ansatz)
    
    def test_vqe_h2_molecule(self):
        """测试氢分子"""
        molecule = MoleculeBuilder.create_h2_molecule()
        
        vqe = VQE(optimizer="spsa")
        result = vqe.compute_energy(molecule)
        
        self.assertIn('ground_state_energy', result)
        self.assertIn('num_iterations', result)
        self.assertIn('num_qubits', result)
    
    def test_vqe_lih_molecule(self):
        """测试LiH分子"""
        molecule = MoleculeBuilder.create_lih_molecule()
        
        vqe = VQE(optimizer="cobyla")
        result = vqe.compute_energy(molecule)
        
        self.assertIn('ground_state_energy', result)
        self.assertGreater(result['num_qubits'], 0)
    
    def test_vqe_custom_hamiltonian(self):
        """测试自定义哈密顿量"""
        hamiltonian = {
            "Z_0": -1.0,
            "Z_1": -1.0,
            "ZZ_0_1": 0.5
        }
        
        vqe = VQE(optimizer="spsa")
        result = vqe.compute_energy_custom_hamiltonian(
            hamiltonian=hamiltonian,
            num_qubits=2
        )
        
        self.assertIn('ground_state_energy', result)
        self.assertIn('optimal_params', result)
    
    def test_vqe_set_ansatz(self):
        """测试设置变分形式"""
        vqe = VQE(optimizer="spsa")
        
        new_ansatz = UCCSD(num_qubits=4, num_electrons=2)
        vqe.set_ansatz(new_ansatz)
        
        self.assertEqual(vqe.ansatz.num_qubits, 4)
    
    def test_vqe_history(self):
        """测试历史记录"""
        molecule = MoleculeBuilder.create_h2_molecule()
        
        vqe = VQE(optimizer="cobyla", config=QuantumOptimizerConfig(max_iterations=50))
        result = vqe.compute_energy(molecule)
        
        history = vqe.get_optimization_history()
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)


class TestVariationalForms(unittest.TestCase):
    """变分形式测试类"""
    
    def test_hardware_efficient_ansatz(self):
        """测试Hardware-Efficient变分形式"""
        ansatz = HardwareEfficientAnsatz(num_qubits=4, depth=3)
        
        self.assertEqual(ansatz.num_parameters, ansatz.num_qubits * (ansatz.depth + 1))
        
        params = ansatz.initial_parameters()
        self.assertEqual(len(params), ansatz.num_parameters)
        
        circuit = ansatz.get_circuit(params)
        self.assertEqual(circuit['type'], 'hardware_efficient')
    
    def test_uccsd_ansatz(self):
        """测试UCCSD变分形式"""
        ansatz = UCCSD(num_qubits=4, num_electrons=2)
        
        self.assertGreater(ansatz.num_parameters, 0)
        
        params = ansatz.initial_parameters()
        self.assertEqual(len(params), ansatz.num_parameters)
        
        circuit = ansatz.get_circuit(params)
        self.assertEqual(circuit['type'], 'uccsd')
    
    def test_qaoa_ansatz(self):
        """测试QAOA变分形式"""
        ansatz = QAOAAnsatz(num_qubits=4, p_layers=3)
        
        self.assertEqual(ansatz.num_parameters, 2 * 3)
        
        params = ansatz.initial_parameters()
        self.assertEqual(len(params), ansatz.num_parameters)
        
        circuit = ansatz.get_circuit(params)
        self.assertEqual(circuit['type'], 'qaoa')
    
    def test_variational_form_factory(self):
        """测试变分形式工厂"""
        forms = VariationalFormFactory.get_available_forms()
        
        self.assertIn('hardware_efficient', forms)
        self.assertIn('uccsd', forms)
        self.assertIn('qaoa', forms)
        
        # 创建变分形式
        ansatz = VariationalFormFactory.create_variational_form(
            'hardware_efficient',
            num_qubits=4,
            depth=2
        )
        self.assertIsInstance(ansatz, HardwareEfficientAnsatz)


class TestOptimizers(unittest.TestCase):
    """优化器测试类"""
    
    def test_cobyla_optimizer(self):
        """测试COBYLA优化器"""
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        optimizer = COBYLA(maxiter=100)
        result = optimizer.minimize(objective, np.array([0.0, 0.0]))
        
        self.assertIn('x', result)
        self.assertIn('fun', result)
        self.assertTrue(result['fun'] < 1.0)
    
    def test_spsa_optimizer(self):
        """测试SPSA优化器"""
        def objective(x):
            return np.sum(x**2)
        
        optimizer = SPSA(maxiter=100, learning_rate=0.1)
        result = optimizer.minimize(objective, np.array([1.0, 1.0, 1.0]))
        
        self.assertIn('x', result)
        self.assertIn('fun', result)
    
    def test_gradient_descent(self):
        """测试梯度下降优化器"""
        def objective(x):
            return (x[0] - 1)**2
        
        optimizer = GradientDescent(maxiter=100, learning_rate=0.1)
        result = optimizer.minimize(objective, np.array([0.0]))
        
        self.assertIn('x', result)
        self.assertTrue(abs(result['x'][0] - 1) < 0.5)
    
    def test_natural_gradient(self):
        """测试自然梯度优化器"""
        def objective(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        optimizer = NaturalGradient(maxiter=100, learning_rate=0.5)
        result = optimizer.minimize(objective, np.array([0.0, 0.0]))
        
        self.assertIn('x', result)
    
    def test_optimizer_factory(self):
        """测试优化器工厂"""
        optimizers = OptimizerFactory.get_available_optimizers()
        
        self.assertIn('cobyla', optimizers)
        self.assertIn('spsa', optimizers)
        
        opt = OptimizerFactory.create_optimizer('cobyla')
        self.assertIsInstance(opt, COBYLA)
    
    def test_optimizer_factory_errors(self):
        """测试优化器工厂错误"""
        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer('unknown_optimizer')


class TestConfig(unittest.TestCase):
    """配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = QuantumOptimizerConfig()
        
        self.assertEqual(config.optimizer, "cobyla")
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.qaoa_layers, 3)
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "optimizer": "spsa",
            "max_iterations": 500,
            "qaoa_layers": 2
        }
        
        config = QuantumOptimizerConfig.from_dict(config_dict)
        
        self.assertEqual(config.optimizer, "spsa")
        self.assertEqual(config.max_iterations, 500)
        self.assertEqual(config.qaoa_layers, 2)
    
    def test_config_to_dict(self):
        """测试配置转字典"""
        config = QuantumOptimizerConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('optimizer', config_dict)
        self.assertIn('max_iterations', config_dict)
    
    def test_default_qaoa_config(self):
        """测试默认QAOA配置"""
        config = QuantumOptimizerConfig.default_qaoa()
        
        self.assertEqual(config.optimizer, "cobyla")
        self.assertEqual(config.qaoa_layers, 3)
    
    def test_default_vqe_config(self):
        """测试默认VQE配置"""
        config = QuantumOptimizerConfig.default_vqe(num_qubits=4)
        
        self.assertEqual(config.optimizer, "spsa")
        self.assertEqual(config.num_qubits, 4)
    
    def test_presets(self):
        """测试预设配置"""
        self.assertIn('fast', PRESETS)
        self.assertIn('accurate', PRESETS)
        self.assertIn('large_scale', PRESETS)


class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def test_qaoa_optimization_speed(self):
        """测试QAOA优化速度"""
        graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        qaoa = QAOA(optimizer="cobyla", p_layers=2)
        
        start_time = time.time()
        result = qaoa.max_cut(graph, num_nodes=4)
        elapsed = time.time() - start_time
        
        # 应该在合理时间内完成
        self.assertLess(elapsed, 10.0)  # 小于10秒
    
    def test_vqe_convergence(self):
        """测试VQE收敛性"""
        molecule = MoleculeBuilder.create_h2_molecule()
        
        vqe = VQE(optimizer="spsa", config=QuantumOptimizerConfig(max_iterations=100))
        result = vqe.compute_energy(molecule)
        
        # 应该收敛
        history = result['convergence_history']
        self.assertGreater(len(history), 0)
    
    def test_large_scale_qaoa(self):
        """测试大规模QAOA"""
        # 生成随机图
        num_nodes = 10
        edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes) if np.random.random() < 0.4]
        
        qaoa = QAOA(optimizer="cobyla", p_layers=2, config=QuantumOptimizerConfig(max_iterations=100))
        result = qaoa.max_cut(edges, num_nodes=num_nodes)
        
        self.assertIn('cut_value', result)
        self.assertIn('num_iterations', result)


class TestMoleculeBuilder(unittest.TestCase):
    """分子构建器测试类"""
    
    def test_create_h2(self):
        """测试氢分子创建"""
        molecule = MoleculeBuilder.create_h2_molecule()
        
        self.assertEqual(len(molecule.geometry), 2)
        self.assertEqual(molecule.num_electrons, 2)
        self.assertEqual(molecule.num_orbitals, 2)
    
    def test_create_lih(self):
        """测试LiH分子创建"""
        molecule = MoleculeBuilder.create_lih_molecule()
        
        self.assertEqual(len(molecule.geometry), 2)
        self.assertEqual(molecule.num_electrons, 4)
    
    def test_create_custom_molecule(self):
        """测试自定义分子创建"""
        atoms = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.0, 0.0, 0.0))
        ]
        
        molecule = MoleculeBuilder.create_custom_molecule(
            atoms=atoms,
            num_electrons=6,
            num_orbitals=4
        )
        
        self.assertEqual(len(molecule.geometry), 2)
        self.assertEqual(molecule.num_electrons, 6)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestQAOA,
        TestVQE,
        TestVariationalForms,
        TestOptimizers,
        TestConfig,
        TestPerformance,
        TestMoleculeBuilder
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_benchmark():
    """运行性能基准测试"""
    print("\n" + "=" * 60)
    print(" Performance Benchmark")
    print("=" * 60)
    
    results = {}
    
    # QAOA基准测试
    print("\n1. QAOA Max-Cut Benchmark:")
    start = time.time()
    for i in range(10):
        graph = [(j, k) for j in range(5) for k in range(j+1, 5) if np.random.random() < 0.5]
        qaoa = QAOA(optimizer="cobyla", p_layers=2)
        result = qaoa.max_cut(graph, num_nodes=5, 
                              config=QuantumOptimizerConfig(max_iterations=50))
    qaoa_time = (time.time() - start) / 10
    print(f"   Average time: {qaoa_time:.4f}s")
    results['qaoa'] = qaoa_time
    
    # VQE基准测试
    print("\n2. VQE Energy Computation Benchmark:")
    start = time.time()
    for i in range(5):
        molecule = MoleculeBuilder.create_h2_molecule()
        vqe = VQE(optimizer="spsa", config=QuantumOptimizerConfig(max_iterations=50))
        result = vqe.compute_energy(molecule)
    vqe_time = (time.time() - start) / 5
    print(f"   Average time: {vqe_time:.4f}s")
    results['vqe'] = vqe_time
    
    # 优化器基准测试
    print("\n3. Optimizer Benchmark:")
    def test_function(x):
        return np.sum((x - 1)**2)
    
    optimizers = [
        ("COBYLA", COBYLA(maxiter=100)),
        ("SPSA", SPSA(maxiter=100, learning_rate=0.1)),
        ("GradientDescent", GradientDescent(maxiter=100, learning_rate=0.1)),
        ("NaturalGradient", NaturalGradient(maxiter=100, learning_rate=0.1))
    ]
    
    for name, opt in optimizers:
        start = time.time()
        result = opt.minimize(test_function, np.zeros(4))
        elapsed = time.time() - start
        print(f"   {name}: {elapsed:.4f}s")
        results[name.lower()] = elapsed
    
    print("\n" + "=" * 60)
    print(" Benchmark completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            run_tests()
        elif command == "benchmark":
            run_benchmark()
        else:
            print("Usage: python test_quantum_optimizer.py [test|benchmark]")
    else:
        run_tests()
