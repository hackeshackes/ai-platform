"""
Quantum Simulator Tests - 测试用例
"""

import unittest
import numpy as np

import sys
sys.path.insert(0, '/Users/yubao/.openclaw/projects/ai-platform/backend')

from core.quantum_simulator.quantum_circuit import QuantumCircuit, bell_circuit, ghz_circuit, random_circuit
from core.quantum_simulator.quantum_state import QuantumState
from core.quantum_simulator.quantum_gates import H, X, Y, Z, S, T, CNOT, CZ, SWAP, QuantumGate
from core.quantum_simulator.noise_models import DepolarizingNoise, PhaseNoise, AmplitudeDamping
from core.quantum_simulator.api import run_circuit, estimate_resources, QuantumSimulator


class TestQuantumGates(unittest.TestCase):
    """测试量子门"""
    
    def test_hadamard_gate(self):
        """测试Hadamard门"""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.h(0)
        
        result = run_circuit(circuit, shots=1000)
        
        # |+>态, 测量0和1应该各约50%
        self.assertAlmostEqual(result.probabilities.get(0, 0), 0.5, delta=0.1)
        self.assertAlmostEqual(result.probabilities.get(1, 0), 0.5, delta=0.1)
    
    def test_pauli_gates(self):
        """测试Pauli门"""
        # X门翻转
        circuit = QuantumCircuit(n_qubits=1)
        circuit.x(0)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        self.assertAlmostEqual(result.probabilities.get(1, 0), 1.0, delta=0.05)
    
    def test_cnot_gate(self):
        """测试CNOT门"""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.h(0)
        circuit.cnot(0, 1)  # 创建Bell态
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        
        # 应该是|00>或|11>
        p00 = result.probabilities.get(0, 0)
        p11 = result.probabilities.get(3, 0)
        
        self.assertAlmostEqual(p00 + p11, 1.0, delta=0.05)
        self.assertAlmostEqual(p00, 0.5, delta=0.1)
        self.assertAlmostEqual(p11, 0.5, delta=0.1)
    
    def test_swap_gate(self):
        """测试SWAP门"""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.x(0)  # |10>
        circuit.swap(0, 1)  # 应该变成|01>
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        self.assertAlmostEqual(result.probabilities.get(2, 0), 1.0, delta=0.05)
    
    def test_rotation_gates(self):
        """测试旋转门"""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.rx(0, np.pi / 2)
        circuit.ry(0, np.pi / 2)
        circuit.rz(0, np.pi / 4)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, delta=0.01)


class TestQuantumCircuits(unittest.TestCase):
    """测试量子电路"""
    
    def test_bell_circuit(self):
        """测试Bell电路"""
        circuit = bell_circuit(n_qubits=2)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        
        # 验证纠缠
        p00 = result.probabilities.get(0, 0)
        p11 = result.probabilities.get(3, 0)
        self.assertAlmostEqual(p00 + p11, 1.0, delta=0.05)
    
    def test_ghz_circuit(self):
        """测试GHZ电路"""
        circuit = ghz_circuit(n_qubits=3)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=1000)
        
        p000 = result.probabilities.get(0, 0)
        p111 = result.probabilities.get(7, 0)
        self.assertAlmostEqual(p000 + p111, 1.0, delta=0.05)
    
    def test_random_circuit(self):
        """测试随机电路"""
        circuit = random_circuit(n_qubits=5, depth=3, seed=42)
        
        self.assertEqual(circuit.n_qubits, 5)
        self.assertGreater(circuit.num_gates, 0)
    
    def test_circuit_depth(self):
        """测试电路深度"""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)
        
        self.assertEqual(circuit.depth, 3)
    
    def test_circuit_gate_counts(self):
        """测试门计数"""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.h(0)
        circuit.h(1)
        circuit.cnot(0, 1)
        
        counts = circuit.gate_counts
        self.assertEqual(counts.get('h', 0), 2)
        self.assertEqual(counts.get('cnot', 0), 1)
    
    def test_qft_circuit(self):
        """测试QFT电路"""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.qft()
        
        result = run_circuit(circuit, shots=100)
        
        self.assertTrue(len(result.probabilities) > 0)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, delta=0.01)


class TestQuantumState(unittest.TestCase):
    """测试量子态"""
    
    def test_zero_state(self):
        """测试|0>态"""
        state = QuantumState.zero_state(3)
        
        self.assertEqual(state.n_qubits, 3)
        self.assertAlmostEqual(state.state_vector[0], 1.0)
        self.assertAlmostEqual(np.linalg.norm(state.state_vector), 1.0)
    
    def test_plus_state(self):
        """测试|+>态"""
        state = QuantumState.plus_state(2)
        
        self.assertAlmostEqual(np.linalg.norm(state.state_vector), 1.0)
    
    def test_state_normalization(self):
        """测试态归一化"""
        state = QuantumState.zero_state(2)
        state.state_vector[1] = 0.5
        
        state.normalize()
        
        self.assertAlmostEqual(np.linalg.norm(state.state_vector), 1.0)
    
    def test_bell_state(self):
        """测试Bell态"""
        state = QuantumState.bell_state(2)
        
        self.assertAlmostEqual(abs(state.state_vector[0]), 1/np.sqrt(2))
        self.assertAlmostEqual(abs(state.state_vector[3]), 1/np.sqrt(2))
    
    def test_measurement_probability(self):
        """测试测量概率"""
        state = QuantumState.zero_state(1)
        
        p0 = state.probability(0)
        p1 = state.probability(1)
        
        self.assertAlmostEqual(p0, 1.0)
        self.assertAlmostEqual(p1, 0.0)


class TestNoiseModels(unittest.TestCase):
    """测试噪声模型"""
    
    def test_depolarizing_noise(self):
        """测试去极化噪声"""
        noise = DepolarizingNoise(probability=0.1)
        
        Kraus = noise.Kraus_operators()
        
        self.assertTrue(len(Kraus) > 0)
    
    def test_phase_noise(self):
        """测试相位噪声"""
        noise = PhaseNoise(probability=0.05)
        
        Kraus = noise.Kraus_operators()
        
        self.assertTrue(len(Kraus) == 2)
    
    def test_amplitude_damping(self):
        """测试振幅阻尼"""
        noise = AmplitudeDamping(probability=0.1)
        
        Kraus = noise.Kraus_operators()
        
        self.assertTrue(len(Kraus) == 2)


class TestAPIFunctions(unittest.TestCase):
    """测试API函数"""
    
    def test_run_circuit(self):
        """测试运行电路"""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=100)
        
        self.assertTrue(result.success)
        self.assertEqual(result.shots, 100)
        self.assertGreater(len(result.counts), 0)
    
    def test_estimate_resources(self):
        """测试资源估算"""
        circuit = QuantumCircuit(n_qubits=20)
        circuit.ansatz(depth=2)
        
        resources = estimate_resources(circuit)
        
        self.assertEqual(resources['n_qubits'], 20)
        self.assertTrue(resources['suitable_for_simulation'])
    
    def test_simulator_result(self):
        """测试模拟结果"""
        circuit = QuantumCircuit(n_qubits=1)
        circuit.h(0)
        circuit.measure_all()
        
        result = run_circuit(circuit, shots=100)
        
        most_prob = result.most_probable(2)
        self.assertTrue(len(most_prob) > 0)


class TestLargeScale(unittest.TestCase):
    """大规模测试"""
    
    def test_50_qubits(self):
        """测试50量子比特"""
        circuit = QuantumCircuit(n_qubits=50)
        circuit.ansatz(depth=2)
        
        resources = estimate_resources(circuit)
        
        self.assertEqual(resources['n_qubits'], 50)
        # 50量子比特: 2^50 * 16 bytes = 16 PB = 16000000 TB
        # 实际上50量子比特状态无法完整存储在内存中
        print(f"50 qubits memory: {resources['state_memory_mb']:.2f} MB")
    
    def test_100_qubits(self):
        """测试100量子比特"""
        circuit = QuantumCircuit(n_qubits=100)
        circuit.ansatz(depth=1)
        
        resources = estimate_resources(circuit)
        
        self.assertEqual(resources['n_qubits'], 100)
        # 100量子比特需要约25MB状态空间 (实际需要更多)
        print(f"100 qubits memory: {resources['state_memory_mb']:.2f} MB")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestQuantumGates,
        TestQuantumCircuits,
        TestQuantumState,
        TestNoiseModels,
        TestAPIFunctions,
        TestLargeScale
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
