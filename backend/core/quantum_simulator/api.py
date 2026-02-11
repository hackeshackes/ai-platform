"""
Quantum Simulator API - 量子模拟器API
提供高层量子电路模拟接口
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import json

from .quantum_circuit import QuantumCircuit
from .quantum_state import QuantumState
from .quantum_gates import QuantumGate, H_MATRIX, X_MATRIX, CNOT_MATRIX
from .noise_models import NoiseModel, DepolarizingNoise


@dataclass
class SimulationResult:
    """模拟结果"""
    counts: Dict[int, int]  # 测量结果计数
    probabilities: Dict[int, float]  # 概率分布
    state_vector: Optional[np.ndarray] = None  # 最终态矢量
    shots: int = 0  # 总测量次数
    success: bool = True
    message: str = ""
    execution_time: float = 0.0
    n_qubits: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'counts': self.counts,
            'probabilities': self.probabilities,
            'shots': self.shots,
            'success': self.success,
            'message': self.message,
            'execution_time': self.execution_time,
            'n_qubits': self.n_qubits
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    def most_probable(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        返回最可能的测量结果
        
        Args:
            top_k: 返回前k个结果
        
        Returns:
            [(状态索引, 概率), ...]
        """
        sorted_probs = sorted(
            self.probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_probs[:top_k]


class QuantumSimulator:
    """
    量子电路模拟器
    支持态矢量模拟和采样
    """
    
    def __init__(self, 
                 noise_model: Optional[NoiseModel] = None,
                 approximation_level: float = 1.0,
                 use_sparse: bool = False):
        """
        初始化模拟器
        
        Args:
            noise_model: 噪声模型
            approximation_level: 近似级别 (0-1), 1表示完整模拟
            use_sparse: 是否使用稀疏矩阵
        """
        self.noise_model = noise_model
        self.approximation_level = approximation_level
        self.use_sparse = use_sparse
        self.state = None
    
    def initialize(self, n_qubits: int, 
                   state_type: str = "zero") -> QuantumState:
        """
        初始化量子态
        
        Args:
            n_qubits: 量子比特数
            state_type: 初始态类型 ('zero', 'plus', 'random')
        
        Returns:
            初始量子态
        """
        if state_type == "zero":
            self.state = QuantumState.zero_state(n_qubits)
        elif state_type == "plus":
            self.state = QuantumState.plus_state(n_qubits)
        elif state_type == "random":
            random_vec = np.random.rand(2 ** n_qubits) + 1j * np.random.rand(2 ** n_qubits)
            self.state = QuantumState.from_state_vector(random_vec)
        else:
            raise ValueError(f"Unknown state type: {state_type}")
        
        return self.state
    
    def run(self, circuit: QuantumCircuit, 
            shots: int = 1024,
            initial_state: Optional[QuantumState] = None) -> SimulationResult:
        """
        运行量子电路
        
        Args:
            circuit: 量子电路
            shots: 测量次数
            initial_state: 初始态 (覆盖initialize设置)
        
        Returns:
            SimulationResult: 模拟结果
        """
        import time
        start_time = time.time()
        
        try:
            n_qubits = circuit.n_qubits
            dim = 2 ** n_qubits
            
            # 初始化态矢量 |0...0>
            state_vector = np.zeros(dim, dtype=complex)
            state_vector[0] = 1.0
            
            # 应用每个门
            for op in circuit.gates:
                gate_matrix = op.gate.matrix
                qubits = op.qubits
                
                if isinstance(qubits, int):
                    # 单量子比特门
                    state_vector = self._apply_single_gate(
                        state_vector, gate_matrix, qubits, n_qubits
                    )
                else:
                    # 双量子比特门
                    state_vector = self._apply_two_qubit_gate(
                        state_vector, gate_matrix, qubits, n_qubits
                    )
            
            # 归一化
            norm = np.linalg.norm(state_vector)
            if norm > 0:
                state_vector /= norm
            
            # 测量采样
            probabilities = np.abs(state_vector) ** 2
            probabilities = probabilities.real
            
            # 检查概率和
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities /= prob_sum
            
            # 采样
            outcomes = np.random.choice(
                dim, size=shots, p=probabilities
            )
            
            # 统计结果
            counts = {}
            for outcome in outcomes:
                counts[outcome] = counts.get(outcome, 0) + 1
            
            # 转换counts为概率
            probabilities_result = {k: v / shots for k, v in counts.items()}
            
            execution_time = time.time() - start_time
            
            return SimulationResult(
                counts=counts,
                probabilities=probabilities_result,
                state_vector=state_vector,
                shots=shots,
                success=True,
                execution_time=execution_time,
                n_qubits=n_qubits
            )
            
        except Exception as e:
            return SimulationResult(
                counts={},
                probabilities={},
                shots=shots,
                success=False,
                message=str(e),
                n_qubits=circuit.n_qubits
            )
    
    def _apply_single_gate(self, state: np.ndarray, 
                          gate: np.ndarray, 
                          qubit: int,
                          n_qubits: int) -> np.ndarray:
        """应用单量子比特门"""
        dim = 2 ** n_qubits
        new_state = np.zeros(dim, dtype=complex)
        
        qubit_shift = n_qubits - 1 - qubit
        mask = 1 << qubit_shift
        
        for i in range(dim):
            # 检查该量子比特是0还是1
            bit = (i & mask) >> qubit_shift
            
            if bit == 0:
                # 如果是|0>分支
                new_state[i] += gate[0, 0] * state[i]
                # 映射到|1>分支
                new_idx = i | mask
                new_state[new_idx] += gate[1, 0] * state[i]
            else:
                # 如果是|1>分支
                new_state[i] += gate[1, 1] * state[i]
                # 映射到|0>分支
                new_idx = i & ~mask
                new_state[new_idx] += gate[0, 1] * state[i]
        
        return new_state
    
    def _apply_two_qubit_gate(self, state: np.ndarray,
                             gate: np.ndarray,
                             qubits: Tuple[int, int],
                             n_qubits: int) -> np.ndarray:
        """应用双量子比特门"""
        dim = 2 ** n_qubits
        new_state = np.zeros(dim, dtype=complex)
        
        control, target = qubits
        control_shift = n_qubits - 1 - control
        target_shift = n_qubits - 1 - target
        
        control_mask = 1 << control_shift
        target_mask = 1 << target_shift
        
        # 处理CNOT和其他双门
        gate_name = None
        if gate.shape == (4, 4):
            # 检查是否为CNOT
            if np.allclose(gate, CNOT_MATRIX):
                gate_name = 'cnot'
            elif np.allclose(gate, np.eye(4, dtype=complex)):
                gate_name = 'identity'
        
        for i in range(dim):
            control_bit = (i & control_mask) >> control_shift
            
            if gate_name == 'cnot':
                if control_bit == 1:
                    # 翻转目标位
                    new_idx = i ^ target_mask
                    new_state[new_idx] = state[i]
                else:
                    new_state[i] = state[i]
            elif gate_name == 'identity':
                new_state[i] = state[i]
            else:
                # 通用双门处理 (简化)
                target_bit = (i & target_mask) >> target_shift
                
                # 构建双量子比特索引
                two_bit_idx = (control_bit << 1) | target_bit
                
                # 计算新状态
                for j in range(4):
                    new_two_bit = j
                    new_control_bit = (new_two_bit >> 1) & 1
                    new_target_bit = new_two_bit & 1
                    
                    # 检查是否只有目标位改变
                    if new_control_bit == control_bit:
                        new_idx = i
                        if new_target_bit != target_bit:
                            if new_target_bit == 1:
                                new_idx = i | target_mask
                            else:
                                new_idx = i & ~target_mask
                        
                        new_state[new_idx] += gate[two_bit_idx, new_two_bit] * state[i]
        
        return new_state
    
    def _apply_noise(self, state: np.ndarray,
                     noise_model: NoiseModel,
                     qubit: int,
                     n_qubits: int) -> np.ndarray:
        """应用噪声"""
        # 简化实现
        return state
    
    def get_state(self) -> Optional[QuantumState]:
        """获取当前量子态"""
        return self.state
    
    def reset(self):
        """重置模拟器"""
        self.state = None


def run_circuit(circuit: QuantumCircuit,
               shots: int = 1024,
               noise_model: Optional[NoiseModel] = None,
               verbose: bool = False) -> SimulationResult:
    """
    运行量子电路的便捷函数
    
    Args:
        circuit: 量子电路
        shots: 测量次数
        noise_model: 噪声模型
        verbose: 详细输出
    
    Returns:
        SimulationResult
    """
    simulator = QuantumSimulator(noise_model=noise_model)
    result = simulator.run(circuit, shots=shots)
    
    if verbose:
        print(f"Circuit: {circuit.name}")
        print(f"Qubits: {circuit.n_qubits}")
        print(f"Depth: {circuit.depth}")
        print(f"Gates: {circuit.num_gates}")
        print(f"Execution time: {result.execution_time:.4f}s")
        print(f"Results: {result.counts}")
    
    return result


def estimate_resources(circuit: QuantumCircuit) -> Dict:
    """
    估算电路资源需求
    
    Args:
        circuit: 量子电路
    
    Returns:
        资源估算字典
    """
    n_qubits = circuit.n_qubits
    dim = 2 ** n_qubits
    
    # 内存需求 - complex128 = 16 bytes
    state_memory = dim * 16
    
    resources = {
        'n_qubits': n_qubits,
        'hilbert_space_dim': dim,
        'state_memory_bytes': state_memory,
        'state_memory_mb': state_memory / (1024 * 1024),
        'gate_count': circuit.num_gates,
        'circuit_depth': circuit.depth,
        'estimated_gates_per_qubit': circuit.num_gates / n_qubits if n_qubits > 0 else 0,
        'gate_counts': circuit.gate_counts,
        'suitable_for_simulation': dim <= 2 ** 20,  # 约1M维度
        'recommendation': 'full_simulation' if dim <= 2 ** 20 else 'use_sampling'
    }
    
    if dim > 2 ** 20:
        resources['recommendation'] = 'hybrid_approach'
        resources['warning'] = 'Large Hilbert space, consider sampling or tensor networks'
    
    return resources


class BatchSimulator:
    """批量模拟器"""
    
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.circuits = []
    
    def add_circuit(self, circuit: QuantumCircuit, shots: int = 1024):
        """添加要模拟的电路"""
        self.circuits.append((circuit, shots))
    
    def run_all(self) -> List[SimulationResult]:
        """运行所有电路"""
        results = []
        for circuit, shots in self.circuits:
            result = run_circuit(circuit, shots=shots)
            results.append(result)
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 预定义电路模板
def create_quantum_volume_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """创建量子体积电路"""
    circuit = QuantumCircuit(n_qubits=n_qubits, name="QuantumVolume")
    
    for _ in range(depth):
        # 随机单门
        for qubit in range(n_qubits):
            gate_type = np.random.choice(['h', 't'])
            if gate_type == 'h':
                circuit.h(qubit)
            else:
                circuit.t(qubit)
        
        # 随机纠缠
        for qubit in range(n_qubits):
            target = np.random.randint(n_qubits)
            if target != qubit:
                circuit.cnot(qubit, target)
    
    return circuit


def create_qaoa_circuit(n_qubits: int, 
                       p: int,
                       gamma: List[float],
                       beta: List[float],
                       problem_graph: Optional[np.ndarray] = None) -> QuantumCircuit:
    """创建QAOA电路"""
    circuit = QuantumCircuit(n_qubits=n_qubits, name="QAOA")
    
    # 初始叠加态
    for qubit in range(n_qubits):
        circuit.h(qubit)
    
    # QAOA层
    for i in range(p):
        # 成本层
        if problem_graph is not None:
            for edge in np.argwhere(problem_graph > 0):
                circuit.rz(edge[1], 2 * gamma[i])
                circuit.cnot(edge[0], edge[1])
                circuit.rz(edge[1], -2 * gamma[i])
                circuit.cnot(edge[0], edge[1])
        
        # 混合层
        for qubit in range(n_qubits):
            circuit.rx(qubit, 2 * beta[i])
    
    return circuit


# 简单的主函数示例
if __name__ == "__main__":
    # 示例1: Bell态
    print("=== Bell State Example ===")
    circuit = QuantumCircuit(n_qubits=2, name="Bell")
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()
    
    result = run_circuit(circuit, shots=1000, verbose=True)
    print()
    
    # 示例2: GHZ态
    print("=== GHZ State Example ===")
    circuit2 = QuantumCircuit(n_qubits=3, name="GHZ")
    circuit2.ghz_state()
    circuit2.measure_all()
    
    result2 = run_circuit(circuit2, shots=500, verbose=True)
    print()
    
    # 示例3: 资源估算
    print("=== Resource Estimation ===")
    circuit3 = QuantumCircuit(n_qubits=50)
    circuit3.ansatz(depth=3)
    resources = estimate_resources(circuit3)
    print(f"Qubits: {resources['n_qubits']}")
    print(f"Gate count: {resources['gate_count']}")
    print(f"State memory: {resources['state_memory_mb']:.2f} MB")
