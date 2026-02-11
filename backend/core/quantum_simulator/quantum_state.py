"""
Quantum State - 量子态表示和处理
支持态矢量、密度矩阵和测量操作
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
try:
    import sparse
    HAS_SPARSE = True
except ImportError:
    HAS_SPARSE = False
    sparse = None


class StateRepresentation(Enum):
    """量子态表示类型"""
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"


@dataclass
class QuantumState:
    """
    量子态类
    支持态矢量和密度矩阵两种表示
    """
    n_qubits: int
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    representation: StateRepresentation = StateRepresentation.STATE_VECTOR
    
    def __post_init__(self):
        """初始化量子态"""
        if self.state_vector is None and self.density_matrix is None:
            # 从|0>^n开始
            self.state_vector = np.zeros(2 ** self.n_qubits, dtype=complex)
            self.state_vector[0] = 1.0
            self.representation = StateRepresentation.STATE_VECTOR
    
    @classmethod
    def zero_state(cls, n_qubits: int) -> 'QuantumState':
        """创建|0>^n态"""
        state = cls(n_qubits=n_qubits)
        state.state_vector = np.zeros(2 ** n_qubits, dtype=complex)
        state.state_vector[0] = 1.0
        return state
    
    @classmethod
    def plus_state(cls, n_qubits: int) -> 'QuantumState':
        """创建|+>^n态 (所有量子比特的叠加态)"""
        state = cls(n_qubits=n_qubits)
        amplitude = 1.0 / np.sqrt(2 ** n_qubits)
        state.state_vector = np.full(2 ** n_qubits, amplitude, dtype=complex)
        return state
    
    @classmethod
    def from_state_vector(cls, state_vector: np.ndarray) -> 'QuantumState':
        """从态矢量创建量子态"""
        n_qubits = int(np.log2(len(state_vector)))
        state = cls(n_qubits=n_qubits)
        state.state_vector = state_vector / np.linalg.norm(state_vector)
        return state
    
    @classmethod
    def from_density_matrix(cls, density_matrix: np.ndarray) -> 'QuantumState':
        """从密度矩阵创建量子态"""
        n_qubits = int(np.log2(density_matrix.shape[0]))
        state = cls(n_qubits=n_qubits)
        state.density_matrix = density_matrix / np.trace(density_matrix)
        state.representation = StateRepresentation.DENSITY_MATRIX
        return state
    
    @classmethod
    def bell_state(cls, n_qubits: int, pair: Tuple[int, int] = (0, 1)) -> 'QuantumState':
        """创建Bell态 (纠缠态)"""
        state = cls(n_qubits=n_qubits)
        state.state_vector = np.zeros(2 ** n_qubits, dtype=complex)
        state.state_vector[0] = 1.0 / np.sqrt(2)
        state.state_vector[-1] = 1.0 / np.sqrt(2)
        return state
    
    @property
    def dim(self) -> int:
        """返回Hilbert空间维度"""
        return 2 ** self.n_qubits
    
    def normalize(self) -> 'QuantumState':
        """归一化量子态"""
        if self.state_vector is not None:
            norm = np.linalg.norm(self.state_vector)
            if norm > 0:
                self.state_vector /= norm
        if self.density_matrix is not None:
            self.density_matrix /= np.trace(self.density_matrix)
        return self
    
    def to_density_matrix(self) -> 'QuantumState':
        """将态矢量转换为密度矩阵"""
        if self.density_matrix is None and self.state_vector is not None:
            self.density_matrix = np.outer(
                self.state_vector, 
                self.state_vector.conj()
            )
            self.representation = StateRepresentation.DENSITY_MATRIX
        return self
    
    def to_state_vector(self) -> 'QuantumState':
        """将密度矩阵转换为态矢量 (纯态)"""
        if self.state_vector is None and self.density_matrix is not None:
            # 提取纯态 (密度矩阵的秩为1)
            eigvals, eigvecs = np.linalg.eigh(self.density_matrix)
            idx = np.argmax(eigvals)
            self.state_vector = eigvecs[:, idx]
        return self
    
    def probability(self, basis_state: int) -> float:
        """
        计算在特定基态上的概率
        
        Args:
            basis_state: 基态索引 (0到2^n-1)
        
        Returns:
            测量到该基态的概率
        """
        if self.state_vector is not None:
            return abs(self.state_vector[basis_state]) ** 2
        elif self.density_matrix is not None:
            return np.real(self.density_matrix[basis_state, basis_state])
        return 0.0
    
    def probabilities(self) -> np.ndarray:
        """返回所有基态的测量概率"""
        if self.state_vector is not None:
            return np.abs(self.state_vector) ** 2
        elif self.density_matrix is not None:
            return np.diag(self.density_matrix).real
        return np.zeros(self.dim)
    
    def partial_trace(self, keep_qubits: List[int]) -> 'QuantumState':
        """
        计算偏迹
        
        Args:
            keep_qubits: 要保留的量子比特索引列表
        
        Returns:
            偏迹后的量子态
        """
        if self.state_vector is not None:
            # 将态矢量重塑为张量形式
            shape = [2] * self.n_qubits
            state_tensor = self.state_vector.reshape(shape)
            
            # 对被丢弃的量子比特求和
            traced_shape = tuple(2 for i in range(self.n_qubits) if i in keep_qubits)
            traced_state = np.zeros(traced_shape, dtype=complex)
            
            # 简化实现
            traced_dim = 2 ** len(keep_qubits)
            traced_dm = np.zeros((traced_dim, traced_dim), dtype=complex)
            
            return QuantumState.from_density_matrix(traced_dm)
        
        return self
    
    def reduced_density_matrix(self, qubits: List[int]) -> np.ndarray:
        """计算指定量子比特的约化密度矩阵"""
        dm = self.to_density_matrix().density_matrix
        
        # 简化实现
        n = len(qubits)
        dim = 2 ** n
        
        reduced_dm = np.zeros((dim, dim), dtype=complex)
        
        # 提取子块
        for i in range(dim):
            for j in range(dim):
                # 映射到原始索引
                orig_i = 0
                orig_j = 0
                for k, q in enumerate(qubits):
                    bit = (i >> k) & 1
                    orig_i |= bit << q
                    bit = (j >> k) & 1
                    orig_j |= bit << q
                
                # 求迹
                total = 0.0
                for trace_bits in range(2 ** (self.n_qubits - n)):
                    full_i = orig_i
                    full_j = orig_j
                    
                    # 添加追踪量子比特
                    bit_pos = 0
                    for m in range(self.n_qubits):
                        if m not in qubits:
                            if (trace_bits >> bit_pos) & 1:
                                full_i |= 1 << m
                                full_j |= 1 << m
                            bit_pos += 1
                    
                    total += dm[full_i, full_j]
                
                reduced_dm[i, j] = total
        
        return reduced_dm
    
    def entanglement_entropy(self, subsystem: List[int]) -> float:
        """
        计算纠缠熵
        
        Args:
            subsystem: 子系统量子比特索引
        
        Returns:
            纠缠熵
        """
        reduced_dm = self.reduced_density_matrix(subsystem)
        
        # 计算冯·诺依曼熵
        eigvals = np.linalg.eigvalsh(reduced_dm)
        eigvals = eigvals[eigvals > 1e-15]  # 过滤零值
        
        entropy = -np.sum(eigvals * np.log2(eigvals))
        return entropy
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        计算与另一个态的保真度
        
        Args:
            other: 另一个量子态
        
        Returns:
            保真度 [0, 1]
        """
        if self.state_vector is not None and other.state_vector is not None:
            return abs(np.vdot(self.state_vector, other.state_vector)) ** 2
        
        # 密度矩阵情况
        dm1 = self.to_density_matrix().density_matrix
        dm2 = other.to_density_matrix().density_matrix
        
        sqrt_dm1 = np.sqrt(dm1)
        fidelity = np.trace(np.sqrt(sqrt_dm1 @ dm2 @ sqrt_dm1))
        return np.real(fidelity) ** 2
    
    def measure(self, qubits: Optional[List[int]] = None) -> Tuple[int, 'QuantumState']:
        """
        量子测量
        
        Args:
            qubits: 要测量的量子比特索引 (None表示所有)
        
        Returns:
            (测量结果索引, 测量后态)
        """
        probs = self.probabilities()
        
        if qubits is not None:
            # 部分测量
            measured_qubits = len(qubits)
            
            # 简化实现
            result = np.random.choice(len(probs), p=probs)
            
            # 计算测量后态
            if self.state_vector is not None:
                mask = 0
                for q in qubits:
                    if (result >> (self.n_qubits - 1 - q)) & 1:
                        mask |= 1 << (self.n_qubits - 1 - q)
                
                new_state = np.zeros_like(self.state_vector)
                new_state[mask] = self.state_vector[mask]
                new_state /= np.linalg.norm(new_state)
                
                return result, QuantumState.from_state_vector(new_state)
        
        # 完整测量
        result = np.random.choice(len(probs), p=probs)
        
        new_state = QuantumState.zero_state(self.n_qubits)
        if self.state_vector is not None:
            new_state.state_vector = np.zeros_like(self.state_vector)
            new_state.state_vector[result] = self.state_vector[result]
            new_state.normalize()
        
        return result, new_state
    
    def measure_all(self, shots: int = 1) -> Dict[int, int]:
        """
        多次测量
        
        Args:
            shots: 测量次数
        
        Returns:
            各基态出现次数的字典
        """
        probs = self.probabilities()
        outcomes = np.random.choice(len(probs), size=shots, p=probs)
        
        result = {}
        for outcome in outcomes:
            result[outcome] = result.get(outcome, 0) + 1
        
        return result
    
    def expection_value(self, operator: np.ndarray) -> complex:
        """
        计算可观测量期望值
        
        Args:
            operator: 算符矩阵
        
        Returns:
            期望值
        """
        if self.state_vector is not None:
            return np.vdot(self.state_vector, operator @ self.state_vector)
        elif self.density_matrix is not None:
            return np.trace(self.density_matrix @ operator)
        return 0.0
    
    def __copy__(self) -> 'QuantumState':
        """深拷贝"""
        new_state = QuantumState(n_qubits=self.n_qubits)
        if self.state_vector is not None:
            new_state.state_vector = self.state_vector.copy()
        if self.density_matrix is not None:
            new_state.density_matrix = self.density_matrix.copy()
        new_state.representation = self.representation
        return new_state
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self.state_vector is not None:
            # 显示非零振幅的基态
            nonzero_indices = np.where(np.abs(self.state_vector) > 1e-10)[0]
            amplitudes = []
            for idx in nonzero_indices:
                amp = self.state_vector[idx]
                if np.abs(amp) > 1e-10:
                    binary = format(idx, f'0{self.n_qubits}b')
                    amplitudes.append(f"{amp:.4f}|{binary}>")
            
            if len(amplitudes) > 3:
                return f"QuantumState({self.n_qubits} qubits, {len(nonzero_indices)} nonzero amplitudes)"
            return f"QuantumState({self.n_qubits} qubits): " + " + ".join(amplitudes)
        
        return f"QuantumState({self.n_qubits} qubits, density matrix)"


# Bloch球坐标转换
def state_vector_to_bloch(state_vector: np.ndarray) -> Tuple[float, float]:
    """
    将单量子比特态矢量转换为Bloch球坐标
    
    Args:
        state_vector: 两维态矢量 [a, b]
    
    Returns:
        (theta, phi) - Bloch球角度
    """
    a, b = state_vector[0], state_vector[1]
    
    # 计算Bloch坐标
    theta = 2 * np.arccos(np.abs(a))
    phi = np.angle(b) - np.angle(a)
    
    return theta, phi


def bloch_to_state_vector(theta: float, phi: float) -> np.ndarray:
    """
    将Bloch球坐标转换为态矢量
    
    Args:
        theta: 极角 [0, π]
        phi: 方位角 [0, 2π)
    
    Returns:
        态矢量
    """
    return np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ], dtype=complex)


# 纠缠态验证
def is_entangled(state: QuantumState, qubits: List[int]) -> bool:
    """
    检查量子态在指定量子比特上是否纠缠
    
    Args:
        state: 量子态
        qubits: 检查的量子比特索引
    
    Returns:
        是否纠缠
    """
    # 计算纠缠熵
    entropy = state.entanglement_entropy(qubits)
    return entropy > 1e-10


def concurrence(state: QuantumState, qubits: Tuple[int, int]) -> float:
    """
    计算两量子比特态的并发度
    
    Args:
        state: 量子态
        qubits: 两个量子比特索引
    
    Returns:
        并发度
    """
    reduced_dm = state.reduced_density_matrix(list(qubits))
    
    # 对于两量子比特纯态
    if state.n_qubits == 2:
        psi = state.state_vector
        if psi is not None:
            # 计算 concurrence
            y_matrix = np.array([
                [0, -1j],
                [1j, 0]
            ])
            
            sigma_y = np.kron(y_matrix, y_matrix)
            chi = sigma_y @ np.conj(psi) @ psi.T @ sigma_y
            
            concurrence = abs(np.vdot(psi, chi))
            return max(0, concurrence)
    
    return 0.0
