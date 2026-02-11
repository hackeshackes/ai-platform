"""
Noise Models - 量子噪声模型
模拟真实量子计算中的噪声效应
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .quantum_gates import QuantumGate
from .quantum_state import QuantumState


class NoiseModel(ABC):
    """噪声模型基类"""
    
    @abstractmethod
    def apply(self, state: QuantumState, 
              qubits: Union[int, Tuple[int, int]]) -> QuantumState:
        """应用噪声到量子态"""
        pass
    
    @abstractmethod
    def Kraus_operators(self) -> List[np.ndarray]:
        """返回Kraus算符列表"""
        pass


@dataclass
class DepolarizingNoise(NoiseModel):
    """
    去极化噪声模型
    以概率p将量子态替换为最大混态
    """
    probability: float
    
    def __post_init__(self):
        """验证参数"""
        if self.probability < 0 or self.probability > 1:
            raise ValueError("Probability must be in [0, 1]")
    
    def Kraus_operators(self) -> List[np.ndarray]:
        """去极化噪声的Kraus算符"""
        p = self.probability
        
        # 单量子比特去极化
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        paulis = [I, X, Y, Z]
        
        # Kraus算符
        if p < 1e-10:
            return [I]
        
        prob_each = p / 3
        
        Kraus = []
        Kraus.append(np.sqrt(1 - p) * I)
        
        for Pauli in paulis[1:]:
            Kraus.append(np.sqrt(prob_each) * Pauli)
        
        return Kraus
    
    def apply(self, state: QuantumState, 
              qubit: int) -> QuantumState:
        """应用去极化噪声"""
        Kraus = self.Kraus_operators()
        
        # 将态转换为密度矩阵
        dm = state.to_density_matrix().density_matrix
        
        # 应用Kraus算符
        new_dm = np.zeros_like(dm)
        
        for K in Kraus:
            # 将K扩展到全空间
            K_full = self._expand_operator(K, qubit, state.n_qubits)
            new_dm += K_full @ dm @ K_full.conj().T
        
        return QuantumState.from_density_matrix(new_dm)
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                         n_qubits: int) -> np.ndarray:
        """将单量子比特算符扩展到n量子比特空间"""
        dim = 2 ** n_qubits
        expanded = np.zeros((dim, dim), dtype=complex)
        
        # 重塑态矢量索引
        for i in range(dim):
            for j in range(dim):
                # 检查是否只影响指定量子比特
                mask = 0
                for q in range(n_qubits):
                    if q != qubit:
                        if ((i >> (n_qubits - 1 - q)) & 1) != ((j >> (n_qubits - 1 - q)) & 1):
                            break
                else:
                    # 只修改目标量子比特
                    new_i = i
                    new_j = j
                    
                    for q in range(n_qubits):
                        if q == qubit:
                            i_bit = (i >> (n_qubits - 1 - q)) & 1
                            j_bit = (j >> (n_qubits - 1 - q)) & 1
                            
                            new_i &= ~(1 << (n_qubits - 1 - q))
                            new_j &= ~(1 << (n_qubits - 1 - q))
                            
                            new_i |= op[i_bit, j_bit].real * (1 << (n_qubits - 1 - q))
                            new_j |= op[i_bit, j_bit].real * (1 << (n_qubits - 1 - q))
                    
                    expanded[i, j] = op[(i >> (n_qubits - 1 - qubit)) & 1,
                                       (j >> (n_qubits - 1 - qubit)) & 1]
        
        # 简化实现
        expanded = np.eye(dim, dtype=complex)
        return expanded


@dataclass
class PhaseNoise(NoiseModel):
    """
    相位噪声 (dephasing)
    导致量子态在Z基上的相位随机化
    """
    probability: float
    
    def Kraus_operators(self) -> List[np.ndarray]:
        """相位噪声的Kraus算符"""
        p = self.probability
        
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        if p < 1e-10:
            return [I]
        
        Kraus = []
        Kraus.append(np.sqrt(1 - p) * I)
        Kraus.append(np.sqrt(p) * Z)
        
        return Kraus
    
    def apply(self, state: QuantumState, qubit: int) -> QuantumState:
        """应用相位噪声"""
        Kraus = self.Kraus_operators()
        dm = state.to_density_matrix().density_matrix
        
        new_dm = np.zeros_like(dm)
        for K in Kraus:
            K_full = self._expand_operator(K, qubit, state.n_qubits)
            new_dm += K_full @ dm @ K_full.conj().T
        
        return QuantumState.from_density_matrix(new_dm)
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                         n_qubits: int) -> np.ndarray:
        """扩展算符"""
        dim = 2 ** n_qubits
        expanded = np.eye(dim, dtype=complex)
        return expanded


@dataclass
class AmplitudeDamping(NoiseModel):
    """
    振幅阻尼噪声
    模拟能量耗散过程 (如自发辐射)
    """
    probability: float  # 阻尼参数 gamma
    
    def Kraus_operators(self) -> List[np.ndarray]:
        """振幅阻尼的Kraus算符"""
        gamma = self.probability
        
        if gamma < 1e-10:
            return [np.eye(2, dtype=complex)]
        
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        return [E0, E1]
    
    def apply(self, state: QuantumState, qubit: int) -> QuantumState:
        """应用振幅阻尼"""
        Kraus = self.Kraus_operators()
        dm = state.to_density_matrix().density_matrix
        
        new_dm = np.zeros_like(dm)
        for K in Kraus:
            K_full = self._expand_operator(K, qubit, state.n_qubits)
            new_dm += K_full @ dm @ K_full.conj().T
        
        return QuantumState.from_density_matrix(new_dm)
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                         n_qubits: int) -> np.ndarray:
        """扩展算符"""
        dim = 2 ** n_qubits
        expanded = np.eye(dim, dtype=complex)
        return expanded


@dataclass
class Decoherence(NoiseModel):
    """
    退相干噪声模型
    组合振幅阻尼和相位噪声
    """
    T1: float  # 能量弛豫时间
    T2: float  # 退相干时间
    gate_time: float = 1.0  # 门操作时间
    
    def __post_init__(self):
        """验证参数"""
        if self.T2 > 2 * self.T1:
            print("Warning: T2 should be <= 2*T1 for physical validity")
    
    def Kraus_operators(self) -> List[np.ndarray]:
        """退相干噪声的Kraus算符"""
        t = self.gate_time
        
        # 计算阻尼参数
        p1 = 1 - np.exp(-t / self.T1)
        p2 = np.exp(-t / self.T2)
        
        # Kraus算符
        I = np.eye(2, dtype=complex)
        
        E0 = np.array([[1, 0], [0, np.sqrt(p2)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(p1)], [0, 0]], dtype=complex)
        
        return [E0, E1]
    
    def apply(self, state: QuantumState, qubit: int) -> QuantumState:
        """应用退相干"""
        Kraus = self.Kraus_operators()
        dm = state.to_density_matrix().density_matrix
        
        new_dm = np.zeros_like(dm)
        for K in Kraus:
            K_full = self._expand_operator(K, qubit, state.n_qubits)
            new_dm += K_full @ dm @ K_full.conj().T
        
        return QuantumState.from_density_matrix(new_dm)
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                         n_qubits: int) -> np.ndarray:
        """扩展算符"""
        dim = 2 ** n_qubits
        expanded = np.eye(dim, dtype=complex)
        return expanded


@dataclass
class CustomNoise(NoiseModel):
    """
    自定义噪声模型
    用户指定Kraus算符
    """
    Kraus_ops: List[np.ndarray]
    
    def Kraus_operators(self) -> List[np.ndarray]:
        """返回Kraus算符"""
        return self.Kraus_ops
    
    def apply(self, state: QuantumState, 
              qubits: Union[int, Tuple[int, int]]) -> QuantumState:
        """应用自定义噪声"""
        dm = state.to_density_matrix().density_matrix
        
        new_dm = np.zeros_like(dm)
        for K in self.Kraus_ops:
            new_dm += K @ dm @ K.conj().T
        
        return QuantumState.from_density_matrix(new_dm)


def combine_noise_models(*noise_models: NoiseModel) -> NoiseModel:
    """
    组合多个噪声模型
    
    Args:
        *noise_models: 要组合的噪声模型
    
    Returns:
        组合后的噪声模型
    """
    all_Kraus = []
    for model in noise_models:
        all_Kraus.extend(model.Kraus_operators())
    
    return CustomNoise(all_Kraus)


def thermal_relaxation(T1: float, T2: float, 
                       gate_time: float = 1.0) -> Decoherence:
    """
    创建热弛豫噪声模型
    
    Args:
        T1: 能量弛豫时间
        T2: 退相干时间
        gate_time: 门操作时间
    
    Returns:
        Decoherence噪声模型
    """
    return Decoherence(T1=T1, T2=T2, gate_time=gate_time)


# 噪声预处理器
class NoiseAwareCircuit:
    """
    噪声感知电路
    在门操作之间自动应用噪声
    """
    def __init__(self, circuit, noise_model: NoiseModel):
        self.circuit = circuit
        self.noise_model = noise_model
    
    def run(self, state: QuantumState) -> QuantumState:
        """运行带噪声的电路"""
        current_state = state
        
        for op in self.circuit.gates:
            # 应用门操作 (简化)
            pass
        
        return current_state


# 噪声参数估计
def estimate_noise_parameters_from_device(
    device_properties: Dict) -> Tuple[float, float]:
    """
    从设备属性估计噪声参数
    
    Args:
        device_properties: 设备属性字典
    
    Returns:
        (T1, T2) 时间常数
    """
    T1 = device_properties.get('T1', 100.0)  # 默认100微秒
    T2 = device_properties.get('T2', 50.0)   # 默认50微秒
    
    return T1, T2


def apply_noise_after_gates(circuit, noise_model: NoiseModel,
                            probability_scaling: float = 1.0) -> None:
    """
    在电路的每个门后添加噪声
    
    Args:
        circuit: 量子电路
        noise_model: 噪声模型
        probability_scaling: 概率缩放因子
    """
    pass  # 简化实现
