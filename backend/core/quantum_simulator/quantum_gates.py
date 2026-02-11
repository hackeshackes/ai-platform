"""
Quantum Gates - 量子门实现
提供所有标准量子门和自定义门的功能
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
import cmath


# 基础数学常量
SQRT2 = np.sqrt(2)
SQRT1_2 = 1.0 / SQRT2
I = 1j  # 虚数单位


@dataclass
class QuantumGate:
    """
    量子门数据类
    """
    name: str
    matrix: np.ndarray
    qubits: Union[int, Tuple[int, int]]
    parameters: Optional[List[float]] = None
    
    @property
    def num_qubits(self) -> int:
        """返回门的量子位数"""
        if isinstance(self.qubits, tuple):
            return 2
        return 1
    
    def adjoint(self) -> 'QuantumGate':
        """返回门的共轭转置"""
        return QuantumGate(
            name=f"{self.name}\\^\\dagger",
            matrix=self.matrix.conj().T,
            qubits=self.qubits,
            parameters=self.parameters
        )


# 单量子比特门矩阵
# Hadamard门
H_MATRIX = np.array([
    [SQRT1_2, SQRT1_2],
    [SQRT1_2, -SQRT1_2]
], dtype=complex)

# Pauli-X门 (量子NOT)
X_MATRIX = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

# Pauli-Y门
Y_MATRIX = np.array([
    [0, -I],
    [I, 0]
], dtype=complex)

# Pauli-Z门
Z_MATRIX = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# S门 (相位门)
S_MATRIX = np.array([
    [1, 0],
    [0, I]
], dtype=complex)

# T门 (π/8门)
T_MATRIX = np.array([
    [1, 0],
    [0, cmath.exp(I * np.pi / 4)]
], dtype=complex)

# 根NOT门
X_SQRT_MATRIX = np.array([
    [0.5 + 0.5j, 0.5 - 0.5j],
    [0.5 - 0.5j, 0.5 + 0.5j]
], dtype=complex)

# Y根门
Y_SQRT_MATRIX = np.array([
    [0.5 + 0.5j, -0.5 - 0.5j],
    [0.5 - 0.5j, 0.5 + 0.5j]
], dtype=complex)

# Z根门
Z_SQRT_MATRIX = np.array([
    [1, 0],
    [0, 1j]
], dtype=complex)


# 双量子比特门矩阵
# CNOT门 (控制NOT)
CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# CZ门 (控制Z)
CZ_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

# SWAP门
SWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# iSWAP门
ISWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

# CNOT门 (另一种编号约定)
CNOT_01_MATRIX = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=complex)


def create_rotation_gate(axis: str, angle: float) -> np.ndarray:
    """
    创建单量子比特旋转门
    
    Args:
        axis: 旋转轴 ('x', 'y', 'z')
        angle: 旋转角度 (弧度)
    
    Returns:
        旋转门矩阵
    """
    cos_half = np.cos(angle / 2)
    sin_half = np.sin(angle / 2)
    
    if axis.lower() == 'x':
        return np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
    elif axis.lower() == 'y':
        return np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
    elif axis.lower() == 'z':
        return np.array([
            [cmath.exp(-1j * angle / 2), 0],
            [0, cmath.exp(1j * angle / 2)]
        ], dtype=complex)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def U_gate(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    通用单量子比特门 (IBM Q约定)
    
    Args:
        theta: 参数 [0, π]
        phi: 参数 [0, 2π)
        lam: 参数 [0, 2π)
    
    Returns:
        U门矩阵
    """
    return np.array([
        [np.cos(theta/2), -np.exp(1j * lam) * np.sin(theta/2)],
        [np.exp(1j * phi) * np.sin(theta/2), 
         np.exp(1j * (phi + lam)) * np.cos(theta/2)]
    ], dtype=complex)


def controlled_U(U: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """
    创建受控U门
    
    Args:
        U: 单量子比特门矩阵
        control: 控制量子比特索引
        target: 目标量子比特索引
        n_qubits: 总量子比特数
    
    Returns:
        受控U门的稀疏矩阵表示
    """
    dim = 2 ** n_qubits
    gate = np.eye(dim, dtype=complex)
    
    # 计算受控门的索引
    for i in range(dim):
        # 检查控制量子比特状态
        if (i >> (n_qubits - 1 - control)) & 1:
            # 如果控制位为1，翻转目标位并应用U
            j = i ^ (1 << (n_qubits - 1 - target))
            # 应用受控U变换
            for a in range(2):
                for b in range(2):
                    gate[i, j] = U[a, b]
                    j = i ^ (1 << (n_qubits - 1 - target)) | (a << (n_qubits - 1 - target))
    
    return gate


def get_gate_matrix(name: str) -> np.ndarray:
    """
    获取标准量子门矩阵
    
    Args:
        name: 门名称
    
    Returns:
        门矩阵
    """
    gates = {
        'h': H_MATRIX,
        'x': X_MATRIX,
        'y': Y_MATRIX,
        'z': Z_MATRIX,
        's': S_MATRIX,
        't': T_MATRIX,
        'sqrt(x)': X_SQRT_MATRIX,
        'sqrt(y)': Y_SQRT_MATRIX,
        'sqrt(z)': Z_SQRT_MATRIX,
        'cnot': CNOT_MATRIX,
        'cx': CNOT_MATRIX,
        'cz': CZ_MATRIX,
        'swap': SWAP_MATRIX,
        'iswap': ISWAP_MATRIX,
    }
    
    name_lower = name.lower()
    if name_lower not in gates:
        raise ValueError(f"Unknown gate: {name}")
    
    return gates[name_lower]


# gate_matrix是get_gate_matrix的别名
gate_matrix = get_gate_matrix


def create_custom_gate(matrix: np.ndarray) -> QuantumGate:
    """
    创建自定义量子门
    
    Args:
        matrix: 门的矩阵表示
    
    Returns:
        QuantumGate对象
    """
    n = int(np.log2(matrix.shape[0]))
    if 2 ** n != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Invalid gate matrix dimensions")
    
    return QuantumGate(
        name="custom",
        matrix=matrix,
        qubits=n
    )


# 便捷函数：创建标准门
def H(qubit: int) -> QuantumGate:
    """创建Hadamard门"""
    return QuantumGate("h", H_MATRIX, qubit)

def X(qubit: int) -> QuantumGate:
    """创建Pauli-X门"""
    return QuantumGate("x", X_MATRIX, qubit)

def Y(qubit: int) -> QuantumGate:
    """创建Pauli-Y门"""
    return QuantumGate("y", Y_MATRIX, qubit)

def Z(qubit: int) -> QuantumGate:
    """创建Pauli-Z门"""
    return QuantumGate("z", Z_MATRIX, qubit)

def S(qubit: int) -> QuantumGate:
    """创建S门"""
    return QuantumGate("s", S_MATRIX, qubit)

def T(qubit: int) -> QuantumGate:
    """创建T门"""
    return QuantumGate("t", T_MATRIX, qubit)

def CNOT(control: int, target: int) -> QuantumGate:
    """创建CNOT门"""
    return QuantumGate("cnot", CNOT_MATRIX, (control, target))

def CZ(control: int, target: int) -> QuantumGate:
    """创建CZ门"""
    return QuantumGate("cz", CZ_MATRIX, (control, target))

def SWAP(qubit1: int, qubit2: int) -> QuantumGate:
    """创建SWAP门"""
    return QuantumGate("swap", SWAP_MATRIX, (qubit1, qubit2))


def R_x(qubit: int, angle: float) -> QuantumGate:
    """创建Rx旋转门"""
    return QuantumGate(
        f"rx({angle:.4f})",
        create_rotation_gate('x', angle),
        qubit,
        parameters=[angle]
    )

def R_y(qubit: int, angle: float) -> QuantumGate:
    """创建Ry旋转门"""
    return QuantumGate(
        f"ry({angle:.4f})",
        create_rotation_gate('y', angle),
        qubit,
        parameters=[angle]
    )

def R_z(qubit: int, angle: float) -> QuantumGate:
    """创建Rz旋转门"""
    return QuantumGate(
        f"rz({angle:.4f})",
        create_rotation_gate('z', angle),
        qubit,
        parameters=[angle]
    )


# 门组合和分解
def decompose_to_clifford(gate_matrix: np.ndarray) -> List[np.ndarray]:
    """
    将门分解为Clifford门集合
    (简化实现)
    
    Args:
        gate_matrix: 输入门矩阵
    
    Returns:
        Clifford门矩阵列表
    """
    # 简化实现 - 实际需要更复杂的分解算法
    return [gate_matrix]


def get_gate_depth(gates: List[QuantumGate]) -> int:
    """
    计算电路深度
    
    Args:
        gates: 门列表
    
    Returns:
        电路深度
    """
    if not gates:
        return 0
    
    # 简化实现
    return len(gates)
