"""
Quantum Circuit - 量子电路构建和优化
支持100+量子比特的高效电路模拟
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from .quantum_gates import (
    QuantumGate, H, X, Y, Z, S, T, CNOT, CZ, SWAP,
    create_rotation_gate, R_x, R_y, R_z, get_gate_matrix
)
from .quantum_state import QuantumState


class CircuitType(Enum):
    """电路类型"""
    CIRCUIT = "circuit"
    ACO = "aco"  # 量子近似优化算法
    QFT = "qft"  # 量子傅里叶变换
    VQE = "vqe"  # 变分量子本征求解器


@dataclass
class GateOperation:
    """门操作记录"""
    gate: QuantumGate
    qubits: Union[int, Tuple[int, int]]
    layer: int
    parameters: Optional[Dict] = None


@dataclass
class QuantumCircuit:
    """
    量子电路类
    支持量子比特的注册、门的添加、电路优化和可视化
    """
    n_qubits: int
    gates: List[GateOperation] = field(default_factory=list)
    measurements: List[Tuple[int, int]] = field(default_factory=list)
    circuit_type: CircuitType = CircuitType.CIRCUIT
    name: str = "quantum_circuit"
    
    def __post_init__(self):
        """初始化量子电路"""
        if self.n_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
    
    def add_gate(self, gate: QuantumGate, 
                 qubits: Union[int, Tuple[int, int]], 
                 parameters: Optional[Dict] = None) -> 'QuantumCircuit':
        """
        添加门到电路
        
        Args:
            gate: 量子门
            qubits: 量子比特索引
            parameters: 额外参数
        
        Returns:
            self (链式调用)
        """
        # 验证量子比特索引
        if isinstance(qubits, int):
            if qubits < 0 or qubits >= self.n_qubits:
                raise ValueError(f"Qubit index {qubits} out of range [0, {self.n_qubits-1}]")
        else:
            for q in qubits:
                if q < 0 or q >= self.n_qubits:
                    raise ValueError(f"Qubit index {q} out of range [0, {self.n_qubits-1}]")
        
        operation = GateOperation(
            gate=gate,
            qubits=qubits,
            layer=len(self.gates),
            parameters=parameters
        )
        
        self.gates.append(operation)
        return self
    
    # 单量子比特门
    def h(self, qubit: int) -> 'QuantumCircuit':
        """添加Hadamard门"""
        return self.add_gate(H(qubit), qubit)
    
    def x(self, qubit: int) -> 'QuantumCircuit':
        """添加Pauli-X门"""
        return self.add_gate(X(qubit), qubit)
    
    def y(self, qubit: int) -> 'QuantumCircuit':
        """添加Pauli-Y门"""
        return self.add_gate(Y(qubit), qubit)
    
    def z(self, qubit: int) -> 'QuantumCircuit':
        """添加Pauli-Z门"""
        return self.add_gate(Z(qubit), qubit)
    
    def s(self, qubit: int) -> 'QuantumCircuit':
        """添加S门"""
        return self.add_gate(S(qubit), qubit)
    
    def t(self, qubit: int) -> 'QuantumCircuit':
        """添加T门"""
        return self.add_gate(T(qubit), qubit)
    
    # 旋转门
    def rx(self, qubit: int, angle: float) -> 'QuantumCircuit':
        """添加Rx旋转门"""
        return self.add_gate(R_x(qubit, angle), qubit, {'angle': angle})
    
    def ry(self, qubit: int, angle: float) -> 'QuantumCircuit':
        """添加Ry旋转门"""
        return self.add_gate(R_y(qubit, angle), qubit, {'angle': angle})
    
    def rz(self, qubit: int, angle: float) -> 'QuantumCircuit':
        """添加Rz旋转门"""
        return self.add_gate(R_z(qubit, angle), qubit, {'angle': angle})
    
    def u(self, qubit: int, theta: float, phi: float, lam: float) -> 'QuantumCircuit':
        """添加通用U门"""
        from .quantum_gates import U_gate
        matrix = U_gate(theta, phi, lam)
        gate = QuantumGate(f"u({theta:.4f},{phi:.4f},{lam:.4f})", matrix, qubit, 
                          [theta, phi, lam])
        return self.add_gate(gate, qubit)
    
    # 双量子比特门
    def cnot(self, control: int, target: int) -> 'QuantumCircuit':
        """添加CNOT门"""
        return self.add_gate(CNOT(control, target), (control, target))
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """添加CNOT门 (别名)"""
        return self.cnot(control, target)
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """添加CZ门"""
        return self.add_gate(CZ(control, target), (control, target))
    
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """添加SWAP门"""
        return self.add_gate(SWAP(qubit1, qubit2), (qubit1, qubit2))
    
    def iswap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """添加iSWAP门"""
        from .quantum_gates import ISWAP_MATRIX
        gate = QuantumGate("iswap", ISWAP_MATRIX, (qubit1, qubit2))
        return self.add_gate(gate, (qubit1, qubit2))
    
    # 三量子比特门
    def toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """添加Toffoli门 (CCNOT)"""
        # Toffoli门分解
        self.h(target)
        self.cnot(control2, target)
        self.t(target)
        self.cnot(control1, target)
        self.t(target)
        self.cnot(control2, target)
        self.h(target)
        self.cnot(control1, target)
        self.cnot(control2, target)
        self.t(control1)
        self.t(control2)
        self.t(target)
        self.cnot(control1, control2)
        self.t(control2)
        self.cnot(control1, control2)
        return self
    
    def fredkin(self, control: int, target1: int, target2: int) -> 'QuantumCircuit':
        """添加Fredkin门 (CSWAP)"""
        # Fredkin门分解
        self.toffoli(control, target2, target1)
        return self
    
    # 测量
    def measure(self, qubit: int, classical_bit: Optional[int] = None) -> 'QuantumCircuit':
        """添加测量"""
        classical = classical_bit if classical_bit is not None else qubit
        self.measurements.append((qubit, classical))
        return self
    
    def measure_all(self) -> 'QuantumCircuit':
        """测量所有量子比特"""
        for qubit in range(self.n_qubits):
            self.measure(qubit)
        return self
    
    # 常用电路构造
    def bell_pair(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """创建Bell对"""
        self.h(qubit1)
        self.cnot(qubit1, qubit2)
        return self
    
    def ghz_state(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """创建GHZ态 (Greenberger-Horne-Zeilinger)"""
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        
        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])
        return self
    
    def w_state(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """创建W态"""
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        n = len(qubits)
        if n < 2:
            raise ValueError("W state requires at least 2 qubits")
        
        # 递归构造W态
        def add_w(self_circuit: QuantumCircuit, qubits_list: List[int]):
            if len(qubits_list) == 1:
                self_circuit.x(qubits_list[0])
                return
            
            n_q = len(qubits_list)
            q0, rest = qubits_list[0], qubits_list[1:]
            
            angle = 2 * np.arccos(1 / np.sqrt(n_q))
            self_circuit.ry(q0, angle)
            
            for q in rest:
                self_circuit.cnot(q0, q)
                self_circuit.ry(q, -angle / 2)
                self_circuit.cnot(q0, q)
            
            add_w(self_circuit, rest)
        
        add_w(self, qubits)
        return self
    
    def qft(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """量子傅里叶变换"""
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        n = len(qubits)
        
        for j in range(n):
            self.h(qubits[j])
            for k in range(j + 1, n):
                angle = 2 * np.pi / (2 ** (k - j + 1))
                self.cu1(qubits[j], qubits[k], angle)
        
        # 逆序SWAP
        for j in range(n // 2):
            self.swap(qubits[j], qubits[n - 1 - j])
        
        return self
    
    def inverse_qft(self, qubits: Optional[List[int]] = None) -> 'QuantumCircuit':
        """逆量子傅里叶变换"""
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        n = len(qubits)
        
        # 逆序SWAP
        for j in range(n // 2):
            self.swap(qubits[j], qubits[n - 1 - j])
        
        for j in range(n - 1, -1, -1):
            for k in range(j):
                angle = -2 * np.pi / (2 ** (j - k + 1))
                self.cu1(qubits[j], qubits[k], angle)
            self.h(qubits[j])
        
        return self
    
    def cu1(self, control: int, target: int, angle: float) -> 'QuantumCircuit':
        """控制相位门"""
        rz_gate = R_z(target, angle)
        
        # 分解为基本门
        self.h(target)
        self.cnot(control, target)
        self.rz(target, -angle)
        self.cnot(control, target)
        self.rz(target, angle)
        self.h(target)
        return self
    
    # 变分电路层
    def entangling_layer(self, pattern: str = "linear") -> 'QuantumCircuit':
        """添加纠缠层"""
        if pattern == "linear":
            for i in range(self.n_qubits - 1):
                self.cnot(i, i + 1)
        elif pattern == "circular":
            for i in range(self.n_qubits):
                self.cnot(i, (i + 1) % self.n_qubits)
        elif pattern == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    self.cnot(i, j)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        return self
    
    def rotation_layer(self, qubits: Optional[List[int]] = None, 
                       param_func: Optional[Callable[[int], Tuple[float, float, float]]] = None) -> 'QuantumCircuit':
        """添加旋转层"""
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        if param_func is None:
            def default_param(i):
                return (np.random.rand() * np.pi, 
                       np.random.rand() * 2 * np.pi, 
                       np.random.rand() * 2 * np.pi)
            param_func = default_param
        
        for i, qubit in enumerate(qubits):
            theta, phi, lam = param_func(i)
            self.u(qubit, theta, phi, lam)
        
        return self
    
    def ansatz(self, depth: int = 3, pattern: str = "linear") -> 'QuantumCircuit':
        """构建变分ansatz电路"""
        for _ in range(depth):
            self.rotation_layer()
            self.entangling_layer(pattern)
        return self
    
    # 电路属性
    @property
    def depth(self) -> int:
        """计算电路深度"""
        if not self.gates:
            return 0
        
        # 计算每层的最大索引
        qubit_layers = defaultdict(int)
        max_depth = 0
        
        for op in self.gates:
            if isinstance(op.qubits, int):
                qubit_indices = [op.qubits]
            else:
                qubit_indices = list(op.qubits)
            
            layer = max(qubit_layers[q] for q in qubit_indices) + 1
            for q in qubit_indices:
                qubit_layers[q] = layer
            max_depth = max(max_depth, layer)
        
        return max_depth
    
    @property
    def gate_counts(self) -> Dict[str, int]:
        """统计门的数量"""
        counts = defaultdict(int)
        for op in self.gates:
            counts[op.gate.name] += 1
        return dict(counts)
    
    @property
    def num_gates(self) -> int:
        """返回门的总数"""
        return len(self.gates)
    
    # 电路优化
    def optimize(self, level: int = 1) -> 'QuantumCircuit':
        """
        电路优化
        
        Args:
            level: 优化级别 (1-3)
        """
        if level >= 1:
            self._remove_idle_gates()
        
        if level >= 2:
            self._merge_rotations()
        
        if level >= 3:
            self._cancel_cnot_pairs()
        
        return self
    
    def _remove_idle_gates(self):
        """移除空闲门 (简化实现)"""
        pass
    
    def _merge_rotations(self):
        """合并相邻旋转门"""
        pass
    
    def _cancel_cnot_pairs(self):
        """消除相邻CNOT对"""
        pass
    
    # 电路转换
    def to_qasm(self) -> str:
        """导出为QASM格式"""
        qasm = f"OPENQASM 2.0;\n"
        qasm += f'include "qelib1.inc";\n'
        qasm += f'qreg q[{self.n_qubits}];\n'
        qasm += f'creg c[{self.n_qubits}];\n\n'
        
        for op in self.gates:
            if isinstance(op.qubits, int):
                qasm += f'{op.gate.name} q[{op.qubits}];\n'
            else:
                qasm += f'{op.gate.name} q[{op.qubits[0]}], q[{op.qubits[1]}];\n'
        
        for qubit, classical in self.measurements:
            qasm += f'meas q[{qubit}] -> c[{classical}];\n'
        
        return qasm
    
    def from_qasm(self, qasm_string: str):
        """从QASM格式加载电路"""
        # 简化实现
        pass
    
    def to_json(self) -> str:
        """导出为JSON格式"""
        circuit_dict = {
            'n_qubits': self.n_qubits,
            'name': self.name,
            'circuit_type': self.circuit_type.value,
            'gates': [
                {
                    'name': op.gate.name,
                    'qubits': op.qubits if isinstance(op.qubits, int) else list(op.qubits),
                    'layer': op.layer
                }
                for op in self.gates
            ],
            'measurements': self.measurements
        }
        return json.dumps(circuit_dict, indent=2)
    
    def from_json(self, json_string: str):
        """从JSON格式加载电路"""
        circuit_dict = json.loads(json_string)
        self.n_qubits = circuit_dict['n_qubits']
        self.name = circuit_dict.get('name', 'quantum_circuit')
        
        for gate_data in circuit_dict.get('gates', []):
            qubits = gate_data['qubits']
            if isinstance(qubits, int):
                gate = QuantumGate(gate_data['name'], 
                                  get_gate_matrix(gate_data['name']), 
                                  qubits)
            else:
                gate = QuantumGate(gate_data['name'],
                                  get_gate_matrix(gate_data['name']),
                                  tuple(qubits))
            self.gates.append(GateOperation(gate, qubits, gate_data['layer']))
        
        self.measurements = circuit_dict.get('measurements', [])
        return self
    
    # 电路可视化
    def draw(self) -> str:
        """绘制电路 ASCII 表示"""
        lines = []
        
        # 初始化量子比特线
        qubit_lines = []
        for i in range(self.n_qubits):
            qubit_lines.append([' ' for _ in range(self._estimate_width())])
        
        # 简化实现 - 返回基本描述
        lines.append(f"Quantum Circuit: {self.name}")
        lines.append(f"Qubits: {self.n_qubits}")
        lines.append(f"Depth: {self.depth}")
        lines.append(f"Gates: {self.num_gates}")
        lines.append("-" * 40)
        
        for i, op in enumerate(self.gates[:10]):  # 只显示前10个门
            if isinstance(op.qubits, int):
                lines.append(f"Step {i}: {op.gate.name} on qubit {op.qubits}")
            else:
                lines.append(f"Step {i}: {op.gate.name} on qubits {op.qubits}")
        
        if len(self.gates) > 10:
            lines.append(f"... and {len(self.gates) - 10} more gates")
        
        return '\n'.join(lines)
    
    def _estimate_width(self) -> int:
        """估计电路显示宽度"""
        return max(10, len(self.gates) * 4)
    
    def __str__(self) -> str:
        return self.draw()
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(n_qubits={self.n_qubits}, gates={self.num_gates}, depth={self.depth})"


def qft_circuit(n_qubits: int) -> QuantumCircuit:
    """创建n量子比特的QFT电路"""
    circuit = QuantumCircuit(n_qubits=n_qubits, name="QFT")
    return circuit.qft()


def bell_circuit(n_qubits: int) -> QuantumCircuit:
    """创建n量子比特的Bell电路"""
    circuit = QuantumCircuit(n_qubits=n_qubits, name="Bell")
    for i in range(0, n_qubits, 2):
        if i + 1 < n_qubits:
            circuit.bell_pair(i, i + 1)
    return circuit


def ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """创建n量子比特的GHZ电路"""
    circuit = QuantumCircuit(n_qubits=n_qubits, name="GHZ")
    return circuit.ghz_state()


def random_circuit(n_qubits: int, depth: int, 
                   seed: Optional[int] = None) -> QuantumCircuit:
    """创建随机电路"""
    if seed is not None:
        np.random.seed(seed)
    
    circuit = QuantumCircuit(n_qubits=n_qubits, name="Random")
    
    for _ in range(depth):
        # 随机单门
        qubit = np.random.randint(n_qubits)
        gate_type = np.random.choice(['h', 't', 'rx', 'ry', 'rz'])
        
        if gate_type == 'h':
            circuit.h(qubit)
        elif gate_type == 't':
            circuit.t(qubit)
        else:
            angle = np.random.rand() * 2 * np.pi
            if gate_type == 'rx':
                circuit.rx(qubit, angle)
            elif gate_type == 'ry':
                circuit.ry(qubit, angle)
            else:
                circuit.rz(qubit, angle)
        
        # 随机双门
        if n_qubits > 1:
            q1, q2 = np.random.choice(n_qubits, 2, replace=False)
            if np.random.rand() < 0.5:
                circuit.cnot(q1, q2)
            else:
                circuit.cz(q1, q2)
    
    return circuit
