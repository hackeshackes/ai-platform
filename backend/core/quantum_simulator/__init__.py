"""
Quantum Simulator - 量子计算模拟器
支持100+量子位的高性能量子电路模拟器
"""

from .quantum_circuit import QuantumCircuit
from .quantum_state import QuantumState
from .quantum_gates import QuantumGate, gate_matrix, H, X, Y, Z, S, T, CNOT, CZ, SWAP
from .noise_models import NoiseModel, DepolarizingNoise, PhaseNoise, AmplitudeDamping, Decoherence
from .api import QuantumSimulator, run_circuit, estimate_resources

__version__ = "1.0.0"
__author__ = "AI Platform"

__all__ = [
    'QuantumCircuit',
    'QuantumState',
    'QuantumGate',
    'gate_matrix',
    'H', 'X', 'Y', 'Z', 'S', 'T',
    'CNOT', 'CZ', 'SWAP',
    'NoiseModel',
    'DepolarizingNoise',
    'PhaseNoise', 
    'AmplitudeDamping',
    'Decoherence',
    'QuantumSimulator',
    'run_circuit',
    'estimate_resources'
]
