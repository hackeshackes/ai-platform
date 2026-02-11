"""
Quantum Simulator Examples - 使用示例
"""

from .quantum_circuit import QuantumCircuit, bell_circuit, ghz_circuit, random_circuit
from .quantum_state import QuantumState
from .api import run_circuit, estimate_resources
from .noise_models import DepolarizingNoise


def example_bell_state():
    """Bell态示例"""
    circuit = QuantumCircuit(n_qubits=2, name="Bell")
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()
    
    print("Bell Circuit:")
    print(circuit)
    print()
    
    result = run_circuit(circuit, shots=1000)
    print(f"Results: {result.counts}")
    print()


def example_ghz_state():
    """GHZ态示例"""
    circuit = QuantumCircuit(n_qubits=3, name="GHZ")
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(0, 2)
    circuit.measure_all()
    
    print("GHZ Circuit:")
    print(circuit)
    print()
    
    result = run_circuit(circuit, shots=500)
    print(f"Results: {result.counts}")
    print()


def example_qft():
    """量子傅里叶变换示例"""
    circuit = QuantumCircuit(n_qubits=4, name="QFT")
    circuit.qft()
    
    print("QFT Circuit:")
    print(circuit)
    print()
    
    result = run_circuit(circuit, shots=100)
    print(f"Most probable: {result.most_probable(5)}")
    print()


def example_random_circuit():
    """随机电路示例"""
    circuit = random_circuit(n_qubits=10, depth=5, seed=42)
    
    print("Random Circuit:")
    print(circuit)
    print()
    
    resources = estimate_resources(circuit)
    print(f"Estimated memory: {resources['state_memory_mb']:.2f} MB")
    print()


def example_variational_circuit():
    """变分电路示例"""
    circuit = QuantumCircuit(n_qubits=4, name="Variational")
    
    # 初始层
    for i in range(4):
        circuit.h(i)
    
    # 变分层
    for _ in range(2):
        circuit.entangling_layer(pattern="linear")
        for i in range(4):
            circuit.ry(i, 0.5)
            circuit.rz(i, 0.3)
    
    circuit.measure_all()
    
    print("Variational Circuit:")
    print(circuit)
    print()
    
    result = run_circuit(circuit, shots=1000)
    print(f"Gate counts: {circuit.gate_counts}")
    print()


def example_noisy_simulation():
    """带噪声模拟示例"""
    circuit = QuantumCircuit(n_qubits=2, name="NoisyBell")
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()
    
    noise_model = DepolarizingNoise(probability=0.01)
    
    print("Noisy Bell Circuit:")
    print(circuit)
    print()
    
    result = run_circuit(circuit, shots=1000, noise_model=noise_model)
    print(f"Results with noise: {result.counts}")
    print()


def example_large_circuit():
    """大电路示例"""
    circuit = QuantumCircuit(n_qubits=50)
    circuit.ansatz(depth=2)
    
    print("Large Circuit (50 qubits):")
    print(f"  Gates: {circuit.num_gates}")
    print(f"  Depth: {circuit.depth}")
    print()
    
    resources = estimate_resources(circuit)
    print("Resource estimation:")
    for key, value in resources.items():
        print(f"  {key}: {value}")
    print()


def run_all_examples():
    """运行所有示例"""
    examples = [
        ("Bell State", example_bell_state),
        ("GHZ State", example_ghz_state),
        ("QFT", example_qft),
        ("Random Circuit", example_random_circuit),
        ("Variational Circuit", example_variational_circuit),
        ("Noisy Simulation", example_noisy_simulation),
        ("Large Circuit", example_large_circuit)
    ]
    
    print("=" * 50)
    print("Quantum Simulator Examples")
    print("=" * 50)
    print()
    
    for name, func in examples:
        print(f"\n{'=' * 40}")
        print(f"Example: {name}")
        print(f"{'=' * 40}")
        func()


if __name__ == "__main__":
    run_all_examples()
