"""
示例代码
Examples

提供QAOA和VQE算法的使用示例
"""

from typing import List, Dict, Any
import numpy as np

from .qaoa import QAOA
from .vqe import VQE, MoleculeData, MoleculeBuilder
from .variational_forms import (
    HardwareEfficientAnsatz, 
    UCCSD, 
    QAOAAnsatz
)
from .optimizers import (
    COBYLA, 
    SPSA, 
    GradientDescent, 
    NaturalGradient
)


def example_qaoa_max_cut():
    """QAOA最大割示例"""
    print("=" * 50)
    print("QAOA Max-Cut Example")
    print("=" * 50)
    
    # 定义图（4节点，5条边）
    graph = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3)
    ]
    
    print(f"Graph: {graph}")
    print(f"Number of nodes: 4")
    print(f"Number of edges: {len(graph)}")
    
    # 创建QAOA实例
    qaoa = QAOA(optimizer="cobyla", p_layers=3)
    
    # 求解最大割
    result = qaoa.max_cut(graph, num_nodes=4)
    
    print(f"\nResults:")
    print(f"  Optimal cut value: {result['cut_value']}")
    print(f"  Approximation ratio: {result['approximation_ratio']:.4f}")
    print(f"  Number of iterations: {result['num_iterations']}")
    print(f"  Final energy: {result['final_energy']:.6f}")
    print(f"  Success: {result['success']}")
    
    return result


def example_qaoa_sparse_graph():
    """QAOA稀疏图示例"""
    print("\n" + "=" * 50)
    print("QAOA Sparse Graph Example")
    print("=" * 50)
    
    # 稀疏图
    edges = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3)]
    weights = [1.0, 1.5, 2.0, 0.5, 1.0]
    num_nodes = 6
    
    print(f"Edges: {edges}")
    print(f"Weights: {weights}")
    
    # 使用SPSA优化器
    qaoa = QAOA(optimizer="spsa", p_layers=2)
    
    result = qaoa.max_cut_sparse(edges, weights, num_nodes)
    
    print(f"\nResults:")
    print(f"  Final energy: {result['final_energy']:.6f}")
    print(f"  Iterations: {result['num_iterations']}")
    
    return result


def example_qaoa_custom_optimizer():
    """QAOA自定义优化器示例"""
    print("\n" + "=" * 50)
    print("QAOA Custom Optimizer Example")
    print("=" * 50)
    
    graph = [(0, 1), (1, 2), (2, 0)]
    
    # 使用自然梯度优化器
    qaoa = QAOA(optimizer="natural_gradient", p_layers=2)
    
    result = qaoa.max_cut(graph, num_nodes=3)
    
    print(f"Graph: {graph}")
    print(f"Results:")
    print(f"  Cut value: {result['cut_value']}")
    print(f"  Iterations: {result['num_iterations']}")
    
    return result


def example_vqe_h2_molecule():
    """VQE氢分子示例"""
    print("\n" + "=" * 50)
    print("VQE H2 Molecule Example")
    print("=" * 50)
    
    # 创建氢分子
    molecule = MoleculeBuilder.create_h2_molecule(bond_length=0.74)
    
    print(f"Molecule: H2")
    print(f"Bond length: 0.74 Angstrom")
    print(f"Basis: sto-3g")
    
    # 创建VQE实例
    vqe = VQE(optimizer="spsa")
    
    # 计算能量
    result = vqe.compute_energy(molecule)
    
    print(f"\nResults:")
    print(f"  Ground state energy: {result['ground_state_energy']:.6f}")
    print(f"  Number of iterations: {result['num_iterations']}")
    print(f"  Number of qubits: {result['num_qubits']}")
    print(f"  Success: {result['success']}")
    
    # 分析收敛
    from .vqe import VQEAnalyzer
    analysis = VQEAnalyzer.analyze_convergence(result['convergence_history'])
    print(f"  Converged: {analysis['converged']}")
    
    return result


def example_vqe_lih_molecule():
    """VQE LiH分子示例"""
    print("\n" + "=" * 50)
    print("VQE LiH Molecule Example")
    print("=" * 50)
    
    # 创建LiH分子
    molecule = MoleculeBuilder.create_lih_molecule(bond_length=1.45)
    
    print(f"Molecule: LiH")
    print(f"Bond length: 1.45 Angstrom")
    
    # 使用COBYLA优化器
    vqe = VQE(optimizer="cobyla")
    
    result = vqe.compute_energy(molecule)
    
    print(f"\nResults:")
    print(f"  Ground state energy: {result['ground_state_energy']:.6f}")
    print(f"  Iterations: {result['num_iterations']}")
    
    return result


def example_vqe_custom_hamiltonian():
    """VQE自定义哈密顿量示例"""
    print("\n" + "=" * 50)
    print("VQE Custom Hamiltonian Example")
    print("=" * 50)
    
    # 自定义哈密顿量
    hamiltonian = {
        "Z_0": -1.0,
        "Z_1": -1.0,
        "ZZ_0_1": 0.5,
        "I": -0.5
    }
    
    print(f"Hamiltonian: {hamiltonian}")
    
    vqe = VQE(optimizer="spsa")
    
    result = vqe.compute_energy_custom_hamiltonian(
        hamiltonian=hamiltonian,
        num_qubits=2
    )
    
    print(f"\nResults:")
    print(f"  Ground state energy: {result['ground_state_energy']:.6f}")
    print(f"  Iterations: {result['num_iterations']}")
    
    return result


def example_variational_forms():
    """变分形式示例"""
    print("\n" + "=" * 50)
    print("Variational Forms Example")
    print("=" * 50)
    
    # Hardware-Efficient Ansatz
    print("\n1. Hardware-Efficient Ansatz:")
    ansatz1 = HardwareEfficientAnsatz(num_qubits=4, depth=3)
    print(f"   Number of parameters: {ansatz1.num_parameters}")
    
    params1 = ansatz1.initial_parameters()
    print(f"   Initial params shape: {params1.shape}")
    
    circuit1 = ansatz1.get_circuit(params1)
    print(f"   Circuit type: {circuit1['type']}")
    
    # UCCSD
    print("\n2. UCCSD Ansatz:")
    ansatz2 = UCCSD(num_qubits=4, num_electrons=2)
    print(f"   Number of parameters: {ansatz2.num_parameters}")
    
    params2 = ansatz2.initial_parameters()
    print(f"   Initial params shape: {params2.shape}")
    
    # QAOA
    print("\n3. QAOA Ansatz:")
    ansatz3 = QAOAAnsatz(num_qubits=4, p_layers=3)
    print(f"   Number of parameters: {ansatz3.num_parameters}")
    
    params3 = ansatz3.initial_parameters()
    print(f"   Initial params shape: {params3.shape}")
    
    return {
        "hardware_efficient": ansatz1,
        "uccsd": ansatz2,
        "qaoa_ansatz": ansatz3
    }


def example_optimizers():
    """优化器示例"""
    print("\n" + "=" * 50)
    print("Optimizers Example")
    print("=" * 50)
    
    # 测试函数
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    
    initial_params = np.array([-1.5, 2.0, 0.5, 1.0])
    
    # COBYLA
    print("\n1. COBYLA:")
    cobyla = COBYLA(maxiter=100)
    result1 = cobyla.minimize(rosenbrock, initial_params)
    print(f"   Final value: {result1.fun:.6f}")
    print(f"   Iterations: {result1.nit}")
    
    # SPSA
    print("\n2. SPSA:")
    spsa = SPSA(maxiter=100, learning_rate=0.1)
    result2 = spsa.minimize(rosenbrock, initial_params)
    print(f"   Final value: {result2.fun:.6f}")
    print(f"   Iterations: {result2.nit}")
    
    # Gradient Descent
    print("\n3. Gradient Descent:")
    gd = GradientDescent(maxiter=100, learning_rate=0.001)
    result3 = gd.minimize(rosenbrock, initial_params)
    print(f"   Final value: {result3.fun:.6f}")
    print(f"   Iterations: {result3.nit}")
    
    # Natural Gradient
    print("\n4. Natural Gradient:")
    ng = NaturalGradient(maxiter=100, learning_rate=0.1)
    result4 = ng.minimize(rosenbrock, initial_params)
    print(f"   Final value: {result4.fun:.6f}")
    print(f"   Iterations: {result4.nit}")
    
    return {
        "cobyla": result1,
        "spsa": result2,
        "gradient_descent": result3,
        "natural_gradient": result4
    }


def example_large_scale_qaoa():
    """大规模QAOA示例"""
    print("\n" + "=" * 50)
    print("Large-Scale QAOA Example")
    print("=" * 50)
    
    # 生成随机图
    num_nodes = 20
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if np.random.random() < 0.3:  # 30% 连接概率
                edges.append((i, j))
    
    print(f"Graph: {num_nodes} nodes, {len(edges)} edges")
    
    # 使用快速配置
    from .config import QuantumOptimizerConfig
    
    config = QuantumOptimizerConfig.default_qaoa()
    config.max_iterations = 200
    config.qaoa_layers = 2
    
    qaoa = QAOA(optimizer="cobyla", config=config, p_layers=2)
    
    result = qaoa.max_cut(edges, num_nodes=num_nodes)
    
    print(f"\nResults:")
    print(f"  Cut value: {result['cut_value']}")
    print(f"  Iterations: {result['num_iterations']}")
    print(f"  Energy: {result['final_energy']:.6f}")
    
    return result


def example_all():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print(" Quantum Optimization Algorithms - All Examples")
    print("=" * 60)
    
    results = {}
    
    # QAOA examples
    results['qaoa_max_cut'] = example_qaoa_max_cut()
    results['qaoa_sparse'] = example_qaoa_sparse_graph()
    results['qaoa_natural_gradient'] = example_qaoa_custom_optimizer()
    
    # VQE examples
    results['vqe_h2'] = example_vqe_h2_molecule()
    results['vqe_lih'] = example_vqe_lih_molecule()
    results['vqe_custom_hamiltonian'] = example_vqe_custom_hamiltonian()
    
    # Variational forms
    results['variational_forms'] = example_variational_forms()
    
    # Optimizers
    results['optimizers'] = example_optimizers()
    
    # Large scale
    results['large_scale_qaoa'] = example_large_scale_qaoa()
    
    print("\n" + "=" * 60)
    print(" All examples completed!")
    print("=" * 60)
    
    return results


# 快捷运行函数
def run_example(example_name: str):
    """运行指定示例"""
    examples = {
        'qaoa_max_cut': example_qaoa_max_cut,
        'qaoa_sparse': example_qaoa_sparse_graph,
        'qaoa_custom': example_qaoa_custom_optimizer,
        'vqe_h2': example_vqe_h2_molecule,
        'vqe_lih': example_vqe_lih_molecule,
        'vqe_custom': example_vqe_custom_hamiltonian,
        'variational_forms': example_variational_forms,
        'optimizers': example_optimizers,
        'large_scale': example_large_scale_qaoa,
        'all': example_all
    }
    
    if example_name not in examples:
        print(f"Available examples: {list(examples.keys())}")
        return None
    
    return examples[example_name]()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        run_example(example_name)
    else:
        example_all()
