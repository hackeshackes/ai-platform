"""
API接口模块
API Module

提供RESTful API接口用于量子优化服务
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from flask import Flask, jsonify, request
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .qaoa import QAOA
from .vqe import VQE, MoleculeData, MoleculeBuilder
from .config import QuantumOptimizerConfig


app = Flask(__name__)

# 全局变量存储任务状态
task_results = {}
task_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class APIResponse:
    """API响应类"""
    success: bool
    data: Any
    message: str
    task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "task_id": self.task_id
        }


def create_qaoa_blueprint():
    """创建QAOA蓝图"""
    from flask import Blueprint
    
    qaoa_bp = Blueprint('qaoa', __name__)
    
    @qaoa_bp.route('/max_cut', methods=['POST'])
    def solve_max_cut():
        """求解最大割问题"""
        try:
            data = request.json
            
            graph = data.get('graph', [])
            num_nodes = data.get('num_nodes')
            p_layers = data.get('p_layers', 3)
            optimizer = data.get('optimizer', 'cobyla')
            
            # 创建QAOA实例
            qaoa = QAOA(optimizer=optimizer, p_layers=p_layers)
            
            # 异步执行
            task_id = f"qaoa_{len(task_results)}"
            
            def run_optimization():
                result = qaoa.max_cut(graph, num_nodes=num_nodes)
                task_results[task_id] = result
            
            task_executor.submit(run_optimization)
            
            return jsonify({
                "success": True,
                "task_id": task_id,
                "message": "Optimization started"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
    
    @qaoa_bp.route('/result/<task_id>', methods=['GET'])
    def get_qaoa_result(task_id):
        """获取QAOA优化结果"""
        if task_id in task_results:
            result = task_results[task_id]
            return jsonify({
                "success": True,
                "data": result
            })
        else:
            return jsonify({
                "success": False,
                "message": "Task not found"
            }), 404
    
    @qaoa_bp.route('/custom', methods=['POST'])
    def solve_custom():
        """求解自定义优化问题"""
        try:
            data = request.json
            
            cost_function_data = data.get('cost_function')
            num_params = data.get('num_params', 2)
            optimizer = data.get('optimizer', 'cobyla')
            
            qaoa = QAOA(optimizer=optimizer)
            
            # 简化的自定义优化
            def simple_cost(params):
                return np.sum(params**2)
            
            result = qaoa.solve_custom(
                simple_cost,
                num_params,
                initial_params=None
            )
            
            return jsonify({
                "success": True,
                "data": result
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
    
    return qaoa_bp


def create_vqe_blueprint():
    """创建VQE蓝图"""
    from flask import Blueprint
    
    vqe_bp = Blueprint('vqe', __name__)
    
    @vqe_bp.route('/energy', methods=['POST'])
    def compute_energy():
        """计算分子能量"""
        try:
            data = request.json
            
            # 构建分子数据
            molecule = MoleculeData(
                geometry=data.get('geometry', []),
                basis=data.get('basis', 'sto-3g'),
                num_electrons=data.get('num_electrons', 2),
                num_orbitals=data.get('num_orbitals', 2)
            )
            
            optimizer = data.get('optimizer', 'spsa')
            
            # 创建VQE实例
            vqe = VQE(optimizer=optimizer)
            
            # 异步执行
            task_id = f"vqe_{len(task_results)}"
            
            def run_vqe():
                result = vqe.compute_energy(molecule)
                task_results[task_id] = result
            
            task_executor.submit(run_vqe)
            
            return jsonify({
                "success": True,
                "task_id": task_id,
                "message": "VQE computation started"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
    
    @vqe_bp.route('/preset/h2', methods=['GET'])
    def compute_h2_energy():
        """计算氢分子能量"""
        try:
            molecule = MoleculeBuilder.create_h2_molecule()
            vqe = VQE(optimizer="spsa")
            
            result = vqe.compute_energy(molecule)
            
            return jsonify({
                "success": True,
                "data": result
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
    
    @vqe_bp.route('/custom_hamiltonian', methods=['POST'])
    def compute_custom_hamiltonian():
        """计算自定义哈密顿量能量"""
        try:
            data = request.json
            
            hamiltonian = data.get('hamiltonian', {})
            num_qubits = data.get('num_qubits', 2)
            optimizer = data.get('optimizer', 'spsa')
            
            vqe = VQE(optimizer=optimizer)
            
            result = vqe.compute_energy_custom_hamiltonian(
                hamiltonian=hamiltonian,
                num_qubits=num_qubits
            )
            
            return jsonify({
                "success": True,
                "data": result
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
    
    @vqe_bp.route('/result/<task_id>', methods=['GET'])
    def get_vqe_result(task_id):
        """获取VQE计算结果"""
        if task_id in task_results:
            result = task_results[task_id]
            return jsonify({
                "success": True,
                "data": result
            })
        else:
            return jsonify({
                "success": False,
                "message": "Task not found"
            }), 404
    
    return vqe_bp


# 注册蓝图
app.register_blueprint(create_qaoa_blueprint(), url_prefix='/api/qaoa')
app.register_blueprint(create_vqe_blueprint(), url_prefix='/api/vqe')


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "quantum-optimizer"
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务状态"""
    return jsonify({
        "status": "running",
        "pending_tasks": len(task_results),
        "available_optimizers": ["cobyla", "spsa", "gradient_descent", "natural_gradient"],
        "available_ansatz": ["hardware_efficient", "uccsd", "qaoa"]
    })


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    return jsonify({
        "tasks": list(task_results.keys()),
        "count": len(task_results)
    })


@app.route('/api/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """删除任务结果"""
    if task_id in task_results:
        del task_results[task_id]
        return jsonify({
            "success": True,
            "message": f"Task {task_id} deleted"
        })
    else:
        return jsonify({
            "success": False,
            "message": "Task not found"
        }), 404


class QuantumOptimizerAPI:
    """量子优化器API类"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.app = app
    
    def run(self, debug: bool = False):
        """运行API服务器"""
        app.run(host=self.host, port=self.port, debug=debug)
    
    def run_background(self):
        """在后台运行API服务器"""
        import threading
        
        thread = threading.Thread(
            target=self.run,
            kwargs={"debug": False},
            daemon=True
        )
        thread.start()
        return thread


# 便捷函数
def start_api_server(host: str = "0.0.0.0", port: int = 5000) -> QuantumOptimizerAPI:
    """启动API服务器"""
    api = QuantumOptimizerAPI(host=host, port=port)
    api.run_background()
    return api


def qaoa_max_cut_sync(
    graph: List[Tuple[int, int]],
    num_nodes: Optional[int] = None,
    p_layers: int = 3,
    optimizer: str = "cobyla"
) -> Dict[str, Any]:
    """同步QAOA最大割求解"""
    qaoa = QAOA(optimizer=optimizer, p_layers=p_layers)
    return qaoa.max_cut(graph, num_nodes=num_nodes)


def vqe_energy_sync(
    molecule_data: MoleculeData,
    optimizer: str = "spsa"
) -> Dict[str, Any]:
    """同步VQE能量计算"""
    vqe = VQE(optimizer=optimizer)
    return vqe.compute_energy(molecule_data)


# 导出API
__all__ = [
    "app",
    "QuantumOptimizerAPI",
    "start_api_server",
    "qaoa_max_cut_sync",
    "vqe_energy_sync"
]
