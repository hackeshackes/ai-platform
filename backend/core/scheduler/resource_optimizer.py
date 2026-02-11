"""
智能调度系统 - 资源优化器

CPU/内存/GPU/存储/网络带宽优化
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class OptimizationStrategy(Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    BALANCED = "balanced"
    ENERGY = "energy"


@dataclass
class Workload:
    """工作负载定义"""
    id: str
    name: str
    resource_requirements: Dict[str, float]
    priority: int = 5  # 1-10, 10最高
    is_gpu_required: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class NodeResources:
    """节点资源"""
    node_id: str
    cpu_total: float
    cpu_available: float
    memory_total: float
    memory_available: float
    gpu_total: int = 0
    gpu_available: int = 0
    storage_total: float = 0.0
    storage_available: float = 0.0
    network_bandwidth: float = 0.0  # Mbps


class ResourceOptimizer:
    """资源优化器"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.nodes: Dict[str, NodeResources] = {}
        self.workloads: Dict[str, Workload] = {}
        self.allocation_history: List[Dict] = []
        
    def register_node(self, node: NodeResources) -> bool:
        """注册节点"""
        self.nodes[node.node_id] = node
        logger.info(f"节点注册成功: {node.node_id}")
        return True
    
    def register_workload(self, workload: Workload) -> bool:
        """注册工作负载"""
        self.workloads[workload.id] = workload
        logger.info(f"工作负载注册成功: {workload.id}")
        return True
    
    def optimize(
        self,
        workloads: List[Workload],
        constraints: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict]:
        """
        优化资源分配
        
        Args:
            workloads: 待分配工作负载列表
            constraints: 约束条件，如 {"cpu": "max", "cost": "min"}
            
        Returns:
            分配结果 {workload_id: {node_id, resources_allocated}}
        """
        if constraints is None:
            constraints = {}
        
        constraints = {
            "cpu": "max",
            "memory": "balanced", 
            "gpu": "efficient",
            "cost": "min",
            **(constraints or {})
        }
        
        # 按优先级排序工作负载
        sorted_workloads = sorted(workloads, key=lambda w: -w.priority)
        
        allocation = {}
        
        for workload in sorted_workloads:
            best_node = self._find_best_node(workload, constraints)
            
            if best_node:
                allocation[workload.id] = {
                    "node_id": best_node.node_id,
                    "resources": self._allocate_resources(best_node, workload),
                    "strategy": self.strategy.value
                }
                logger.info(f"工作负载 {workload.id} 分配到节点 {best_node.node_id}")
            else:
                allocation[workload.id] = {
                    "node_id": None,
                    "resources": {},
                    "strategy": "pending",
                    "reason": "无可用节点"
                }
                logger.warning(f"工作负载 {workload.id} 无法分配: 无可用节点")
        
        self.allocation_history.append({
            "timestamp": "now",
            "allocations": allocation,
            "constraints": constraints
        })
        
        return allocation
    
    def _find_best_node(
        self, 
        workload: Workload, 
        constraints: Dict
    ) -> Optional[NodeResources]:
        """查找最佳节点"""
        candidates = []
        
        for node_id, node in self.nodes.items():
            if self._can_accommodate(node, workload):
                score = self._calculate_node_score(node, workload, constraints)
                candidates.append((score, node))
        
        if not candidates:
            return None
        
        # 返回分数最高的节点
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    
    def _can_accommodate(self, node: NodeResources, workload: Workload) -> bool:
        """检查节点是否能容纳工作负载"""
        req = workload.resource_requirements
        
        # CPU检查
        if req.get("cpu", 0) > node.cpu_available:
            return False
        
        # 内存检查
        if req.get("memory", 0) > node.memory_available:
            return False
        
        # GPU检查
        if workload.is_gpu_required and req.get("gpu", 0) > node.gpu_available:
            return False
        
        return True
    
    def _calculate_node_score(
        self, 
        node: NodeResources, 
        workload: Workload,
        constraints: Dict
    ) -> float:
        """计算节点评分"""
        score = 100.0
        
        # CPU利用率评分 (越高越好)
        cpu_utilization = (node.cpu_total - node.cpu_available) / node.cpu_total
        if constraints.get("cpu") == "max":
            score += (1 - cpu_utilization) * 50
        else:
            score += (1 - abs(0.7 - cpu_utilization)) * 50
        
        # 内存利用率评分
        mem_utilization = (node.memory_total - node.memory_available) / node.memory_total
        score += (1 - abs(0.7 - mem_utilization)) * 30
        
        # GPU工作负载优先分配到有GPU的节点
        if workload.is_gpu_required:
            if node.gpu_available > 0:
                score += 100
            else:
                score -= 1000
        
        # 成本优化: 利用率高的节点优先
        if constraints.get("cost") == "min":
            total_util = (cpu_utilization + mem_utilization) / 2
            score += total_util * 20
        
        return score
    
    def _allocate_resources(
        self, 
        node: NodeResources, 
        workload: Workload
    ) -> Dict:
        """分配资源"""
        req = workload.resource_requirements
        
        allocated = {}
        
        # CPU分配
        if "cpu" in req:
            allocated["cpu"] = min(req["cpu"], node.cpu_available)
            node.cpu_available -= allocated["cpu"]
        
        # 内存分配
        if "memory" in req:
            allocated["memory"] = min(req["memory"], node.memory_available)
            node.memory_available -= allocated["memory"]
        
        # GPU分配
        if workload.is_gpu_required and "gpu" in req:
            allocated["gpu"] = min(req["gpu"], node.gpu_available)
            node.gpu_available -= allocated["gpu"]
        
        return allocated
    
    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        total_cpu = sum(n.cpu_total for n in self.nodes.values())
        available_cpu = sum(n.cpu_available for n in self.nodes.values())
        total_mem = sum(n.memory_total for n in self.nodes.values())
        available_mem = sum(n.memory_available for n in self.nodes.values())
        total_gpu = sum(n.gpu_total for n in self.nodes.values())
        available_gpu = sum(n.gpu_available for n in self.nodes.values())
        
        return {
            "cpu_utilization": (total_cpu - available_cpu) / total_cpu if total_cpu > 0 else 0,
            "memory_utilization": (total_mem - available_mem) / total_mem if total_mem > 0 else 0,
            "gpu_utilization": (total_gpu - available_gpu) / total_gpu if total_gpu > 0 else 0,
            "total_nodes": len(self.nodes),
            "total_workloads": len(self.workloads),
            "strategy": self.strategy.value
        }
    
    def optimize_gpu_schedule(self, gpu_workloads: List[Workload]) -> Dict[str, str]:
        """GPU调度优化"""
        gpu_nodes = {
            node_id: node for node_id, node in self.nodes.items()
            if node.gpu_total > 0
        }
        
        allocation = {}
        
        for workload in gpu_workloads:
            if not workload.is_gpu_required:
                continue
                
            best_node = None
            best_score = -float('inf')
            
            for node_id, node in gpu_nodes.items():
                if node.gpu_available >= workload.resource_requirements.get("gpu", 1):
                    # 评分: 优先选择GPU利用率低的节点
                    gpu_util = (node.gpu_total - node.gpu_available) / node.gpu_total
                    score = (1 - gpu_util) * 100 - node_id.lower().encode()[0] % 10
                    
                    if score > best_score:
                        best_score = score
                        best_node = node_id
            
            if best_node:
                allocation[workload.id] = best_node
                self.nodes[best_node].gpu_available -= workload.resource_requirements.get("gpu", 1)
        
        return allocation
    
    def optimize_network_bandwidth(self) -> Dict[str, float]:
        """网络带宽优化"""
        allocations = {}
        
        for workload_id, workload in self.workloads.items():
            if "network" in workload.resource_requirements:
                required = workload.resource_requirements["network"]
                available_nodes = [
                    (nid, node) for nid, node in self.nodes.items()
                    if node.network_bandwidth >= required
                ]
                
                if available_nodes:
                    # 选择网络带宽最充裕的节点
                    best_node = max(available_nodes, key=lambda x: x[1].network_bandwidth)
                    allocations[workload_id] = {
                        "node_id": best_node[0],
                        "bandwidth": required
                    }
        
        return allocations
