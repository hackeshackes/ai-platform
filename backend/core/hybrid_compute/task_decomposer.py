"""
任务分解器 - 负责将复杂问题分解为量子/经典子任务
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import networkx as nx


class TaskType(Enum):
    """任务类型枚举"""
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"
    DATA_PREPROCESSING = "data_preprocessing"
    RESULT_POSTPROCESSING = "result_postprocessing"


class DependencyType(Enum):
    """依赖类型枚举"""
    DATA_FLOW = "data_flow"      # 数据流依赖
    CONTROL_FLOW = "control_flow" # 控制流依赖
    RESOURCE = "resource"         # 资源依赖


@dataclass
class SubTask:
    """子任务数据类"""
    task_id: str
    name: str
    task_type: TaskType
    description: str = ""
    input_specs: Dict[str, Any] = field(default_factory=dict)
    output_specs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: float = 1.0
    estimated_time: float = 1.0
    priority: int = 0
    quantum_config: Optional[Dict[str, Any]] = None
    classical_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TaskGraph:
    """任务图数据结构"""
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    edges: List[Tuple[str, str, DependencyType]] = field(default_factory=list)
    
    def add_task(self, task: SubTask) -> None:
        """添加任务节点"""
        self.tasks[task.task_id] = task
    
    def add_dependency(self, from_id: str, to_id: str, dep_type: DependencyType = DependencyType.DATA_FLOW) -> None:
        """添加依赖边"""
        self.edges.append((from_id, to_id, dep_type))
    
    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（拓扑排序，按层级分组）"""
        # 构建图
        G = nx.DiGraph()
        for task_id in self.tasks:
            G.add_node(task_id)
        for from_id, to_id, _ in self.edges:
            G.add_edge(from_id, to_id)
        
        # 拓扑排序
        try:
            levels = []
            temp_graph = G.copy()
            
            while temp_graph.nodes():
                # 找到入度为0的节点
                zero_indegree = [n for n in temp_graph.nodes() if temp_graph.in_degree(n) == 0]
                if not zero_indegree:
                    raise ValueError("任务图存在循环依赖")
                
                levels.append(zero_indegree)
                temp_graph.remove_nodes_from(zero_indegree)
            
            return levels
        except nx.NetworkXUnfeasible:
            # 如果存在循环依赖，返回任意顺序
            return [list(self.tasks.keys())]
    
    def get_task(self, task_id: str) -> Optional[SubTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_quantum_tasks(self) -> List[SubTask]:
        """获取所有量子任务"""
        return [t for t in self.tasks.values() if t.task_type == TaskType.QUANTUM]
    
    def get_classical_tasks(self) -> List[SubTask]:
        """获取所有经典任务"""
        return [t for t in self.tasks.values() if t.task_type == TaskType.CLASSICAL]


class TaskDecomposer:
    """
    任务分解器核心类
    
    负责将复杂问题分解为可执行的量子/经典子任务，
    分析任务间依赖关系，优化任务划分策略。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化任务分解器"""
        self.config = config or {}
        self.strategy = self.config.get('strategy', 'auto')
        self.quantum_threshold = self.config.get('quantum_threshold', 0.7)  # 量子加速阈值
        
        # 问题类型识别规则
        self.problem_rules = {
            'optimization': {
                'quantum_tasks': ['sampling', 'vqe_iteration', 'QAOA_layer'],
                'classical_tasks': ['objective_eval', 'gradient_compute', 'convergence_check'],
                'expected_speedup': 10
            },
            'quantum_chemistry': {
                'quantum_tasks': ['hamiltonian_measurement', 'ansatz_evaluation', 'ground_state_search'],
                'classical_tasks': ['post_processing', 'energy_extrapolation', 'molecular_properties'],
                'expected_speedup': 100
            },
            'machine_learning': {
                'quantum_tasks': ['kernel_computation', 'feature_map_evaluation', 'quantum_svm_train'],
                'classical_tasks': ['data_preprocessing', 'model_training', 'hyperparameter_tuning'],
                'expected_speedup': 50
            },
            'cryptography': {
                'quantum_tasks': ['grover_search', 'amplitude_amplification', 'oracle_evaluation'],
                'classical_tasks': ['key_verification', 'cryptanalysis', 'security_analysis'],
                'expected_speedup': 1000  # 指数级
            }
        }
    
    def decompose(self, problem: Dict[str, Any], strategy: str = "auto") -> TaskGraph:
        """
        分解问题为子任务图
        
        Args:
            problem: 问题描述字典，应包含 'type', 'data', 'constraints' 等字段
            strategy: 分解策略 ('auto', 'quantum_heavy', 'classical_heavy', 'balanced')
            
        Returns:
            TaskGraph: 包含所有子任务和依赖关系的任务图
        """
        # 解析问题类型
        problem_type = problem.get('type', 'optimization')
        problem_data = problem.get('data', {})
        constraints = problem.get('constraints', {})
        
        # 创建任务图
        task_graph = TaskGraph()
        
        # 根据问题类型选择分解规则
        rules = self.problem_rules.get(problem_type, self.problem_rules['optimization'])
        
        # 根据策略调整任务划分
        if strategy == "quantum_heavy":
            self._decompose_quantum_heavy(task_graph, rules, problem_data, constraints)
        elif strategy == "classical_heavy":
            self._decompose_classical_heavy(task_graph, rules, problem_data, constraints)
        else:  # auto or balanced
            self._decompose_balanced(task_graph, rules, problem_data, constraints)
        
        # 分析并优化依赖关系
        self._analyze_dependencies(task_graph)
        
        return task_graph
    
    def _decompose_quantum_heavy(self, task_graph: TaskGraph, rules: Dict, 
                                  problem_data: Dict, constraints: Dict) -> None:
        """量子优先的分解策略"""
        # 数据预处理（经典）
        preprocess_task = SubTask(
            task_id="preprocess_001",
            name="数据预处理",
            task_type=TaskType.DATA_PREPROCESSING,
            description="输入数据的预处理和标准化",
            estimated_cost=0.1,
            estimated_time=0.1,
            priority=0
        )
        task_graph.add_task(preprocess_task)
        
        # 量子核心任务
        for i, qt_name in enumerate(rules['quantum_tasks']):
            quantum_task = SubTask(
                task_id=f"quantum_{i:03d}",
                name=f"量子任务: {qt_name}",
                task_type=TaskType.QUANTUM,
                description=f"执行{qt_name}量子计算",
                dependencies=["preprocess_001"],
                estimated_cost=0.8,
                estimated_time=0.8,
                priority=10 - i,
                quantum_config=self._get_quantum_config(qt_name, problem_data)
            )
            task_graph.add_task(quantum_task)
        
        # 经典后处理
        postprocess_task = SubTask(
            task_id="postprocess_001",
            name="结果后处理",
            task_type=TaskType.RESULT_POSTPROCESSING,
            description="量子计算结果的后处理",
            dependencies=[f"quantum_{i:03d}" for i in range(len(rules['quantum_tasks']))],
            estimated_cost=0.1,
            estimated_time=0.1,
            priority=-1
        )
        task_graph.add_task(postprocess_task)
    
    def _decompose_classical_heavy(self, task_graph: TaskGraph, rules: Dict,
                                   problem_data: Dict, constraints: Dict) -> None:
        """经典优先的分解策略"""
        # 数据预处理
        preprocess_task = SubTask(
            task_id="preprocess_001",
            name="数据预处理",
            task_type=TaskType.DATA_PREPROCESSING,
            description="输入数据的预处理和标准化",
            estimated_cost=0.2,
            estimated_time=0.2,
            priority=0
        )
        task_graph.add_task(preprocess_task)
        
        # 经典子任务
        for i, ct_name in enumerate(rules['classical_tasks']):
            if i == 0:
                deps = ["preprocess_001"]
            else:
                deps = [f"classical_{i-1:03d}"]
            
            classical_task = SubTask(
                task_id=f"classical_{i:03d}",
                name=f"经典任务: {ct_name}",
                task_type=TaskType.CLASSICAL,
                description=f"执行{ct_name}经典计算",
                dependencies=deps,
                estimated_cost=0.15,
                estimated_time=0.2,
                priority=5 - i,
                classical_config=self._get_classical_config(ct_name, problem_data)
            )
            task_graph.add_task(classical_task)
        
        # 轻量级量子任务
        for i, qt_name in enumerate(rules['quantum_tasks'][:2]):  # 只保留核心量子任务
            quantum_task = SubTask(
                task_id=f"quantum_{i:03d}",
                name=f"量子任务: {qt_name}",
                task_type=TaskType.QUANTUM,
                description=f"执行{qt_name}量子计算",
                dependencies=[f"classical_{len(rules['classical_tasks'])-1:03d}"],
                estimated_cost=0.4,
                estimated_time=0.4,
                priority=5,
                quantum_config=self._get_quantum_config(qt_name, problem_data)
            )
            task_graph.add_task(quantum_task)
        
        # 最终后处理
        postprocess_task = SubTask(
            task_id="postprocess_001",
            name="结果后处理",
            task_type=TaskType.RESULT_POSTPROCESSING,
            description="最终结果处理",
            dependencies=["quantum_000", "quantum_001"],
            estimated_cost=0.1,
            estimated_time=0.1,
            priority=-1
        )
        task_graph.add_task(postprocess_task)
    
    def _decompose_balanced(self, task_graph: TaskGraph, rules: Dict,
                            problem_data: Dict, constraints: Dict) -> None:
        """平衡分解策略"""
        # 数据预处理
        preprocess_task = SubTask(
            task_id="preprocess_001",
            name="数据预处理",
            task_type=TaskType.DATA_PREPROCESSING,
            description="输入数据的预处理和标准化",
            estimated_cost=0.15,
            estimated_time=0.15,
            priority=0
        )
        task_graph.add_task(preprocess_task)
        
        # 量子-经典交替执行
        total_phases = max(len(rules['quantum_tasks']), len(rules['classical_tasks']))
        
        for phase in range(total_phases):
            # 经典任务（如果存在）
            if phase < len(rules['classical_tasks']):
                deps = ["preprocess_001"] if phase == 0 else [f"quantum_{phase-1:03d}"]
                classical_task = SubTask(
                    task_id=f"classical_{phase:03d}",
                    name=f"经典任务: {rules['classical_tasks'][phase]}",
                    task_type=TaskType.CLASSICAL,
                    description=f"执行{rules['classical_tasks'][phase]}经典计算",
                    dependencies=deps,
                    estimated_cost=0.15,
                    estimated_time=0.15,
                    priority=5,
                    classical_config=self._get_classical_config(rules['classical_tasks'][phase], problem_data)
                )
                task_graph.add_task(classical_task)
            
            # 量子任务（如果存在）
            if phase < len(rules['quantum_tasks']):
                if phase == 0:
                    deps = ["preprocess_001"]
                elif phase < len(rules['classical_tasks']):
                    deps = [f"classical_{phase:03d}"]
                else:
                    deps = [f"quantum_{phase-1:03d}"]
                
                quantum_task = SubTask(
                    task_id=f"quantum_{phase:03d}",
                    name=f"量子任务: {rules['quantum_tasks'][phase]}",
                    task_type=TaskType.QUANTUM,
                    description=f"执行{rules['quantum_tasks'][phase]}量子计算",
                    dependencies=deps,
                    estimated_cost=0.3,
                    estimated_time=0.3,
                    priority=5,
                    quantum_config=self._get_quantum_config(rules['quantum_tasks'][phase], problem_data)
                )
                task_graph.add_task(quantum_task)
        
        # 后处理
        postprocess_task = SubTask(
            task_id="postprocess_001",
            name="结果后处理",
            task_type=TaskType.RESULT_POSTPROCESSING,
            description="最终结果处理和格式化",
            dependencies=[f"quantum_{(total_phases-1):03d}"] if total_phases > 0 else ["preprocess_001"],
            estimated_cost=0.1,
            estimated_time=0.1,
            priority=-1
        )
        task_graph.add_task(postprocess_task)
    
    def _analyze_dependencies(self, task_graph: TaskGraph) -> None:
        """分析并优化依赖关系"""
        # 构建反向依赖图
        reverse_deps = defaultdict(list)
        for from_id, to_id, dep_type in task_graph.edges:
            reverse_deps[to_id].append(from_id)
        
        # 优化：移除冗余依赖
        for task_id in task_graph.tasks:
            task = task_graph.tasks[task_id]
            direct_deps = set(task.dependencies)
            
            # 如果A依赖B，B依赖C，那么A可以直接依赖C
            optimized_deps = set()
            for dep in direct_deps:
                if dep not in reverse_deps:
                    optimized_deps.add(dep)
                else:
                    # 检查是否所有依赖项都有相同的间接依赖
                    has_common_indirect = True
                    for indirect in reverse_deps[dep]:
                        if indirect not in direct_deps:
                            has_common_indirect = False
                            break
                    if not has_common_indirect:
                        optimized_deps.add(dep)
            
            task.dependencies = list(optimized_deps)
    
    def _get_quantum_config(self, task_name: str, problem_data: Dict) -> Dict[str, Any]:
        """获取量子任务配置"""
        return {
            'task_name': task_name,
            'qubits': problem_data.get('n_qubits', 4),
            'depth': problem_data.get('circuit_depth', 10),
            'shots': problem_data.get('shots', 1024),
            'backend': problem_data.get('quantum_backend', 'simulator')
        }
    
    def _get_classical_config(self, task_name: str, problem_data: Dict) -> Dict[str, Any]:
        """获取经典任务配置"""
        return {
            'task_name': task_name,
            'algorithm': task_name,
            'precision': problem_data.get('precision', 'double'),
            'max_iterations': problem_data.get('max_iter', 1000)
        }
    
    def estimate_speedup(self, task_graph: TaskGraph) -> float:
        """
        估算混合计算的加速比
        
        Returns:
            float: 估算的加速比
        """
        quantum_time = sum(t.estimated_time for t in task_graph.get_quantum_tasks())
        classical_time = sum(t.estimated_time for t in task_graph.get_classical_tasks())
        
        if classical_time == 0:
            return float('inf') if quantum_time > 0 else 1.0
        
        # 考虑量子加速因子
        quantum_speedup = 10  # 假设量子计算比经典快10倍
        effective_quantum_time = quantum_time / quantum_speedup
        
        total_original = quantum_time + classical_time
        total_hybrid = effective_quantum_time + classical_time
        
        return total_original / total_hybrid if total_hybrid > 0 else 1.0
    
    def optimize_partition(self, task_graph: TaskGraph) -> TaskGraph:
        """
        优化任务划分以提高混合效率
        
        Args:
            task_graph: 原始任务图
            
        Returns:
            TaskGraph: 优化后的任务图
        """
        # 合并连续相同类型的任务
        optimized = TaskGraph()
        merged_tasks = {}
        
        current_type = None
        current_tasks = []
        
        for task_id in task_graph.get_execution_order():
            task = task_graph.get_task(task_id)
            if task is None:
                continue
            
            if current_type is None:
                current_type = task.task_type
                current_tasks = [task]
            elif task.task_type == current_type:
                current_tasks.append(task)
            else:
                # 合并连续相同类型的任务
                merged_id = f"merged_{current_type.value}_{len(merged_tasks)}"
                merged_task = self._merge_tasks(merged_id, current_type, current_tasks)
                merged_tasks[merged_id] = merged_task
                
                current_type = task.task_type
                current_tasks = [task]
        
        # 处理最后一批任务
        if current_tasks:
            merged_id = f"merged_{current_type.value}_{len(merged_tasks)}"
            merged_task = self._merge_tasks(merged_id, current_type, current_tasks)
            merged_tasks[merged_id] = merged_task
        
        # 复制依赖关系
        for task_id, task in merged_tasks.items():
            optimized.add_task(task)
        
        for from_id, to_id, dep_type in task_graph.edges:
            if from_id in merged_tasks and to_id in merged_tasks:
                optimized.add_dependency(from_id, to_id, dep_type)
        
        return optimized
    
    def _merge_tasks(self, task_id: str, task_type: TaskType, tasks: List[SubTask]) -> SubTask:
        """合并多个任务为一个"""
        merged_dependencies = set()
        total_cost = 0
        total_time = 0
        
        for task in tasks:
            merged_dependencies.update(task.dependencies)
            total_cost += task.estimated_cost
            total_time += task.estimated_time
        
        return SubTask(
            task_id=task_id,
            name=f"合并后的{task_type.value}任务",
            task_type=task_type,
            description=f"合并了{len(tasks)}个{task_type.value}任务",
            dependencies=list(merged_dependencies),
            estimated_cost=total_cost * 0.9,  # 合并略有优化
            estimated_time=total_time * 0.85,  # 减少通信开销
            priority=min(t.priority for t in tasks)
        )
