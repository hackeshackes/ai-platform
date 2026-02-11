"""
Self Organization - 自组织系统
神经网络自组织、突触可塑性、结构涌现、动态架构
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import random
from collections import defaultdict


class NetworkType(Enum):
    """网络类型"""
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    CONVOLUTIONAL = "convolutional"
    ATTENTION = "attention"
    HYBRID = "hybrid"


class PlasticityRule(Enum):
    """可塑性规则"""
    HEBBIAN = "hebbian"                    # Hebbian学习
    STDP = "stdp"                          # 脉冲时序依赖性可塑性
    HOMEOSTATIC = "homeostatic"            # 稳态可塑性
    METAPLASTICITY = "metaplasticity"      # 元可塑性


@dataclass
class Neuron:
    """神经元"""
    neuron_id: str
    layer: int
    position: Tuple[int, ...]
    activation: float = 0.0
    threshold: float = 0.5
    decay_rate: float = 0.1
    
    
@dataclass
class Synapse:
    """突触连接"""
    pre_neuron_id: str
    post_neuron_id: str
    weight: float
    plasticity_rule: PlasticityRule
    eligibility_trace: float = 0.0
    last_update: int = 0
    
    
@dataclass
class NetworkStructure:
    """网络结构"""
    network_type: NetworkType
    layers: Dict[int, List[Neuron]]
    connections: List[Synapse]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfOrganization:
    """
    自组织系统
    实现神经网络的自组织学习、突触可塑性、结构涌现和动态架构调整
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[Tuple[str, str], Synapse] = {}
        self.structure = NetworkStructure(
            network_type=NetworkType.HYBRID,
            layers={},
            connections=[]
        )
        self.time_step = 0
        self.activity_history: List[Dict] = []
        
        # 可塑性参数
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.plasticity_scale = self.config.get('plasticity_scale', 1.0)
        self.structural_plasticity_rate = self.config.get('structural_plasticity_rate', 0.001)
        
    def _default_config(self) -> Dict:
        return {
            'learning_rate': 0.01,
            'plasticity_scale': 1.0,
            'structural_plasticity_rate': 0.001,
            'max_neurons_per_layer': 100,
            'max_connections_per_neuron': 50,
            'pruning_threshold': 0.01,
            'emergence_window': 1000
        }
    
    def organize(self, neural_network: Any = None) -> NetworkStructure:
        """
        执行自组织过程
        
        Args:
            neural_network: 可选的神经网络实例
            
        Returns:
            更新后的网络结构
        """
        # 更新突触可塑性
        self._update_synaptic_plasticity()
        
        # 执行结构可塑性（生长/修剪）
        self._update_structural_plasticity()
        
        # 更新网络活动
        self._update_activity()
        
        # 检测结构涌现
        self._detect_structure_emergence()
        
        # 更新层结构
        self._update_layer_structure()
        
        # 创建网络结构对象
        self.structure = NetworkStructure(
            network_type=self._determine_network_type(),
            layers=self._get_layer_neurons(),
            connections=list(self.synapses.values()),
            metadata={
                'time_step': self.time_step,
                'total_neurons': len(self.neurons),
                'total_connections': len(self.synapses),
                'emergence_detected': self._check_emergence()
            }
        )
        
        self.time_step += 1
        return self.structure
    
    def _update_synaptic_plasticity(self):
        """更新突触可塑性"""
        for (pre_id, post_id), synapse in self.synapses.items():
            if pre_id not in self.neurons or post_id not in self.neurons:
                continue
                
            pre_neuron = self.neurons[pre_id]
            post_neuron = self.neurons[post_id]
            
            # 应用不同的可塑性规则
            if synapse.plasticity_rule == PlasticityRule.HEBBIAN:
                delta_w = self._hebbian_rule(pre_neuron, post_neuron)
            elif synapse.plasticity_rule == PlasticityRule.STSP:
                delta_w = self._stdp_rule(pre_neuron, post_neuron)
            elif synapse.plasticity_rule == PlasticityRule.HOMEOSTATIC:
                delta_w = self._homeostatic_rule(pre_neuron, post_neuron)
            else:
                delta_w = self._metaplasticity_rule(pre_neuron, post_neuron)
            
            # 更新权重
            synapse.weight += self.learning_rate * delta_w * self.plasticity_scale
            
            # 权重限制
            synapse.weight = np.clip(synapse.weight, -1.0, 1.0)
            
            # 更新 eligibility trace
            synapse.eligibility_trace = 0.9 * synapse.eligibility_trace + pre_neuron.activation * post_neuron.activation
            synapse.last_update = self.time_step
    
    def _hebbian_rule(self, pre: Neuron, post: Neuron) -> float:
        """Hebbian学习规则: fire together, wire together"""
        return pre.activation * post.activation - 0.01 * self.neurons[post.neuron_id].decay_rate
    
    def _stdp_rule(self, pre: Neuron, post: Neuron) -> float:
        """脉冲时序依赖性可塑性"""
        time_diff = post.threshold - pre.threshold  # 简化的时间差
        return np.exp(-abs(time_diff)) * pre.activation * post.activation
    
    def _homeostatic_rule(self, pre: Neuron, post: Neuron) -> float:
        """稳态可塑性"""
        target_activity = 0.5
        current_activity = post.activation
        return -0.1 * (current_activity - target_activity) * pre.activation
    
    def _metaplasticity_rule(self, pre: Neuron, post: Neuron) -> float:
        """元可塑性 - 根据突触历史调整可塑性"""
        base_plasticity = self._hebbian_rule(pre, post)
        history_factor = 1.0 - 0.5 * abs(pre.activation - 0.5)
        return base_plasticity * history_factor
    
    def _update_structural_plasticity(self):
        """更新结构可塑性（生长和修剪）"""
        # 修剪弱连接
        self._prune_weak_connections()
        
        # 生长新连接
        self._grow_new_connections()
        
        # 可能创建新神经元
        self._maybe_create_neuron()
    
    def _prune_weak_connections(self):
        """修剪弱突触连接"""
        weak_connections = []
        for (pre_id, post_id), synapse in self.synapses.items():
            if abs(synapse.weight) < self.config['pruning_threshold']:
                weak_connections.append((pre_id, post_id))
                
        for conn in weak_connections:
            del self.synapses[conn]
    
    def _grow_new_connections(self):
        """生长新连接"""
        if len(self.synapses) >= len(self.neurons) * self.config['max_connections_per_neuron']:
            return
            
        # 找到未连接的神经元对
        potential_connections = []
        for pre_id in self.neurons:
            for post_id in self.neurons:
                if pre_id != post_id and (pre_id, post_id) not in self.synapses:
                    potential_connections.append((pre_id, post_id))
                    
        # 随机选择一些进行连接
        num_to_grow = min(
            len(potential_connections),
            int(self.structural_plasticity_rate * len(self.neurons) * 10)
        )
        
        for pre_id, post_id in random.sample(potential_connections, num_to_grow):
            synapse = Synapse(
                pre_neuron_id=pre_id,
                post_neuron_id=post_id,
                weight=random.uniform(-0.1, 0.1),
                plasticity_rule=PlasticityRule.HEBBIAN
            )
            self.synapses[(pre_id, post_id)] = synapse
    
    def _maybe_create_neuron(self):
        """可能创建新神经元"""
        total_neurons = len(self.neurons)
        max_neurons = sum(self.config['max_neurons_per_layer'] for _ in range(3))
        
        if total_neurons >= max_neurons:
            return
            
        # 基于活动模式决定是否创建
        if len(self.activity_history) > 10:
            recent_activity = np.mean([a['total_activity'] for a in self.activity_history[-10:]])
            if recent_activity > 0.3:
                # 创建新神经元
                new_neuron_id = f"neuron_{total_neurons}"
                new_neuron = Neuron(
                    neuron_id=new_neuron_id,
                    layer=random.randint(0, 2),
                    position=(random.randint(0, 10), random.randint(0, 10))
                )
                self.neurons[new_neuron_id] = new_neuron
    
    def _update_activity(self):
        """更新网络活动"""
        total_activity = 0.0
        
        for neuron in self.neurons.values():
            # 计算输入
            input_sum = 0.0
            for (pre_id, post_id), synapse in self.synapses.items():
                if post_id == neuron.neuron_id and pre_id in self.neurons:
                    input_sum += synapse.weight * self.neurons[pre_id].activation
            
            # 更新激活
            neuron.activation = np.tanh(input_sum)
            total_activity += neuron.activation
            
        self.activity_history.append({
            'time_step': self.time_step,
            'total_activity': total_activity,
            'neuron_activities': {
                nid: n.activation for nid, n in self.neurons.items()
            }
        })
        
        # 限制历史长度
        if len(self.activity_history) > self.config['emergence_window']:
            self.activity_history = self.activity_history[-self.config['emergence_window']:]
    
    def _detect_structure_emergence(self):
        """检测结构涌现"""
        if len(self.activity_history) < 100:
            return
            
        # 检查是否有新的活动模式
        recent = self.activity_history[-100:]
        
        # 检测模块化结构
        modularity = self._calculate_modularity()
        
        # 检测层次结构
        hierarchy = self._calculate_hierarchy()
        
        # 记录涌现事件
        if modularity > 0.5 or hierarchy > 0.3:
            self._record_emergence_event('structure', {
                'modularity': modularity,
                'hierarchy': hierarchy,
                'time_step': self.time_step
            })
    
    def _calculate_modularity(self) -> float:
        """计算模块化程度"""
        if len(self.neurons) < 10:
            return 0.0
            
        # 简化的模块化计算
        layer_groups = defaultdict(list)
        for neuron in self.neurons.values():
            layer_groups[neuron.layer].append(neuron.neuron_id)
            
        within_connections = 0
        between_connections = 0
        
        for (pre_id, post_id), synapse in self.synapses.items():
            if self.neurons[pre_id].layer == self.neurons[post_id].layer:
                within_connections += 1
            else:
                between_connections += 1
                
        total = within_connections + between_connections
        if total == 0:
            return 0.0
            
        return within_connections / total
    
    def _calculate_hierarchy(self) -> float:
        """计算层次结构"""
        layers = sorted(set(n.layer for n in self.neurons.values()))
        if len(layers) < 2:
            return 0.0
            
        # 检查信息是否从低层流向高层
        flow_count = 0
        for (pre_id, post_id), synapse in self.synapses.items():
            if self.neurons[pre_id].layer < self.neurons[post_id].layer:
                flow_count += 1
                
        total = len(self.synapses)
        if total == 0:
            return 0.0
            
        return flow_count / total
    
    def _update_layer_structure(self):
        """更新层结构"""
        for neuron in self.neurons.values():
            if neuron.layer not in self.structure.layers:
                self.structure.layers[neuron.layer] = []
            self.structure.layers[neuron.layer].append(neuron)
    
    def _determine_network_type(self) -> NetworkType:
        """确定网络类型"""
        layer_count = len(self.structure.layers)
        recurrent_connections = sum(
            1 for (pre_id, post_id) in self.synapses
            if self.neurons[pre_id].layer == self.neurons[post_id].layer
        )
        
        if recurrent_connections > len(self.synapses) * 0.3:
            return NetworkType.RECURRENT
        elif layer_count >= 3:
            return NetworkType.HYBRID
        else:
            return NetworkType.FEEDFORWARD
    
    def _get_layer_neurons(self) -> Dict[int, List[Neuron]]:
        """获取按层分组的神经元"""
        layers = defaultdict(list)
        for neuron in self.neurons.values():
            layers[neuron.layer].append(neuron)
        return dict(layers)
    
    def _check_emergence(self) -> bool:
        """检查是否检测到涌现"""
        if len(self.activity_history) < 100:
            return False
            
        recent = self.activity_history[-100:]
        variance = np.var([a['total_activity'] for a in recent])
        
        return variance > 0.1
    
    def _record_emergence_event(self, event_type: str, data: Dict):
        """记录涌现事件"""
        # 这里可以连接到监控系统
        pass
    
    def add_neuron(self, layer: int, position: Tuple[int, ...]) -> Neuron:
        """添加神经元"""
        neuron_id = f"neuron_{len(self.neurons)}"
        neuron = Neuron(
            neuron_id=neuron_id,
            layer=layer,
            position=position
        )
        self.neurons[neuron_id] = neuron
        return neuron
    
    def add_synapse(self, pre_id: str, post_id: str, 
                   weight: float = None, 
                   plasticity_rule: PlasticityRule = PlasticityRule.HEBBIAN) -> Synapse:
        """添加突触连接"""
        if weight is None:
            weight = random.uniform(-0.1, 0.1)
            
        synapse = Synapse(
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
            weight=weight,
            plasticity_rule=plasticity_rule
        )
        self.synapses[(pre_id, post_id)] = synapse
        return synapse
    
    def reset(self):
        """重置系统"""
        self.neurons = {}
        self.synapses = {}
        self.structure = NetworkStructure(
            network_type=NetworkType.HYBRID,
            layers={},
            connections=[]
        )
        self.time_step = 0
        self.activity_history = []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_neurons': len(self.neurons),
            'total_connections': len(self.synapses),
            'network_type': self.structure.network_type.value,
            'layers': list(self.structure.layers.keys()),
            'time_step': self.time_step,
            'recent_activity_mean': np.mean([
                a['total_activity'] for a in self.activity_history[-10:]
            ]) if self.activity_history else 0
        }
