"""
AutoML NAS - 神经架构搜索
Neural Architecture Search Module for AutoML

提供神经网络架构自动搜索功能:
- 搜索空间定义
- 架构编码与解码
- 进化算法搜索
- 性能预测
- 搜索结果导出
"""
from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from uuid import uuid4
import json
import random
import copy
from pathlib import Path

class NASTask(Enum):
    """NAS任务类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"

class LayerType(Enum):
    """层类型"""
    DENSE = "dense"
    CONV2D = "conv2d"
    CONV2D_TRANSPOSE = "conv2d_transpose"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    POOLING = "pooling"
    RESIDUAL = "residual"

@dataclass
class LayerSpec:
    """
    层规格定义
    
    Attributes:
        layer_id: 层ID
        type: 层类型
        config: 层配置参数字典
        input_shape: 输入形状
        output_shape: 输出形状
    """
    layer_id: str
    type: LayerType
    config: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def to_dict(self) -> Dict:
        return {
            "layer_id": self.layer_id,
            "type": self.type.value,
            "config": self.config,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape
        }

@dataclass
class ArchitectureGenome:
    """
    架构基因组 (神经架构的编码表示)
    
    Attributes:
        genome_id: 基因组ID
        layers: 层列表
        connections: 连接关系
        fitness: 适应度分数
        accuracy: 验证准确率
        latency: 推理延迟(ms)
        params: 参数量
    """
    genome_id: str
    layers: List[LayerSpec]
    connections: List[Tuple[str, str]] = field(default_factory=list)  # (from_layer, to_layer)
    fitness: float = 0.0
    accuracy: float = 0.0
    latency: float = 0.0
    params: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "genome_id": self.genome_id,
            "layers": [l.to_dict() for l in self.layers],
            "connections": self.connections,
            "fitness": self.fitness,
            "accuracy": self.accuracy,
            "latency": self.latency,
            "params": self.params,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ArchitectureGenome":
        """从字典创建"""
        data["layers"] = [LayerSpec(**l) if isinstance(l, dict) else l for l in data["layers"]]
        return cls(**data)

@dataclass
class NASSearchSpace:
    """
    NAS搜索空间定义
    
    Attributes:
        space_id: 搜索空间ID
        task_type: 任务类型
        input_shape: 输入形状
        max_layers: 最大层数
        available_layers: 可用的层类型
        layer_configs: 层配置选项
    """
    space_id: str
    task_type: NASTask
    input_shape: Tuple[int, ...]
    max_layers: int = 10
    available_layers: List[LayerType] = field(default_factory=list)
    layer_configs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 默认搜索空间
        if not self.available_layers:
            self.available_layers = [
                LayerType.DENSE,
                LayerType.CONV2D,
                LayerType.DROPOUT,
                LayerType.BATCH_NORM,
                LayerType.POOLING
            ]
        
        # 默认层配置
        if not self.layer_configs:
            self.layer_configs = {
                LayerType.DENSE.value: {
                    "units": [32, 64, 128, 256, 512],
                    "activation": ["relu", "tanh", "sigmoid"]
                },
                LayerType.CONV2D.value: {
                    "filters": [16, 32, 64, 128, 256],
                    "kernel_size": [(3, 3), (5, 5)],
                    "strides": [(1, 1), (2, 2)],
                    "activation": ["relu", "swish"]
                },
                LayerType.POOLING.value: {
                    "pool_size": [(2, 2)],
                    "strides": [(2, 2), None]
                },
                LayerType.DROPOUT.value: {
                    "rate": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
    
    def to_dict(self) -> Dict:
        return {
            "space_id": self.space_id,
            "task_type": self.task_type.value,
            "input_shape": self.input_shape,
            "max_layers": self.max_layers,
            "available_layers": [l.value for l in self.available_layers],
            "layer_configs": self.layer_configs
        }


@dataclass
class NASResult:
    """
    NAS搜索结果
    
    Attributes:
        search_id: 搜索任务ID
        search_space: 搜索空间
        population: 种群列表
        best_genome: 最佳架构
        all_genomes: 所有评估的架构
        generations: 迭代次数
        total_time: 总耗时
        created_at: 创建时间
    """
    search_id: str
    search_space: NASSearchSpace
    best_genome: Optional[ArchitectureGenome]
    all_genomes: List[ArchitectureGenome]
    generations: int
    total_time: float
    fitness_history: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "search_id": self.search_id,
            "search_space": self.search_space.to_dict(),
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "all_genomes": [g.to_dict() for g in self.all_genomes],
            "generations": self.generations,
            "total_time": self.total_time,
            "fitness_history": self.fitness_history,
            "created_at": self.created_at.isoformat()
        }


class NeuralArchitectureSearcher:
    """
    神经架构搜索器
    
    使用进化算法自动搜索最优神经网络架构。
    
    Usage:
        nas = NeuralArchitectureSearcher()
        
        space = nas.create_search_space(
            task_type=NASTask.CLASSIFICATION,
            input_shape=(28, 28, 1),
            max_layers=8
        )
        
        result = await nas.search(
            evaluate_fn=lambda arch: evaluate(arch, X_train, y_train),
            search_space=space,
            population_size=20,
            generations=30
        )
        
        print(f"Best architecture: {result.best_genome}")
    """
    
    def __init__(self, verbose: bool = True):
        """初始化搜索器"""
        self.verbose = verbose
        self.searches: Dict[str, NASResult] = {}
    
    def create_search_space(
        self,
        task_type: NASTask,
        input_shape: Tuple[int, ...],
        max_layers: int = 8,
        custom_layers: Optional[List[LayerType]] = None,
        custom_configs: Optional[Dict[str, Any]] = None
    ) -> NASSearchSpace:
        """
        创建搜索空间
        
        Args:
            task_type: 任务类型
            input_shape: 输入形状
            max_layers: 最大层数
            custom_layers: 自定义可用层
            custom_configs: 自定义层配置
            
        Returns:
            NASSearchSpace: 搜索空间定义
        """
        space_id = str(uuid4())
        
        return NASSearchSpace(
            space_id=space_id,
            task_type=task_type,
            input_shape=input_shape,
            max_layers=max_layers,
            available_layers=custom_layers or [],
            layer_configs=custom_configs or {}
        )
    
    async def search(
        self,
        evaluate_fn: callable,
        search_space: NASSearchSpace,
        population_size: int = 20,
        generations: int = 30,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2,
        elitism: int = 2,
        time_budget: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> NASResult:
        """
        执行神经架构搜索
        
        Args:
            evaluate_fn: 架构评估函数，接受ArchitectureGenome返回(fitness, accuracy, latency)
            search_space: 搜索空间
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elitism: 精英保留数量
            time_budget: 时间预算(秒)
            save_dir: 保存目录
            
        Returns:
            NASResult: 搜索结果
        """
        search_id = str(uuid4())
        started_at = datetime.utcnow()
        
        if self.verbose:
            print(f"[NAS] Starting search {search_id}")
            print(f"[NAS] Space: {search_space.task_type.value}, layers={search_space.max_layers}")
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            genome = self._random_genome(search_space)
            population.append(genome)
        
        best_genome = None
        fitness_history = []
        
        # 进化迭代
        for gen in range(generations):
            # 检查时间预算
            if time_budget:
                elapsed = (datetime.utcnow() - started_at).total_seconds()
                if elapsed > time_budget:
                    if self.verbose:
                        print(f"[NAS] Time budget reached at generation {gen}")
                    break
            
            if self.verbose:
                print(f"[NAS] Generation {gen + 1}/{generations}")
            
            # 评估种群
            for genome in population:
                if genome.fitness == 0:  # 未评估
                    try:
                        fitness, accuracy, latency = await evaluate_fn(genome)
                        genome.fitness = fitness
                        genome.accuracy = accuracy
                        genome.latency = latency
                    except Exception as e:
                        if self.verbose:
                            print(f"[NAS] Evaluation failed: {str(e)}")
                        genome.fitness = 0.0
            
            # 找到最佳个体
            population.sort(key=lambda g: g.fitness, reverse=True)
            current_best = population[0]
            
            if not best_genome or current_best.fitness > best_genome.fitness:
                best_genome = copy.deepcopy(current_best)
            
            fitness_history.append(current_best.fitness)
            
            if self.verbose:
                print(f"[NAS] Best fitness: {current_best.fitness:.4f}, accuracy: {current_best.accuracy:.4f}")
            
            # 生成新一代
            new_population = []
            
            # 精英保留
            for i in range(elitism):
                new_population.append(copy.deepcopy(population[i]))
            
            # 交叉和变异
            while len(new_population) < population_size:
                # 选择父代
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # 交叉
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, search_space)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # 变异
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, search_space)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, search_space)
                
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            population = new_population
        
        total_time = (datetime.utcnow() - started_at).total_seconds()
        
        # 收集所有评估过的架构
        all_genomes = list(set(population + [best_genome]))
        all_genomes.sort(key=lambda g: g.fitness, reverse=True)
        
        result = NASResult(
            search_id=search_id,
            search_space=search_space,
            best_genome=best_genome,
            all_genomes=all_genomes[:population_size],  # 保存top结果
            generations=generations,
            total_time=total_time,
            fitness_history=fitness_history
        )
        
        self.searches[search_id] = result
        
        # 保存结果
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / f"{search_id}.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        
        if self.verbose:
            print(f"[NAS] Done! Best accuracy: {best_genome.accuracy:.4f}")
            print(f"[NAS] Total time: {total_time:.1f}s")
        
        return result
    
    def _random_genome(self, space: NASSearchSpace) -> ArchitectureGenome:
        """生成随机架构基因组"""
        genome_id = str(uuid4())
        num_layers = random.randint(2, space.max_layers)
        
        layers = []
        current_shape = space.input_shape
        
        for i in range(num_layers):
            layer_type = random.choice(space.available_layers)
            
            if layer_type == LayerType.DENSE:
                units = random.choice(space.layer_configs.get("dense", {}).get("units", [64]))
                config = {"units": units}
            elif layer_type == LayerType.CONV2D:
                filters = random.choice(space.layer_configs.get("conv2d", {}).get("filters", [32]))
                kernel_size = random.choice(space.layer_configs.get("conv2d", {}).get("kernel_size", [(3, 3)]))
                config = {"filters": filters, "kernel_size": kernel_size}
            elif layer_type == LayerType.DROPOUT:
                rate = random.choice(space.layer_configs.get("dropout", {}).get("rate", [0.3]))
                config = {"rate": rate}
            elif layer_type == LayerType.BATCH_NORM:
                config = {}
            elif layer_type == LayerType.POOLING:
                pool_size = random.choice(space.layer_configs.get("pooling", {}).get("pool_size", [(2, 2)]))
                config = {"pool_size": pool_size}
            else:
                config = {}
            
            layer = LayerSpec(
                layer_id=f"layer_{i}",
                type=layer_type,
                config=config,
                input_shape=current_shape
            )
            
            # 模拟计算输出形状
            current_shape = self._compute_output_shape(layer, current_shape)
            layer.output_shape = current_shape
            
            layers.append(layer)
        
        return ArchitectureGenome(
            genome_id=genome_id,
            layers=layers
        )
    
    def _compute_output_shape(
        self,
        layer: LayerSpec,
        input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """计算层输出形状"""
        h, w = input_shape[0], input_shape[1]
        
        if layer.type == LayerType.CONV2D:
            filters = layer.config.get("filters", 32)
            kernel = layer.config.get("kernel_size", (3, 3))
            stride = layer.config.get("strides", (1, 1))
            
            h = (h - kernel[0]) // stride[0] + 1 if stride[0] else h
            w = (w - kernel[1]) // stride[1] + 1 if stride[1] else w
            
            return (h, w, filters)
        
        elif layer.type == LayerType.POOLING:
            pool_size = layer.config.get("pool_size", (2, 2))
            stride = layer.config.get("strides", pool_size)
            
            h = h // pool_size[0]
            w = w // pool_size[1]
            
            return (h, w, input_shape[-1])
        
        elif layer.type == LayerType.DENSE:
            units = layer.config.get("units", 64)
            return (units,)
        
        elif layer.type in [LayerType.DROPOUT, LayerType.BATCH_NORM]:
            return input_shape
        
        return input_shape
    
    def _tournament_select(
        self,
        population: List[ArchitectureGenome],
        tournament_size: int = 3
    ) -> ArchitectureGenome:
        """锦标赛选择"""
        selected = random.sample(population, min(tournament_size, len(population)))
        return max(selected, key=lambda g: g.fitness)
    
    def _crossover(
        self,
        parent1: ArchitectureGenome,
        parent2: ArchitectureGenome,
        space: NASSearchSpace
    ) -> Tuple[ArchitectureGenome, ArchitectureGenome]:
        """单点交叉"""
        # 简化: 交换后半部分层
        split1 = len(parent1.layers) // 2
        split2 = len(parent2.layers) // 2
        
        child1_layers = parent1.layers[:split1] + parent2.layers[split2:]
        child2_layers = parent2.layers[:split2] + parent1.layers[split1:]
        
        child1 = ArchitectureGenome(
            genome_id=str(uuid4()),
            layers=child1_layers
        )
        child2 = ArchitectureGenome(
            genome_id=str(uuid4()),
            layers=child2_layers
        )
        
        return child1, child2
    
    def _mutate(
        self,
        genome: ArchitectureGenome,
        space: NASSearchSpace
    ) -> ArchitectureGenome:
        """变异操作"""
        mutated = copy.deepcopy(genome)
        
        mutation_type = random.choice(["layer_add", "layer_remove", "layer_modify"])
        
        if mutation_type == "layer_add" and len(genome.layers) < space.max_layers:
            # 添加新层
            new_layer = self._random_layer(len(genome.layers), space)
            insert_pos = random.randint(0, len(genome.layers))
            mutated.layers.insert(insert_pos, new_layer)
        
        elif mutation_type == "layer_remove" and len(genome.layers) > 2:
            # 移除层
            remove_pos = random.randint(0, len(genome.layers) - 1)
            del mutated.layers[remove_pos]
        
        else:
            # 修改层参数
            if mutated.layers:
                modify_pos = random.randint(0, len(mutated.layers) - 1)
                layer = mutated.layers[modify_pos]
                
                if layer.type == LayerType.DENSE:
                    layer.config["units"] = random.choice(
                        space.layer_configs.get("dense", {}).get("units", [64])
                    )
                elif layer.type == LayerType.CONV2D:
                    layer.config["filters"] = random.choice(
                        space.layer_configs.get("conv2d", {}).get("filters", [32])
                    )
        
        return mutated
    
    def _random_layer(self, layer_id: int, space: NASSearchSpace) -> LayerSpec:
        """生成随机层"""
        layer_type = random.choice(space.available_layers)
        
        if layer_type == LayerType.DENSE:
            units = random.choice(space.layer_configs.get("dense", {}).get("units", [64]))
            config = {"units": units}
        elif layer_type == LayerType.CONV2D:
            filters = random.choice(space.layer_configs.get("conv2d", {}).get("filters", [32]))
            config = {"filters": filters}
        elif layer_type == LayerType.DROPOUT:
            rate = random.choice(space.layer_configs.get("dropout", {}).get("rate", [0.3]))
            config = {"rate": rate}
        else:
            config = {}
        
        return LayerSpec(layer_id=f"layer_{layer_id}", type=layer_type, config=config)
    
    def export_keras(
        self,
        genome: ArchitectureGenome,
        input_shape: Tuple[int, ...],
        num_classes: int = 10
    ) -> str:
        """
        将基因组导出为Keras模型代码
        
        Args:
            genome: 架构基因组
            input_shape: 输入形状
            num_classes: 输出类别数
            
        Returns:
            str: Keras模型代码
        """
        code_lines = [
            "import tensorflow as tf",
            "from tensorflow import keras",
            "from tensorflow.keras import layers",
            "",
            "def build_model(input_shape, num_classes):",
            "    inputs = keras.Input(shape=input_shape)",
            "    x = inputs",
            ""
        ]
        
        has_flatten = False
        
        for layer in genome.layers:
            if layer.type == LayerType.CONV2D:
                filters = layer.config.get("filters", 32)
                kernel = layer.config.get("kernel_size", (3, 3))
                act = layer.config.get("activation", "relu")
                code_lines.append(f"    x = layers.Conv2D({filters}, {kernel}, activation='{act}')(x)")
            
            elif layer.type == LayerType.POOLING:
                pool = layer.config.get("pool_size", (2, 2))
                code_lines.append(f"    x = layers.MaxPooling2D({pool})(x)")
            
            elif layer.type == LayerType.DROPOUT:
                rate = layer.config.get("rate", 0.3)
                code_lines.append(f"    x = layers.Dropout({rate})(x)")
            
            elif layer.type == LayerType.BATCH_NORM:
                code_lines.append("    x = layers.BatchNormalization()(x)")
            
            elif layer.type == LayerType.DENSE:
                units = layer.config.get("units", 64)
                code_lines.append(f"    x = layers.Dense({units}, activation='relu')(x)")
                
                # 检查是否需要flatten
                if not has_flatten and len(layer.input_shape) > 2 if layer.input_shape else True:
                    if not has_flatten:
                        code_lines.insert(-1, "    x = layers.Flatten()(x)")
                        has_flatten = True
        
        # 添加输出层
        if not has_flatten:
            code_lines.append("    x = layers.Flatten()(x)")
        
        code_lines.append(f"    outputs = layers.Dense(num_classes, activation='softmax')(x)")
        code_lines.append("")
        code_lines.append("    return keras.Model(inputs, outputs)")
        
        return "\n".join(code_lines)
    
    def get_search(self, search_id: str) -> Optional[NASResult]:
        """获取搜索任务结果"""
        return self.searches.get(search_id)
    
    def list_searches(self, limit: int = 20) -> List[Dict]:
        """列出所有搜索任务"""
        result = []
        for search_id, res in self.searches.items():
            result.append({
                "search_id": search_id,
                "task_type": res.search_space.task_type.value,
                "best_accuracy": res.best_genome.accuracy if res.best_genome else None,
                "generations": res.generations,
                "total_time": res.total_time,
                "created_at": res.created_at.isoformat()
            })
        
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result[:limit]


# 默认NAS搜索器实例
default_nas = NeuralArchitectureSearcher(verbose=True)
