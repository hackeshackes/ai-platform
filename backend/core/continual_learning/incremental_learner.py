"""
增量学习器 (Incremental Learner)

支持顺序学习、任务序列管理、在线学习和样本高效学习。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import copy
import numpy as np
from datetime import datetime


@dataclass
class Task:
    """任务定义"""
    task_id: str
    name: str
    dataset: Dataset
    num_samples: int
    difficulty: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")


@dataclass
class LearningResult:
    """学习结果"""
    task_id: str
    success: bool
    loss: float
    accuracy: float
    time_seconds: float
    memory_used_mb: float
    samples_processed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)


class IncrementalLearner:
    """
    增量学习器
    
    支持：
    - 顺序学习：依次学习新任务
    - 任务序列：管理多个任务
    - 在线学习：实时更新模型
    - 样本高效学习：利用少量样本学习
    
    Attributes:
        model: 学习模型
        memory_replay: 记忆回放实例
        knowledge_consolidation: 知识巩固实例
        device: 计算设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_replay: Optional['MemoryReplay'] = None,
        knowledge_consolidation: Optional['KnowledgeConsolidation'] = None,
        device: Optional[torch.device] = None,
        config: Optional['ContinualLearningConfig'] = None
    ):
        self.model = model
        self.memory_replay = memory_replay
        self.knowledge_consolidation = knowledge_consolidation
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or ContinualLearningConfig()
        
        # 任务管理
        self.tasks: Dict[str, Task] = {}
        self.task_order: List[str] = []
        self.current_task_id: Optional[str] = None
        
        # 模型状态
        self.model.to(self.device)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # 训练历史
        self.training_history: List[LearningResult] = []
        self.learned_parameters: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # 性能指标
        self.forgetting_ratio: float = 0.0
        self.average_accuracy: float = 0.0
        
    def set_optimizer(self, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """设置优化器和学习率调度器"""
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def add_task(self, task_data: Union[Dataset, Dict], task_id: str, name: Optional[str] = None, 
                 difficulty: float = 1.0, metadata: Optional[Dict] = None) -> Task:
        """
        添加新任务
        
        Args:
            task_data: 任务数据 (Dataset或字典)
            task_id: 任务唯一标识
            name: 任务名称
            difficulty: 任务难度 (0.1-2.0)
            metadata: 额外元数据
            
        Returns:
            Task对象
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
            
        if isinstance(task_data, dict):
            # 从字典创建数据集
            dataset = self._create_dataset_from_dict(task_data)
        else:
            dataset = task_data
            
        num_samples = len(dataset)
        
        task = Task(
            task_id=task_id,
            name=name or f"Task_{task_id}",
            dataset=dataset,
            num_samples=num_samples,
            difficulty=difficulty,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.task_order.append(task_id)
        
        # 如果使用知识巩固，初始化任务参数
        if self.knowledge_consolidation is not None:
            self.knowledge_consolidation.register_task(task_id)
            
        return task
    
    def _create_dataset_from_dict(self, data: Dict) -> Dataset:
        """从字典创建数据集（简单实现）"""
        from torch.utils.data import TensorDataset
        
        features = torch.tensor(data.get('features', []), dtype=torch.float32)
        labels = torch.tensor(data.get('labels', []), dtype=torch.long)
        
        if len(features) == 0:
            raise ValueError("No features provided")
            
        return TensorDataset(features, labels)
    
    def train(
        self,
        task_id: str,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_replay: bool = True,
        use_consolidation: bool = True,
        val_dataset: Optional[Dataset] = None,
        callback: Optional[Callable] = None
    ) -> LearningResult:
        """
        训练单个任务
        
        Args:
            task_id: 任务ID
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            use_replay: 是否使用记忆回放
            use_consolidation: 是否使用知识巩固
            val_dataset: 验证数据集
            callback: 回调函数
            
        Returns:
            LearningResult: 训练结果
        """
        import time
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        self.current_task_id = task_id
        
        start_time = time.time()
        samples_processed = 0
        
        # 设置优化器
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                
        # 准备数据加载器
        dataloader = DataLoader(
            task.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 保存当前模型状态（用于遗忘检测）
        if len(self.learned_parameters) > 0:
            prev_state = copy.deepcopy(self.model.state_dict())
            
        # 训练循环
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                samples_processed += len(data)
                
                self.optimizer.zero_grad()
                
                # 获取Replay样本
                replay_loss = torch.tensor(0.0, device=self.device)
                if use_replay and self.memory_replay is not None:
                    replay_data, replay_target = self.memory_replay.sample(batch_size)
                    if len(replay_data) > 0:
                        replay_data = replay_data.to(self.device)
                        replay_target = replay_target.to(self.device)
                        
                        # 计算Replay损失
                        replay_output = self.model(replay_data)
                        replay_loss = nn.functional.cross_entropy(replay_output, replay_target)
                
                # 计算当前任务损失
                output = self.model(data)
                task_loss = nn.functional.cross_entropy(output, target)
                
                # 总损失
                total_loss_val = task_loss + replay_loss * self.config.replay_weight
                
                # 添加知识巩固损失
                consolidation_loss = torch.tensor(0.0, device=self.device)
                if use_consolidation and self.knowledge_consolidation is not None:
                    consolidation_loss = self.knowledge_consolidation.compute_consolidation_loss(
                        self.model, task_id
                    )
                
                final_loss = total_loss_val + consolidation_loss * self.config.consolidation_weight
                
                final_loss.backward()
                self.optimizer.step()
                
                epoch_loss += final_loss.item()
                num_batches += 1
                
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
                
            total_loss += epoch_loss / num_batches if num_batches > 0 else 0
            
            # 回调函数
            if callback:
                callback(epoch, epoch_loss / num_batches if num_batches > 0 else 0)
                
        # 计算最终指标
        avg_loss = total_loss / epochs if epochs > 0 else 0
        accuracy = self._evaluate(task.dataset) if val_dataset is None else self._evaluate(val_dataset)
        
        # 存储到记忆回放
        if self.memory_replay is not None:
            self.memory_replay.store(task.dataset, task_id)
            
        # 保存当前任务参数（用于EWC等方法）
        self.learned_parameters[task_id] = copy.deepcopy(self.model.state_dict())
        
        # 计算遗忘率
        if len(self.learned_parameters) > 1:
            self.forgetting_ratio = self._calculate_forgetting()
            
        time_elapsed = time.time() - start_time
        
        # 创建结果
        result = LearningResult(
            task_id=task_id,
            success=True,
            loss=avg_loss,
            accuracy=accuracy,
            time_seconds=time_elapsed,
            memory_used_mb=self._get_memory_usage(),
            samples_processed=samples_processed,
            details={
                "epochs": epochs,
                "batch_size": batch_size,
                "use_replay": use_replay,
                "use_consolidation": use_consolidation,
                "forgetting_ratio": self.forgetting_ratio
            }
        )
        
        self.training_history.append(result)
        self.average_accuracy = np.mean([r.accuracy for r in self.training_history])
        
        return result
    
    def train_task_sequence(
        self,
        task_ids: Optional[List[str]] = None,
        epochs_per_task: int = 10,
        **kwargs
    ) -> List[LearningResult]:
        """
        按顺序训练多个任务
        
        Args:
            task_ids: 任务ID列表（None表示使用添加顺序）
            epochs_per_task: 每个任务的训练轮数
            **kwargs: train()的其他参数
            
        Returns:
            学习结果列表
        """
        if task_ids is None:
            task_ids = self.task_order
            
        results = []
        
        for task_id in task_ids:
            print(f"Training task: {task_id}")
            result = self.train(task_id, epochs=epochs_per_task, **kwargs)
            results.append(result)
            
            # 打印进度
            print(f"  - Loss: {result.loss:.4f}, Accuracy: {result.accuracy:.4f}")
            
        return results
    
    def _evaluate(self, dataset: Dataset) -> float:
        """评估模型"""
        self.model.eval()
        
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        self.model.train()
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_forgetting(self) -> float:
        """计算遗忘率"""
        if len(self.training_history) < 2:
            return 0.0
            
        forgetting = 0.0
        num_comparisons = 0
        
        for i in range(len(self.training_history) - 1):
            past_task_id = self.training_history[i].task_id
            current_task_id = self.training_history[-1].task_id
            
            if past_task_id != current_task_id:
                # 简单遗忘计算（实际应用中需要更复杂的评估）
                past_acc = self.training_history[i].accuracy
                current_acc = self._evaluate(self.tasks[past_task_id].dataset)
                
                forgetting += max(0, past_acc - current_acc)
                num_comparisons += 1
                
        return forgetting / num_comparisons if num_comparisons > 0 else 0.0
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_model(self) -> nn.Module:
        """获取当前模型"""
        return self.model
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tasks': {k: {
                'task_id': v.task_id,
                'name': v.name,
                'num_samples': v.num_samples,
                'difficulty': v.difficulty
            } for k, v in self.tasks.items()},
            'task_order': self.task_order,
            'training_history': [
                {
                    'task_id': r.task_id,
                    'success': r.success,
                    'loss': r.loss,
                    'accuracy': r.accuracy,
                    'time_seconds': r.time_seconds
                } for r in self.training_history
            ],
            'learned_parameters': {
                k: {pk: pv.cpu() for pk, pv in v.items()} 
                for k, v in self.learned_parameters.items()
            }
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.task_order = checkpoint['task_order']
        self.training_history = [
            LearningResult(**r) for r in checkpoint['training_history']
        ]
        self.learned_parameters = checkpoint['learned_parameters']
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.training_history),
            "average_accuracy": self.average_accuracy,
            "forgetting_ratio": self.forgetting_ratio,
            "total_samples_processed": sum(r.samples_processed for r in self.training_history),
            "total_training_time": sum(r.time_seconds for r in self.training_history),
            "current_task": self.current_task_id,
            "memory_usage_mb": self._get_memory_usage()
        }
    
    def online_update(self, data_batch: Any, target_batch: Any):
        """
        在线学习更新
        
        Args:
            data_batch: 数据批次
            target_batch: 目标批次
        """
        self.model.train()
        
        data = data_batch.to(self.device)
        target = target_batch.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        
    def sample_efficient_learning(self, task_id: str, sample_ratio: float = 0.1, **kwargs):
        """
        样本高效学习
        
        Args:
            task_id: 任务ID
            sample_ratio: 采样比例
            **kwargs: train()的其他参数
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        
        # 采样子集
        num_samples = max(1, int(len(task.dataset) * sample_ratio))
        
        # 简单随机采样
        indices = np.random.choice(len(task.dataset), num_samples, replace=False)
        
        from torch.utils.data import Subset
        subset_dataset = Subset(task.dataset, indices)
        
        # 使用子集训练
        original_dataset = task.dataset
        task.dataset = subset_dataset
        
        try:
            result = self.train(task_id, **kwargs)
        finally:
            task.dataset = original_dataset
            
        return result
