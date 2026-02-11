"""
课程学习系统 (Curriculum Learning System)

提供多种课程学习策略：
- Difficulty Scheduling: 难度调度
- Progressive Learning: 渐进式学习
- Active Learning: 主动学习
- Adaptive Curriculum: 自适应课程
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from collections import defaultdict
import copy


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    name: str
    dataset: Dataset
    difficulty: float  # 0.0 - 2.0
    estimated_samples: int
    priority: int = 0
    completed: bool = False
    accuracy: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SampleInfo:
    """样本信息"""
    index: int
    difficulty: float
    loss: float = float('inf')
    uncertainty: float = 0.0
    task_id: str = ""
    used: bool = False


class CurriculumStrategy(ABC):
    """课程策略基类"""
    
    @abstractmethod
    def get_order(self, tasks: List[TaskInfo]) -> List[str]:
        """获取任务顺序"""
        pass
    
    @abstractmethod
    def update_progress(self, task_id: str, metrics: Dict[str, float]):
        """更新进度"""
        pass


class DifficultyScheduler(CurriculumStrategy):
    """
    难度调度器
    
    根据任务难度排序，从易到难学习
    """
    
    def __init__(
        self,
        sort_ascending: bool = True,
        warmup_tasks: int = 1,
        difficulty_scaling: float = 1.0
    ):
        self.sort_ascending = sort_ascending
        self.warmup_tasks = warmup_tasks
        self.difficulty_scaling = difficulty_scaling
        self.task_difficulties: Dict[str, float] = {}
        self.completion_order: List[str] = []
        
    def get_order(self, tasks: List[TaskInfo]) -> List[str]:
        """获取按难度排序的任务顺序"""
        # 按难度排序
        sorted_tasks = sorted(tasks, key=lambda t: t.difficulty, reverse=not self.sort_ascending)
        
        order = [t.task_id for t in sorted_tasks]
        
        return order
    
    def update_progress(self, task_id: str, metrics: Dict[str, float]):
        """更新任务进度"""
        if task_id not in self.completion_order:
            self.completion_order.append(task_id)
            
        # 更新难度估计
        if 'accuracy' in metrics:
            # 如果准确率高，可能意味着任务较简单
            base_difficulty = 1.0 - metrics['accuracy']
            self.task_difficulties[task_id] = base_difficulty * self.difficulty_scaling
            
    def add_task_difficulty(self, task_id: str, difficulty: float):
        """添加任务难度"""
        self.task_difficulties[task_id] = difficulty
        
    def get_current_difficulty(self) -> float:
        """获取当前难度级别"""
        if not self.completion_order:
            return 0.5
            
        completed = len(self.completion_order)
        return min(2.0, 0.5 + completed * 0.1 * self.difficulty_scaling)


class ProgressiveDifficultyScheduler(DifficultyScheduler):
    """
    渐进式难度调度
    
    难度随时间逐渐增加
    """
    
    def __init__(
        self,
        base_difficulty: float = 0.3,
        max_difficulty: float = 2.0,
        difficulty_increment: float = 0.1,
        patience: int = 2
    ):
        super().__init__(sort_ascending=True)
        self.base_difficulty = base_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.patience = patience
        
        self.current_difficulty = base_difficulty
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.successful_tasks: List[str] = []
        
    def get_order(self, tasks: List[TaskInfo]) -> List[str]:
        """获取渐进式任务顺序"""
        # 过滤可用任务
        available_tasks = [
            t for t in tasks 
            if t.difficulty <= self.current_difficulty or t.task_id in self.successful_tasks
        ]
        
        if not available_tasks:
            # 如果没有可用的，提高难度
            available_tasks = tasks
            
        # 按难度排序
        sorted_tasks = sorted(available_tasks, key=lambda t: t.difficulty)
        
        return [t.task_id for t in sorted_tasks]
    
    def update_progress(self, task_id: str, metrics: Dict[str, float]):
        """更新进度并调整难度"""
        if 'accuracy' in metrics:
            if metrics['accuracy'] >= 0.8:
                # 任务成功，增加难度
                if task_id not in self.successful_tasks:
                    self.successful_tasks.append(task_id)
                    self.current_difficulty = min(
                        self.max_difficulty,
                        self.current_difficulty + self.difficulty_increment
                    )
            else:
                # 任务失败，降低难度或保持
                self.failed_attempts[task_id] += 1
                
                if self.failed_attempts[task_id] >= self.patience:
                    self.current_difficulty = max(
                        self.base_difficulty,
                        self.current_difficulty - self.difficulty_increment
                    )
                    self.failed_attempts[task_id] = 0


class ActiveLearning(CurriculumStrategy):
    """
    主动学习策略
    
    选择最有价值的样本进行学习
    """
    
    def __init__(
        self,
        model: nn.Module,
        selection_strategy: str = "uncertainty",  # uncertainty, diversity, expected_gradient
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.selection_strategy = selection_strategy
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        self.sample_scores: Dict[str, List[SampleInfo]] = {}
        self.selected_samples: Dict[str, List[int]] = defaultdict(list)
        
    def compute_sample_scores(
        self,
        task_id: str,
        dataset: Dataset,
        unlabeled_indices: Optional[List[int]] = None
    ) -> List[SampleInfo]:
        """计算样本分数"""
        self.model.eval()
        
        if unlabeled_indices is None:
            unlabeled_indices = list(range(len(dataset)))
            
        sample_infos = []
        
        dataloader = DataLoader(
            Subset(dataset, unlabeled_indices),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                    
                data = data.to(self.device)
                
                # 获取模型输出
                output = self.model(data)
                
                if self.selection_strategy == "uncertainty":
                    # 使用熵作为不确定性度量
                    probs = F.softmax(output, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    
                    for i, idx in enumerate(unlabeled_indices[:len(entropy)]):
                        sample_info = SampleInfo(
                            index=idx,
                            difficulty=entropy[i].item(),
                            uncertainty=entropy[i].item(),
                            task_id=task_id
                        )
                        sample_infos.append(sample_info)
                        
                elif self.selection_strategy == "margin":
                    # 使用边缘分数
                    probs = F.softmax(output, dim=1)
                    top2, _ = probs.topk(2, dim=1)
                    margin = top2[:, 0] - top2[:, 1]
                    
                    for i, idx in enumerate(unlabeled_indices[:len(margin)]):
                        sample_info = SampleInfo(
                            index=idx,
                            difficulty=margin[i].item(),
                            uncertainty=1 - margin[i].item(),
                            task_id=task_id
                        )
                        sample_infos.append(sample_info)
                        
                elif self.selection_strategy == "confidence":
                    # 使用置信度
                    probs = F.softmax(output, dim=1)
                    confidence, _ = probs.max(dim=1)
                    
                    for i, idx in enumerate(unlabeled_indices[:len(confidence)]):
                        sample_info = SampleInfo(
                            index=idx,
                            difficulty=1 - confidence[i].item(),
                            uncertainty=1 - confidence[i].item(),
                            task_id=task_id
                        )
                        sample_infos.append(sample_info)
                        
        return sample_infos
    
    def select_samples(
        self,
        task_id: str,
        sample_infos: List[SampleInfo],
        num_samples: int
    ) -> List[int]:
        """选择最有价值的样本"""
        if not sample_infos:
            return []
            
        # 按不确定性排序
        if self.selection_strategy in ["uncertainty", "confidence"]:
            # 高不确定性优先
            sorted_infos = sorted(
                sample_infos, 
                key=lambda s: s.difficulty, 
                reverse=True
            )
        else:
            sorted_infos = sorted(sample_infos, key=lambda s: s.difficulty)
            
        selected = [s.index for s in sorted_infos[:num_samples]]
        
        # 标记已选样本
        for s in sorted_infos[:num_samples]:
            s.used = True
            
        self.selected_samples[task_id].extend(selected)
        
        return selected
    
    def get_order(self, tasks: List[TaskInfo]) -> List[str]:
        """主动学习不改变任务顺序"""
        return [t.task_id for t in tasks]
    
    def update_progress(self, task_id: str, metrics: Dict[str, float]):
        """更新进度"""
        pass
    
    def get_labeled_samples(self, task_id: str) -> List[int]:
        """获取已标记的样本"""
        return self.selected_samples[task_id]
    
    def get_unlabeled_samples(self, task_id: str, total: int) -> List[int]:
        """获取未标记的样本"""
        labeled = set(self.selected_samples[task_id])
        return [i for i in range(total) if i not in labeled]


class AdaptiveCurriculum(CurriculumStrategy):
    """
    自适应课程
    
    根据学习进度动态调整课程
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 5,
        adaptation_rate: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.device = device or torch.device('cpu')
        
        # 任务性能历史
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # 当前难度权重
        self.difficulty_weights: Dict[str, float] = {}
        
        # 样本难度缓存
        self.sample_difficulties: Dict[str, np.ndarray] = {}
        
    def get_order(self, tasks: List[TaskInfo]) -> List[str]:
        """获取自适应排序的任务"""
        if not tasks:
            return []
            
        # 计算每个任务的加权分数
        task_scores = []
        
        for task in tasks:
            base_difficulty = task.difficulty
            
            # 根据历史性能调整
            if task.task_id in self.performance_history:
                history = self.performance_history[task.task_id]
                if len(history) >= self.window_size:
                    recent_avg = np.mean(history[-self.window_size:])
                    # 性能好，增加权重；性能差，减少权重
                    adjustment = 1.0 - self.adaptation_rate * (1.0 - recent_avg)
                    base_difficulty *= adjustment
                    
            # 根据完成状态调整
            if task.completed:
                base_difficulty *= 0.5
                
            # 根据优先级调整
            priority_boost = task.priority * 0.1
            base_difficulty += priority_boost
            
            self.difficulty_weights[task.task_id] = base_difficulty
            
            task_scores.append((task.task_id, base_difficulty))
            
        # 按调整后的难度排序
        sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in sorted_tasks]
    
    def update_progress(self, task_id: str, metrics: Dict[str, float]):
        """更新性能历史"""
        if 'accuracy' in metrics:
            self.performance_history[task_id].append(metrics['accuracy'])
            
            # 保持历史窗口大小
            if len(self.performance_history[task_id]) > self.window_size * 2:
                self.performance_history[task_id] = \
                    self.performance_history[task_id][-self.window_size:]
                    
    def update_sample_difficulties(
        self,
        task_id: str,
        losses: List[float],
        indices: Optional[List[int]] = None
    ):
        """更新样本难度"""
        if indices is None:
            indices = list(range(len(losses)))
            
        self.sample_difficulties[task_id] = np.array([
            losses[i] for i in range(len(losses))
        ])
    
    def get_next_samples(
        self,
        task_id: str,
        num_samples: int,
        strategy: str = "easy_first"
    ) -> List[int]:
        """获取下一批样本"""
        if task_id not in self.sample_difficulties:
            # 随机选择
            return random.sample(range(len(self.sample_difficulties.get(task_id, []))), 
                                min(num_samples, len(self.sample_difficulties.get(task_id, []))))
            
        difficulties = self.sample_difficulties[task_id]
        
        if strategy == "easy_first":
            # 简单样本优先
            sorted_indices = np.argsort(difficulties)
        elif strategy == "hard_first":
            # 困难样本优先
            sorted_indices = np.argsort(difficulties)[::-1]
        elif strategy == "balanced":
            # 平衡选择
            sorted_indices = np.random.permutation(len(difficulties))
        else:
            sorted_indices = np.arange(len(difficulties))
            
        return sorted_indices[:num_samples].tolist()


class CurriculumLearning:
    """
    课程学习主类
    
    整合多种课程学习策略
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        strategy: str = "difficulty",
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.strategy = strategy
        self.config = config or {}
        self.device = device or torch.device('cpu')
        
        if model is not None:
            model.to(self.device)
            
        # 任务管理
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_order: List[str] = []
        
        # 策略实例
        self._init_strategy()
        
        # 统计
        self.statistics: Dict[str, Any] = {
            "total_samples_processed": 0,
            "curriculum_changes": 0
        }
        
    def _init_strategy(self):
        """初始化策略"""
        if self.strategy == "difficulty":
            self.curriculum_strategy = DifficultyScheduler(
                sort_ascending=self.config.get('sort_ascending', True),
                warmup_tasks=self.config.get('warmup_tasks', 1)
            )
        elif self.strategy == "progressive":
            self.curriculum_strategy = ProgressiveDifficultyScheduler(
                base_difficulty=self.config.get('base_difficulty', 0.3),
                max_difficulty=self.config.get('max_difficulty', 2.0)
            )
        elif self.strategy == "active":
            if self.model is None:
                raise ValueError("Model required for active learning")
            self.curriculum_strategy = ActiveLearning(
                model=self.model,
                selection_strategy=self.config.get('selection_strategy', 'uncertainty'),
                batch_size=self.config.get('batch_size', 32),
                device=self.device
            )
        elif self.strategy == "adaptive":
            if self.model is None:
                raise ValueError("Model required for adaptive curriculum")
            self.curriculum_strategy = AdaptiveCurriculum(
                model=self.model,
                window_size=self.config.get('window_size', 5),
                adaptation_rate=self.config.get('adaptation_rate', 0.1),
                device=self.device
            )
        else:
            self.curriculum_strategy = DifficultyScheduler()
            
    def add_task(
        self,
        task_id: str,
        dataset: Dataset,
        difficulty: float = 1.0,
        name: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> TaskInfo:
        """添加任务"""
        task = TaskInfo(
            task_id=task_id,
            name=name or f"Task_{task_id}",
            dataset=dataset,
            difficulty=difficulty,
            estimated_samples=len(dataset),
            priority=priority,
            metadata=kwargs
        )
        
        self.tasks[task_id] = task
        self.task_order.append(task_id)
        
        return task
    
    def set_difficulty(self, tasks: List[TaskInfo], sorted: bool = True) -> List[TaskInfo]:
        """设置任务难度并排序"""
        if sorted:
            # 按难度排序
            sorted_tasks = sorted(tasks, key=lambda t: t.difficulty)
            self.task_order = [t.task_id for t in sorted_tasks]
        else:
            self.task_order = [t.task_id for t in tasks]
            
        return sorted_tasks if sorted else tasks
    
    def get_ordered_tasks(self) -> List[TaskInfo]:
        """获取排序后的任务列表"""
        order = self.curriculum_strategy.get_order(list(self.tasks.values()))
        
        # 如果策略返回了顺序，使用它
        if order:
            return [self.tasks[task_id] for task_id in order if task_id in self.tasks]
        
        # 否则使用默认顺序
        return [self.tasks[task_id] for task_id in self.task_order]
    
    def get_next_task(self) -> Optional[TaskInfo]:
        """获取下一个任务"""
        ordered_tasks = self.get_ordered_tasks()
        
        for task in ordered_tasks:
            if not task.completed:
                return task
                
        return None
    
    def complete_task(self, task_id: str, metrics: Dict[str, float]):
        """标记任务完成"""
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            self.tasks[task_id].accuracy = metrics.get('accuracy', 0.0)
            
            # 更新策略
            self.curriculum_strategy.update_progress(task_id, metrics)
            
            self.statistics["curriculum_changes"] += 1
            
    def select_samples(
        self,
        task_id: str,
        num_samples: int = 32,
        strategy: Optional[str] = None
    ) -> List[int]:
        """选择样本"""
        if task_id not in self.tasks:
            return []
            
        task = self.tasks[task_id]
        
        if isinstance(self.curriculum_strategy, ActiveLearning):
            # 计算分数并选择
            sample_infos = self.curriculum_strategy.compute_sample_scores(
                task_id,
                task.dataset
            )
            return self.curriculum_strategy.select_samples(
                task_id, 
                sample_infos, 
                num_samples
            )
        elif isinstance(self.curriculum_strategy, AdaptiveCurriculum):
            return self.curriculum_strategy.get_next_samples(
                task_id,
                num_samples,
                strategy or "easy_first"
            )
        else:
            # 随机选择
            return random.sample(range(task.estimated_samples), min(num_samples, task.estimated_samples))
    
    def get_curriculum(self, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取当前课程"""
        ordered_tasks = self.get_ordered_tasks()
        
        if max_tasks:
            ordered_tasks = ordered_tasks[:max_tasks]
            
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "estimated_samples": t.estimated_samples,
                "priority": t.priority,
                "completed": t.completed
            }
            for t in ordered_tasks
        ]
    
    def update_sample_difficulties(
        self,
        task_id: str,
        losses: List[float],
        indices: Optional[List[int]] = None
    ):
        """更新样本难度"""
        if isinstance(self.curriculum_strategy, AdaptiveCurriculum):
            self.curriculum_strategy.update_sample_difficulties(
                task_id,
                losses,
                indices
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.statistics.copy()
        stats.update({
            "strategy": self.strategy,
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.completed),
            "current_difficulty": self.curriculum_strategy.get_current_difficulty() 
                if hasattr(self.curriculum_strategy, 'get_current_difficulty') else None
        })
        
        return stats
    
    def reset(self):
        """重置课程"""
        self.tasks = {}
        self.task_order = []
        self._init_strategy()
        self.statistics = {
            "total_samples_processed": 0,
            "curriculum_changes": 0
        }


# 辅助函数
def estimate_difficulty(
    model: nn.Module,
    dataset: Dataset,
    device: Optional[torch.device] = None,
    sample_size: int = 100
) -> float:
    """
    估计数据集难度
    
    通过在小样本上测试模型性能来估计
    """
    model.eval()
    device = device or torch.device('cpu')
    
    # 采样
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=min(32, len(subset)))
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                data, target = batch[0], batch[1]
            else:
                data, target = batch, None
                
            data = data.to(device)
            
            output = model(data)
            
            if target is not None:
                loss = F.cross_entropy(output, target.to(device))
                total_loss += loss.item() * len(data)
                total_samples += len(data)
                
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    # 损失越大，难度越高
    difficulty = min(2.0, max(0.1, avg_loss))
    
    return difficulty


def auto_generate_curriculum(
    tasks: List[Tuple[str, Dataset, float]],
    model: nn.Module,
    device: Optional[torch.device] = None
) -> List[TaskInfo]:
    """
    自动生成课程
    
    根据估计的难度自动排序任务
    """
    device = device or torch.device('cpu')
    model.to(device)
    
    task_infos = []
    
    for task_id, dataset, base_difficulty in tasks:
        # 估计实际难度
        estimated_difficulty = estimate_difficulty(model, dataset, device)
        
        # 结合基础难度和估计难度
        final_difficulty = (base_difficulty + estimated_difficulty) / 2
        
        task_info = TaskInfo(
            task_id=task_id,
            name=f"Task_{task_id}",
            dataset=dataset,
            difficulty=final_difficulty,
            estimated_samples=len(dataset)
        )
        
        task_infos.append(task_info)
        
    # 按难度排序
    task_infos.sort(key=lambda t: t.difficulty)
    
    return task_infos


# 导入必要的模块
import torch.nn.functional as F
from torch.utils.data import Subset
