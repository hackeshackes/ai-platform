"""
任务生成器 - Task Generator
实现任务分布采样、难度梯度、任务相似度计算
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


@dataclass
class Task:
    """任务数据类"""
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    task_id: str
    difficulty: float = 0.5
    metadata: Optional[Dict] = None


@dataclass
class TaskDistribution:
    """任务分布配置"""
    n_way: int = 5
    k_shot: int = 1
    query_size: int = 15
    difficulty_range: Tuple[float, float] = (0.0, 1.0)
    similarity_threshold: float = 0.8


class DifficultyGrader:
    """
    任务难度分级器
    """
    
    def __init__(self, difficulty_bins: int = 5):
        self.difficulty_bins = difficulty_bins
        self.bin_edges = np.linspace(0, 1, difficulty_bins + 1)
        
    def compute_difficulty(self, 
                          task_data: torch.Tensor, 
                          task_labels: torch.Tensor) -> float:
        """
        计算任务难度
        
        难度指标基于：
        1. 类内方差（类内越分散越难）
        2. 类间重叠度（类间越重叠越难）
        3. 样本数量（样本越少越难）
        """
        unique_labels = torch.unique(task_labels)
        
        # 计算类内方差
        intra_class_variance = 0.0
        for label in unique_labels:
            class_data = task_data[task_labels == label]
            if len(class_data) > 1:
                class_mean = class_data.mean(dim=0)
                variance = ((class_data - class_mean) ** 2).mean()
                intra_class_variance += variance.item()
        
        intra_class_variance /= len(unique_labels)
        
        # 计算类间重叠度
        class_means = []
        for label in unique_labels:
            class_data = task_data[task_labels == label]
            class_mean = class_data.mean(dim=0)
            class_means.append(class_mean.reshape(1, -1))  # 确保是2D
        
        if len(class_means) > 1:
            class_means = np.vstack(class_means)  # 垂直拼接成2D数组
            distances = cdist(class_means, class_means)
            np.fill_diagonal(distances, np.inf)
            min_distances = distances.min(axis=1)
            inter_class_overlap = 1.0 / (min_distances.mean() + 1e-8)
            inter_class_overlap = min(inter_class_overlap, 1.0)
        else:
            inter_class_overlap = 0.0
        
        # 综合难度
        difficulty = (0.4 * intra_class_variance + 
                     0.4 * inter_class_overlap + 
                     0.2 * (1.0 / len(task_data)))
        
        return min(max(difficulty, 0.0), 1.0)
    
    def get_difficulty_bin(self, difficulty: float) -> int:
        """获取难度等级"""
        for i in range(len(self.bin_edges) - 1):
            if self.bin_edges[i] <= difficulty < self.bin_edges[i + 1]:
                return i
        return len(self.bin_edges) - 1


class TaskSimilarity:
    """
    任务相似度计算器
    """
    
    def __init__(self, method: str = "wasserstein"):
        self.method = method
        
    def compute_similarity(self, task1: Task, task2: Task) -> float:
        """计算两个任务的相似度"""
        if self.method == "wasserstein":
            return self._wasserstein_similarity(task1, task2)
        elif self.method == "distribution":
            return self._distribution_similarity(task1, task2)
        elif self.method == "label":
            return self._label_similarity(task1, task2)
        else:
            return self._feature_similarity(task1, task2)
    
    def _wasserstein_similarity(self, task1: Task, task2: Task) -> float:
        """基于Wasserstein距离的相似度"""
        support1_flat = task1.support_x.reshape(len(task1.support_x), -1).numpy()
        support2_flat = task2.support_x.reshape(len(task2.support_x), -1).numpy()
        
        mean1, std1 = support1_flat.mean(axis=0), support1_flat.std(axis=0)
        mean2, std2 = support2_flat.mean(axis=0), support2_flat.std(axis=0)
        
        distance = np.sqrt(((mean1 - mean2) ** 2).sum() + ((std1 - std2) ** 2).sum())
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def _distribution_similarity(self, task1: Task, task2: Task) -> float:
        """基于数据分布的相似度"""
        all_data1 = task1.support_x.reshape(len(task1.support_x), -1)
        all_data2 = task2.support_x.reshape(len(task2.support_x), -1)
        
        p1 = np.histogram(all_data1, bins=50, density=True)[0]
        p2 = np.histogram(all_data2, bins=50, density=True)[0]
        
        p1 = p1 / (p1.sum() + 1e-8)
        p2 = p2 / (p2.sum() + 1e-8)
        
        # JS散度
        m = 0.5 * (p1 + p2)
        js = 0.5 * np.sum(p1 * np.log((p1 + 1e-8) / (m + 1e-8))) + \
             0.5 * np.sum(p2 * np.log((p2 + 1e-8) / (m + 1e-8)))
        
        return 1.0 - np.clip(js, 0, 1)
    
    def _label_similarity(self, task1: Task, task2: Task) -> float:
        """基于标签分布的相似度"""
        labels1 = task1.support_y.numpy()
        labels2 = task2.support_y.numpy()
        
        unique1, counts1 = np.unique(labels1, return_counts=True)
        unique2, counts2 = np.unique(labels2, return_counts=True)
        
        dist1 = np.zeros(max(unique1.max(), unique2.max()) + 1)
        dist2 = np.zeros(max(unique1.max(), unique2.max()) + 1)
        
        for label, count in zip(unique1, counts1):
            dist1[label] = count
        for label, count in zip(unique2, counts2):
            dist2[label] = count
        
        dist1 = dist1 / (dist1.sum() + 1e-8)
        dist2 = dist2 / (dist2.sum() + 1e-8)
        
        # 余弦相似度
        similarity = np.dot(dist1, dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2) + 1e-8)
        
        return similarity
    
    def _feature_similarity(self, task1: Task, task2: Task) -> float:
        """基于特征均值的相似度"""
        mean1 = task1.support_x.reshape(len(task1.support_x), -1).mean(axis=0).numpy()
        mean2 = task2.support_x.reshape(len(task2.support_x), -1).mean(axis=0).numpy()
        
        cosine_sim = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2) + 1e-8)
        
        return cosine_sim


class TaskGenerator:
    """
    任务生成器
    
    功能：
    - 从数据集生成episode
    - 支持难度控制
    - 支持任务相似度计算
    - 支持 Curriculum Learning（课程学习）
    """
    
    def __init__(self,
                 dataset: Dict[int, torch.Tensor],
                 labels: Dict[int, torch.Tensor],
                 config: Optional[TaskDistribution] = None):
        """
        Args:
            dataset: 类别到数据的映射
            labels: 类别到标签的映射
            config: 任务分布配置
        """
        self.dataset = dataset
        self.labels = labels
        self.config = config or TaskDistribution()
        self.classes = list(dataset.keys())
        
        self.difficulty_grader = DifficultyGrader()
        self.similarity_calculator = TaskSimilarity()
        
        # 任务缓存
        self.task_cache: Dict[str, Task] = {}
        
    def sample_task(self, 
                   n_way: Optional[int] = None,
                   k_shot: Optional[int] = None,
                   difficulty: Optional[float] = None,
                   exclude_tasks: Optional[List[str]] = None) -> Task:
        """
        采样一个任务
        
        Args:
            n_way: 类别数
            k_shot: 每个类别的样本数
            difficulty: 目标难度
            exclude_tasks: 排除的任务ID列表
        
        Returns:
            采样的任务
        """
        n_way = n_way or self.config.n_way
        k_shot = k_shot or self.config.k_shot
        exclude_tasks = exclude_tasks or []
        
        # 随机选择类别
        selected_classes = np.random.choice(self.classes, n_way, replace=False)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(selected_classes):
            class_data = self.dataset[cls]
            class_labels = self.labels[cls]
            
            # 随机选择样本
            indices = np.random.permutation(len(class_data))
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:k_shot + self.config.query_size]
            
            support_x.append(class_data[support_idx])
            support_y.append(torch.full((k_shot,), i, dtype=torch.long))
            
            query_x.append(class_data[query_idx])
            query_y.append(torch.full((len(query_idx),), i, dtype=torch.long))
        
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.cat(support_y, dim=0)
        query_x = torch.cat(query_x, dim=0)
        query_y = torch.cat(query_y, dim=0)
        
        # 生成任务ID
        task_id = self._generate_task_id(selected_classes)
        
        # 计算难度
        all_data = torch.cat([support_x, query_x], dim=0)
        all_labels = torch.cat([support_y, query_y], dim=0)
        actual_difficulty = self.difficulty_grader.compute_difficulty(all_data, all_labels)
        
        task = Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            task_id=task_id,
            difficulty=actual_difficulty,
            metadata={
                "classes": selected_classes.tolist(),
                "n_way": n_way,
                "k_shot": k_shot
            }
        )
        
        return task
    
    def sample_task_batch(self,
                         batch_size: int,
                         n_way: Optional[int] = None,
                         k_shot: Optional[int] = None,
                         difficulty_range: Optional[Tuple[float, float]] = None,
                         use_curriculum: bool = False,
                         curriculum_difficulty: float = 0.5) -> List[Task]:
        """
        采样一批任务
        
        Args:
            batch_size: 任务数量
            n_way: 类别数
            k_shot: 每个类别的样本数
            difficulty_range: 难度范围
            use_curriculum: 是否使用课程学习
            curriculum_difficulty: 当前课程难度
        
        Returns:
            任务列表
        """
        tasks = []
        
        for _ in range(batch_size):
            if use_curriculum:
                # 课程学习：逐渐增加难度
                difficulty = np.random.uniform(0, curriculum_difficulty)
            elif difficulty_range:
                difficulty = np.random.uniform(*difficulty_range)
            else:
                difficulty = None
            
            task = self.sample_task(n_way, k_shot, difficulty)
            tasks.append(task)
        
        return tasks
    
    def generate_episode(self,
                        n_way: int,
                        k_shot: int,
                        num_episodes: int = 100) -> List[Task]:
        """
        生成一个完整的episode（用于评估）
        
        Args:
            n_way: 类别数
            k_shot: 每个类别的样本数
            num_episodes: 生成的任务数
        
        Returns:
            任务列表
        """
        return [self.sample_task(n_way, k_shot) for _ in range(num_episodes)]
    
    def _generate_task_id(self, classes: np.ndarray) -> str:
        """生成唯一任务ID"""
        return f"task_{'_'.join(map(str, sorted(classes)))}"
    
    def compute_task_similarity(self, task1: Task, task2: Task) -> float:
        """计算两个任务的相似度"""
        return self.similarity_calculator.compute_similarity(task1, task2)
    
    def find_similar_tasks(self,
                          task: Task,
                          task_pool: List[Task],
                          threshold: float = 0.8,
                          max_similar: int = 5) -> List[Tuple[Task, float]]:
        """
        查找相似任务
        
        Args:
            task: 目标任务
            task_pool: 任务池
            threshold: 相似度阈值
            max_similar: 最大返回数量
        
        Returns:
            相似任务列表（相似度降序）
        """
        similarities = []
        
        for other_task in task_pool:
            if other_task.task_id == task.task_id:
                continue
            
            sim = self.compute_task_similarity(task, other_task)
            if sim >= threshold:
                similarities.append((other_task, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:max_similar]
    
    def get_task_distribution_stats(self, tasks: List[Task]) -> Dict:
        """
        获取任务分布统计信息
        
        Args:
            tasks: 任务列表
        
        Returns:
            统计信息字典
        """
        difficulties = [t.difficulty for t in tasks]
        
        return {
            "n_tasks": len(tasks),
            "mean_difficulty": np.mean(difficulties),
            "std_difficulty": np.std(difficulties),
            "min_difficulty": np.min(difficulties),
            "max_difficulty": np.max(difficulties),
            "difficulty_histogram": np.histogram(difficulties, bins=10)[0].tolist()
        }
