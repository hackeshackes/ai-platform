"""
元学习器 - Meta Learner
实现MAML、Prototypical Networks、Reptile等元学习算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from copy import deepcopy


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML)
    
    实现能够快速适应新任务的元学习器
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr  # 内层学习率（任务适应）
        self.outer_lr = outer_lr  # 外层学习率（元更新）
        
    def forward(self, x):
        return self.model(x)
    
    def inner_update(self, support_x: torch.Tensor, support_y: torch.Tensor, 
                    device: torch.device) -> nn.Module:
        """
        在支持集上进行内层更新（任务适应）
        """
        self.model = self.model.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # 克隆模型以避免影响原始模型
        adapted_model = deepcopy(self.model)
        adapted_model = adapted_model.to(device)
        
        # 计算损失并进行梯度下降
        logits = adapted_model(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        # 计算梯度并更新参数
        grads = torch.autograd.grad(loss, adapted_model.parameters(), 
                                   create_graph=True, allow_unused=True)
        
        adapted_params = []
        for param, grad in zip(adapted_model.parameters(), grads):
            if grad is not None:
                adapted_params.append(param - self.inner_lr * grad)
            else:
                adapted_params.append(param)
        
        # 更新模型参数
        with torch.no_grad():
            for i, param in enumerate(adapted_model.parameters()):
                param.copy_(adapted_params[i])
        
        return adapted_model
    
    def meta_update(self, query_loss: torch.Tensor):
        """
        元更新：基于查询集损失更新元参数
        """
        self.optimizer.zero_grad()
        query_loss.backward()
        self.optimizer.step()


class PrototypicalNetworks(nn.Module):
    """
    Prototypical Networks - 原型网络
    
    通过计算类别原型进行少样本分类
    """
    
    def __init__(self, encoder: nn.Module, distance_metric: str = "euclidean"):
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
        
        if distance_metric == "euclidean":
            self.distance = self._euclidean_distance
        elif distance_metric == "cosine":
            self.distance = self._cosine_distance
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    def forward(self, x):
        return self.encoder(x)
    
    def _euclidean_distance(self, query_embeddings: torch.Tensor, 
                           prototypes: torch.Tensor) -> torch.Tensor:
        """计算欧氏距离"""
        n_way = prototypes.size(0)
        query_embeddings = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)
        prototypes = prototypes.unsqueeze(0).expand_as(query_embeddings)
        return torch.pow(query_embeddings - prototypes, 2).sum(dim=-1)
    
    def _cosine_distance(self, query_embeddings: torch.Tensor,
                        prototypes: torch.Tensor) -> torch.Tensor:
        """计算余弦距离"""
        # 归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        return 1 - torch.mm(query_embeddings, prototypes.t())
    
    def get_prototypes(self, support_embeddings: torch.Tensor, 
                      support_labels: torch.Tensor, 
                      n_way: int) -> torch.Tensor:
        """
        计算每个类别的原型
        
        Args:
            support_embeddings: 支持集嵌入 [n_support, embedding_dim]
            support_labels: 支持集标签 [n_support]
            n_way: 类别数
        
        Returns:
            prototypes: 原型向量 [n_way, embedding_dim]
        """
        classes = torch.arange(n_way, device=support_embeddings.device)
        prototypes = []
        
        for c in classes:
            class_embeddings = support_embeddings[support_labels == c]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def classify(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """基于原型进行分类"""
        distances = self.distance(query_embeddings, prototypes)
        return -distances  # 距离越小，分类概率越高


class Reptile(nn.Module):
    """
    Reptile算法 - 简化的元学习方法
    
    通过在多个任务上执行SGD并直接更新初始参数
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.1, outer_lr: float = 0.001):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        
    def forward(self, x):
        return self.model(x)
    
    def inner_update(self, support_x: torch.Tensor, support_y: torch.Tensor,
                    device: torch.device, steps: int = 5) -> Dict[str, torch.Tensor]:
        """
        在任务上进行多步SGD
        """
        self.model = self.model.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # 保存初始参数
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # 执行多步SGD
        adapted_model = deepcopy(self.model)
        adapted_model = adapted_model.to(device)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 返回参数更新方向
        final_params = {name: param for name, param in adapted_model.named_parameters()}
        
        return {
            "initial": initial_params,
            "final": final_params,
            "model": adapted_model
        }


class MetaLearner:
    """
    元学习器主类
    
    整合MAML、Prototypical Networks、Reptile等算法
    支持学习策略自动选择
    """
    
    SUPPORTED_ALGORITHMS = ["maml", "prototypical", "reptile", "baseline"]
    
    def __init__(self, 
                 algorithm: str = "maml",
                 model: Optional[nn.Module] = None,
                 n_way: int = 5,
                 k_shot: int = 1,
                 inner_lr: float = 0.01,
                 outer_lr: float = 0.001,
                 device: Optional[torch.device] = None):
        """
        Args:
            algorithm: 元学习算法
            model: 基础模型
            n_way: 分类类别数
            k_shot: 每个类别的样本数
            inner_lr: 内层学习率
            outer_lr: 外层学习率
            device: 计算设备
        """
        self.algorithm = algorithm.lower()
        self.n_way = n_way
        self.k_shot = k_shot
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported. Choose from: {self.SUPPORTED_ALGORITHMS}")
        
        # 初始化模型
        self.model = model
        if model is None:
            self.model = self._create_default_encoder()
        
        # 初始化算法特定组件
        if self.algorithm == "maml":
            self.meta_learner = MAML(self.model, inner_lr, outer_lr)
        elif self.algorithm == "prototypical":
            self.meta_learner = PrototypicalNetworks(self.model)
        elif self.algorithm == "reptile":
            self.meta_learner = Reptile(self.model, inner_lr, outer_lr)
        else:
            self.meta_learner = self.model
        
        self.meta_learner = self.meta_learner.to(self.device)
        self.optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=outer_lr)
        
    def _create_default_encoder(self) -> nn.Module:
        """创建默认特征编码器"""
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def select_algorithm(self, task_distribution: Dict) -> str:
        """
        根据任务分布自动选择最佳算法
        
        Args:
            task_distribution: 任务分布统计信息
        
        Returns:
            推荐的算法名称
        """
        # 基于任务数量和样本数选择算法
        n_tasks = task_distribution.get("n_tasks", 100)
        avg_samples = task_distribution.get("avg_samples_per_class", 5)
        
        if avg_samples <= 5:
            return "prototypical"
        elif n_tasks > 1000:
            return "reptile"
        else:
            return "maml"
    
    def train_epoch(self, 
                    support_sets: List[torch.Tensor], 
                    query_sets: List[torch.Tensor],
                    labels_support: List[torch.Tensor],
                    labels_query: List[torch.Tensor]) -> float:
        """
        训练一个epoch
        
        Args:
            support_sets: 支持集列表
            query_sets: 查询集列表
            labels_support: 支持集标签
            labels_query: 查询集标签
        """
        total_loss = 0.0
        n_tasks = len(support_sets)
        
        for i in range(n_tasks):
            support_x, support_y = support_sets[i].to(self.device), labels_support[i].to(self.device)
            query_x, query_y = query_sets[i].to(self.device), labels_query[i].to(self.device)
            
            if self.algorithm == "maml":
                # MAML训练
                adapted_model = self.meta_learner.inner_update(support_x, support_y, self.device)
                query_logits = adapted_model(query_x)
                query_loss = F.cross_entropy(query_logits, query_y)
                self.meta_learner.meta_update(query_loss)
                
            elif self.algorithm == "prototypical":
                # Prototypical Networks训练
                support_embeddings = self.meta_learner.encoder(support_x)
                prototypes = self.meta_learner.get_prototypes(support_embeddings, support_y, self.n_way)
                query_embeddings = self.meta_learner.encoder(query_x)
                logits = self.meta_learner.classify(query_embeddings, prototypes)
                query_loss = F.cross_entropy(logits, query_y)
                
                self.optimizer.zero_grad()
                query_loss.backward()
                self.optimizer.step()
                
            elif self.algorithm == "reptile":
                # Reptile训练
                result = self.meta_learner.inner_update(support_x, support_y, self.device, steps=5)
                final_params = result["final"]
                
                # 计算元梯度并更新
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in final_params:
                            grad = param - final_params[name]
                            param.add_(grad * self.meta_learner.outer_lr)
            
            total_loss += query_loss.item()
        
        return total_loss / n_tasks
    
    def adapt(self, 
              support_x: torch.Tensor, 
              support_y: torch.Tensor,
              steps: int = 5) -> nn.Module:
        """
        快速适应新任务
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            steps: 适应步数
        
        Returns:
            适应后的模型
        """
        self.meta_learner.eval()
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        if self.algorithm == "maml":
            adapted_model = self.meta_learner.inner_update(support_x, support_y, self.device)
            return adapted_model
            
        elif self.algorithm == "prototypical":
            embeddings = self.meta_learner.encoder(support_x)
            prototypes = self.meta_learner.get_prototypes(embeddings, support_y, self.n_way)
            return {"prototypes": prototypes, "encoder": self.meta_learner.encoder}
            
        elif self.algorithm == "reptile":
            result = self.meta_learner.inner_update(support_x, support_y, self.device, steps)
            return result["model"]
        
        return self.meta_learner
    
    def evaluate(self, 
                model: Union[nn.Module, Dict],
                query_x: torch.Tensor, 
                query_y: torch.Tensor) -> float:
        """
        评估模型性能
        
        Args:
            model: 适应后的模型或字典
            query_x: 查询集输入
            query_y: 查询集标签
        
        Returns:
            准确率
        """
        self.meta_learner.eval()
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)
        
        with torch.no_grad():
            if self.algorithm in ["maml", "reptile"]:
                logits = model(query_x)
                predictions = logits.argmax(dim=-1)
            elif self.algorithm == "prototypical":
                embeddings = model["encoder"](query_x)
                logits = model["prototypes"](embeddings)
                predictions = logits.argmax(dim=-1)
            else:
                logits = self.meta_learner(query_x)
                predictions = logits.argmax(dim=-1)
            
            accuracy = (predictions == query_y).float().mean().item()
        
        return accuracy
