"""
少样本学习器 - Few-Shot Learner
实现N-way K-shot学习、对比学习、度量学习
目标：1-5样本达到90%精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class NWayKShotDataset(Dataset):
    """
    N-way K-shot 数据集
    """
    
    def __init__(self, 
                 data: Dict[int, torch.Tensor], 
                 labels: Dict[int, torch.Tensor],
                 n_way: int = 5,
                 k_shot: int = 1,
                 query_size: int = 15,
                 transform: Optional[Callable] = None):
        """
        Args:
            data: 类别到数据的映射
            labels: 类别到标签的映射
            n_way: 类别数
            k_shot: 每个类别的支持样本数
            query_size: 每个类别的查询样本数
            transform: 数据变换
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.transform = transform
        self.classes = list(data.keys())
        
    def __len__(self):
        return 1000  # 返回足够多的episode数
    
    def __getitem__(self, idx):
        """生成一个episode"""
        # 随机选择n_way个类别
        selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(selected_classes):
            class_data = self.data[cls]
            class_labels = self.labels[cls]
            
            # 随机选择样本
            indices = np.random.permutation(len(class_data))
            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:self.k_shot + self.query_size]
            
            support_x.append(class_data[support_idx])
            support_y.append(torch.full((self.k_shot,), i, dtype=torch.long))
            
            query_x.append(class_data[query_idx])
            query_y.append(torch.full((self.query_size,), i, dtype=torch.long))
        
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.cat(support_y, dim=0)
        query_x = torch.cat(query_x, dim=0)
        query_y = torch.cat(query_y, dim=0)
        
        # 打乱查询集
        perm = torch.randperm(len(query_x))
        query_x, query_y = query_x[perm], query_y[perm]
        
        if self.transform:
            support_x = self.transform(support_x)
            query_x = self.transform(query_x)
        
        return support_x, support_y, query_x, query_y


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                query_embeddings: torch.Tensor, 
                support_embeddings: torch.Tensor,
                query_labels: torch.Tensor,
                support_labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            query_embeddings: 查询集嵌入
            support_embeddings: 支持集嵌入
            query_labels: 查询集标签
            support_labels: 支持集标签
        
        Returns:
            对比损失
        """
        # 归一化嵌入
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        support_embeddings = F.normalize(support_embeddings, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.mm(query_embeddings, support_embeddings.t()) / self.temperature
        
        # 创建正样本掩码
        labels_expanded = query_labels.unsqueeze(1) == support_labels.unsqueeze(0)
        positive_mask = labels_expanded.float()
        
        # 避免标签泄露：排除自身
        for i in range(len(query_labels)):
            if query_labels[i] in support_labels:
                pos_indices = (support_labels == query_labels[i]).nonzero(as_tuple=True)[0]
                for j in pos_indices:
                    if torch.allclose(query_embeddings[i], support_embeddings[j]):
                        positive_mask[i, j] = 0
                        break
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 只计算正样本的损失
        loss = -(positive_mask * log_prob).sum() / (positive_mask.sum() + 1e-8)
        
        return loss


class MetricLearner(nn.Module):
    """
    度量学习模块
    
    学习任务相关的度量函数
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 可学习的度量矩阵
        self.metric_matrix = nn.Parameter(torch.eye(embedding_dim))
        
    def forward(self, x):
        return self.embedding_net(x)
    
    def compute_distance(self, 
                        embeddings1: torch.Tensor, 
                        embeddings2: torch.Tensor,
                        method: str = "mahalanobis") -> torch.Tensor:
        """
        计算样本间距离
        
        Args:
            embeddings1: 第一组嵌入
            embeddings2: 第二组嵌入
            method: 距离度量方法
        
        Returns:
            距离矩阵
        """
        if method == "euclidean":
            return torch.cdist(embeddings1, embeddings2)
        
        elif method == "cosine":
            embeddings1 = F.normalize(embeddings1, p=2, dim=-1)
            embeddings2 = F.normalize(embeddings2, p=2, dim=-1)
            return 1 - torch.mm(embeddings1, embeddings2.t())
        
        elif method == "mahalanobis":
            # 马氏距离
            embeddings1 = self.embedding_net(embeddings1)
            embeddings2 = self.embedding_net(embeddings2)
            L = self.metric_matrix @ self.metric_matrix.t()
            diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)
            return torch.sqrt((diff @ L * diff).sum(dim=-1) + 1e-8)


class FewShotLearner(nn.Module):
    """
    少样本学习器
    
    整合N-way K-shot学习、对比学习、度量学习
    目标：1-5样本达到90%精度
    """
    
    def __init__(self,
                 n_way: int = 5,
                 k_shot: int = 1,
                 embedding_dim: int = 64,
                 use_contrastive: bool = True,
                 use_metric: bool = True,
                 temperature: float = 0.07):
        super().__init__()
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.embedding_dim = embedding_dim
        
        # 特征编码器
        self.encoder = nn.Sequential(
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
            nn.Linear(128, embedding_dim)
        )
        
        # 度量学习器
        self.metric_learner = MetricLearner(embedding_dim) if use_metric else None
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature) if use_contrastive else None
        
        # 分类器
        self.classifier = nn.Linear(embedding_dim, n_way)
        
    def forward(self, x):
        embeddings = self.encoder(x)
        return embeddings
    
    def get_prototypes(self,
                      support_embeddings: torch.Tensor,
                      support_labels: torch.Tensor) -> torch.Tensor:
        """
        计算类别原型
        
        Args:
            support_embeddings: 支持集嵌入
            support_labels: 支持集标签
        
        Returns:
            类别原型
        """
        prototypes = []
        for i in range(self.n_way):
            mask = support_labels == i
            class_embeddings = support_embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def classify_with_prototypes(self,
                                query_embeddings: torch.Tensor,
                                prototypes: torch.Tensor,
                                method: str = "euclidean") -> torch.Tensor:
        """
        基于原型进行分类
        
        Args:
            query_embeddings: 查询集嵌入
            prototypes: 类别原型
            method: 距离度量方法
        
        Returns:
            分类logits
        """
        if method == "euclidean":
            distances = torch.cdist(query_embeddings, prototypes)
            return -distances
        
        elif method == "cosine":
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            return torch.mm(query_embeddings, prototypes.t())
        
        elif method == "mahalanobis" and self.metric_learner is not None:
            distances = self.metric_learner.compute_distance(
                query_embeddings, prototypes, method
            )
            return -distances
        
        return torch.mm(query_embeddings, prototypes.t())
    
    def adapt(self,
             support_x: torch.Tensor,
             support_y: torch.Tensor,
             query_x: Optional[torch.Tensor] = None,
             query_y: Optional[torch.Tensor] = None,
             lr: float = 0.1,
             steps: int = 10) -> Tuple[torch.Tensor, float]:
        """
        快速适应新任务
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            query_x: 查询集输入（可选，用于验证）
            query_y: 查询集标签（可选）
            lr: 学习率
            steps: 适应步数
        
        Returns:
            适应后的模型参数和验证准确率
        """
        self.eval()
        
        # 编码支持集
        with torch.no_grad():
            support_embeddings = self.encoder(support_x)
        
        # 计算原型
        prototypes = self.get_prototypes(support_embeddings, support_y)
        
        # 在查询集上评估
        if query_x is not None:
            with torch.no_grad():
                query_embeddings = self.encoder(query_x)
                logits = self.classify_with_prototypes(query_embeddings, prototypes)
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == query_y).float().mean().item()
            return prototypes, accuracy
        
        return prototypes, 0.0
    
    def fine_tune(self,
                  support_x: torch.Tensor,
                  support_y: torch.Tensor,
                  lr: float = 0.01,
                  epochs: int = 50) -> float:
        """
        对支持集进行微调
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            lr: 学习率
            epochs: 训练轮数
        
        Returns:
            最终损失
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        support_embeddings = self.encoder(support_x.detach())
        prototypes = self.get_prototypes(support_embeddings, support_y)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 计算当前嵌入
            current_embeddings = self.encoder(support_x)
            logits = self.classify_with_prototypes(current_embeddings, prototypes)
            
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def meta_train_epoch(self,
                        support_sets: List[torch.Tensor],
                        query_sets: List[torch.Tensor],
                        labels_support: List[torch.Tensor],
                        labels_query: List[torch.Tensor]) -> Dict[str, float]:
        """
        元训练一个epoch
        
        Args:
            support_sets: 支持集列表
            query_sets: 查询集列表
            labels_support: 支持集标签
            labels_query: 查询集标签
        
        Returns:
            训练统计信息
        """
        total_loss = 0.0
        total_accuracy = 0.0
        n_tasks = len(support_sets)
        
        for i in range(n_tasks):
            support_x, support_y = support_sets[i], labels_support[i]
            query_x, query_y = query_sets[i], labels_query[i]
            
            # 编码
            support_embeddings = self.encoder(support_x)
            query_embeddings = self.encoder(query_x)
            
            # 计算原型
            prototypes = self.get_prototypes(support_embeddings, support_y)
            
            # 分类
            logits = self.classify_with_prototypes(query_embeddings, prototypes)
            
            # 计算损失
            loss = F.cross_entropy(logits, query_y)
            
            # 对比学习损失
            if self.contrastive_loss is not None:
                contrastive = self.contrastive_loss(
                    query_embeddings, support_embeddings, query_y, support_y
                )
                loss = loss + 0.1 * contrastive
            
            # 反向传播
            self.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
            # 计算准确率
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == query_y).float().mean().item()
            
            total_loss += loss.item()
            total_accuracy += accuracy
        
        return {
            "loss": total_loss / n_tasks,
            "accuracy": total_accuracy / n_tasks
        }
    
    def evaluate_episode(self,
                        support_x: torch.Tensor,
                        support_y: torch.Tensor,
                        query_x: torch.Tensor,
                        query_y: torch.Tensor,
                        method: str = "prototype") -> Dict[str, float]:
        """
        评估单个episode
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            query_x: 查询集输入
            query_y: 查询集标签
            method: 评估方法
        
        Returns:
            评估结果
        """
        self.eval()
        
        with torch.no_grad():
            support_embeddings = self.encoder(support_x)
            query_embeddings = self.encoder(query_x)
            
            if method == "prototype":
                prototypes = self.get_prototypes(support_embeddings, support_y)
                logits = self.classify_with_prototypes(query_embeddings, prototypes)
            
            elif method == "matching":
                logits = self.matching_network(support_embeddings, query_embeddings, support_y)
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == query_y).float().mean().item()
            
            # 计算每个类别的准确率
            class_accuracies = {}
            for i in range(self.n_way):
                mask = query_y == i
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == query_y[mask]).float().mean().item()
                    class_accuracies[f"class_{i}"] = class_acc
        
        return {
            "overall_accuracy": accuracy,
            "class_accuracies": class_accuracies
        }
