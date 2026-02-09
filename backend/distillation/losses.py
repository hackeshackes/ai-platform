"""
Distillation Losses - v3.0 Core Feature

蒸馏损失函数实现

Note: This module requires PyTorch to be installed for actual loss computation.
When PyTorch is not available, it provides interface classes and metadata only.
"""
from typing import Dict, List, Optional, Union, Callable, TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

# Runtime imports with graceful fallback
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


class DistillationLoss:
    """
    蒸馏损失基类
    
    提供多种蒸馏损失函数实现
    
    Note: This is a base class. When PyTorch is not available,
    loss computation will be simulated.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """
        初始化蒸馏损失
        
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            beta: 原始任务损失权重
        """
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        student_labels=None,
        target_labels=None
    ) -> Dict[str, float]:
        """
        计算蒸馏损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            student_labels: 学生模型真实标签
            target_labels: 目标任务标签
            
        Returns:
            损失字典
        """
        if not HAS_TORCH:
            # Return mock values when torch is not available
            return {
                "total_loss": 0.5,
                "distillation_loss": 0.3,
                "task_loss": 0.2,
                "alpha": self.alpha,
                "beta": self.beta
            }
        
        # Actual torch implementation
        raise NotImplementedError


class KLDivergenceLoss(DistillationLoss):
    """
    KL散度蒸馏损失
    
    使用KL散度度量教师和学生输出分布的差异
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = "batchmean"
    ):
        super().__init__(temperature, alpha, beta)
        self.reduction = reduction
        
        if HAS_TORCH:
            self.kl_loss = nn.KLDivLoss(
                reduction=reduction,
                log_target=False
            )
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        student_labels=None,
        target_labels=None
    ) -> Dict[str, float]:
        """计算KL散度蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.45 * self.alpha + 0.3 * self.beta,
                "kl_loss": 0.45,
                "task_loss": 0.3,
                "alpha": self.alpha,
                "beta": self.beta
            }
        
        # Apply temperature scaling
        student_soft = student_logits / self.temperature
        teacher_soft = teacher_logits / self.temperature
        
        # Compute soft target probabilities
        student_probs = F.log_softmax(student_soft, dim=-1)
        teacher_probs = F.softmax(teacher_soft, dim=-1)
        
        # KL divergence loss
        kl_loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Task loss
        task_loss = torch.tensor(0.0)
        if student_labels is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                student_labels.view(-1)
            )
        
        # Total loss
        total_loss = self.alpha * kl_loss + self.beta * task_loss
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "kl_loss": kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss,
            "task_loss": task_loss.item() if hasattr(task_loss, 'item') else task_loss,
            "alpha": self.alpha,
            "beta": self.beta
        }


class MSELoss(DistillationLoss):
    """
    均方误差蒸馏损失
    
    直接比较教师和学生的logits
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        normalize: bool = True
    ):
        super().__init__(temperature, alpha, beta)
        self.normalize = normalize
        
        if HAS_TORCH:
            self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        student_labels=None,
        target_labels=None
    ) -> Dict[str, float]:
        """计算MSE蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.35 * self.alpha + 0.3 * self.beta,
                "mse_loss": 0.35,
                "task_loss": 0.3,
                "alpha": self.alpha,
                "beta": self.beta
            }
        
        # Normalize logits
        if self.normalize:
            student_logits = F.normalize(student_logits, p=2, dim=-1)
            teacher_logits = F.normalize(teacher_logits, p=2, dim=-1)
        
        # MSE loss
        mse_loss = self.mse_loss(student_logits, teacher_logits)
        
        # Task loss
        task_loss = torch.tensor(0.0)
        if student_labels is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                student_labels.view(-1)
            )
        
        total_loss = self.alpha * mse_loss + self.beta * task_loss
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "mse_loss": mse_loss.item() if hasattr(mse_loss, 'item') else mse_loss,
            "task_loss": task_loss.item() if hasattr(task_loss, 'item') else task_loss,
            "alpha": self.alpha,
            "beta": self.beta
        }


class CosineEmbeddingLoss(DistillationLoss):
    """
    余弦相似度蒸馏损失
    
    使用余弦相似度保持教师和学生输出方向一致
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        margin: float = 0.0
    ):
        super().__init__(temperature, alpha, beta)
        self.margin = margin
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        student_labels=None,
        target_labels=None
    ) -> Dict[str, float]:
        """计算余弦相似度蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.4 * self.alpha + 0.3 * self.beta,
                "cosine_loss": 0.4,
                "task_loss": 0.3,
                "alpha": self.alpha,
                "beta": self.beta
            }
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            student_logits,
            teacher_logits,
            dim=-1
        )
        
        # Contrastive loss
        cosine_loss = (1 - cos_sim).mean()
        
        # Task loss
        task_loss = torch.tensor(0.0)
        if student_labels is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                student_labels.view(-1)
            )
        
        total_loss = self.alpha * cosine_loss + self.beta * task_loss
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "cosine_loss": cosine_loss.item() if hasattr(cosine_loss, 'item') else cosine_loss,
            "task_loss": task_loss.item() if hasattr(task_loss, 'item') else task_loss,
            "alpha": self.alpha,
            "beta": self.beta
        }


class AttentionBasedLoss(DistillationLoss):
    """
    基于注意力的蒸馏损失
    
    匹配教师和学生的注意力权重
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        attention_layers: Optional[List[int]] = None,
        normalize: bool = True
    ):
        super().__init__(temperature, alpha, beta)
        self.attention_layers = attention_layers or list(range(12))
        self.normalize = normalize
        
        if HAS_TORCH:
            self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        student_attentions=None,
        teacher_attentions=None,
        student_logits=None,
        teacher_logits=None,
        student_labels=None
    ) -> Dict[str, float]:
        """计算注意力蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.38 * self.alpha + 0.3 * self.beta,
                "attention_loss": 0.38,
                "task_loss": 0.3,
                "layers_matched": len(self.attention_layers)
            }
        
        attention_loss = torch.tensor(0.0)
        num_layers = 0
        
        if student_attentions and teacher_attentions:
            for layer in self.attention_layers:
                if layer in student_attentions and layer in teacher_attentions:
                    s_attn = student_attentions[layer]
                    t_attn = teacher_attentions[layer]
                    
                    # Normalize
                    if self.normalize:
                        s_attn = s_attn / (s_attn.sum(dim=-1, keepdim=True) + 1e-8)
                        t_attn = t_attn / (t_attn.sum(dim=-1, keepdim=True) + 1e-8)
                    
                    attention_loss += self.mse_loss(s_attn, t_attn)
                    num_layers += 1
            
            if num_layers > 0:
                attention_loss = attention_loss / num_layers
        
        # Task loss
        task_loss = torch.tensor(0.0)
        if student_labels is not None and student_logits is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                student_labels.view(-1)
            )
        
        total_loss = self.alpha * attention_loss + self.beta * task_loss
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "attention_loss": attention_loss.item() if hasattr(attention_loss, 'item') else attention_loss,
            "task_loss": task_loss.item() if hasattr(task_loss, 'item') else task_loss,
            "layers_matched": num_layers,
            "alpha": self.alpha,
            "beta": self.beta
        }


class HiddenStateLoss(DistillationLoss):
    """
    隐藏状态蒸馏损失
    
    匹配教师和学生的中间层表示
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        hidden_layers: Optional[List[int]] = None,
        projection_size: Optional[int] = None
    ):
        super().__init__(temperature, alpha, beta)
        self.hidden_layers = hidden_layers or list(range(12))
        self.projection_size = projection_size
        
        if HAS_TORCH and projection_size:
            self.projector = nn.ModuleDict()
    
    def forward(
        self,
        student_hidden=None,
        teacher_hidden=None
    ) -> Dict[str, float]:
        """计算隐藏状态蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.42 * self.alpha,
                "hidden_loss": 0.42,
                "layers_matched": len(self.hidden_layers)
            }
        
        hidden_loss = torch.tensor(0.0)
        num_layers = 0
        
        if student_hidden and teacher_hidden:
            for layer in self.hidden_layers:
                if layer in student_hidden and layer in teacher_hidden:
                    s_hidden = student_hidden[layer]
                    t_hidden = teacher_hidden[layer]
                    
                    # Project if needed
                    if self.projection_size and f"layer_{layer}" in self.projector:
                        s_hidden = self.projector[f"layer_{layer}"](s_hidden)
                        t_hidden = self.projector[f"layer_{layer}"](t_hidden)
                    
                    # L2 distance
                    layer_loss = F.mse_loss(s_hidden, t_hidden)
                    hidden_loss += layer_loss
                    num_layers += 1
            
            if num_layers > 0:
                hidden_loss = hidden_loss / num_layers
        
        total_loss = self.alpha * hidden_loss
        
        return {
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "hidden_loss": hidden_loss.item() if hasattr(hidden_loss, 'item') else hidden_loss,
            "layers_matched": num_layers,
            "alpha": self.alpha,
            "beta": self.beta
        }


class CombinedLoss(DistillationLoss):
    """
    组合蒸馏损失
    
    结合多种蒸馏损失
    """
    
    def __init__(
        self,
        losses: Optional[List] = None,
        weights: Optional[List[float]] = None,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        super().__init__(temperature=2.0, alpha=alpha, beta=beta)
        self.losses = losses or []
        self.weights = weights or [1.0] * len(losses) if losses else []
        
        if HAS_TORCH:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        student_labels=None,
        **kwargs
    ) -> Dict[str, float]:
        """计算组合蒸馏损失"""
        if not HAS_TORCH:
            return {
                "total_loss": 0.35 * self.alpha + 0.3 * self.beta,
                "distillation_loss": 0.35,
                "task_loss": 0.3
            }
        
        distillation_loss = torch.tensor(0.0)
        loss_components = {}
        
        for i, loss_fn in enumerate(self.losses):
            loss_output = loss_fn(
                student_logits,
                teacher_logits,
                student_labels=student_labels,
                **kwargs
            )
            
            dl = loss_output.get("total_loss", 0.0)
            distillation_loss = distillation_loss + self.weights[i] * dl
            loss_components[f"loss_{i}"] = dl
        
        # Task loss
        task_loss = torch.tensor(0.0)
        if student_labels is not None:
            task_loss = self.ce_loss(
                student_logits.view(-1, student_logits.size(-1)),
                student_labels.view(-1)
            )
        
        total_loss = self.alpha * distillation_loss + self.beta * task_loss
        
        loss_components["total_loss"] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        loss_components["task_loss"] = task_loss.item() if hasattr(task_loss, 'item') else task_loss
        loss_components["distillation_loss"] = distillation_loss.item() if hasattr(distillation_loss, 'item') else distillation_loss
        loss_components["alpha"] = self.alpha
        loss_components["beta"] = self.beta
        
        return loss_components


class LossFactory:
    """蒸馏损失函数工厂"""
    
    @staticmethod
    def create_loss(
        loss_type: str,
        temperature: float = 2.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        **kwargs
    ):
        """
        创建蒸馏损失函数
        
        Args:
            loss_type: 损失类型 ('kl', 'mse', 'cosine', 'attention', 'hidden', 'combined')
            temperature: 温度
            alpha: 蒸馏损失权重
            beta: 原始损失权重
            **kwargs: 额外参数
            
        Returns:
            损失模块
        """
        loss_type = loss_type.lower()
        
        if loss_type in ("kl", "kldivergence"):
            return KLDivergenceLoss(temperature, alpha, beta)
        elif loss_type in ("mse", "mean_squared"):
            return MSELoss(temperature, alpha, beta)
        elif loss_type in ("cosine", "cosine_embedding"):
            return CosineEmbeddingLoss(temperature, alpha, beta)
        elif loss_type == "attention":
            return AttentionBasedLoss(temperature, alpha, beta, **kwargs)
        elif loss_type == "hidden":
            return HiddenStateLoss(temperature, alpha, beta, **kwargs)
        elif loss_type == "combined":
            return CombinedLoss([], weights=[], alpha=alpha, beta=beta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def get_available_losses() -> List[Dict]:
        """获取可用的损失函数"""
        return [
            {
                "id": "kl",
                "name": "KL Divergence",
                "description": "Knowledge distillation using Kullback-Leibler divergence",
                "type": "soft_target"
            },
            {
                "id": "mse",
                "name": "Mean Squared Error",
                "description": "Direct logits comparison using MSE",
                "type": "logits_matching"
            },
            {
                "id": "cosine",
                "name": "Cosine Similarity",
                "description": "Output direction preservation",
                "type": "similarity"
            },
            {
                "id": "attention",
                "name": "Attention Transfer",
                "description": "Matches attention patterns between teacher and student",
                "type": "feature_based"
            },
            {
                "id": "hidden",
                "name": "Hidden State Transfer",
                "description": "Matches intermediate representations",
                "type": "feature_based"
            }
        ]


# Export information
__all__ = [
    "DistillationLoss",
    "KLDivergenceLoss",
    "MSELoss",
    "CosineEmbeddingLoss",
    "AttentionBasedLoss",
    "HiddenStateLoss",
    "CombinedLoss",
    "LossFactory",
    "HAS_TORCH",
]
