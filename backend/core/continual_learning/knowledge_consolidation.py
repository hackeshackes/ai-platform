"""
知识巩固系统 (Knowledge Consolidation System)

提供多种知识巩固方法：
- Elastic Weight Consolidation (EWC): 弹性权重巩固
- Knowledge Distillation: 知识蒸馏
- Regularization: 正则化
- Meta-Learning Consolidation: 元学习巩固
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import math


@dataclass
class ConsolidationConfig:
    """知识巩固配置"""
    method: str = "ewc"
    importance_weight: float = 3000
    fisher_update_period: int = 100
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    regularization_weight: float = 0.01
    use_omega: bool = False
    use_lambda: bool = True
    meta_lr: float = 0.001


class KnowledgeConsolidation(ABC):
    """知识巩固基类"""
    
    @abstractmethod
    def register_task(self, task_id: str):
        """注册新任务"""
        pass
    
    @abstractmethod
    def compute_consolidation_loss(self, model: nn.Module, task_id: str) -> torch.Tensor:
        """计算巩固损失"""
        pass
    
    @abstractmethod
    def after_training(self, task_id: str):
        """训练后处理"""
        pass


class EWC(KnowledgeConsolidation):
    """
    弹性权重巩固 (Elastic Weight Consolidation)
    
    核心思想：
    - 计算每个参数对旧任务的重要性（Fisher信息矩阵）
    - 对重要参数施加更大的惩罚，防止它们被大幅修改
    
    优点：
    - 实现简单
    - 理论基础扎实
    - 适用于多种模型
    
    缺点：
    - Fisher信息矩阵计算开销大
    - 只能保护单任务知识
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_weight: float = 3000,
        fisher_update_period: int = 100,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.importance_weight = importance_weight
        self.fisher_update_period = fisher_update_period
        self.device = device or torch.device('cpu')
        
        # 存储旧任务的参数和Fisher信息
        self.task_parameters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_fisher: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_ids: List[str] = []
        
        # 样本计数器
        self.sample_count = 0
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.task_parameters[task_id] = {}
            self.task_fisher[task_id] = {}
            
            # 复制当前参数
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()
                    self.task_fisher[task_id][name] = torch.zeros_like(param.data)
                    
    def update_fisher(self, dataset: Optional[Dataset] = None, dataloader = None):
        """更新Fisher信息矩阵"""
        if dataloader is None and dataset is not None:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
        if dataloader is None:
            return
            
        # 计算Fisher信息
        self.model.eval()
        fisher_accum = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data)
                
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                data, target = batch[0], batch[1]
            else:
                data, target = batch, None
                
            data = data.to(self.device)
            self.model.zero_grad()
            
            output = self.model(data)
            
            if target is not None:
                loss = F.nll_loss(output, target.to(self.device))
            else:
                loss = output.mean()
                
            loss.backward()
            
            # 累加梯度的平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data ** 2
                    
        # 平均并保存
        num_samples = len(dataloader)
        for name in fisher_accum:
            fisher_accum[name] /= num_samples
            
        # 更新最后任务的Fisher信息
        if self.task_ids:
            current_task = self.task_ids[-1]
            self.task_fisher[current_task] = fisher_accum
            
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算EWC损失"""
        if task_id is None:
            task_id = self.task_ids[-1] if self.task_ids else None
            
        if task_id is None or task_id not in self.task_parameters:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        
        # 计算所有旧任务的EWC损失
        for old_task_id in self.task_ids:
            if old_task_id == task_id:
                continue
                
            for name, param in model.named_parameters():
                if name in self.task_parameters[old_task_id]:
                    # 获取旧参数和Fisher信息
                    old_param = self.task_parameters[old_task_id][name]
                    fisher = self.task_fisher[old_task_id].get(name, torch.ones_like(param))
                    
                    # EWC损失: sum_i F_i * (theta_i - theta_i*)^2
                    diff = param - old_param
                    loss += (fisher * diff ** 2).sum()
                    
        return self.importance_weight * loss
    
    def compute_online_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """计算在线EWC损失（累积所有任务）"""
        loss = torch.tensor(0.0, device=self.device)
        
        for old_task_id in self.task_ids:
            for name, param in model.named_parameters():
                if name in self.task_parameters[old_task_id]:
                    old_param = self.task_parameters[old_task_id][name]
                    fisher = self.task_fisher[old_task_id].get(name, torch.ones_like(param))
                    
                    diff = param - old_param
                    loss += (fisher * diff ** 2).sum()
                    
        return self.importance_weight * loss
    
    def after_training(self, task_id: str):
        """训练后更新参数和Fisher"""
        if task_id in self.task_ids:
            # 更新参数
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()
                    
    def set_importance(self, task_id: str, importance: float):
        """设置任务重要性"""
        self.importance_weight = importance
        
    def get_fisher_norm(self, task_id: str) -> float:
        """获取Fisher信息范数"""
        if task_id not in self.task_fisher:
            return 0.0
            
        total_norm = 0.0
        for name, fisher in self.task_fisher[task_id].items():
            total_norm += fisher.sum().item()
            
        return total_norm


class SI(KnowledgeConsolidation):
    """
    突触重要性 (Synaptic Intelligence)
    
    类似于EWC，但使用不同的重要性度量
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_weight: float = 1000,
        epsilon: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.importance_weight = importance_weight
        self.epsilon = epsilon
        self.device = device or torch.device('cpu')
        
        self.task_parameters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_omega: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_ids: List[str] = []
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.task_parameters[task_id] = {}
            self.task_omega[task_id] = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()
                    self.task_omega[task_id][name] = torch.zeros_like(param.data)
                    
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算SI损失"""
        if task_id is None:
            task_id = self.task_ids[-1] if self.task_ids else None
            
        if task_id is None:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        
        for old_task_id in self.task_ids:
            if old_task_id == task_id:
                continue
                
            for name, param in model.named_parameters():
                if name in self.task_omega[old_task_id]:
                    old_param = self.task_parameters[old_task_id][name]
                    omega = self.task_omega[old_task_id][name]
                    
                    diff = param - old_param
                    # SI损失: sum_i omega_i * (theta_i - theta_i*)^2 / (epsilon + theta_i* - theta_i)^2
                    loss += (omega * diff ** 2 / (self.epsilon + diff ** 2)).sum()
                    
        return self.importance_weight * loss
    
    def update_omega(self, task_id: str, dataloader):
        """更新omega值"""
        if task_id not in self.task_ids:
            return
            
        # 记录训练前后的参数变化
        param_before = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # 一次训练步骤
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                data, target = batch[0], batch[1]
            else:
                data, target = batch, None
                
            data = data.to(self.device)
            self.model.zero_grad()
            
            output = self.model(data)
            if target is not None:
                loss = F.nll_loss(output, target.to(self.device))
            else:
                loss = output.mean()
                
            loss.backward()
            self.modelOptimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.modelOptimizer.step()
            break
            
        # 计算omega
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.task_omega[task_id]:
                diff = param.data - param_before[name]
                omega = torch.abs(param_before[name].grad * diff)
                self.task_omega[task_id][name] += omega
                
    def after_training(self, task_id: str):
        """训练后保存参数"""
        if task_id in self.task_ids:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()


class KnowledgeDistillation(KnowledgeConsolidation):
    """
    知识蒸馏巩固
    
    核心思想：
    - 使用旧模型的输出作为软目标
    - 通过KL散度保持旧知识
    
    优点：
    - 保持模型的输出分布
    - 可以与任意模型一起使用
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device or torch.device('cpu')
        
        # 旧模型缓存
        self.task_models: Dict[str, nn.Module] = {}
        self.task_ids: List[str] = []
        
    def register_task(self, task_id: str):
        """注册新任务（保存旧模型）"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            # 深拷贝当前模型
            old_model = copy.deepcopy(self.model)
            old_model.eval()
            
            for param in old_model.parameters():
                param.requires_grad = False
                
            self.task_models[task_id] = old_model
            
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算蒸馏损失"""
        if task_id is None:
            task_id = self.task_ids[-1] if self.task_ids else None
            
        if task_id is None:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        
        # 蒸馏损失
        for old_task_id in self.task_models:
            if old_task_id == task_id:
                continue
                
            old_model = self.task_models[old_task_id]
            
            # 获取旧模型的软输出
            with torch.no_grad():
                old_outputs = old_model(model.input_buffer)
                
            # 计算KL散度
            soft_targets = F.softmax(old_outputs / self.temperature, dim=1)
            soft_outputs = F.log_softmax(model.output_buffer / self.temperature, dim=1)
            
            distillation_loss = F.kl_div(
                soft_outputs, 
                soft_targets, 
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            loss += distillation_loss
            
        return self.alpha * loss
    
    def distill_from_model(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        input_data: torch.Tensor
    ) -> torch.Tensor:
        """从教师模型蒸馏"""
        teacher_model.eval()
        
        with torch.no_grad():
            teacher_outputs = teacher_model(input_data.to(self.device))
            
        student_outputs = model(input_data.to(self.device))
        
        # 蒸馏损失
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss
    
    def after_training(self, task_id: str):
        """训练后更新模型"""
        pass


class LwF(KnowledgeDistillation):
    """
    学习而不遗忘 (Learning without Forgetting)
    
    使用蒸馏损失保持旧任务性能
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, temperature, alpha, device)
        
        # 每个任务的输出头
        self.task_outputs: Dict[str, nn.Module] = {}
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            
            # 保存当前模型用于蒸馏
            old_model = copy.deepcopy(self.model)
            for param in old_model.parameters():
                param.requires_grad = False
            self.task_models[task_id] = old_model
            
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None,
        inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算LwF损失"""
        if task_id is None or inputs is None:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        
        # 对每个旧任务计算蒸馏损失
        for old_task_id in self.task_models:
            old_model = self.task_models[old_task_id]
            
            with torch.no_grad():
                old_outputs = old_model(inputs)
                
            new_outputs = model(inputs)
            
            # 分类损失（仅针对当前任务的类）
            # 蒸馏损失（针对所有类）
            soft_targets = F.softmax(old_outputs / self.temperature, dim=1)
            soft_outputs = F.log_softmax(new_outputs / self.temperature, dim=1)
            
            distillation_loss = F.kl_div(
                soft_outputs,
                soft_targets,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            loss += (1 - self.alpha) * distillation_loss
            
        return loss


class MetaLearningConsolidation(KnowledgeConsolidation):
    """
    元学习巩固
    
    通过元学习快速适应新任务同时保持旧知识
    """
    
    def __init__(
        self,
        model: nn.Module,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.device = device or torch.device('cpu')
        
        # 元学习器
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
        # 任务适应器
        self.adapters: Dict[str, nn.Module] = {}
        self.task_ids: List[str] = []
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.adapters[task_id] = copy.deepcopy(self.model)
            
            for param in self.adapters[task_id].parameters():
                param.requires_grad = False
                
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算元学习巩固损失"""
        loss = torch.tensor(0.0, device=self.device)
        
        for adapter_task_id, adapter in self.adapters.items():
            if adapter_task_id == task_id:
                continue
                
            # 对齐损失
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), 
                adapter.named_parameters()
            ):
                if name1 == name2:
                    loss += F.mse_loss(param1, param2)
                    
        return loss
    
    def meta_update(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        query_labels: torch.Tensor
    ):
        """执行元更新"""
        self.model.train()
        
        # 内循环：在支持集上适应
        adapted_params = {}
        for name, param in self.model.named_parameters():
            adapted_params[name] = param - self.inner_lr * param.grad
            
        # 外循环：在查询集上评估
        # 使用适应后的参数计算损失
        output = self.model._forward_with_params(query_data, adapted_params)
        loss = F.cross_entropy(output, query_labels)
        
        # 更新元模型
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
    def after_training(self, task_id: str):
        """训练后更新适配器"""
        if task_id in self.task_ids:
            self.adapters[task_id] = copy.deepcopy(self.model)
            for param in self.adapters[task_id].parameters():
                param.requires_grad = False


class Regularization(KnowledgeConsolidation):
    """
    L2正则化巩固
    
    通过权重衰减防止参数大幅变化
    """
    
    def __init__(
        self,
        model: nn.Module,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.weight_decay = weight_decay
        self.device = device or torch.device('cpu')
        
        self.task_parameters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_ids: List[str] = []
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.task_parameters[task_id] = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()
                    
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算L2正则化损失"""
        loss = torch.tensor(0.0, device=self.device)
        
        # 累积所有旧任务的L2损失
        for old_task_id in self.task_parameters:
            for name, param in model.named_parameters():
                if name in self.task_parameters[old_task_id]:
                    old_param = self.task_parameters[old_task_id][name]
                    loss += torch.sum((param - old_param) ** 2)
                    
        return self.weight_decay * loss
    
    def after_training(self, task_id: str):
        """训练后保存参数"""
        if task_id in self.task_parameters:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_parameters[task_id][name] = param.data.clone()


class MAS(KnowledgeConsolidation):
    """
    记忆感知突触 (Memory Aware Synapses)
    
    使用模型输出的幅度作为参数重要性
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_weight: float = 1000,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.importance_weight = importance_weight
        self.device = device or torch.device('cpu')
        
        self.task_omega: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_ids: List[str] = []
        
    def register_task(self, task_id: str):
        """注册新任务"""
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.task_omega[task_id] = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.task_omega[task_id][name] = torch.zeros_like(param.data)
                    
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算MAS损失"""
        loss = torch.tensor(0.0, device=self.device)
        
        for old_task_id in self.task_omega:
            if old_task_id == task_id:
                continue
                
            for name, param in model.named_parameters():
                if name in self.task_omega[old_task_id]:
                    omega = self.task_omega[old_task_id][name]
                    loss += (omega * (param ** 2)).sum()
                    
        return self.importance_weight * loss
    
    def update_omega(
        self,
        task_id: str,
        dataloader: torch.utils.data.DataLoader
    ):
        """更新omega（基于输出幅度）"""
        if task_id not in self.task_omega:
            return
            
        self.model.eval()
        omega_accum = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                omega_accum[name] = torch.zeros_like(param.data)
                
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                data = batch[0]
            else:
                data = batch
                
            data = data.to(self.device)
            
            # 获取模型输出
            with torch.enable_grad():
                output = self.model(data)
                
                # 对输出求平方
                output_squared = output ** 2
                
                # 反向传播到参数
                output_squared.sum().backward(retain_graph=True)
                
                # 累加 |dL/df| * |df/dtheta|
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        omega_accum[name] += torch.abs(param.grad * param)
                        
        # 更新omega
        num_samples = len(dataloader)
        for name in omega_accum:
            self.task_omega[task_id][name] = omega_accum[name] / num_samples
            
    def after_training(self, task_id: str):
        pass


class KnowledgeConsolidation:
    """
    统一知识巩固接口
    
    支持多种巩固方法的切换
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "ewc",
        config: Optional[ConsolidationConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.config = config or ConsolidationConfig(method=method)
        
        if method == "ewc":
            self.consolidator = EWC(model, config.importance_weight if config else 3000, device=self.device)
        elif method == "si":
            self.consolidator = SI(model, config.importance_weight if config else 1000, device=self.device)
        elif method == "distillation":
            self.consolidator = KnowledgeDistillation(model, device=self.device)
        elif method == "lwf":
            self.consolidator = LwF(model, device=self.device)
        elif method == "regularization":
            self.consolidator = Regularization(model, config.weight_decay if config else 0.01, device=self.device)
        elif method == "mas":
            self.consolidator = MAS(model, config.importance_weight if config else 1000, device=self.device)
        elif method == "meta":
            self.consolidator = MetaLearningConsolidation(model, device=self.device)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def register_task(self, task_id: str):
        """注册新任务"""
        self.consolidator.register_task(task_id)
        
    def compute_consolidation_loss(
        self,
        model: nn.Module,
        task_id: str = None
    ) -> torch.Tensor:
        """计算巩固损失"""
        return self.consolidator.compute_consolidation_loss(model, task_id)
    
    def update_fisher(self, dataset: Dataset = None, dataloader = None):
        """更新Fisher信息"""
        if hasattr(self.consolidator, 'update_fisher'):
            self.consolidator.update_fisher(dataset, dataloader)
            
    def update_omega(self, task_id: str, dataloader):
        """更新omega"""
        if hasattr(self.consolidator, 'update_omega'):
            self.consolidator.update_omega(task_id, dataloader)
            
    def after_training(self, task_id: str):
        """训练后处理"""
        self.consolidator.after_training(task_id)
        
    def switch_method(self, method: str, **kwargs):
        """切换巩固方法"""
        if method == "ewc":
            self.consolidator = EWC(self.model, **kwargs)
        elif method == "si":
            self.consolidator = SI(self.model, **kwargs)
        elif method == "distillation":
            self.consolidator = KnowledgeDistillation(self.model, **kwargs)
        elif method == "regularization":
            self.consolidator = Regularization(self.model, **kwargs)
        elif method == "mas":
            self.consolidator = MAS(self.model, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "method": self.config.method,
            "num_tasks": len(getattr(self.consolidator, 'task_ids', []))
        }
        
        if hasattr(self.consolidator, 'task_omega'):
            total_importance = 0
            for task_omega in self.consolidator.task_omega.values():
                for name, omega in task_omega.items():
                    total_importance += omega.sum().item()
            stats["total_importance"] = total_importance
            
        return stats
