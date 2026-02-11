"""
适应引擎 - Adaptation Engine
实现快速参数更新、梯度适配、知识迁移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Union, Callable
from copy import deepcopy
import numpy as np
from collections import defaultdict


class GradientAdapter:
    """
    梯度适配器
    """
    
    def __init__(self, 
                 lr: float = 0.01,
                 grad_clip: float = 1.0,
                 use_second_order: bool = True):
        self.lr = lr
        self.grad_clip = grad_clip
        self.use_second_order = use_second_order
        
    def adapt(self,
             model: nn.Module,
             support_x: torch.Tensor,
             support_y: torch.Tensor,
             device: torch.device) -> nn.Module:
        """
        通过梯度下降适配模型
        
        Args:
            model: 原始模型
            support_x: 支持集输入
            support_y: 支持集标签
            device: 计算设备
        
        Returns:
            适应后的模型
        """
        model = model.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        adapted_model = deepcopy(model)
        adapted_model = adapted_model.to(device)
        
        # 多次梯度更新
        for _ in range(5):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, 
                adapted_model.parameters(), 
                create_graph=self.use_second_order
            )
            
            # 裁剪梯度
            if self.grad_clip > 0:
                grads = tuple(g.clamp(min=-self.grad_clip, max=self.grad_clip) 
                           for g in grads)
            
            # 更新参数
            adapted_params = []
            for param, grad in zip(adapted_model.parameters(), grads):
                if grad is not None:
                    adapted_params.append(param - self.lr * grad)
                else:
                    adapted_params.append(param)
            
            # 应用更新
            with torch.no_grad():
                for i, param in enumerate(adapted_model.parameters()):
                    param.copy_(adapted_params[i])
        
        return adapted_model
    
    def get_parameter_delta(self,
                           model: nn.Module,
                           support_x: torch.Tensor,
                           support_y: torch.Tensor,
                           device: torch.device) -> Dict[str, torch.Tensor]:
        """
        获取参数变化量
        
        Args:
            model: 原始模型
            support_x: 支持集输入
            support_y: 支持集标签
            device: 计算设备
        
        Returns:
            参数变化量字典
        """
        model = model.to(device)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        adapted_model = self.adapt(model, support_x, support_y, device)
        
        deltas = {}
        for name, param in model.named_parameters():
            deltas[name] = adapted_model.state_dict()[name] - param
        
        return deltas


class KnowledgeTransfer:
    """
    知识迁移模块
    """
    
    def __init__(self, 
                 transfer_method: str = "feature",
                 alpha: float = 0.1,
                 temperature: float = 1.0):
        """
        Args:
            transfer_method: 迁移方法（feature, weight, gradient）
            alpha: 知识蒸馏权重
            temperature: 蒸馏温度
        """
        self.transfer_method = transfer_method
        self.alpha = alpha
        self.temperature = temperature
        
    def transfer(self,
                source_model: nn.Module,
                target_model: nn.Module,
                data: torch.Tensor) -> nn.Module:
        """
        从源模型迁移知识到目标模型
        
        Args:
            source_model: 源模型
            target_model: 目标模型
            data: 迁移数据
        
        Returns:
            更新后的目标模型
        """
        source_model.eval()
        target_model.train()
        
        if self.transfer_method == "feature":
            return self._feature_transfer(source_model, target_model, data)
        elif self.transfer_method == "weight":
            return self._weight_transfer(source_model, target_model)
        elif self.transfer_method == "gradient":
            return self._gradient_transfer(source_model, target_model, data)
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")
    
    def _feature_transfer(self,
                         source_model: nn.Module,
                         target_model: nn.Module,
                         data: torch.Tensor) -> nn.Module:
        """特征迁移"""
        with torch.no_grad():
            source_features = self._get_features(source_model, data)
        
        target_features = self._get_features(target_model, data)
        
        # 特征对齐损失
        loss = F.mse_loss(source_features, target_features)
        
        # 反向传播
        target_model.zero_grad()
        loss.backward()
        
        # 冻结源模型，只更新目标模型
        return target_model
    
    def _weight_transfer(self,
                        source_model: nn.Module,
                        target_model: nn.Module) -> nn.Module:
        """权重迁移：线性组合源模型和目标模型权重"""
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()
        
        for name in target_state:
            if name in source_state:
                target_state[name] = (1 - self.alpha) * target_state[name] + \
                                     self.alpha * source_state[name]
        
        target_model.load_state_dict(target_state)
        return target_model
    
    def _gradient_transfer(self,
                          source_model: nn.Module,
                          target_model: nn.Module,
                          data: torch.Tensor) -> nn.Module:
        """梯度迁移"""
        # 计算源模型的梯度
        source_model.eval()
        with torch.no_grad():
            source_logits = source_model(data)
            source_loss = F.cross_entropy(source_logits, source_logits.argmax(dim=-1))
            source_grads = torch.autograd.grad(source_loss, source_model.parameters())
        
        # 将源模型梯度作为目标
        target_model.train()
        target_logits = target_model(data)
        target_loss = F.cross_entropy(target_logits, target_logits.argmax(dim=-1))
        target_grads = torch.autograd.grad(target_loss, target_model.parameters())
        
        # 梯度对齐
        loss = 0
        for tg, sg in zip(target_grads, source_grads):
            if sg is not None:
                loss += F.mse_loss(tg, sg)
        
        target_model.zero_grad()
        loss.backward()
        
        return target_model
    
    def _get_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """提取模型特征"""
        x = model.conv1(data)
        x = model.bn1(x)
        x = F.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.avgpool(x)
        return x.flatten(1)


class FastAdapter:
    """
    快速适配器 - 整合多种适配策略
    """
    
    def __init__(self,
                 strategy: str = "maml",
                 lr: float = 0.01,
                 steps: int = 5,
                 use_early_stopping: bool = True,
                 patience: int = 3,
                 device: Optional[torch.device] = None):
        """
        Args:
            strategy: 适配策略（maml, reptile, fim, latent）
            lr: 学习率
            steps: 适配步数
            use_early_stopping: 是否使用早停
            patience: 早停耐心值
            device: 计算设备
        """
        self.strategy = strategy
        self.lr = lr
        self.steps = steps
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gradient_adapter = GradientAdapter(lr)
        self.knowledge_transfer = KnowledgeTransfer()
        
    def adapt(self,
             model: nn.Module,
             support_x: torch.Tensor,
             support_y: torch.Tensor,
             query_x: Optional[torch.Tensor] = None,
             query_y: Optional[torch.Tensor] = None,
             metadata: Optional[Dict] = None) -> Dict:
        """
        快速适应新任务
        
        Args:
            model: 基础模型
            support_x: 支持集输入
            support_y: 支持集标签
            query_x: 查询集输入（可选，用于验证）
            query_y: 查询集标签（可选）
            metadata: 任务元数据
        
        Returns:
            适应结果字典
        """
        if self.strategy == "maml":
            return self._adapt_maml(model, support_x, support_y, query_x, query_y)
        elif self.strategy == "reptile":
            return self._adapt_reptile(model, support_x, support_y)
        elif self.strategy == "fim":
            return self._adapt_fim(model, support_x, support_y)
        elif self.strategy == "latent":
            return self._adapt_latent(model, support_x, support_y, metadata)
        else:
            return self._adapt_gradient(model, support_x, support_y)
    
    def _adapt_maml(self,
                   model: nn.Module,
                   support_x: torch.Tensor,
                   support_y: torch.Tensor,
                   query_x: Optional[torch.Tensor],
                   query_y: Optional[torch.Tensor]) -> Dict:
        """MAML风格适配"""
        adapted_model = self.gradient_adapter.adapt(
            model, support_x, support_y, self.device
        )
        
        result = {"model": adapted_model}
        
        if query_x is not None:
            adapted_model.eval()
            with torch.no_grad():
                query_x = query_x.to(self.device)
                logits = adapted_model(query_x)
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == query_y.to(self.device)).float().mean().item()
            result["accuracy"] = accuracy
        
        return result
    
    def _adapt_reptile(self,
                      model: nn.Module,
                      support_x: torch.Tensor,
                      support_y: torch.Tensor) -> Dict:
        """Reptile风格适配"""
        model = model.to(self.device)
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        # 保存初始参数
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # 多步SGD
        adapted_model = deepcopy(model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.lr)
        
        for _ in range(self.steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 计算参数更新
        final_params = {name: param for name, param in adapted_model.named_parameters()}
        param_updates = {}
        for name in initial_params:
            param_updates[name] = final_params[name] - initial_params[name]
        
        # 应用元更新
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in param_updates:
                    param.add_(param_updates[name] * 0.1)  # 元学习率
        
        return {"model": model, "updates": param_updates}
    
    def _adapt_fim(self,
                  model: nn.Module,
                  support_x: torch.Tensor,
                  support_y: torch.Tensor) -> Dict:
        """Fisher Information Matrix近似适配"""
        model = model.to(self.device)
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        # 计算Fisher Information
        model.eval()
        with torch.no_grad():
            logits = model(support_x)
            probs = F.softmax(logits, dim=-1)
            targets = support_y
        
        fisher = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            # 简化的Fisher计算
            grad = param.grad
            fisher[name] = (grad ** 2).mean()
        
        # 使用Fisher信息调整学习率
        adapted_model = deepcopy(model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.lr)
        
        for _ in range(self.steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            
            # Fisher加权梯度
            for name, param in adapted_model.named_parameters():
                if name in fisher and param.grad is not None:
                    param.grad = param.grad / (fisher[name] + 1e-8)
            
            optimizer.step()
        
        return {"model": adapted_model, "fisher": fisher}
    
    def _adapt_latent(self,
                     model: nn.Module,
                     support_x: torch.Tensor,
                     support_y: torch.Tensor,
                     metadata: Optional[Dict]) -> Dict:
        """潜在适应 - 基于任务元数据"""
        # 简化实现：使用任务ID作为潜在变量
        task_id = metadata.get("task_id", "unknown") if metadata else "unknown"
        
        # 基于任务ID生成适配参数
        adapted_model = deepcopy(model)
        
        # 冻结大部分参数，只微调最后几层
        for param in adapted_model.parameters():
            param.requires_grad = False
        
        # 找到分类层
        for name, param in adapted_model.named_parameters():
            if "classifier" in name or "fc" in name or "linear" in name:
                param.requires_grad = True
        
        # 快速微调
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adapted_model.parameters()), lr=0.1)
        
        for _ in range(20):  # 更多步数，因为只有少量参数
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return {"model": adapted_model, "task_id": task_id}
    
    def _adapt_gradient(self,
                       model: nn.Module,
                       support_x: torch.Tensor,
                       support_y: torch.Tensor) -> Dict:
        """简单梯度下降适配"""
        model = model.to(self.device)
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        adapted_model = deepcopy(model)
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.lr)
        
        for _ in range(self.steps):
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return {"model": adapted_model}


class AdaptationEngine:
    """
    适应引擎主类
    
    整合梯度适配、知识迁移、快速参数更新
    """
    
    def __init__(self,
                 strategy: str = "auto",
                 device: Optional[torch.device] = None):
        """
        Args:
            strategy: 默认适配策略
            device: 计算设备
        """
        self.strategy = strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.adapter = FastAdapter(strategy=strategy if strategy != "auto" else "maml")
        
    def adapt(self,
             base_model: nn.Module,
             new_task_data: Dict[str, torch.Tensor],
             config: Optional[Dict] = None) -> Dict:
        """
        适应新任务
        
        Args:
            base_model: 基础模型
            new_task_data: 新任务数据（包含support_x, support_y, query_x, query_y）
            config: 配置参数
        
        Returns:
            适应结果
        """
        config = config or {}
        
        strategy = config.get("strategy", self.strategy)
        if strategy == "auto":
            # 根据任务自动选择策略
            strategy = self._select_strategy(new_task_data)
        
        # 更新适配器策略
        self.adapter.strategy = strategy
        self.adapter.device = self.device
        
        support_x = new_task_data["support_x"]
        support_y = new_task_data["support_y"]
        query_x = new_task_data.get("query_x")
        query_y = new_task_data.get("query_y")
        
        result = self.adapter.adapt(
            base_model,
            support_x,
            support_y,
            query_x,
            query_y,
            new_task_data.get("metadata")
        )
        
        return {
            "adapted_model": result.get("model"),
            "accuracy": result.get("accuracy", 0.0),
            "strategy": strategy,
            "task_info": result.get("task_info", {})
        }
    
    def _select_strategy(self, task_data: Dict) -> str:
        """根据任务特点自动选择适配策略"""
        support_size = len(task_data["support_x"])
        
        if support_size <= 5:
            return "latent"  # 极少样本使用潜在适应
        elif support_size <= 20:
            return "maml"   # 少量样本使用MAML
        else:
            return "gradient"  # 更多样本使用简单梯度下降
    
    def batch_adapt(self,
                    base_model: nn.Module,
                    task_batch: List[Dict],
                    config: Optional[Dict] = None) -> List[Dict]:
        """
        批量适应多个任务
        
        Args:
            base_model: 基础模型
            task_batch: 任务批次
            config: 配置参数
        
        Returns:
            适应结果列表
        """
        results = []
        
        for task_data in task_batch:
            result = self.adapt(base_model, task_data, config)
            results.append(result)
        
        return results
    
    def evaluate_adaptation(self,
                           adapted_model: nn.Module,
                           test_data: Dict[str, torch.Tensor]) -> Dict:
        """
        评估适应后的模型
        
        Args:
            adapted_model: 适应后的模型
            test_data: 测试数据
        
        Returns:
            评估结果
        """
        adapted_model.eval()
        
        test_x = test_data["test_x"].to(self.device)
        test_y = test_data["test_y"]
        
        with torch.no_grad():
            logits = adapted_model(test_x)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == test_y.to(self.device)).float().mean().item()
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for i in torch.unique(test_y):
            mask = test_y == i
            class_acc = (predictions[mask.to(self.device)] == i.to(self.device)).float().mean().item()
            class_accuracies[f"class_{i.item()}"] = class_acc
        
        return {
            "overall_accuracy": accuracy,
            "class_accuracies": class_accuracies
        }
