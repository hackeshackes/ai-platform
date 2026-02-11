"""
记忆回放系统 (Memory Replay System)

提供多种记忆回放策略：
- Experience Replay: 经验回放
- Importance Sampling: 重要性采样
- Generative Replay: 生成回放
- Compressed Replay: 压缩回放
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import random
from abc import ABC, abstractmethod


@dataclass
class Experience:
    """经验样本"""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    task_id: str
    importance: float = 1.0
    timestamp: int = 0
    
    def __post_init__(self):
        self.timestamp = self.timestamp or torch.randint(0, 1000000, ()).item()


class ReplayBuffer(ABC):
    """回放缓冲区抽象基类"""
    
    @abstractmethod
    def store(self, data: Any, task_id: str = "default") -> int:
        """存储经验"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样批次"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回缓冲区大小"""
        pass
    
    @abstractmethod
    def clear(self):
        """清空缓冲区"""
        pass


class ExperienceReplay(ReplayBuffer):
    """
    标准经验回放
    
    特点：
    - 随机采样
    - 固定大小的循环缓冲区
    - 简单高效
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        device: Optional[torch.device] = None,
        store_states: bool = True
    ):
        self.capacity = capacity
        self.device = device or torch.device('cpu')
        self.store_states = store_states
        
        # 存储数据结构
        self.states: deque = deque(maxlen=capacity)
        self.actions: deque = deque(maxlen=capacity)
        self.rewards: deque = deque(maxlen=capacity)
        self.next_states: deque = deque(maxlen=capacity)
        self.dones: deque = deque(maxlen=capacity)
        self.task_ids: deque = deque(maxlen=capacity)
        self.importances: deque = deque(maxlen=capacity)
        
        self._position = 0
        
    def store(
        self,
        data: Union[Dataset, List[Tuple], torch.Tensor, Dict],
        task_id: str = "default"
    ) -> int:
        """
        存储经验
        
        Args:
            data: 经验数据 (Dataset, List of tuples, Tensor, or Dict)
            task_id: 任务ID
            
        Returns:
            存储的样本数
        """
        samples = self._extract_samples(data)
        count = 0
        
        for sample in samples:
            if len(sample) >= 2:
                state, action = sample[0], sample[1]
                reward = sample[2] if len(sample) > 2 else 0.0
                next_state = sample[3] if len(sample) > 3 else None
                done = sample[4] if len(sample) > 4 else False
                
                self._add_single(state, action, reward, next_state, done, task_id)
                count += 1
                
        return count
    
    def _extract_samples(self, data: Any) -> List[Any]:
        """从不同格式提取样本"""
        if isinstance(data, Dataset):
            return list(data)
        elif isinstance(data, torch.Tensor):
            return [(data[i],) for i in range(len(data))]
        elif isinstance(data, dict):
            if 'features' in data and 'labels' in data:
                features = data['features']
                labels = data['labels']
                return list(zip(features, labels))
            else:
                return [(data,)]
        elif isinstance(data, (list, tuple)):
            return data if isinstance(data[0], (list, tuple)) else [data]
        else:
            return []
    
    def _add_single(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        task_id: str
    ):
        """添加单个经验"""
        # 转换为张量并移动到设备
        state_t = self._to_tensor(state).to(self.device)
        action_t = self._to_tensor(action).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        if next_state is not None:
            next_state_t = self._to_tensor(next_state).to(self.device)
        else:
            next_state_t = torch.zeros_like(state_t)
            
        done_t = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        if len(self.states) < self.capacity:
            self.states.append(state_t)
            self.actions.append(action_t)
            self.rewards.append(reward_t)
            self.next_states.append(next_state_t)
            self.dones.append(done_t)
            self.task_ids.append(task_id)
            self.importances.append(torch.tensor(1.0, device=self.device))
        else:
            self.states[self._position] = state_t
            self.actions[self._position] = action_t
            self.rewards[self._position] = reward_t
            self.next_states[self._position] = next_state_t
            self.dones[self._position] = done_t
            self.task_ids[self._position] = task_id
            self.importances[self._position] = torch.tensor(1.0, device=self.device)
            
        self._position = (self._position + 1) % self.capacity
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """转换为张量"""
        if isinstance(data, torch.Tensor):
            return data.clone()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            return torch.tensor(data, dtype=torch.float32)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采样"""
        if len(self) == 0:
            return torch.tensor([]), torch.tensor([])
            
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        
        states = torch.stack([self.states[i] for i in indices])
        targets = torch.stack([self.actions[i] for i in indices])
        
        return states, targets
    
    def sample_with_rewards(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样并返回奖励信息"""
        if len(self) == 0:
            empty = torch.tensor([])
            return empty, empty, empty, empty, empty
            
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        
        states = torch.stack([self.states[i] for i in indices])
        actions = torch.stack([self.actions[i] for i in indices])
        rewards = torch.stack([self.rewards[i] for i in indices])
        next_states = torch.stack([self.next_states[i] for i in indices])
        dones = torch.stack([self.dones[i] for i in indices])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.states)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.task_ids.clear()
        self.importances.clear()
        self._position = 0
    
    def get_all_data(self, task_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取所有数据"""
        if task_id is None:
            states = torch.stack(list(self.states))
            targets = torch.stack(list(self.actions))
        else:
            indices = [i for i, t in enumerate(self.task_ids) if t == task_id]
            states = torch.stack([self.states[i] for i in indices])
            targets = torch.stack([self.actions[i] for i in indices])
            
        return states, targets


class ImportanceSamplingReplay(ReplayBuffer):
    """
    优先级经验回放 (Prioritized Experience Replay)
    
    特点：
    - 优先级采样
    - TD误差作为优先级指标
    - importance sampling weights 减少偏差
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        device: Optional[torch.device] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.001,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.device = device or torch.device('cpu')
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样指数
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        
        # 优先级树 (简化实现使用排序)
        self.priorities: deque = deque(maxlen=capacity)
        
        # 存储数据
        self.states: deque = deque(maxlen=capacity)
        self.actions: deque = deque(maxlen=capacity)
        self.rewards: deque = deque(maxlen=capacity)
        self.next_states: deque = deque(maxlen=capacity)
        self.dones: deque = deque(maxlen=capacity)
        self.task_ids: deque = deque(maxlen=capacity)
        
        self._position = 0
        
    def store(
        self,
        data: Union[Dataset, List[Tuple], torch.Tensor],
        task_id: str = "default",
        priorities: Optional[List[float]] = None
    ) -> int:
        """存储经验"""
        samples = self._extract_samples(data)
        count = 0
        
        for i, sample in enumerate(samples):
            if len(sample) >= 2:
                state, action = sample[0], sample[1]
                reward = sample[2] if len(sample) > 2 else 0.0
                next_state = sample[3] if len(sample) > 3 else None
                done = sample[4] if len(sample) > 4 else False
                
                # 使用提供的优先级或默认优先级
                priority = priorities[i] if priorities else 1.0
                
                self._add_single(state, action, reward, next_state, done, task_id, priority)
                count += 1
                
        return count
    
    def _add_single(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        task_id: str,
        priority: float
    ):
        """添加单个经验"""
        state_t = self._to_tensor(state).to(self.device)
        action_t = self._to_tensor(action).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        if next_state is not None:
            next_state_t = self._to_tensor(next_state).to(self.device)
        else:
            next_state_t = torch.zeros_like(state_t)
            
        done_t = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # 计算优先级 (TD误差的绝对值 + epsilon)
        priority_val = (abs(reward_t.item()) + self.epsilon) ** self.alpha
        
        if len(self.states) < self.capacity:
            self.states.append(state_t)
            self.actions.append(action_t)
            self.rewards.append(reward_t)
            self.next_states.append(next_state_t)
            self.dones.append(done_t)
            self.task_ids.append(task_id)
            self.priorities.append(priority_val)
        else:
            self.states[self._position] = state_t
            self.actions[self._position] = action_t
            self.rewards[self._position] = reward_t
            self.next_states[self._position] = next_state_t
            self.dones[self._position] = done_t
            self.task_ids[self._position] = task_id
            self.priorities[self._position] = priority_val
            
        self._position = (self._position + 1) % self.capacity
    
    def _to_tensor(self, data: Any) -> torch.Tensor:
        """转换为张量"""
        if isinstance(data, torch.Tensor):
            return data.clone()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            return torch.tensor(data, dtype=torch.float32)
    
    def _extract_samples(self, data: Any) -> List[Any]:
        """从不同格式提取样本"""
        if isinstance(data, Dataset):
            return list(data)
        elif isinstance(data, torch.Tensor):
            return [(data[i],) for i in range(len(data))]
        elif isinstance(data, (list, tuple)):
            return data if isinstance(data[0], (list, tuple)) else [data]
        else:
            return []
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """优先级采样"""
        if len(self) == 0:
            return torch.tensor([]), torch.tensor([])
            
        # 计算采样概率
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        # 采样
        indices = np.random.choice(
            len(self),
            size=min(batch_size, len(self)),
            replace=False,
            p=probs
        )
        
        # 计算重要性采样权重
        weights = (len(self) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        states = torch.stack([self.states[i] for i in indices])
        targets = torch.stack([self.actions[i] for i in indices])
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        return states, targets
    
    def sample_with_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """采样并返回权重和索引"""
        if len(self) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), np.array([])
            
        priorities = np.array(list(self.priorities))
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(
            len(self),
            size=min(batch_size, len(self)),
            replace=False,
            p=probs
        )
        
        weights = (len(self) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        states = torch.stack([self.states[i] for i in indices])
        targets = torch.stack([self.actions[i] for i in indices])
        
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        return states, targets, torch.tensor(weights, dtype=torch.float32, device=self.device), indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
    
    def __len__(self) -> int:
        return len(self.states)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.task_ids.clear()
        self.priorities.clear()
        self._position = 0
        self.beta = 0.4


class GenerativeReplay(ReplayBuffer):
    """
    生成回放
    
    特点：
    - 使用生成模型生成伪样本
    - 节省存储空间
    - 适用于高维数据
    """
    
    def __init__(
        self,
        generator: nn.Module,
        capacity: int = 10000,
        device: Optional[torch.device] = None,
        latent_dim: int = 100,
        sample_interval: int = 100
    ):
        self.generator = generator
        self.capacity = capacity
        self.device = device or torch.device('cpu')
        self.latent_dim = latent_dim
        self.sample_interval = sample_interval
        self.generator.to(self.device)
        
        # 真实样本缓冲区
        self.real_samples: deque = deque(maxlen=capacity // 2)
        
        # 生成样本
        self.generated_samples: deque = deque(maxlen=capacity // 2)
        
        # 生成计数
        self._generation_count = 0
        
    def store(self, data: Any, task_id: str = "default") -> int:
        """存储真实样本并生成伪样本"""
        if isinstance(data, Dataset):
            samples = list(data)[:self.capacity // 2]
        elif isinstance(data, torch.Tensor):
            samples = [(data[i],) for i in range(min(len(data), self.capacity // 2))]
        else:
            samples = []
            
        # 存储真实样本
        for sample in samples:
            if len(self.real_samples) < self.capacity // 2:
                self.real_samples.append(sample)
                
        # 生成伪样本
        self._generate_samples(len(samples))
        
        return len(samples)
    
    def _generate_samples(self, num_samples: int):
        """生成伪样本"""
        self.generator.eval()
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 生成随机潜在向量
                z = torch.randn(1, self.latent_dim, device=self.device)
                
                # 生成样本
                fake_sample = self.generator(z)
                
                if len(self.generated_samples) < self.capacity // 2:
                    self.generated_samples.append((fake_sample.squeeze(0),))
                    
        self._generation_count += 1
        
        self.generator.train()
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样（混合真实和生成样本）"""
        if len(self) == 0:
            return torch.tensor([]), torch.tensor([])
            
        all_samples = list(self.real_samples) + list(self.generated_samples)
        indices = random.sample(range(len(all_samples)), min(batch_size, len(all_samples)))
        
        states = torch.stack([all_samples[i][0] for i in indices])
        # 使用索引作为伪标签
        targets = torch.tensor(indices, dtype=torch.long)
        
        return states, targets
    
    def sample_generated(self, batch_size: int) -> torch.Tensor:
        """仅采样生成样本"""
        if len(self.generated_samples) == 0:
            return torch.tensor([])
            
        indices = random.sample(
            range(len(self.generated_samples)), 
            min(batch_size, len(self.generated_samples))
        )
        
        samples = torch.stack([self.generated_samples[i][0] for i in indices])
        return samples
    
    def __len__(self) -> int:
        return len(self.real_samples) + len(self.generated_samples)
    
    def clear(self):
        """清空缓冲区"""
        self.real_samples.clear()
        self.generated_samples.clear()
        self._generation_count = 0


class CompressedReplay(ReplayBuffer):
    """
    压缩回放
    
    特点：
    - 使用特征压缩存储
    - 显著减少存储空间
    - 适用于资源受限环境
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        device: Optional[torch.device] = None,
        compression_ratio: float = 0.1,
        encoder: Optional[nn.Module] = None
    ):
        self.capacity = capacity
        self.device = device or torch.device('cpu')
        self.compression_ratio = compression_ratio
        self.encoder = encoder
        self.compression_dim: Optional[int] = None
        
        if encoder is not None:
            self.encoder.to(self.device)
            self.encoder.eval()
        
        # 压缩特征存储
        self.compressed_features: deque = deque(maxlen=capacity)
        self.labels: deque = deque(maxlen=capacity)
        self.task_ids: deque = deque(maxlen=capacity)
        self.original_shapes: deque = deque(maxlen=capacity)
        
        self._position = 0
        
    def store(
        self,
        data: Union[Dataset, torch.Tensor, Dict],
        task_id: str = "default"
    ) -> int:
        """压缩并存储"""
        if isinstance(data, Dataset):
            tensors = [data[i][0] for i in range(len(data))]
            labels = [data[i][1] for i in range(len(data))]
        elif isinstance(data, torch.Tensor):
            tensors = [data[i] for i in range(len(data))]
            labels = [torch.tensor(i) for i in range(len(data))]
        elif isinstance(data, dict) and 'features' in data:
            tensors = data['features']
            labels = data.get('labels', list(range(len(tensors))))
        else:
            return 0
            
        count = 0
        for tensor, label in zip(tensors, labels):
            if len(self.compressed_features) < self.capacity:
                # 压缩
                if self.encoder is not None:
                    with torch.no_grad():
                        compressed = self.encoder(tensor.unsqueeze(0).to(self.device))
                        compressed = compressed.squeeze(0).cpu()
                else:
                    # 简单降采样
                    compressed = self._simple_compress(tensor)
                    
                self.compressed_features.append(compressed)
                self.labels.append(label)
                self.task_ids.append(task_id)
                self.original_shapes.append(tensor.shape)
                count += 1
                
        return count
    
    def _simple_compress(self, tensor: torch.Tensor) -> torch.Tensor:
        """简单压缩（降采样）"""
        if self.compression_ratio >= 1.0:
            return tensor
            
        # 使用平均池化压缩
        dim = int(tensor.numel() * self.compression_ratio)
        if dim < 1:
            dim = 1
            
        # 简单的特征选择（保留最重要的维度）
        abs_vals = torch.abs(tensor)
        _, top_indices = abs_vals.topk(dim)
        
        compressed = torch.zeros(dim, dtype=tensor.dtype)
        compressed[torch.arange(dim)] = tensor[top_indices]
        
        return compressed
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样"""
        if len(self) == 0:
            return torch.tensor([]), torch.tensor([])
            
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        
        features = torch.stack([self.compressed_features[i] for i in indices])
        targets = torch.stack([self.labels[i] if isinstance(self.labels[i], torch.Tensor) 
                              else torch.tensor(self.labels[i]) for i in indices])
        
        return features, targets
    
    def decompress(self, index: int) -> torch.Tensor:
        """解压缩单个样本"""
        compressed = self.compressed_features[index]
        original_shape = self.original_shapes[index]
        
        if self.encoder is not None:
            # 使用解码器（如果有）
            with torch.no_grad():
                decompressed = self.decoder(compressed.unsqueeze(0)).squeeze(0)
        else:
            # 简单解压缩（填充零）
            decompressed = torch.zeros(original_shape, dtype=compressed.dtype)
            dim = min(len(compressed), decompressed.numel())
            decompressed[:dim] = compressed[:dim]
            
        return decompressed
    
    def __len__(self) -> int:
        return len(self.compressed_features)
    
    def clear(self):
        """清空缓冲区"""
        self.compressed_features.clear()
        self.labels.clear()
        self.task_ids.clear()
        self.original_shapes.clear()
        self._position = 0


class MemoryReplay:
    """
    统一记忆回放接口
    
    支持多种回放策略的切换
    """
    
    def __init__(
        self,
        strategy: str = "experience",
        capacity: int = 10000,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Args:
            strategy: 回放策略 ("experience", "importance", "generative", "compressed")
            capacity: 缓冲区容量
            device: 计算设备
            **kwargs: 策略特定参数
        """
        self.strategy = strategy
        self.device = device or torch.device('cpu')
        
        if strategy == "experience":
            self.replay_buffer = ExperienceReplay(capacity, device)
        elif strategy == "importance":
            self.replay_buffer = ImportanceSamplingReplay(capacity, device, **kwargs)
        elif strategy == "generative":
            self.replay_buffer = GenerativeReplay(capacity=capacity, device=device, **kwargs)
        elif strategy == "compressed":
            self.replay_buffer = CompressedReplay(capacity=capacity, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def store(
        self,
        data: Union[Dataset, List[Tuple], torch.Tensor, Dict],
        task_id: str = "default",
        **kwargs
    ) -> int:
        """存储经验"""
        return self.replay_buffer.store(data, task_id, **kwargs)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样批次"""
        return self.replay_buffer.sample(batch_size)
    
    def sample_with_weights(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """采样并返回权重"""
        if hasattr(self.replay_buffer, 'sample_with_weights'):
            return self.replay_buffer.sample_with_weights(batch_size)
        else:
            states, targets = self.sample(batch_size)
            weights = torch.ones(len(states))
            return states, targets, weights, np.arange(len(states))
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        if hasattr(self.replay_buffer, 'update_priorities'):
            self.replay_buffer.update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        return len(self.replay_buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.replay_buffer.clear()
    
    def switch_strategy(self, strategy: str, **kwargs):
        """切换回放策略"""
        capacity = len(self.replay_buffer)
        self.strategy = strategy
        
        if strategy == "experience":
            self.replay_buffer = ExperienceReplay(capacity, self.device, **kwargs)
        elif strategy == "importance":
            self.replay_buffer = ImportanceSamplingReplay(capacity, self.device, **kwargs)
        elif strategy == "generative":
            self.replay_buffer = GenerativeReplay(capacity=capacity, device=self.device, **kwargs)
        elif strategy == "compressed":
            self.replay_buffer = CompressedReplay(capacity=capacity, device=self.device, **kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "strategy": self.strategy,
            "buffer_size": len(self),
            "capacity": self.replay_buffer.capacity
        }
        
        if hasattr(self.replay_buffer, 'priorities'):
            priorities = list(self.replay_buffer.priorities)
            stats["priority_stats"] = {
                "mean": np.mean(priorities) if priorities else 0,
                "std": np.std(priorities) if priorities else 0
            }
            
        return stats
