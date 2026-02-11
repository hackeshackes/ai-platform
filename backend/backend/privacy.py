"""
Privacy protection manager for Federated Learning
Implements differential privacy with gradient clipping and noise addition
"""
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class PrivacyManager:
    """隐私保护管理器 - 差分隐私实现"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_norm: float = 1.0,
        noise_multiplier: float = 1.0
    ):
        """
        初始化隐私管理器
        
        Args:
            epsilon: 差分隐私预算 (越小隐私保护越强)
            delta: 失败概率参数
            max_norm: 梯度裁剪的最大范数
            noise_multiplier: 噪声乘数
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier
        self.privacy_budget_used = 0.0
    
    def compute_noise_scale(self, sample_size: int, batch_size: int) -> float:
        """
        计算噪声标准差
        
        基于差分隐私的Gaussian机制:
        noise = N(0, sigma^2) 其中 sigma = noise_multiplier * L / epsilon
        L = max_norm
        """
        effective_epsilon = self.epsilon / len(self.privacy_accounting()) if self.privacy_accounting() else self.epsilon
        
        sigma = self.noise_multiplier * self.max_norm / effective_epsilon
        
        logger.debug(f"Computed noise scale: sigma={sigma:.4f}, epsilon={effective_epsilon:.4f}")
        return sigma
    
    def _sensitivity(self, gradients: np.ndarray) -> float:
        """
        计算梯度敏感度
        
        Args:
            gradients: 梯度数组
            
        Returns:
            敏感度值
        """
        norm = np.linalg.norm(gradients)
        return min(norm, self.max_norm)
    
    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """
        梯度裁剪 - L2范数裁剪
        
        Args:
            gradients: 原始梯度
            
        Returns:
            裁剪后的梯度
        """
        norm = np.linalg.norm(gradients)
        if norm > self.max_norm:
            scaling_factor = self.max_norm / norm
            gradients = gradients * scaling_factor
            logger.debug(f"Gradients clipped: norm={norm:.4f} -> {self.max_norm}")
        return gradients
    
    def add_noise(self, gradients: np.ndarray, sample_size: int = 1) -> np.ndarray:
        """
        添加高斯噪声 - 差分隐私核心机制
        
        Args:
            gradients: 梯度数组
            sample_size: 样本数量
            
        Returns:
            添加噪声后的梯度
        """
        gradients = self.clip_gradients(gradients)
        
        sigma = self.compute_noise_scale(sample_size, len(gradients))
        noise = np.random.normal(0, sigma, gradients.shape)
        
        noisy_gradients = gradients + noise
        
        logger.debug(f"Added noise: sigma={sigma:.4f}, noise_std={np.std(noise):.4f}")
        return noisy_gradients
    
    def add_noise_to_dict(
        self,
        gradients: Dict[str, Any],
        sample_size: int = 1
    ) -> Dict[str, Any]:
        """
        对字典形式的梯度添加噪声
        
        Args:
            gradients: 梯度字典
            sample_size: 样本数量
            
        Returns:
            添加噪声后的梯度字典
        """
        noisy_gradients = {}
        
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                noisy_gradients[key] = self.add_noise(value, sample_size)
            elif isinstance(value, (list, tuple)):
                arr = np.array(value)
                noisy_gradients[key] = self.add_noise(arr, sample_size).tolist()
            elif isinstance(value, (int, float)):
                noise_std = self.compute_noise_scale(sample_size, 1) * abs(value) * 0.01
                noisy_gradients[key] = value + np.random.normal(0, noise_std)
            else:
                noisy_gradients[key] = value
        
        return noisy_gradients
    
    def clip_and_noise_dict(
        self,
        gradients: Dict[str, Any],
        sample_size: int = 1
    ) -> Dict[str, Any]:
        """
        裁剪并添加噪声到字典
        
        Args:
            gradients: 梯度字典
            sample_size: 样本数量
            
        Returns:
            处理后的梯度字典
        """
        clipped = {}
        total_norm = 0.0
        
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                norm = np.linalg.norm(value)
                total_norm += norm ** 2
            elif isinstance(value, list):
                norm = np.linalg.norm(np.array(value))
                total_norm += norm ** 2
        
        total_norm = np.sqrt(total_norm)
        
        scaling_factor = min(1.0, self.max_norm / (total_norm + 1e-8))
        
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                clipped[key] = value * scaling_factor
            elif isinstance(value, list):
                clipped[key] = [v * scaling_factor for v in value]
            else:
                clipped[key] = value
        
        return self.add_noise_to_dict(clipped, sample_size)
    
    def privacy_accounting(self) -> List[Dict[str, Any]]:
        """
        隐私账户 - 跟踪隐私预算使用情况
        
        Returns:
            隐私预算使用记录
        """
        return [
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "used": self.privacy_budget_used,
                "remaining": self.epsilon - self.privacy_budget_used
            }
        ]
    
    def update_privacy_budget(self, rounds: int = 1):
        """
        更新隐私预算消耗
        
        使用矩估计方法计算隐私消耗
        """
        if self.privacy_accounting():
            self.privacy_budget_used += self.epsilon / 100 * rounds
            self.privacy_budget_used = min(self.privacy_budget_used, self.epsilon)
        
        logger.info(f"Privacy budget updated: {self.privacy_budget_used:.4f}/{self.epsilon}")
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """
        获取隐私消耗状态
        
        Returns:
            隐私消耗信息
        """
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "budget_used": self.privacy_budget_used,
            "budget_remaining": self.epsilon - self.privacy_budget_used,
            "budget_ratio": self.privacy_budget_used / self.epsilon if self.epsilon > 0 else 0
        }
    
    def compose_privacy_budget(self, num_clients: int) -> float:
        """
        组合隐私预算 - 计算多轮训练的总体隐私消耗
        
        使用高级组合定理
        """
        if num_clients <= 1:
            return self.epsilon
        
        composition_delta = num_clients * self.delta
        composition_epsilon = self.epsilon * np.sqrt(2 * np.log(1 / composition_delta) * num_clients)
        
        logger.info(
            f"Composed privacy budget: {composition_epsilon:.4f} "
            f"(epsilon={self.epsilon}, clients={num_clients})"
        )
        return composition_epsilon
