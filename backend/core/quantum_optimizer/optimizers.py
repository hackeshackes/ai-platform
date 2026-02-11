"""
优化器模块
Optimizers Module

提供多种经典优化器用于量子变分算法参数优化
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize, OptimizeResult


class Optimizer(ABC):
    """优化器抽象基类"""
    
    @abstractmethod
    def minimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, int], None]] = None
    ) -> OptimizeResult:
        """最小化目标函数"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取优化器名称"""
        pass


class COBYLA(Optimizer):
    """
    COBYLA优化器 (Constrained Optimization BY Linear Approximation)
    
    适用于无约束优化问题，不需要梯度信息
    """
    
    def __init__(
        self,
        maxiter: int = 1000,
        tol: float = 1e-6,
        rhobeg: float = 0.5,
        rhoend: float = 1e-6
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.rhobeg = rhobeg
        self.rhoend = rhoend
    
    def minimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, int], None]] = None
    ) -> OptimizeResult:
        
        def wrapped_callback(params):
            if callback is not None:
                value = objective_func(params)
                callback(params, value, 0)
        
        result = minimize(
            objective_func,
            initial_params,
            method='COBYLA',
            options={
                'maxiter': self.maxiter,
                'rhobeg': self.rhobeg,
                'rhoend': self.rhoend
            }
        )
        return result
    
    def get_name(self) -> str:
        return "COBYLA"


class SPSA(Optimizer):
    """
    SPSA优化器 (Simultaneous Perturbation Stochastic Approximation)
    
    随机梯度估计方法，每次迭代只需要两次函数评估
    适用于大规模参数优化
    """
    
    def __init__(
        self,
        maxiter: int = 1000,
        learning_rate: float = 0.01,
        perturbation: float = 0.1,
        decay_rate: float = 0.602
    ):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.decay_rate = decay_rate
    
    def minimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, int], None]] = None
    ) -> OptimizeResult:
        
        params = initial_params.copy()
        best_params = params.copy()
        best_value = objective_func(params)
        
        # 动态调整参数
        a = self.learning_rate
        c = self.perturbation
        
        for iteration in range(self.maxiter):
            # 随机扰动
            delta = np.random.choice([-1, 1], size=len(params))
            
            # 估计梯度
            params_plus = params + c * delta
            params_minus = params - c * delta
            
            f_plus = objective_func(params_plus)
            f_minus = objective_func(params_minus)
            
            gradient_estimate = (f_plus - f_minus) / (2 * c) * delta
            
            # 更新参数
            ak = a / ((iteration + 1) ** self.decay_rate)
            ck = c / ((iteration + 1) ** (1/6))
            
            params = params - ak * gradient_estimate
            
            # 更新最优解
            current_value = objective_func(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
            
            # 回调
            if callback is not None:
                callback(params, best_value, iteration)
            
            # 动态调整扰动
            c = ck
        
        return OptimizeResult(
            x=best_params,
            fun=best_value,
            nit=self.maxiter,
            success=True
        )
    
    def get_name(self) -> str:
        return "SPSA"


class GradientDescent(Optimizer):
    """
    梯度下降优化器
    
    使用数值梯度进行参数优化
    """
    
    def __init__(
        self,
        maxiter: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        tol: float = 1e-6
    ):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tol = tol
    
    def minimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, int], None]] = None
    ) -> OptimizeResult:
        
        params = initial_params.copy()
        velocity = np.zeros_like(params)
        best_params = params.copy()
        best_value = objective_func(params)
        
        for iteration in range(self.maxiter):
            # 数值梯度计算
            gradient = self._compute_gradient(objective_func, params)
            
            # 动量更新
            velocity = self.momentum * velocity - self.learning_rate * gradient
            
            # 参数更新
            params = params + velocity
            
            # 约束参数到合理范围
            params = np.clip(params, -2*np.pi, 2*np.pi)
            
            # 更新最优解
            current_value = objective_func(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
            
            # 收敛检查
            if iteration > 0 and abs(best_value - previous_value) < self.tol:
                break
            
            previous_value = best_value
            
            # 回调
            if callback is not None:
                callback(best_params, best_value, iteration)
        
        return OptimizeResult(
            x=best_params,
            fun=best_value,
            nit=iteration + 1,
            success=True
        )
    
    def _compute_gradient(
        self,
        func: Callable[[np.ndarray], float],
        params: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """数值梯度计算"""
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            gradient[i] = (func(params_plus) - func(params_minus)) / (2 * eps)
        return gradient
    
    def get_name(self) -> str:
        return "GradientDescent"


class NaturalGradient(Optimizer):
    """
    自然梯度优化器
    
    使用Fisher信息矩阵进行参数空间的自然梯度下降
    收敛速度更快
    """
    
    def __init__(
        self,
        maxiter: int = 1000,
        learning_rate: float = 0.1,
        tol: float = 1e-6,
        epsilon: float = 1e-8
    ):
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.tol = tol
        self.epsilon = epsilon
    
    def minimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        callback: Optional[Callable[[np.ndarray, float, int], None]] = None
    ) -> OptimizeResult:
        
        params = initial_params.copy()
        best_params = params.copy()
        best_value = objective_func(params)
        
        for iteration in range(self.maxiter):
            # 计算标准梯度
            gradient = self._compute_gradient(objective_func, params)
            
            # 近似Fisher信息矩阵
            fisher_matrix = self._compute_fisher(
                objective_func, params, gradient
            )
            
            # 自然梯度更新
            natural_grad = self._solve_linear_system(
                fisher_matrix, gradient
            )
            
            params = params - self.learning_rate * natural_grad
            
            # 约束参数
            params = np.clip(params, -2*np.pi, 2*np.pi)
            
            # 更新最优解
            current_value = objective_func(params)
            if current_value < best_value:
                best_value = current_value
                best_params = params.copy()
            
            # 收敛检查
            if np.linalg.norm(gradient) < self.tol:
                break
            
            # 回调
            if callback is not None:
                callback(best_params, best_value, iteration)
        
        return OptimizeResult(
            x=best_params,
            fun=best_value,
            nit=iteration + 1,
            success=True
        )
    
    def _compute_gradient(
        self,
        func: Callable[[np.ndarray], float],
        params: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """数值梯度计算"""
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            gradient[i] = (func(params_plus) - func(params_minus)) / (2 * eps)
        return gradient
    
    def _compute_fisher(
        self,
        func: Callable[[np.ndarray], float],
        params: np.ndarray,
        gradient: np.ndarray,
        num_samples: int = 10
    ) -> np.ndarray:
        """Fisher信息矩阵近似"""
        n = len(params)
        fisher = np.eye(n) * self.epsilon
        
        # 使用梯度的外积近似Fisher矩阵
        fisher += np.outer(gradient, gradient)
        
        # 添加正则化
        fisher += np.eye(n) * 0.01
        
        return fisher
    
    def _solve_linear_system(
        self,
        matrix: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        """求解线性系统"""
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, vector, rcond=None)[0]
    
    def get_name(self) -> str:
        return "NaturalGradient"


class OptimizerFactory:
    """优化器工厂类"""
    
    @staticmethod
    def create_optimizer(
        name: str,
        **kwargs
    ) -> Optimizer:
        """创建优化器实例"""
        optimizers = {
            "cobyla": COBYLA,
            "spsa": SPSA,
            "gradient_descent": GradientDescent,
            "natural_gradient": NaturalGradient
        }
        
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        
        return optimizers[name.lower()](**kwargs)
    
    @staticmethod
    def get_available_optimizers() -> list:
        """获取可用的优化器列表"""
        return ["cobyla", "spsa", "gradient_descent", "natural_gradient"]
