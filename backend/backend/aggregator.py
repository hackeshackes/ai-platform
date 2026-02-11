"""
Model aggregator for Federated Learning
Implements FedAvg and other aggregation algorithms
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Aggregator:
    """模型聚合器 - 实现联邦平均(FedAvg)算法"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        """
        初始化聚合器
        
        Args:
            aggregation_method: 聚合方法 ("fedavg", "fedmedian", "fedtrimmedavg")
        """
        self.aggregation_method = aggregation_method
        self.aggregation_history = []
    
    def fedavg(
        self,
        client_weights: List[Dict[str, Any]],
        client_data_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        FedAvg - 联邦平均算法
        
        核心思想: 根据各客户端数据量加权平均模型参数
        
        Args:
            client_weights: 客户端模型权重列表
            client_data_sizes: 各客户端数据量列表
            
        Returns:
            聚合后的全局模型权重
        """
        if not client_weights:
            raise ValueError("No client weights provided for aggregation")
        
        if len(client_weights) == 1:
            logger.info("Single client, returning its weights directly")
            return client_weights[0]
        
        if client_data_sizes is None:
            total_samples = len(client_weights)
            client_data_sizes = [1] * total_samples
        
        total_samples = sum(client_data_sizes)
        
        global_weights = {}
        
        for key in client_weights[0].keys():
            weights_list = []
            
            for i, client_weight in enumerate(client_weights):
                if key in client_weight:
                    weight_value = client_weight[key]
                    weight_array = self._to_numpy_array(weight_value)
                    weights_list.append((weight_array, client_data_sizes[i]))
            
            if weights_list:
                weighted_sum = np.zeros_like(weights_list[0][0])
                total_weight = 0
                
                for weight_array, data_size in weights_list:
                    weight_factor = data_size / total_samples
                    weighted_sum += weight_array * weight_factor
                    total_weight += weight_factor
                
                global_weights[key] = weighted_sum
        
        logger.info(
            f"FedAvg aggregation completed: {len(client_weights)} clients, "
            f"total_samples={total_samples}"
        )
        
        self.aggregation_history.append({
            "method": "fedavg",
            "num_clients": len(client_weights),
            "total_samples": total_samples
        })
        
        return global_weights
    
    def fedmedian(self, client_weights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        FedMedian - 联邦中位数聚合
        
        对每个参数取所有客户端的中位数, 对异常值更鲁棒
        
        Args:
            client_weights: 客户端模型权重列表
            
        Returns:
            聚合后的全局模型权重
        """
        if not client_weights:
            raise ValueError("No client weights provided")
        
        if len(client_weights) == 1:
            return client_weights[0]
        
        global_weights = {}
        
        for key in client_weights[0].keys():
            weights_arrays = []
            
            for client_weight in client_weights:
                if key in client_weight:
                    weight_array = self._to_numpy_array(client_weight[key])
                    weights_arrays.append(weight_array)
            
            if weights_arrays:
                stacked = np.stack(weights_arrays, axis=0)
                median = np.median(stacked, axis=0)
                global_weights[key] = median
        
        logger.info(f"FedMedian aggregation completed: {len(client_weights)} clients")
        
        self.aggregation_history.append({
            "method": "fedmedian",
            "num_clients": len(client_weights)
        })
        
        return global_weights
    
    def fedtrimmedavg(
        self,
        client_weights: List[Dict[str, Any]],
        trim_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        FedTrimmedAvg - 裁剪平均聚合
        
        去除最高和最低的trim_ratio比例值后取平均
        
        Args:
            client_weights: 客户端模型权重列表
            trim_ratio: 裁剪比例 (0.1 表示去掉最高和最低各10%)
            
        Returns:
            聚合后的全局模型权重
        """
        if not client_weights:
            raise ValueError("No client weights provided")
        
        if len(client_weights) == 1:
            return client_weights[0]
        
        if trim_ratio <= 0 or trim_ratio >= 0.5:
            raise ValueError("trim_ratio must be between 0 and 0.5")
        
        global_weights = {}
        
        for key in client_weights[0].keys():
            weights_arrays = []
            
            for client_weight in client_weights:
                if key in client_weight:
                    weight_array = self._to_numpy_array(client_weight[key])
                    weights_arrays.append(weight_array)
            
            if weights_arrays:
                stacked = np.stack(weights_arrays, axis=0)
                n, *shape = stacked.shape
                trimmed = n * trim_ratio
                
                trimmed_low = int(np.ceil(trimmed))
                trimmed_high = n - int(np.ceil(trimmed))
                
                sorted_indices = np.argsort(stacked, axis=0)
                sorted_stacked = np.take_along_axis(stacked, sorted_indices, axis=0)
                
                trimmed_stacked = sorted_stacked[trimmed_low:trimmed_high]
                trimmed_mean = np.mean(trimmed_stacked, axis=0)
                
                global_weights[key] = trimmed_mean
        
        logger.info(
            f"FedTrimmedAvg aggregation completed: {len(client_weights)} clients, "
            f"trim_ratio={trim_ratio}"
        )
        
        self.aggregation_history.append({
            "method": "fedtrimmedavg",
            "num_clients": len(client_weights),
            "trim_ratio": trim_ratio
        })
        
        return global_weights
    
    def weighted_fedavg(
        self,
        client_weights: List[Dict[str, Any]],
        client_scores: List[float]
    ) -> Dict[str, Any]:
        """
        加权FedAvg - 根据客户端性能指标加权
        
        Args:
            client_weights: 客户端模型权重列表
            client_scores: 客户端性能分数列表
            
        Returns:
            聚合后的全局模型权重
        """
        if len(client_weights) != len(client_scores):
            raise ValueError("Number of weights and scores must match")
        
        total_score = sum(client_scores)
        
        if total_score <= 0:
            logger.warning("All client scores are zero, using equal weights")
            return self.fedavg(client_weights)
        
        normalized_scores = [score / total_score for score in client_scores]
        
        global_weights = {}
        
        for key in client_weights[0].keys():
            weighted_sum = np.zeros_like(
                self._to_numpy_array(client_weights[0][key])
            )
            
            for i, client_weight in enumerate(client_weights):
                if key in client_weight:
                    weight_array = self._to_numpy_array(client_weight[key])
                    weighted_sum += weight_array * normalized_scores[i]
            
            global_weights[key] = weighted_sum
        
        logger.info(
            f"Weighted FedAvg aggregation completed: {len(client_weights)} clients"
        )
        
        self.aggregation_history.append({
            "method": "weighted_fedavg",
            "num_clients": len(client_weights)
        })
        
        return global_weights
    
    def _to_numpy_array(self, value: Any) -> np.ndarray:
        """
        将值转换为numpy数组
        
        Args:
            value: 输入值
            
        Returns:
            numpy数组
        """
        if isinstance(value, np.ndarray):
            return value.copy()
        elif isinstance(value, (list, tuple)):
            return np.array(value)
        elif isinstance(value, (int, float)):
            return np.array([value])
        else:
            raise TypeError(f"Unsupported weight type: {type(value)}")
    
    def aggregate(
        self,
        client_weights: List[Dict[str, Any]],
        client_data_sizes: Optional[List[int]] = None,
        client_scores: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        统一的聚合接口
        
        Args:
            client_weights: 客户端模型权重列表
            client_data_sizes: 各客户端数据量列表
            client_scores: 客户端性能分数列表
            
        Returns:
            聚合后的全局模型权重
        """
        if self.aggregation_method == "fedavg":
            if client_scores is not None:
                return self.weighted_fedavg(client_weights, client_scores)
            return self.fedavg(client_weights, client_data_sizes)
        elif self.aggregation_method == "fedmedian":
            return self.fedmedian(client_weights)
        elif self.aggregation_method == "fedtrimmedavg":
            trim_ratio = kwargs.get("trim_ratio", 0.1)
            return self.fedtrimmedavg(client_weights, trim_ratio)
        else:
            logger.warning(f"Unknown method {self.aggregation_method}, using FedAvg")
            return self.fedavg(client_weights, client_data_sizes)
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """
        获取聚合历史
        
        Returns:
            聚合历史记录
        """
        return self.aggregation_history.copy()
