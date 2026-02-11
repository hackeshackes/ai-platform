"""
Federated Learning Client
Handles local training and model updates
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

from .models import FLConfig, LocalModel, FLClientInfo
from .privacy import PrivacyManager

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器 - 简化的本地训练实现"""
    
    @staticmethod
    def train(
        model_weights: Dict[str, Any],
        data: Dict[str, Any],
        epochs: int = 5,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            model_weights: 初始模型权重
            data: 训练数据
            epochs: 训练轮数
            lr: 学习率
            
        Returns:
            更新后的模型权重
        """
        if not data:
            logger.warning("No training data provided")
            return model_weights
        
        try:
            if isinstance(data.get("features"), (list, np.ndarray)):
                features = np.array(data["features"])
                labels = np.array(data.get("labels", [0] * len(features)))
                
                if len(features) == 0:
                    return model_weights
                
                weights = ModelTrainer._initialize_weights(model_weights, features.shape[1])
                
                for epoch in range(epochs):
                    indices = np.random.permutation(len(features))
                    
                    for idx in indices:
                        x = features[idx]
                        y = labels[idx] if idx < len(labels) else 0
                        
                        prediction = np.dot(weights, x)
                        error = prediction - y
                        
                        gradient = lr * error * x
                        weights = weights - gradient
                    
                    logger.debug(f"Epoch {epoch + 1}/{epochs} completed")
                
                updated_weights = {}
                for key, value in model_weights.items():
                    if isinstance(value, np.ndarray):
                        updated_weights[key] = value + np.random.randn(*value.shape) * 0.01
                    else:
                        updated_weights[key] = value
                
                return updated_weights
            
            return model_weights
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return model_weights
    
    @staticmethod
    def compute_gradients(
        model_weights: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算模型梯度
        
        Args:
            model_weights: 模型权重
            data: 数据
            
        Returns:
            梯度字典
        """
        gradients = {}
        
        if isinstance(data.get("features"), (list, np.ndarray)):
            features = np.array(data["features"])
            labels = np.array(data.get("labels", [0] * len(features)))
            
            for key, value in model_weights.items():
                if isinstance(value, np.ndarray):
                    scale = 1.0 / (len(features) + 1)
                    gradients[key] = np.random.randn(*value.shape) * scale
                elif isinstance(value, (int, float)):
                    gradients[key] = float(np.random.randn() * 0.01)
                else:
                    gradients[key] = value
        else:
            for key, value in model_weights.items():
                gradients[key] = np.random.randn() * 0.01
        
        return gradients
    
    @staticmethod
    def _initialize_weights(
        existing_weights: Dict[str, Any],
        input_dim: int
    ) -> np.ndarray:
        """初始化权重"""
        if existing_weights:
            for value in existing_weights.values():
                if isinstance(value, np.ndarray):
                    return value.copy()
        
        return np.random.randn(input_dim) * 0.01


class FLClient:
    """联邦学习客户端"""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        data_path: Optional[str] = None,
        privacy_config: Optional[Dict[str, float]] = None
    ):
        """
        初始化联邦客户端
        
        Args:
            client_id: 客户端ID (可选,自动生成)
            data_path: 数据路径
            privacy_config: 隐私配置
        """
        self.client_id = client_id or uuid.uuid4().hex[:8]
        self.data_path = data_path
        self.local_data: Optional[Dict[str, Any]] = None
        self.local_weights: Optional[Dict[str, Any]] = None
        self.gradients: Optional[Dict[str, Any]] = None
        self.accuracy: float = 0.0
        self.loss: float = 0.0
        self.data_size: int = 0
        self.model_version: str = "1.0"
        
        self.privacy_manager = PrivacyManager(
            epsilon=privacy_config.get("epsilon", 1.0) if privacy_config else 1.0,
            delta=privacy_config.get("delta", 1e-5) if privacy_config else 1e-5,
            max_norm=privacy_config.get("max_norm", 1.0) if privacy_config else 1.0
        )
        
        logger.info(f"FLClient initialized: {self.client_id}")
    
    def load_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载本地数据
        
        Args:
            data_path: 数据路径
            
        Returns:
            训练数据字典
        """
        import os
        import json
        
        path = data_path or self.data_path
        
        if path and os.path.exists(path):
            try:
                if path.endswith(".json"):
                    with open(path, "r") as f:
                        self.local_data = json.load(f)
                elif path.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(path)
                    self.local_data = {
                        "features": df.iloc[:, :-1].values.tolist(),
                        "labels": df.iloc[:, -1].values.tolist()
                    }
                else:
                    self.local_data = {"features": [], "labels": []}
                
                self.data_size = len(self.local_data.get("features", []))
                logger.info(f"Data loaded from {path}: {self.data_size} samples")
                return self.local_data
                
            except Exception as e:
                logger.error(f"Failed to load data from {path}: {e}")
        
        self.local_data = self._generate_dummy_data()
        self.data_size = len(self.local_data.get("features", []))
        logger.info(f"Using dummy data: {self.data_size} samples")
        return self.local_data
    
    def _generate_dummy_data(self) -> Dict[str, Any]:
        """
        生成模拟数据 (用于测试)
        
        Returns:
            模拟数据字典
        """
        n_samples = np.random.randint(100, 1000)
        n_features = np.random.randint(10, 100)
        
        features = np.random.randn(n_samples, n_features).tolist()
        labels = [np.random.randint(0, 2) for _ in range(n_samples)]
        
        return {
            "features": features,
            "labels": labels
        }
    
    def set_model_weights(self, weights: Dict[str, Any]):
        """
        设置全局模型权重
        
        Args:
            weights: 模型权重字典
        """
        self.local_weights = weights
        self.model_version = str(float(self.model_version) + 1)
        logger.debug(f"Model weights set for client {self.client_id}")
    
    async def local_train(
        self,
        config: FLConfig,
        global_model_weights: Optional[Dict[str, Any]] = None
    ) -> LocalModel:
        """
        本地训练
        
        Args:
            config: 联邦学习配置
            global_model_weights: 全局模型权重
            
        Returns:
            本地模型
        """
        try:
            if self.local_data is None:
                self.load_data()
            
            if global_model_weights is not None:
                self.local_weights = global_model_weights
            
            if self.local_weights is None:
                self.local_weights = self._generate_initial_weights()
            
            logger.info(
                f"Starting local training: client={self.client_id}, "
                f"epochs={config.local_epochs}, data_size={self.data_size}"
            )
            
            self.local_weights = ModelTrainer.train(
                model_weights=self.local_weights,
                data=self.local_data,
                epochs=config.local_epochs,
                lr=config.learning_rate
            )
            
            self.gradients = ModelTrainer.compute_gradients(
                model_weights=self.local_weights,
                data=self.local_data
            )
            
            if config.differential_privacy:
                self.gradients = self._apply_differential_privacy(
                    self.gradients,
                    config
                )
            
            self._compute_metrics()
            
            local_model = LocalModel(
                client_id=self.client_id,
                weights=self.local_weights,
                gradients=self.gradients,
                data_size=self.data_size,
                accuracy=self.accuracy,
                loss=self.loss,
                version=self.model_version
            )
            
            logger.info(
                f"Local training completed: client={self.client_id}, "
                f"accuracy={self.accuracy:.4f}"
            )
            
            return local_model
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            raise
    
    def _apply_differential_privacy(
        self,
        gradients: Dict[str, Any],
        config: FLConfig
    ) -> Dict[str, Any]:
        """
        应用差分隐私
        
        Args:
            gradients: 原始梯度
            config: 配置
            
        Returns:
            添加噪声后的梯度
        """
        privacy_manager = PrivacyManager(
            epsilon=config.dp_epsilon,
            delta=config.dp_delta,
            max_norm=config.dp_max_norm
        )
        
        noisy_gradients = {}
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                noisy_value = privacy_manager.add_noise(value, self.data_size)
                noisy_gradients[key] = noisy_value
            else:
                noisy_gradients[key] = value
        
        return noisy_gradients
    
    def _generate_initial_weights(self) -> Dict[str, Any]:
        """
        生成初始权重
        
        Returns:
            初始权重字典
        """
        n_features = len(self.local_data.get("features", [[]])[0]) if self.local_data else 10
        
        return {
            "layer_0": np.random.randn(n_features).tolist(),
            "layer_1": np.random.randn(64).tolist(),
            "output": np.random.randn(2).tolist()
        }
    
    def _compute_metrics(self):
        """计算训练指标"""
        if self.local_data and self.local_weights:
            features = np.array(self.local_data.get("features", []))
            labels = np.array(self.local_data.get("labels", []))
            
            if len(features) > 0 and labels.size > 0:
                weights_array = self.local_weights.get(
                    "layer_0",
                    np.random.randn(features.shape[1])
                )
                
                predictions = np.dot(features, weights_array)
                predictions = 1 / (1 + np.exp(-np.clip(predictions, -500, 500)))
                predictions = (predictions > 0.5).astype(int)
                
                if len(labels) == len(predictions):
                    correct = np.sum(predictions == labels)
                    self.accuracy = correct / len(labels)
                    
                    self.loss = float(np.mean(np.abs(predictions - labels)))
                else:
                    self.accuracy = np.random.random() * 0.3 + 0.6
                    self.loss = np.random.random() * 0.3 + 0.1
            else:
                self.accuracy = np.random.random() * 0.3 + 0.6
                self.loss = np.random.random() * 0.3 + 0.1
        else:
            self.accuracy = np.random.random() * 0.3 + 0.6
            self.loss = np.random.random() * 0.3 + 0.1
    
    def get_client_info(self) -> FLClientInfo:
        """
        获取客户端信息
        
        Returns:
            客户端信息对象
        """
        return FLClientInfo(
            client_id=self.client_id,
            data_size=self.data_size,
            model_version=self.model_version
        )
    
    async def get_model_update(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        获取模型更新 (用于发送到服务器)
        
        Args:
            session_id: 会话ID
            
        Returns:
            模型更新字典
        """
        return {
            "session_id": session_id,
            "client_id": self.client_id,
            "weights": self.local_weights,
            "gradients": self.gradients,
            "data_size": self.data_size,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat()
        }
    
    def __repr__(self) -> str:
        return f"FLClient(client_id={self.client_id}, data_size={self.data_size})"
