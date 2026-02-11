"""
Pydantic models for Federated Learning Platform
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """联邦训练会话状态"""
    PENDING = "pending"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class FLConfig(BaseModel):
    """联邦学习配置"""
    model_name: str = "default_model"
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    min_clients: int = 2
    max_clients: int = 10
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    rounds: int = 10


class FLClientInfo(BaseModel):
    """联邦学习客户端信息"""
    client_id: str
    data_size: int
    model_version: str = "1.0"


class FLSession(BaseModel):
    """联邦训练会话"""
    id: str
    config: FLConfig
    status: SessionStatus
    participants: List[FLClientInfo] = []
    global_model_version: str = "1.0"
    current_round: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class LocalModel(BaseModel):
    """本地模型"""
    client_id: str
    weights: Dict[str, Any]
    gradients: Dict[str, Any] = {}
    data_size: int
    accuracy: float = 0.0
    loss: float = 0.0
    version: str = "1.0"


class GlobalModel(BaseModel):
    """全局模型"""
    session_id: str
    weights: Dict[str, Any]
    version: str
    round_number: int
    created_at: datetime


class TrainingResult(BaseModel):
    """训练结果"""
    session_id: str
    client_id: str
    local_model: LocalModel
    success: bool
    error_message: Optional[str] = None


class AggregationResult(BaseModel):
    """聚合结果"""
    session_id: str
    global_model: GlobalModel
    participating_clients: int
    aggregated_at: datetime
    metrics: Dict[str, float] = {}
