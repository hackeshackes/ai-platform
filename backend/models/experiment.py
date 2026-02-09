"""Experiment model"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    description = Column(Text)
    project_id = Column(Integer, ForeignKey("projects.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # 实验配置
    base_model = Column(String(200))  # 基础模型
    task_type = Column(String(50))  # training, fine-tuning, distillation
    hyperparameters = Column(JSON)  # 超参数配置
    
    # 状态
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    metrics = Column(JSON)  # 评估指标
    
    # 结果
    artifacts_path = Column(String(500))  # 模型保存路径
    logs = Column(Text)  # 训练日志
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="experiments")
    user = relationship("User", back_populates="experiments")
    tasks = relationship("Task", back_populates="experiment")
