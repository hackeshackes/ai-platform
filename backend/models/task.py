"""Task model - 任务调度"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)  # training, inference, data_processing, etc.
    project_id = Column(Integer, ForeignKey("projects.id"))
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # 配置
    config = Column(JSON, nullable=False)  # 任务配置
    command = Column(Text)  # 执行命令
    environment = Column(JSON)  # 环境变量
    
    # 状态
    status = Column(String(50), default="pending")  # pending, queued, running, completed, failed
    priority = Column(Integer, default=0)  # 优先级
    
    # 结果
    result = Column(JSON)  # 执行结果
    error_message = Column(Text)  # 错误信息
    logs = Column(Text)  # 执行日志
    
    # 资源
    gpu_required = Column(Integer, default=0)  # 所需GPU数量
    memory_required = Column(String(50))  # 所需内存
    
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project")
    experiment = relationship("Experiment", back_populates="tasks")
    user = relationship("User", back_populates="tasks")
