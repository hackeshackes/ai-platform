"""Model model - 模型管理"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    description = Column(Text)
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    # 模型信息
    base_model = Column(String(200))  # 基础模型
    model_type = Column(String(50))  # llm, embedding, vision, etc.
    framework = Column(String(50))  # pytorch, tensorflow, etc.
    
    # 版本
    version = Column(Integer, default=1)
    stage = Column(String(50), default="staging")  # staging, production, archived
    
    # 参数
    parameter_size = Column(String(50))  # 7B, 13B, 70B, etc.
    quantization = Column(String(50))  # fp16, int8, int4, etc.
    
    # 存储
    storage_path = Column(String(500))  # 模型文件路径
    size = Column(Integer)  # 模型大小(bytes)
    
    # 性能指标
    metrics = Column(JSON)  # 评估指标
    benchmark_results = Column(JSON)  # 基准测试结果
    
    # 训练信息
    training_dataset_id = Column(Integer)
    training_config = Column(JSON)  # 训练配置
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="models")
