"""Dataset model"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    description = Column(Text)
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    # 数据信息
    data_type = Column(String(50))  # text, image, audio, video
    format = Column(String(50))  # jsonl, csv, parquet, etc.
    size = Column(Integer)  # 数据大小(bytes)
    row_count = Column(Integer)  # 数据行数
    
    # 存储路径
    storage_path = Column(String(500))  # S3路径或本地路径
    schema = Column(JSON)  # 数据Schema
    
    # 版本控制
    version = Column(Integer, default=1)
    parent_id = Column(Integer)  # 父版本ID
    
    # 标注信息
    annotation_status = Column(String(50), default="none")  # none, in_progress, completed
    annotation_project_id = Column(Integer)  # Label Studio项目ID
    
    # 统计
    stats = Column(JSON)  # 数据统计信息
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="datasets")
