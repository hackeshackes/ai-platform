"""Workflow model - 工作流管理"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    description = Column(Text)
    
    # 工作流定义
    definition = Column(JSON, nullable=False)  # 工作流DSL定义
    nodes = Column(JSON)  # 节点列表
    edges = Column(JSON)  # 连接关系
    
    # 状态
    status = Column(String(50), default="draft")  # draft, active, archived
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
