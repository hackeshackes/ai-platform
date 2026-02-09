"""Project model"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True, nullable=False)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String(50), default="active")  # active, archived
    settings = Column(JSON)  # 项目配置
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    experiments = relationship("Experiment", back_populates="project")
    datasets = relationship("Dataset", back_populates="project")
    models = relationship("Model", back_populates="project")
