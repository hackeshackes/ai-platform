"""
AI Platform 数据模型
SQLite + SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    """用户模型"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default='user')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    projects = relationship("Project", back_populates="owner")

class Project(Base):
    """项目模型"""
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey('users.id'))
    status = Column(String(50), default='active')
    config = Column(JSON)  # 项目配置
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="projects")
    experiments = relationship("Experiment", back_populates="project")
    datasets = relationship("Dataset", back_populates="project")
    models = relationship("Model", back_populates="project")

class Experiment(Base):
    """实验模型"""
    __tablename__ = 'experiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    base_model = Column(String(255))
    task_type = Column(String(50))  # fine_tune, train, distill
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    config = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("Project", back_populates="experiments")
    tasks = relationship("Task", back_populates="experiment")

class Dataset(Base):
    """数据集模型"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    file_path = Column(String(500))
    size = Column(Integer)  # bytes
    format = Column(String(50))  # json, csv, parquet
    version = Column(String(50), default='v1')
    status = Column(String(50), default='uploaded')
    quality_report = Column(JSON)  # 质量报告
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="datasets")

class Model(Base):
    """模型模型"""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    version = Column(String(50))
    framework = Column(String(100))  # pytorch, tensorflow, vLLM
    file_path = Column(String(500))
    size = Column(Integer)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    project = relationship("Project", back_populates="models")
    tasks = relationship("Task", back_populates="model")

class Task(Base):
    """任务模型"""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    model_id = Column(Integer, ForeignKey('models.id'))
    type = Column(String(50))  # training, inference, evaluation
    status = Column(String(50), default='pending')
    config = Column(JSON)
    logs = Column(Text)
    progress = Column(Float, default=0.0)
    gpu_usage = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    project = relationship("Project")
    experiment = relationship("Experiment", back_populates="tasks")
    model = relationship("Model", back_populates="tasks")

# v1.1: 数据集版本控制
class DatasetVersion(Base):
    """数据集版本"""
    __tablename__ = "dataset_versions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    version = Column(String(20), nullable=False)  # v1.0, v1.1
    parent_version_id = Column(Integer, nullable=True)
    commit_message = Column(Text, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA256
    row_count = Column(Integer, default=0)
    file_size = Column(Integer, default=0)  # bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=True)

# v1.1: 角色
class Role(Base):
    """角色"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    permissions = Column(Text, nullable=True)  # 逗号分隔的权限列表
    created_at = Column(DateTime, default=datetime.utcnow)

# v1.1: 项目权限
class ProjectPermission(Base):
    """项目权限"""
    __tablename__ = "project_permissions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# 数据库连接
DATABASE_URL = 'sqlite:///ai_platform.db'
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """初始化数据库"""
    Base.metadata.create_all(engine)

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
