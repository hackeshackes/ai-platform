"""
数据集版本管理 API v1.1
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import hashlib

# 动态导入
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from api.endpoints.auth import get_current_user
    from models import get_db, DatasetVersion
except ImportError:
    get_current_user = None
    get_db = None
    DatasetVersion = None

router = APIRouter()

# Pydantic模型
class VersionCreate(BaseModel):
    dataset_id: int
    version: str
    commit_message: Optional[str] = None
    row_count: int = 0
    file_size: int = 0

class VersionResponse(BaseModel):
    id: int
    dataset_id: int
    version: str
    commit_message: Optional[str]
    row_count: int
    file_size: int
    created_at: datetime
    class Config:
        from_attributes = True

@router.get("/datasets/versions", response_model=List[VersionResponse])
def list_versions(
    dataset_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """获取版本列表"""
    query = db.query(DatasetVersion)
    if dataset_id:
        query = query.filter(DatasetVersion.dataset_id == dataset_id)
    return query.order_by(DatasetVersion.created_at.desc()).offset(skip).limit(limit).all()

@router.get("/datasets/versions/{version_id}", response_model=VersionResponse)
def get_version(
    version_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """获取版本详情"""
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="版本不存在")
    return version

@router.post("/datasets/versions", response_model=VersionResponse)
def create_version(
    version: VersionCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """创建新版本"""
    file_hash = hashlib.sha256(f"{version.dataset_id}{version.version}{datetime.now()}".encode()).hexdigest()
    
    db_version = DatasetVersion(
        **version.dict(),
        file_hash=file_hash,
        created_by=current_user.id
    )
    db.add(db_version)
    db.commit()
    db.refresh(db_version)
    return db_version

@router.delete("/datasets/versions/{version_id}")
def delete_version(
    version_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """删除版本"""
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="版本不存在")
    db.delete(version)
    db.commit()
    return {"message": "版本已删除"}
