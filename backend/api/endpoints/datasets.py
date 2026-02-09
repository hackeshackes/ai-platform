"""Dataset management endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid
import os

router = APIRouter()

# 模拟数据库
fake_datasets_db = {
    1: {
        "id": 1,
        "name": "alpaca-zh",
        "description": "中文Alpaca指令数据集",
        "project_id": 1,
        "file_path": "/data/alpaca-zh.jsonl",
        "size": 52428800,  # 50MB
        "format": "jsonl",
        "version": "v1.0",
        "row_count": 52000,
        "status": "ready",
        "quality_report": {
            "null_ratio": 0.02,
            "avg_length": 256,
            "duplicates": 15
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    },
    2: {
        "id": 2,
        "name": "belle-zh",
        "description": "Belle中文对话数据集",
        "project_id": 1,
        "file_path": "/data/belle-zh.jsonl",
        "size": 104857600,  # 100MB
        "format": "jsonl",
        "version": "v2.0",
        "row_count": 100000,
        "status": "ready",
        "quality_report": {
            "null_ratio": 0.01,
            "avg_length": 512,
            "duplicates": 0
        },
        "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "updated_at": (datetime.utcnow() - timedelta(days=1)).isoformat()
    }
}

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    project_id: int
    format: str = "jsonl"

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

@router.get("")
async def list_datasets(project_id: Optional[int] = None, skip: int = 0, limit: int = 50):
    """获取数据集列表"""
    datasets = list(fake_datasets_db.values())
    
    if project_id:
        datasets = [d for d in datasets if d["project_id"] == project_id]
    
    # 添加格式化大小
    for d in datasets:
        d["size_formatted"] = format_size(d["size"])
    
    return {
        "total": len(datasets),
        "datasets": datasets[skip:skip+limit]
    }

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int):
    """获取数据集详情"""
    if dataset_id not in fake_datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = fake_datasets_db[dataset_id]
    dataset["size_formatted"] = format_size(dataset["size"])
    return dataset

@router.post("", response_model=dict, status_code=201)
async def create_dataset(data: DatasetCreate):
    """创建数据集记录"""
    dataset_id = len(fake_datasets_db) + 1
    
    new_dataset = {
        "id": dataset_id,
        "name": data.name,
        "description": data.description or "",
        "project_id": data.project_id,
        "file_path": f"/data/{data.name}.{data.format}",
        "size": 0,
        "format": data.format,
        "version": "v1.0",
        "row_count": 0,
        "status": "pending",
        "quality_report": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    fake_datasets_db[dataset_id] = new_dataset
    return new_dataset

@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: int):
    """删除数据集"""
    if dataset_id not in fake_datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    del fake_datasets_db[dataset_id]
    return None

@router.post("/{dataset_id}/upload")
async def upload_dataset(dataset_id: int, file: UploadFile = File(...)):
    """上传数据集文件"""
    if dataset_id not in fake_datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # 模拟上传
    fake_datasets_db[dataset_id]["status"] = "ready"
    fake_datasets_db[dataset_id]["size"] = 1024 * 1024  # 1MB mock
    fake_datasets_db[dataset_id]["row_count"] = 1000
    fake_datasets_db[dataset_id]["quality_report"] = {
        "null_ratio": 0.01,
        "avg_length": 256,
        "duplicates": 0
    }
    
    return {"message": "Upload successful", "dataset": fake_datasets_db[dataset_id]}
