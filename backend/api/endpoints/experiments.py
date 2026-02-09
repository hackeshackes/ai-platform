"""Experiment endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter()

# 模拟数据库
fake_experiments_db = {
    1: {
        "id": 1,
        "name": "Llama-2-7b LoRA Fine-tuning",
        "description": "使用LoRA方法微调Llama-2-7B模型",
        "project_id": 1,
        "user_id": 1,
        "base_model": "meta-llama/Llama-2-7b-hf",
        "task_type": "fine-tuning",
        "hyperparameters": {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 3
        },
        "status": "completed",
        "metrics": {
            "loss": 0.023,
            "eval_loss": 0.031,
            "perplexity": 1.03
        },
        "artifacts_path": "/models/llama-2-7b-lora-v1",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
}

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    base_model: str
    task_type: str
    hyperparameters: Optional[dict] = None

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    metrics: Optional[dict] = None

@router.get("")
async def list_experiments(project_id: Optional[int] = None, skip: int = 0, limit: int = 100):
    """获取实验列表"""
    experiments = list(fake_experiments_db.values())
    
    if project_id:
        experiments = [e for e in experiments if e["project_id"] == project_id]
    
    return {
        "total": len(experiments),
        "experiments": experiments[skip:skip+limit]
    }

@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int):
    """获取单个实验"""
    if experiment_id not in fake_experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return fake_experiments_db[experiment_id]

@router.post("", response_model=dict, status_code=201)
async def create_experiment(data: ExperimentCreate):
    """创建实验"""
    exp_id = len(fake_experiments_db) + 1
    
    new_exp = {
        "id": exp_id,
        "name": data.name,
        "description": data.description,
        "project_id": 1,
        "user_id": 1,
        "base_model": data.base_model,
        "task_type": data.task_type,
        "hyperparameters": data.hyperparameters or {},
        "status": "pending",
        "metrics": {},
        "artifacts_path": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    fake_experiments_db[exp_id] = new_exp
    return new_exp

@router.put("/{experiment_id}")
async def update_experiment(experiment_id: int, data: ExperimentUpdate):
    """更新实验"""
    if experiment_id not in fake_experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    for field, value in data.dict(exclude_unset=True).items():
        fake_experiments_db[experiment_id][field] = value
    
    fake_experiments_db[experiment_id]["updated_at"] = datetime.utcnow()
    return fake_experiments_db[experiment_id]

@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: int):
    """删除实验"""
    if experiment_id not in fake_experiments_db:
        raise HTTPException(status_code=404, detail="Experiment not found")
    del fake_experiments_db[experiment_id]
    return None
