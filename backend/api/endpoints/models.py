"""Model management endpoints"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime, timedelta

router = APIRouter()

# 模拟模型数据库
MODELS_DB = {
    1: {
        "id": 1,
        "name": "Llama-2-7b-chat-finetuned",
        "project_id": 1,
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "framework": "transformers",
        "version": "v1.0",
        "size": "14GB",
        "status": "ready",
        "metrics": {
            "final_loss": 0.2834,
            "bleu_score": 45.2
        },
        "created_at": datetime.utcnow().isoformat()
    },
    2: {
        "id": 2,
        "name": "Qwen-7B-Lora-test",
        "project_id": 1,
        "base_model": "Qwen/Qwen-7B-Chat",
        "framework": "peft",
        "version": "v2.1",
        "size": "1.2GB",
        "status": "ready",
        "metrics": {
            "final_loss": 0.3121,
            "bleu_score": 42.8
        },
        "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat()
    }
}

class ModelCreate(BaseModel):
    name: str
    project_id: int
    base_model: str
    framework: str = "transformers"
    version: str = "v1.0"

@router.get("")
async def list_models(project_id: Optional[int] = None):
    """获取模型列表"""
    models = list(MODELS_DB.values())
    if project_id:
        models = [m for m in models if m["project_id"] == project_id]
    return {"total": len(models), "models": models}

@router.get("/{model_id}")
async def get_model(model_id: int):
    """获取模型详情"""
    if model_id not in MODELS_DB:
        raise HTTPException(status_code=404, detail="Model not found")
    return MODELS_DB[model_id]

@router.post("", response_model=dict, status_code=201)
async def create_model(data: ModelCreate):
    """创建模型记录"""
    model_id = len(MODELS_DB) + 1
    new_model = {
        "id": model_id,
        "name": data.name,
        "project_id": data.project_id,
        "base_model": data.base_model,
        "framework": data.framework,
        "version": data.version,
        "size": "0B",
        "status": "pending",
        "metrics": None,
        "created_at": datetime.utcnow().isoformat()
    }
    MODELS_DB[model_id] = new_model
    return new_model

@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: int):
    """删除模型"""
    if model_id not in MODELS_DB:
        raise HTTPException(status_code=404, detail="Model not found")
    del MODELS_DB[model_id]
