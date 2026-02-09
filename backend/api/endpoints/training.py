"""Training task submission endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid

router = APIRouter()

# 预配置模型
PREDEFINED_MODELS = [
    {
        "id": "llama2-7b",
        "name": "Llama-2-7b-chat-hf",
        "provider": "meta",
        "size": "7B",
        "type": "base",
        "framework": "transformers"
    },
    {
        "id": "llama2-13b",
        "name": "Llama-2-13b-chat-hf",
        "provider": "meta",
        "size": "13B",
        "type": "base",
        "framework": "transformers"
    },
    {
        "id": "qwen-7b",
        "name": "Qwen-7B-Chat",
        "provider": "alibaba",
        "size": "7B",
        "type": "chat",
        "framework": "transformers"
    },
    {
        "id": "qwen-14b",
        "name": "Qwen-14B-Chat",
        "provider": "alibaba",
        "size": "14B",
        "type": "chat",
        "framework": "transformers"
    },
    {
        "id": "baichuan-7b",
        "name": "Baichuan-7B-Chat",
        "provider": "baichuan",
        "size": "7B",
        "type": "chat",
        "framework": "transformers"
    }
]

# 预配置数据集
PREDEFINED_DATASETS = [
    {"id": 1, "name": "alpaca-zh", "size": "50MB", "rows": "52K"},
    {"id": 2, "name": "belle-zh", "size": "100MB", "rows": "100K"},
    {"id": 3, "name": "sharegpt", "size": "200MB", "rows": "87K"},
]

# 训练配置模板
TRAINING_TEMPLATES = [
    {
        "id": "lora",
        "name": "LoRA (低秩适配)",
        "description": "轻量级微调，适合消费级GPU",
        "params": {
            "learning_rate": 2e-4,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05
        },
        "min_gpu_memory": "8GB"
    },
    {
        "id": "qlora",
        "name": "QLoRA (量化LoRA)",
        "description": "4-bit量化，极低显存需求",
        "params": {
            "learning_rate": 2e-4,
            "r": 64,
            "alpha": 16,
            "quantization": "4bit"
        },
        "min_gpu_memory": "6GB"
    },
    {
        "id": "full-finetune",
        "name": "全参数微调",
        "description": "完整模型微调，效果最佳",
        "params": {
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "max_grad_norm": 1.0
        },
        "min_gpu_memory": "40GB"
    }
]

class TrainingConfig(BaseModel):
    model_id: str
    dataset_id: int
    template_id: str
    project_id: int
    experiment_name: str
    hyperparameters: Optional[dict] = None

@router.get("/models")
async def get_available_models():
    """获取可用模型列表"""
    return {"models": PREDEFINED_MODELS}

@router.get("/datasets")
async def get_training_datasets():
    """获取训练数据集列表"""
    return {"datasets": PREDEFINED_DATASETS}

@router.get("/templates")
async def get_training_templates():
    """获取训练配置模板"""
    return {"templates": TRAINING_TEMPLATES}

@router.post("/submit")
async def submit_training_job(config: TrainingConfig):
    """提交训练任务"""
    job_id = str(uuid.uuid4())[:8]
    
    # 查找模型和模板信息
    model = next((m for m in PREDEFINED_MODELS if m["id"] == config.model_id), None)
    template = next((t for t in TRAINING_TEMPLATES if t["id"] == config.template_id), None)
    
    job = {
        "job_id": job_id,
        "experiment_name": config.experiment_name,
        "model": model,
        "dataset_id": config.dataset_id,
        "template": template,
        "status": "queued",
        "progress": 0,
        "queue_position": 1,
        "estimated_start": datetime.utcnow().isoformat(),
        "config": config.dict()
    }
    
    return {
        "message": "训练任务已提交",
        "job": job
    }

@router.get("/jobs")
async def get_training_jobs(status: Optional[str] = None):
    """获取训练任务列表"""
    # 模拟任务列表
    jobs = [
        {
            "job_id": "abc12345",
            "experiment_name": "Llama-2-7B-LoRA-Test",
            "model": "Llama-2-7b-chat-hf",
            "status": "running",
            "progress": 45,
            "start_time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "eta": "2h 30m"
        },
        {
            "job_id": "def67890",
            "experiment_name": "Qwen-7B-QLoRA",
            "model": "Qwen-7B-Chat",
            "status": "queued",
            "progress": 0,
            "queue_position": 2,
            "estimated_start": datetime.utcnow().isoformat(),
            "eta": "5h 0m"
        }
    ]
    
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    return {"jobs": jobs}

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """获取单个任务状态"""
    return {
        "job_id": job_id,
        "status": "running",
        "progress": 45,
        "current_step": 450,
        "total_steps": 1000,
        "current_loss": 0.4523,
        "best_loss": 0.3210,
        "eta": "2h 30m",
        "logs": [
            {"time": "2026-02-08 18:00:00", "level": "INFO", "message": "Training started"},
            {"time": "2026-02-08 18:05:00", "level": "INFO", "message": "Step 100/1000: loss=1.2345"},
            {"time": "2026-02-08 18:10:00", "level": "INFO", "message": "Step 200/1000: loss=0.9876"},
            {"time": "2026-02-08 18:15:00", "level": "INFO", "message": "Step 300/1000: loss=0.7654"},
        ]
    }
