"""Metrics endpoints for training visualization"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

router = APIRouter()

class LossData(BaseModel):
    step: int
    loss: float
    epoch: Optional[int] = None
    timestamp: datetime

class LossResponse(BaseModel):
    experiment_id: str
    data: List[LossData]
    total_steps: int

@router.get("/loss")
async def get_loss_data(experiment_id: Optional[str] = None, steps: int = 100):
    """获取训练Loss曲线数据"""
    # 模拟真实训练数据 (实际应从TensorBoard/MLflow读取)
    import time
    
    data = []
    for i in range(steps):
        # 模拟loss下降曲线 + 噪声
        base_loss = 2.0 * (0.99 ** i)
        noise = random.uniform(-0.05, 0.05)
        loss = max(0.1, base_loss + noise)
        
        data.append({
            "step": i + 1,
            "loss": round(loss, 4),
            "epoch": (i + 1) // 10,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return {
        "experiment_id": experiment_id or "demo-exp-001",
        "total_steps": steps,
        "data": data,
        "metrics": {
            "initial_loss": round(data[0]['loss'], 4),
            "final_loss": round(data[-1]['loss'], 4),
            "best_loss": round(min(d['loss'] for d in data), 4),
        }
    }

@router.get("/loss/{experiment_id}")
async def get_experiment_loss(experiment_id: str, steps: int = 100):
    """获取特定实验的Loss曲线"""
    return await get_loss_data(experiment_id, steps)

@router.get("/metrics/summary")
async def get_metrics_summary():
    """获取指标概览"""
    return {
        "experiments": [
            {
                "id": "exp-001",
                "name": "Llama 2 Fine-tuning",
                "status": "running",
                "current_step": 150,
                "total_steps": 1000,
                "current_loss": 0.45,
                "best_loss": 0.32,
                "eta": "2h 30m"
            },
            {
                "id": "exp-002",
                "name": "Qwen 7B SFT",
                "status": "completed",
                "current_step": 1000,
                "total_steps": 1000,
                "current_loss": 0.28,
                "best_loss": 0.25,
                "eta": "0s"
            }
        ]
    }
