"""
元学习API接口
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from .learner import MetaLearner
from .few_shot_learner import FewShotLearner
from .task_generator import TaskGenerator
from .adaptation_engine import AdaptationEngine
from .config import MetaLearningConfig

app = FastAPI(title="Meta Learning API", version="1.0.0")

# 全局实例
meta_learner: Optional[MetaLearner] = None
few_shot_learner: Optional[FewShotLearner] = None
adaptation_engine: Optional[AdaptationEngine] = None


# 请求/响应模型
class TrainRequest(BaseModel):
    algorithm: str = "maml"
    epochs: int = 100
    n_way: int = 5
    k_shot: int = 1
    meta_lr: float = 0.001


class AdaptRequest(BaseModel):
    support_x: List[List[float]]
    support_y: List[int]
    query_x: Optional[List[List[float]]] = None
    query_y: Optional[List[int]] = None
    strategy: str = "auto"


class EvaluateRequest(BaseModel):
    model_path: str
    support_x: List[List[float]]
    support_y: List[int]
    query_x: List[List[float]]
    query_y: List[int]


class TaskRequest(BaseModel):
    n_way: int = 5
    k_shot: int = 1
    difficulty: Optional[float] = None
    batch_size: int = 1


class ResponseModel(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None


@app.get("/")
async def root():
    return {"message": "Meta Learning API", "version": "1.0.0"}


@app.post("/api/initialize")
async def initialize(config: TrainRequest) -> ResponseModel:
    """初始化元学习器"""
    global meta_learner, few_shot_learner, adaptation_engine
    
    try:
        meta_learner = MetaLearner(
            algorithm=config.algorithm,
            n_way=config.n_way,
            k_shot=config.k_shot,
            outer_lr=config.meta_lr
        )
        
        few_shot_learner = FewShotLearner(
            n_way=config.n_way,
            k_shot=config.k_shot
        )
        
        adaptation_engine = AdaptationEngine()
        
        return ResponseModel(
            success=True,
            message=f"MetaLearner initialized with {config.algorithm}",
            data={"algorithm": config.algorithm, "n_way": config.n_way, "k_shot": config.k_shot}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train(config: TrainRequest) -> ResponseModel:
    """训练元学习器"""
    global meta_learner
    
    if meta_learner is None:
        raise HTTPException(status_code=400, detail="MetaLearner not initialized")
    
    try:
        # 生成训练任务
        task_dist = {"n_tasks": 100, "avg_samples_per_class": 10}
        best_model = meta_learner.train(task_dist, epochs=config.epochs)
        
        return ResponseModel(
            success=True,
            message=f"Training completed for {config.epochs} epochs",
            data={"final_loss": 0.0}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/adapt")
async def adapt(request: AdaptRequest) -> ResponseModel:
    """快速适应新任务"""
    global adaptation_engine
    
    if adaptation_engine is None:
        adaptation_engine = AdaptationEngine()
    
    try:
        # 转换输入
        support_x = torch.tensor(request.support_x)
        support_y = torch.tensor(request.support_y)
        query_x = torch.tensor(request.query_x) if request.query_x else None
        query_y = torch.tensor(request.query_y) if request.query_y else None
        
        task_data = {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y
        }
        
        result = adaptation_engine.adapt(None, task_data, {"strategy": request.strategy})
        
        return ResponseModel(
            success=True,
            message="Adaptation completed",
            data={
                "accuracy": result.get("accuracy", 0.0),
                "strategy": result.get("strategy", "unknown")
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate_tasks")
async def generate_tasks(request: TaskRequest) -> ResponseModel:
    """生成任务"""
    try:
        tasks = []
        for _ in range(request.batch_size):
            task = {
                "n_way": request.n_way,
                "k_shot": request.k_shot,
                "difficulty": request.difficulty
            }
            tasks.append(task)
        
        return ResponseModel(
            success=True,
            message=f"Generated {request.batch_size} tasks",
            data={"tasks": tasks}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status() -> Dict:
    """获取API状态"""
    return {
        "status": "running",
        "meta_learner": meta_learner is not None,
        "few_shot_learner": few_shot_learner is not None,
        "adaptation_engine": adaptation_engine is not None
    }


def create_api() -> FastAPI:
    """创建并返回FastAPI应用"""
    return app
