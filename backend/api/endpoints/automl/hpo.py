"""
AutoML API端点 v2.0 Phase 3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from automl.hpo import hpo_optimizer, HPOMethod, HPOParam
from api.endpoints.auth import get_current_user

router = APIRouter()

class HPOParamModel(BaseModel):
    name: str
    type: str  # categorical, continuous, integer
    values: List[float]
    log_scale: bool = False

class HPOCreateModel(BaseModel):
    name: str
    params: List[HPOParamModel]
    method: str = "bayesian"
    max_trials: int = 100
    timeout_seconds: int = 3600
    direction: str = "maximize"

class HPOCheckModel(BaseModel):
    params: Dict

@router.post("/hpo/start")
async def start_hpo(
    request: HPOCreateModel,
    current_user = Depends(get_current_user)
):
    """
    启动超参数优化
    
    v2.0 Phase 3: AutoML
    """
    try:
        method = HPOMethod(request.method)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
    
    # 转换参数
    hpo_params = [
        HPOParam(
            name=p.name,
            type=p.type,
            values=p.values,
            log_scale=p.log_scale
        )
        for p in request.params
    ]
    
    # 创建目标函数 (简化)
    async def objective_fn(params: Dict) -> float:
        # 模拟目标函数
        return 0.5 + random.random() * 0.5
    
    import random
    
    result = await hpo_optimizer.optimize(
        objective_fn=objective_fn,
        params=hpo_params,
        method=method,
        max_trials=request.max_trials,
        timeout_seconds=request.timeout_seconds,
        direction=request.direction
    )
    
    return {
        "optimization_id": result.optimization_id,
        "best_trial_id": result.best_trial.trial_id if result.best_trial else None,
        "best_objective": result.best_objective,
        "best_params": result.best_params,
        "total_trials": result.total_trials,
        "status": "completed"
    }

@router.get("/hpo/{optimization_id}")
async def get_hpo_result(optimization_id: str):
    """
    获取HPO结果
    
    v2.0 Phase 3: AutoML
    """
    status = hpo_optimizer.get_optimization_status(optimization_id)
    if not status:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return status

@router.get("/hpo/methods")
async def list_hpo_methods():
    """
    列出HPO方法
    
    v2.0 Phase 3: AutoML
    """
    return {
        "methods": [
            {"id": m.value, "name": m.name}
            for m in HPOMethod
        ]
    }
