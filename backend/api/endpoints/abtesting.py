"""
A/B Testing API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# 直接导入模块
import importlib.util
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'abtesting/engine.py')

spec = importlib.util.spec_from_file_location("abtesting_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    ab_testing_engine = module.ab_testing_engine
    ExperimentStatus = module.ExperimentStatus
except Exception as e:
    print(f"Failed to import abtesting module: {e}")
    ab_testing_engine = None
    ExperimentStatus = None

router = APIRouter()

class CreateExperimentModel(BaseModel):
    name: str
    description: str
    variants: List[Dict]
    target_metrics: Optional[List[str]] = None

class AssignVariantModel(BaseModel):
    user_id: str

class TrackConversionModel(BaseModel):
    user_id: str
    value: float = 1.0

class TrackEventModel(BaseModel):
    user_id: str
    event_name: str
    event_value: float = 0.0

@router.get("/experiments")
async def list_experiments(status: Optional[str] = None):
    """列出实验"""
    estatus = ExperimentStatus(status) if status else None
    experiments = ab_testing_engine.list_experiments(status=estatus)
    
    return {
        "total": len(experiments),
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "description": e.description,
                "status": e.status.value,
                "variants_count": len(e.variants),
                "created_by": e.created_by,
                "created_at": e.created_at.isoformat()
            }
            for e in experiments
        ]
    }

@router.post("/experiments")
async def create_experiment(request: CreateExperimentModel):
    """创建实验"""
    experiment = ab_testing_engine.create_experiment(
        name=request.name,
        description=request.description,
        variants=request.variants,
        target_metrics=request.target_metrics
    )
    
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "message": "Experiment created"
    }

@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """获取实验详情"""
    experiment = ab_testing_engine.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": experiment.status.value,
        "variants": experiment.variants,
        "target_metrics": experiment.target_metrics,
        "start_date": experiment.start_date.isoformat() if experiment.start_date else None,
        "end_date": experiment.end_date.isoformat() if experiment.end_date else None
    }

@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """开始实验"""
    result = ab_testing_engine.start_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment started"}

@router.post("/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    """暂停实验"""
    result = ab_testing_engine.pause_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment paused"}

@router.post("/experiments/{experiment_id}/complete")
async def complete_experiment(experiment_id: str):
    """完成实验"""
    result = ab_testing_engine.complete_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment completed"}

@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """删除实验"""
    result = ab_testing_engine.delete_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment deleted"}

@router.post("/experiments/{experiment_id}/assign")
async def assign_variant(experiment_id: str, request: AssignVariantModel):
    """分配变体"""
    variant_id = ab_testing_engine.assign_variant(experiment_id, request.user_id)
    if not variant_id:
        raise HTTPException(status_code=400, detail="Cannot assign variant")
    return {"variant_id": variant_id}

@router.get("/experiments/{experiment_id}/results")
async def get_results(experiment_id: str):
    """获取实验结果"""
    results = ab_testing_engine.get_results(experiment_id)
    return {
        "total": len(results),
        "results": [
            {
                "variant_id": r.variant_id,
                "sample_size": r.sample_size,
                "conversions": r.conversions,
                "conversion_rate": r.conversion_rate,
                "p_value": r.p_value
            }
            for r in results
        ]
    }

@router.get("/experiments/{experiment_id}/leaderboard")
async def get_leaderboard(experiment_id: str):
    """获取排行榜"""
    leaderboard = ab_testing_engine.get_leaderboard(experiment_id)
    return {"leaderboard": leaderboard}

@router.post("/track/conversion")
async def track_conversion(experiment_id: str, request: TrackConversionModel):
    """跟踪转化"""
    result = ab_testing_engine.track_conversion(experiment_id, request.user_id, request.value)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to track conversion")
    return {"message": "Conversion tracked"}

@router.post("/track/event")
async def track_event(experiment_id: str, request: TrackEventModel):
    """跟踪事件"""
    result = ab_testing_engine.track_event(experiment_id, request.user_id, request.event_name, request.event_value)
    return {"message": "Event tracked"}

@router.get("/summary")
async def get_summary():
    """获取统计"""
    return ab_testing_engine.get_summary()

@router.get("/health")
async def abtesting_health():
    """健康检查"""
    return {
        "status": "healthy",
        "experiments": len(ab_testing_engine.experiments)
    }
