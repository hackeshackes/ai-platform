"""
judges.py - AI Platform v2.3
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'judges/builder.py')

spec = importlib.util.spec_from_file_location("gateway_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    judge_builder = module.judge_builder
except Exception as e:
    print(f"Failed to import module: {e}")
    judge_builder = None

from api.endpoints.auth import get_current_user

router = APIRouter()
class CreateJudgeModel(BaseModel):
    name: str
    description: str
    prompt: str
    instructions: str
    judge_type: str  # llm, rule-based
    criteria: Optional[List[Dict]] = None

class UpdateJudgeModel(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    criteria: Optional[List[Dict]] = None

class CreateRunModel(BaseModel):
    judge_id: str
    traces: List[str]

class TestJudgeModel(BaseModel):
    judge_id: str
    trace_data: Dict

@router.get("/health")
async def judge_health():
    """
    Judge Builder健康检查
    
    v2.3: Judge Builder
    """
    judges = list(judge_builder.judges.values())
    runs = list(judge_builder.runs.values())
    
    return {
        "status": "healthy",
        "judges_count": len(judges),
        "runs_count": len(runs),
        "builtin_judges": len([j for j in judges if j.created_by == "system"])
    }

@router.get("")

@router.get("")
async def list_judges(
    judge_type: Optional[str] = None,
    created_by: Optional[str] = None
):
    """
    列出评估器
    
    v2.3: Judge Builder
    """
    judges = judge_builder.list_judges(
        judge_type=judge_type,
        created_by=created_by
    )
    
    return {
        "total": len(judges),
        "judges": [
            {
                "judge_id": j.judge_id,
                "name": j.name,
                "description": j.description,
                "type": j.judge_type,
                "criteria": j.criteria,
                "created_by": j.created_by,
                "created_at": j.created_at.isoformat()
            }
            for j in judges
        ]
    }

@router.post("")
async def create_judge(
    request: CreateJudgeModel,
    current_user = Depends(get_current_user)
):
    """
    创建评估器
    
    v2.3: Judge Builder
    """
    try:
        judge = judge_builder.create_judge(
            name=request.name,
            description=request.description,
            prompt=request.prompt,
            instructions=request.instructions,
            judge_type=request.judge_type,
            criteria=request.criteria,
            created_by=str(current_user.id)
        )
        
        return {
            "judge_id": judge.judge_id,
            "name": judge.name,
            "message": "Judge created"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates")
async def get_templates():
    """
    获取评估器模板
    
    v2.3: Judge Builder
    """
    templates = judge_builder.get_templates()
    
    return {
        "total": len(templates),
        "templates": templates
    }

@router.get("/{judge_id}")
async def get_judge(judge_id: str):
    """
    获取评估器详情
    
    v2.3: Judge Builder
    """
    judge = judge_builder.get_judge(judge_id)
    if not judge:
        raise HTTPException(status_code=404, detail="Judge not found")
    
    return {
        "judge_id": judge.judge_id,
        "name": judge.name,
        "description": judge.description,
        "prompt": judge.prompt,
        "instructions": judge.instructions,
        "type": judge.judge_type,
        "criteria": judge.criteria,
        "created_by": judge.created_by,
        "created_at": judge.created_at.isoformat(),
        "updated_at": judge.updated_at.isoformat()
    }

@router.put("/{judge_id}")
async def update_judge(
    judge_id: str,
    request: UpdateJudgeModel
):
    """
    更新评估器
    
    v2.3: Judge Builder
    """
    result = judge_builder.update_judge(
        judge_id=judge_id,
        name=request.name,
        prompt=request.prompt,
        instructions=request.instructions,
        criteria=request.criteria
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Judge not found")
    
    return {"message": "Judge updated"}

@router.delete("/{judge_id}")
async def delete_judge(judge_id: str):
    """
    删除评估器
    
    v2.3: Judge Builder
    """
    result = judge_builder.delete_judge(judge_id)
    if not result:
        raise HTTPException(status_code=404, detail="Judge not found")
    
    return {"message": "Judge deleted"}

@router.post("/run")
async def create_run(
    request: CreateRunModel,
    current_user = Depends(get_current_user)
):
    """
    创建评估运行
    
    v2.3: Judge Builder
    """
    try:
        run = await judge_builder.create_run(
            judge_id=request.judge_id,
            traces=request.traces,
            created_by=str(current_user.id)
        )
        
        return {
            "run_id": run.run_id,
            "judge_id": run.judge_id,
            "status": run.status,
            "summary": run.summary,
            "created_at": run.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/runs")
async def list_runs(
    judge_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    列出评估运行
    
    v2.3: Judge Builder
    """
    runs = judge_builder.list_runs(
        judge_id=judge_id,
        status=status,
        limit=limit
    )
    
    return {
        "total": len(runs),
        "runs": [
            {
                "run_id": r.run_id,
                "judge_id": r.judge_id,
                "status": r.status,
                "traces_count": len(r.traces),
                "summary": r.summary,
                "created_at": r.created_at.isoformat()
            }
            for r in runs
        ]
    }

@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """
    获取评估运行详情
    
    v2.3: Judge Builder
    """
    run = judge_builder.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": run.run_id,
        "judge_id": run.judge_id,
        "status": run.status,
        "traces": run.traces,
        "results_count": len(run.results),
        "summary": run.summary,
        "created_at": run.created_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None
    }

@router.get("/runs/{run_id}/results")
async def get_run_results(run_id: str):
    """
    获取评估结果
    
    v2.3: Judge Builder
    """
    results = judge_builder.get_run_results(run_id)
    
    return {
        "total": len(results),
        "results": [
            {
                "result_id": r.result_id,
                "trace_id": r.trace_id,
                "score": r.score,
                "passing": r.passing,
                "feedback": r.feedback,
                "criteria_scores": r.criteria_scores
            }
            for r in results
        ]
    }

@router.post("/test")
async def test_judge(request: TestJudgeModel):
    """
    测试评估器
    
    v2.3: Judge Builder
    """
    try:
        result = await judge_builder.test_judge(
            judge_id=request.judge_id,
            trace_data=request.trace_data
        )
        
        return {
            "result_id": result.result_id,
            "score": result.score,
            "passing": result.passing,
            "feedback": result.feedback,
            "criteria_scores": result.criteria_scores
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

