"""
Pipeline API端点 v2.0
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

from backend.pipeline.dag import pipeline_engine, PipelineStatus
from backend.pipeline.templates.defaults import get_template, list_templates
from backend.core.auth import get_current_user

router = APIRouter()

# 内存存储 (生产环境使用数据库)
pipelines_db = {}

class PipelineCreate(BaseModel):
    """创建Pipeline请求"""
    name: str
    description: Optional[str] = None
    steps: List[dict]
    template: Optional[str] = None
    metadata: Optional[dict] = None

class PipelineExecute(BaseModel):
    """执行Pipeline请求"""
    parameters: Optional[dict] = None

@router.get("")
async def list_pipelines(
    skip: int = 0,
    limit: int = 20,
    current_user = Depends(get_current_user)
):
    """
    获取Pipeline列表
    
    v2.0 Phase 2: Pipeline编排
    """
    pipelines = list(pipelines_db.values())
    
    return {
        "total": len(pipelines),
        "pipelines": pipelines[skip:skip+limit]
    }

@router.post("")
async def create_pipeline(
    request: PipelineCreate,
    current_user = Depends(get_current_user)
):
    """
    创建Pipeline
    
    v2.0 Phase 2: Pipeline编排
    """
    # 如果指定模板，使用模板
    if request.template:
        template = get_template(request.template)
        steps = template["steps"]
        description = request.description or template["description"]
    else:
        steps = request.steps
        description = request.description or ""
    
    pipeline = await pipeline_engine.create_pipeline(
        name=request.name,
        description=description,
        steps=steps,
        metadata={
            "user_id": current_user.id,
            **(request.metadata or {})
        }
    )
    
    # 保存到数据库
    pipelines_db[pipeline.pipeline_id] = {
        "pipeline_id": pipeline.pipeline_id,
        "name": pipeline.name,
        "description": pipeline.description,
        "status": pipeline.status.value,
        "created_at": pipeline.created_at.isoformat(),
        "user_id": current_user.id
    }
    
    return {
        "pipeline_id": pipeline.pipeline_id,
        "name": pipeline.name,
        "status": pipeline.status.value,
        "created_at": pipeline.created_at.isoformat()
    }

@router.get("/{pipeline_id}")
async def get_pipeline(
    pipeline_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取Pipeline详情
    
    v2.0 Phase 2: Pipeline编排
    """
    status = await pipeline_engine.get_status(pipeline_id)
    return status

@router.post("/{pipeline_id}/execute")
async def execute_pipeline(
    pipeline_id: str,
    request: Optional[PipelineExecute] = None,
    current_user = Depends(get_current_user)
):
    """
    执行Pipeline
    
    v2.0 Phase 2: Pipeline编排
    """
    try:
        pipeline = await pipeline_engine.execute(pipeline_id)
        
        # 更新数据库
        if pipeline_id in pipelines_db:
            pipelines_db[pipeline_id]["status"] = pipeline.status.value
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "status": pipeline.status.value,
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{pipeline_id}/status")
async def get_pipeline_status(
    pipeline_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取Pipeline执行状态
    
    v2.0 Phase 2: Pipeline编排
    """
    try:
        return await pipeline_engine.get_status(pipeline_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{pipeline_id}/cancel")
async def cancel_pipeline(
    pipeline_id: str,
    current_user = Depends(get_current_user)
):
    """
    取消Pipeline
    
    v2.0 Phase 2: Pipeline编排
    """
    try:
        await pipeline_engine.cancel(pipeline_id)
        
        # 更新数据库
        if pipeline_id in pipelines_db:
            pipelines_db[pipeline_id]["status"] = PipelineStatus.CANCELLED.value
        
        return {"message": "Pipeline cancelled"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{pipeline_id}/logs")
async def get_pipeline_logs(
    pipeline_id: str,
    step_id: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """
    获取Pipeline日志
    
    v2.0 Phase 2: Pipeline编排
    """
    # TODO: 实现日志获取
    return {
        "pipeline_id": pipeline_id,
        "step_id": step_id,
        "logs": []
    }

@router.get("/templates/list")
async def list_pipeline_templates(
    current_user = Depends(get_current_user)
):
    """
    获取Pipeline模板列表
    
    v2.0 Phase 2: Pipeline编排
    """
    return {
        "templates": list_templates()
    }

@router.post("/templates/{template_name}/create")
async def create_from_template(
    template_name: str,
    name: str,
    description: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """
    从模板创建Pipeline
    
    v2.0 Phase 2: Pipeline编排
    """
    template = get_template(template_name)
    
    pipeline = await pipeline_engine.create_pipeline(
        name=name,
        description=description or template["description"],
        steps=template["steps"],
        metadata={"template": template_name, "user_id": current_user.id}
    )
    
    return {
        "pipeline_id": pipeline.pipeline_id,
        "name": pipeline.name,
        "template": template_name
    }
