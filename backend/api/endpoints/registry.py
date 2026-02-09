"""
Model Registry API端点 v2.1
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from backend.models.registry import model_registry
from api.endpoints.auth import get_current_user

router = APIRouter()

class CreateRegisteredModelModel(BaseModel):
    name: str
    description: str = ""
    project_id: Optional[str] = None

class CreateModelVersionModel(BaseModel):
    model_uri: str
    description: str = ""
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class TransitionStageModel(BaseModel):
    to_stage: str  # Staging, Production, Archived
    comment: str = ""

class SetAliasModel(BaseModel):
    alias: str
    version: int

class SearchModelsModel(BaseModel):
    query: Optional[str] = None
    stage: Optional[str] = None
    status: Optional[str] = None

@router.post("/register")
async def create_registered_model(
    request: CreateRegisteredModelModel,
    current_user = Depends(get_current_user)
):
    """
    创建注册模型
    
    v2.1: Model Registry
    """
    try:
        model = await model_registry.create_registered_model(
            name=request.name,
            description=request.description,
            project_id=request.project_id,
            created_by=str(current_user.id)
        )
        
        return {
            "model_id": model.model_id,
            "name": model.name,
            "description": model.description,
            "created_at": model.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("")
async def list_registered_models(
    project_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    列出注册模型
    
    v2.1: Model Registry
    """
    models = await model_registry.list_registered_models(
        project_id=project_id,
        skip=skip,
        limit=limit
    )
    
    return {
        "total": len(models),
        "models": [
            {
                "model_id": m.model_id,
                "name": m.name,
                "description": m.description,
                "versions_count": len(m.versions),
                "latest_version": max(v.version for v in m.versions) if m.versions else None,
                "created_at": m.created_at.isoformat()
            }
            for m in models
        ]
    }

@router.get("/search")
async def search_models(request: SearchModelsModel = None):
    """
    搜索模型
    
    v2.1: Model Registry
    """
    results = await model_registry.search_models(
        query=request.query if request else None,
        stage=request.stage if request else None,
        status=request.status if request else None
    )
    
    return {"results": results}

@router.get("/{model_id}")
async def get_registered_model(
    model_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取注册模型详情
    
    v2.1: Model Registry
    """
    model = await model_registry.get_registered_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model.model_id,
        "name": model.name,
        "description": model.description,
        "project_id": model.project_id,
        "versions": [
            {
                "version": v.version,
                "status": v.status,
                "current_stage": v.current_stage,
                "created_at": v.created_at.isoformat()
            }
            for v in sorted(model.versions, key=lambda x: x.version, reverse=True)
        ],
        "created_at": model.created_at.isoformat()
    }

@router.get("/{model_id}/versions")
async def get_model_versions(
    model_id: str,
    stage: Optional[str] = None,
    status: Optional[str] = None
):
    """
    获取模型版本列表
    
    v2.1: Model Registry
    """
    versions = await model_registry.get_latest_versions(model_id, status=status)
    
    if stage:
        versions = [v for v in versions if v.current_stage == stage]
    
    return {
        "total": len(versions),
        "versions": [
            {
                "version": v.version,
                "status": v.status,
                "current_stage": v.current_stage,
                "description": v.description,
                "created_at": v.created_at.isoformat()
            }
            for v in versions
        ]
    }

@router.get("/{model_id}/versions/{version}")
async def get_model_version(
    model_id: str,
    version: int
):
    """
    获取模型版本详情
    
    v2.1: Model Registry
    """
    v = await model_registry.get_model_version(model_id, version)
    if not v:
        raise HTTPException(status_code=404, detail="Model version not found")
    
    return {
        "version_id": v.version_id,
        "model_id": v.model_id,
        "version": v.version,
        "model_uri": v.model_uri,
        "status": v.status,
        "current_stage": v.current_stage,
        "description": v.description,
        "metadata": v.metadata,
        "run_id": v.run_id,
        "created_at": v.created_at.isoformat(),
        "updated_at": v.updated_at.isoformat()
    }

@router.post("/{model_id}/versions")
async def create_model_version(
    model_id: str,
    request: CreateModelVersionModel,
    current_user = Depends(get_current_user)
):
    """
    创建模型版本
    
    v2.1: Model Registry
    """
    try:
        version = await model_registry.create_model_version(
            model_id=model_id,
            model_uri=request.model_uri,
            description=request.description,
            run_id=request.run_id,
            metadata=request.metadata
        )
        
        return {
            "version_id": version.version_id,
            "model_id": version.model_id,
            "version": version.version,
            "status": version.status,
            "created_at": version.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{model_id}/versions/{version}/stage")
async def transition_stage(
    model_id: str,
    version: int,
    request: TransitionStageModel,
    current_user = Depends(get_current_user)
):
    """
    切换模型阶段
    
    v2.1: Model Registry
    
    阶段: None -> Staging -> Production -> Archived
    """
    try:
        version = await model_registry.transition_stage(
            model_id=model_id,
            version=version,
            to_stage=request.to_stage,
            transitioned_by=str(current_user.id),
            comment=request.comment
        )
        
        return {
            "version": version.version,
            "from_stage": request.to_stage,
            "to_stage": version.current_stage,
            "transitioned_at": version.updated_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{model_id}/alias")
async def set_alias(
    model_id: str,
    request: SetAliasModel,
    current_user = Depends(get_current_user)
):
    """
    设置模型别名
    
    v2.1: Model Registry
    """
    try:
        alias = await model_registry.set_alias(
            model_id=model_id,
            alias=request.alias,
            version=request.version
        )
        
        return {
            "alias": alias.alias,
            "model_id": alias.model_id,
            "version": alias.version,
            "created_at": alias.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{model_id}/alias/{alias}")
async def get_by_alias(
    model_id: str,
    alias: str
):
    """
    根据别名获取模型版本
    
    v2.1: Model Registry
    """
    version = await model_registry.get_by_alias(model_id, alias)
    if not version:
        raise HTTPException(status_code=404, detail="Alias not found")
    
    return {
        "alias": alias,
        "version": version.version,
        "model_uri": version.model_uri,
        "current_stage": version.current_stage
    }

@router.get("/{model_id}/versions/{version}/history")
async def get_transition_history(
    model_id: str,
    version: int
):
    """
    获取版本转换历史
    
    v2.1: Model Registry
    """
    history = await model_registry.get_transition_history(model_id, version)
    
    return {
        "total": len(history),
        "history": [
            {
                "from_stage": h.from_stage,
                "to_stage": h.to_stage,
                "transitioned_by": h.transitioned_by,
                "comment": h.comment,
                "transitioned_at": h.transitioned_at.isoformat()
            }
            for h in history
        ]
    }

@router.post("/{model_id}/versions/{version}/archive")
async def archive_model(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """
    归档模型
    
    v2.1: Model Registry
    """
    try:
        version = await model_registry.archive_model(model_id, version)
        
        return {
            "version": version.version,
            "current_stage": version.current_stage
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
