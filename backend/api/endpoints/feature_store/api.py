"""
Feature Store API端点 v2.1
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from backend.feature_store.store import feature_store
from backend.feature_store.ingestion import ingestion_service
from backend.feature_store.serving import serving_service
from api.endpoints.auth import get_current_user

router = APIRouter()

# 初始化服务
ingestion_service.feature_store = feature_store
serving_service.feature_store = feature_store

class CreateFeatureGroupModel(BaseModel):
    name: str
    description: str = ""
    features: List[Dict[str, Any]]
    source_type: str = "batch"
    source_uri: str = ""

class AddFeaturesModel(BaseModel):
    features: List[Dict[str, Any]]

class IngestFeaturesModel(BaseModel):
    entity_id: str
    features: Dict[str, Any]
    event_time: Optional[str] = None

class BatchIngestModel(BaseModel):
    records: List[Dict[str, Any]]  # [{entity_id, features, event_time}]

class SetOnlineFeaturesModel(BaseModel):
    entity_id: str
    features: Dict[str, Any]

class GetFeaturesModel(BaseModel):
    entity_ids: List[str]
    feature_names: Optional[List[str]] = None

@router.post("/groups")
async def create_feature_group(
    request: CreateFeatureGroupModel,
    current_user = Depends(get_current_user)
):
    """
    创建特征组
    
    v2.1: Feature Store
    """
    group = await feature_store.create_feature_group(
        name=request.name,
        description=request.description,
        owner_id=str(current_user.id),
        features=request.features,
        source_type=request.source_type,
        source_uri=request.source_uri
    )
    
    return {
        "group_id": group.group_id,
        "name": group.name,
        "features_count": len(group.features),
        "created_at": group.created_at.isoformat()
    }

@router.get("/groups")
async def list_feature_groups(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    列出特征组
    
    v2.1: Feature Store
    """
    groups = await feature_store.list_feature_groups(
        owner_id=str(current_user.id),
        skip=skip,
        limit=limit
    )
    
    return {
        "total": len(groups),
        "groups": [
            {
                "group_id": g.group_id,
                "name": g.name,
                "description": g.description,
                "features_count": len(g.features),
                "source_type": g.source_type,
                "created_at": g.created_at.isoformat()
            }
            for g in groups
        ]
    }

@router.get("/groups/{group_id}")
async def get_feature_group(
    group_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取特征组详情
    
    v2.1: Feature Store
    """
    group = await feature_store.get_feature_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="FeatureGroup not found")
    
    return {
        "group_id": group.group_id,
        "name": group.name,
        "description": group.description,
        "features": [
            {
                "name": f.name,
                "dtype": f.dtype,
                "description": f.description,
                "version": f.version
            }
            for f in group.features
        ],
        "source_type": group.source_type,
        "created_at": group.created_at.isoformat()
    }

@router.get("/groups/{group_id}/schema")
async def get_feature_schema(group_id: str):
    """
    获取特征模式
    
    v2.1: Feature Store
    """
    try:
        schema = await feature_store.get_schema(group_id)
        return schema
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/groups/{group_id}/features")
async def add_features(
    group_id: str,
    request: AddFeaturesModel,
    current_user = Depends(get_current_user)
):
    """
    添加特征
    
    v2.1: Feature Store
    """
    try:
        group = await feature_store.add_features(group_id, request.features)
        return {
            "group_id": group.group_id,
            "features_count": len(group.features)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/ingest")
async def ingest_features(request: IngestFeaturesModel):
    """
    摄入特征数据
    
    v2.1: Feature Store
    """
    event_time = None
    if request.event_time:
        event_time = datetime.fromisoformat(request.event_time)
    
    result = await ingestion_service.ingest_features(
        feature_group_id="default",  # 需要前端传入
        entity_id=request.entity_id,
        features=request.features,
        event_time=event_time
    )
    
    return result

@router.post("/batch-ingest")
async def batch_ingest(group_id: str, request: BatchIngestModel):
    """
    批量摄入特征数据
    
    v2.1: Feature Store
    """
    result = await ingestion_service.batch_ingest(group_id, request.records)
    return result

@router.post("/online")
async def set_online_features(
    group_id: str,
    request: SetOnlineFeaturesModel
):
    """
    设置在线特征
    
    v2.1: Feature Store
    """
    result = await serving_service.set_online_features(
        feature_group_id=group_id,
        entity_id=request.entity_id,
        features=request.features
    )
    
    return result

@router.get("/online/{group_id}/{entity_id}")
async def get_online_features(
    group_id: str,
    entity_id: str,
    features: Optional[str] = None
):
    """
    获取在线特征
    
    v2.1: Feature Store
    """
    feature_names = features.split(",") if features else None
    
    features = await serving_service.get_online_features(
        feature_group_id=group_id,
        entity_id=entity_id,
        feature_names=feature_names
    )
    
    if not features:
        raise HTTPException(status_code=404, detail="Features not found")
    
    return {
        "entity_id": entity_id,
        "features": features
    }

@router.post("/online/batch")
async def batch_get_online_features(
    group_id: str,
    request: GetFeaturesModel
):
    """
    批量获取在线特征
    
    v2.1: Feature Store
    """
    result = await serving_service.batch_get_online_features(
        feature_group_id=group_id,
        entity_ids=request.entity_ids,
        feature_names=request.feature_names
    )
    
    return result

@router.get("/stats")
async def get_feature_store_stats():
    """
    获取Feature Store统计
    
    v2.1: Feature Store
    """
    serving_stats = await serving_service.get_stats()
    
    return {
        "feature_groups": len(feature_store.feature_groups),
        "ingestion_history": len(ingestion_service.ingestion_history),
        "serving": serving_stats
    }
