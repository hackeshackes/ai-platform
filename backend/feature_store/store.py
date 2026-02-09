"""
Feature Store核心模块
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Feature:
    """特征定义"""
    name: str
    dtype: str  # int32, float64, string, bool
    description: str = ""
    version: int = 1

@dataclass
class FeatureGroup:
    """特征组"""
    group_id: str
    name: str
    description: str
    owner_id: str
    features: List[Feature]
    source_type: str = "batch"  # batch, stream
    source_uri: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FeatureMaterialization:
    """特征物化"""
    materialization_id: str
    feature_group_id: str
    storage_type: str  # online, offline
    schedule: str = ""
    last_run: Optional[datetime] = None
    status: str = "pending"

class FeatureStore:
    """特征存储服务"""
    
    def __init__(self):
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.materializations: Dict[str, FeatureMaterialization] = {}
    
    async def create_feature_group(
        self,
        name: str,
        description: str,
        owner_id: str,
        features: List[Dict[str, Any]],
        source_type: str = "batch",
        source_uri: str = ""
    ) -> FeatureGroup:
        """创建特征组"""
        group_id = str(uuid4())
        
        feature_objects = [
            Feature(
                name=f["name"],
                dtype=f["dtype"],
                description=f.get("description", ""),
                version=f.get("version", 1)
            )
            for f in features
        ]
        
        group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            owner_id=owner_id,
            features=feature_objects,
            source_type=source_type,
            source_uri=source_uri
        )
        
        self.feature_groups[group_id] = group
        return group
    
    async def get_feature_group(self, group_id: str) -> Optional[FeatureGroup]:
        """获取特征组"""
        return self.feature_groups.get(group_id)
    
    async def list_feature_groups(
        self,
        owner_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[FeatureGroup]:
        """列出特征组"""
        groups = list(self.feature_groups.values())
        
        if owner_id:
            groups = [g for g in groups if g.owner_id == owner_id]
        
        return groups[skip:skip+limit]
    
    async def add_features(
        self,
        group_id: str,
        features: List[Dict[str, Any]]
    ) -> FeatureGroup:
        """添加特征"""
        group = self.feature_groups.get(group_id)
        if not group:
            raise ValueError(f"FeatureGroup {group_id} not found")
        
        for f in features:
            group.features.append(Feature(
                name=f["name"],
                dtype=f["dtype"],
                description=f.get("description", ""),
                version=f.get("version", 1)
            ))
        
        group.updated_at = datetime.utcnow()
        return group
    
    async def create_materialization(
        self,
        feature_group_id: str,
        storage_type: str,
        schedule: str = ""
    ) -> FeatureMaterialization:
        """创建物化任务"""
        materialization = FeatureMaterialization(
            materialization_id=str(uuid4()),
            feature_group_id=feature_group_id,
            storage_type=storage_type,
            schedule=schedule
        )
        
        self.materializations[materialization.materialization_id] = materialization
        return materialization
    
    async def run_materialization(self, materialization_id: str) -> bool:
        """执行物化"""
        materialization = self.materializations.get(materialization_id)
        if not materialization:
            raise ValueError(f"Materialization {materialization_id} not found")
        
        # 模拟物化过程
        materialization.status = "running"
        materialization.last_run = datetime.utcnow()
        materialization.status = "completed"
        
        return True
    
    async def get_schema(self, feature_group_id: str) -> Dict[str, Any]:
        """获取特征模式"""
        group = self.feature_groups.get(feature_group_id)
        if not group:
            raise ValueError(f"FeatureGroup {feature_group_id} not found")
        
        return {
            "feature_group_id": group.group_id,
            "name": group.name,
            "features": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "description": f.description,
                    "version": f.version
                }
                for f in group.features
            ]
        }

# Feature Store实例
feature_store = FeatureStore()
