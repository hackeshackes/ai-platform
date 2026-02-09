"""
Model Registry模块
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    model_id: str
    version: int
    model_uri: str
    status: str = "pending"  # pending, ready, failed, archived
    current_stage: str = "None"  # None, Staging, Production, Archived
    description: str = ""
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RegisteredModel:
    """注册模型"""
    model_id: str
    name: str
    description: str = ""
    project_id: Optional[str] = None
    created_by: Optional[str] = None
    versions: List[ModelVersion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ModelStageTransition:
    """阶段转换记录"""
    transition_id: str
    model_version_id: str
    from_stage: str
    to_stage: str
    transitioned_by: Optional[str] = None
    comment: str = ""
    transitioned_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ModelAlias:
    """模型别名"""
    alias_id: str
    model_id: str
    alias: str
    version: int
    created_at: datetime = field(default_factory=datetime.utcnow)

class ModelRegistry:
    """模型注册中心"""
    
    def __init__(self):
        self.registered_models: Dict[str, RegisteredModel] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        self.transitions: List[ModelStageTransition] = []
        self.aliases: Dict[str, ModelAlias] = {}
    
    async def create_registered_model(
        self,
        name: str,
        description: str = "",
        project_id: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> RegisteredModel:
        """创建注册模型"""
        # 检查名称唯一
        for model in self.registered_models.values():
            if model.name == name:
                raise ValueError(f"Model {name} already exists")
        
        model_id = str(uuid4())
        
        model = RegisteredModel(
            model_id=model_id,
            name=name,
            description=description,
            project_id=project_id,
            created_by=created_by
        )
        
        self.registered_models[model_id] = model
        return model
    
    async def get_registered_model(self, model_id: str) -> Optional[RegisteredModel]]:
        """获取注册模型"""
        return self.registered_models.get(model_id)
    
    async def get_registered_model_by_name(self, name: str) -> Optional[RegisteredModel]]:
        """根据名称获取注册模型"""
        for model in self.registered_models.values():
            if model.name == name:
                return model
        return None
    
    async def list_registered_models(
        self,
        project_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[RegisteredModel]:
        """列出注册模型"""
        models = list(self.registered_models.values())
        
        if project_id:
            models = [m for m in models if m.project_id == project_id]
        
        return models[skip:skip+limit]
    
    async def create_model_version(
        self,
        model_id: str,
        model_uri: str,
        description: str = "",
        metadata: Optional[Dict] = None,
        run_id: Optional[str] = None
    ) -> ModelVersion:
        """创建模型版本"""
        model = self.registered_models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        version_id = str(uuid4())
        version_number = len(model.versions) + 1
        
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version=version_number,
            model_uri=model_uri,
            description=description,
            metadata=metadata or {},
            run_id=run_id
        )
        
        model.versions.append(version)
        self.model_versions[version_id] = version
        
        # 自动设置状态为ready
        version.status = "ready"
        
        return version
    
    async def get_model_version(
        self,
        model_id: str,
        version: int
    ) -> Optional[ModelVersion]:
        """获取模型版本"""
        model = self.registered_models.get(model_id)
        if not model:
            return None
        
        for v in model.versions:
            if v.version == version:
                return v
        return None
    
    async def get_latest_versions(
        self,
        model_id: str,
        status: Optional[str] = None
    ) -> List[ModelVersion]:
        """获取最新版本"""
        model = self.registered_models.get(model_id)
        if not model:
            return []
        
        versions = sorted(
            model.versions,
            key=lambda v: v.version,
            reverse=True
        )
        
        if status:
            versions = [v for v in versions if v.status == status]
        
        return versions
    
    async def transition_stage(
        self,
        model_id: str,
        version: int,
        to_stage: str,
        transitioned_by: Optional[str] = None,
        comment: str = ""
    ) -> ModelVersion:
        """切换模型阶段"""
        version_obj = await self.get_model_version(model_id, version)
        if not version_obj:
            raise ValueError(f"Model version {model_id}:{version} not found")
        
        from_stage = version_obj.current_stage
        
        # 验证阶段转换
        valid_transitions = {
            "None": ["Staging"],
            "Staging": ["Production", "Archived"],
            "Production": ["Archived"],
            "Archived": []
        }
        
        if to_stage not in valid_transitions.get(from_stage, []):
            raise ValueError(
                f"Invalid transition from {from_stage} to {to_stage}"
            )
        
        # 执行转换
        version_obj.current_stage = to_stage
        version_obj.updated_at = datetime.utcnow()
        
        # 记录转换历史
        transition = ModelStageTransition(
            transition_id=str(uuid4()),
            model_version_id=version_obj.version_id,
            from_stage=from_stage,
            to_stage=to_stage,
            transitioned_by=transitioned_by,
            comment=comment
        )
        self.transitions.append(transition)
        
        return version_obj
    
    async def set_alias(
        self,
        model_id: str,
        alias: str,
        version: int
    ) -> ModelAlias:
        """设置模型别名"""
        # 验证模型版本存在
        version_obj = await self.get_model_version(model_id, version)
        if not version_obj:
            raise ValueError(f"Model version {model_id}:{version} not found")
        
        # 删除旧别名
        old_alias_id = f"{model_id}:{alias}"
        if old_alias_id in self.aliases:
            del self.aliases[old_alias_id]
        
        # 创建新别名
        model_alias = ModelAlias(
            alias_id=str(uuid4()),
            model_id=model_id,
            alias=alias,
            version=version
        )
        
        self.aliases[f"{model_id}:{alias}"] = model_alias
        return model_alias
    
    async def get_by_alias(self, model_id: str, alias: str) -> Optional[ModelVersion]:
        """根据别名获取模型"""
        alias_key = f"{model_id}:{alias}"
        model_alias = self.aliases.get(alias_key)
        if not model_alias:
            return None
        
        return await self.get_model_version(model_id, model_alias.version)
    
    async def archive_model(self, model_id: str, version: int) -> ModelVersion:
        """归档模型"""
        return await self.transition_stage(
            model_id=model_id,
            version=version,
            to_stage="Archived",
            comment="Archived"
        )
    
    async def search_models(
        self,
        query: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索模型"""
        results = []
        
        for model in self.registered_models.values():
            # 搜索过滤
            if query and query.lower() not in model.name.lower():
                continue
            
            # 版本过滤
            versions = model.versions
            if stage:
                versions = [v for v in versions if v.current_stage == stage]
            if status:
                versions = [v for v in versions if v.status == status]
            
            if versions or not (query or stage or status):
                results.append({
                    "model_id": model.model_id,
                    "name": model.name,
                    "description": model.description,
                    "latest_version": max(v.version for v in model.versions) if model.versions else None,
                    "versions_count": len(versions),
                    "current_stage": versions[0].current_stage if versions else "None"
                })
        
        return results
    
    async def get_transition_history(
        self,
        model_id: str,
        version: int
    ) -> List[ModelStageTransition]:
        """获取转换历史"""
        version_obj = await self.get_model_version(model_id, version)
        if not version_obj:
            return []
        
        return [
            t for t in self.transitions
            if t.model_version_id == version_obj.version_id
        ]

# Model Registry实例
model_registry = ModelRegistry()
