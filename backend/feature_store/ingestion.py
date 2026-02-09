"""
特征摄入模块
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4

class FeatureIngestion:
    """特征摄入服务"""
    
    def __init__(self, feature_store):
        self.feature_store = feature_store
        self.ingestion_history: List[Dict] = []
    
    async def ingest_features(
        self,
        feature_group_id: str,
        entity_id: str,
        features: Dict[str, Any],
        event_time: Optional[datetime] = None
    ) -> Dict:
        """
        摄入特征数据
        
        Args:
            feature_group_id: 特征组ID
            entity_id: 实体ID (如用户ID、商品ID)
            features: 特征值字典
            event_time: 事件时间
        """
        ingestion_id = str(uuid4())
        
        # 验证特征存在
        group = await self.feature_store.get_feature_group(feature_group_id)
        if not group:
            raise ValueError(f"FeatureGroup {feature_group_id} not found")
        
        feature_names = {f.name for f in group.features}
        provided_features = set(features.keys())
        
        # 检查必要特征
        missing_features = feature_names - provided_features
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # 记录摄入历史
        ingestion_record = {
            "ingestion_id": ingestion_id,
            "feature_group_id": feature_group_id,
            "entity_id": entity_id,
            "features": features,
            "event_time": event_time or datetime.utcnow(),
            "status": "completed"
        }
        
        self.ingestion_history.append(ingestion_record)
        
        return {
            "ingestion_id": ingestion_id,
            "status": "completed",
            "entity_id": entity_id,
            "features_count": len(features),
            "timestamp": ingestion_record["event_time"].isoformat()
        }
    
    async def batch_ingest(
        self,
        feature_group_id: str,
        records: List[Dict[str, Any]]
    ) -> Dict:
        """
        批量摄入特征数据
        
        Args:
            feature_group_id: 特征组ID
            records: 记录列表 [{entity_id, features, event_time}]
        """
        ingestion_id = str(uuid4())
        
        results = []
        for record in records:
            try:
                result = await self.ingest_features(
                    feature_group_id=feature_group_id,
                    entity_id=record["entity_id"],
                    features=record["features"],
                    event_time=record.get("event_time")
                )
                results.append({
                    "entity_id": record["entity_id"],
                    "status": "completed",
                    "ingestion_id": result["ingestion_id"]
                })
            except Exception as e:
                results.append({
                    "entity_id": record.get("entity_id", "unknown"),
                    "status": "failed",
                    "error": str(e)
                })
        
        completed = sum(1 for r in results if r["status"] == "completed")
        
        return {
            "batch_id": ingestion_id,
            "total": len(records),
            "completed": completed,
            "failed": len(records) - completed,
            "results": results
        }
    
    async def get_ingestion_history(
        self,
        feature_group_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取摄入历史"""
        history = self.ingestion_history
        
        if feature_group_id:
            history = [h for h in history if h["feature_group_id"] == feature_group_id]
        
        return history[-limit:]
    
    async def validate_features(
        self,
        feature_group_id: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证特征值"""
        group = await self.feature_store.get_feature_group(feature_group_id)
        if not group:
            raise ValueError(f"FeatureGroup {feature_group_id} not found")
        
        validations = []
        errors = []
        
        for feature in group.features:
            if feature.name not in features:
                validations.append({
                    "feature": feature.name,
                    "status": "missing",
                    "error": "Feature not provided"
                })
                errors.append(feature.name)
            else:
                value = features[feature.name]
                # 验证数据类型
                if feature.dtype == "int32" and not isinstance(value, int):
                    validations.append({
                        "feature": feature.name,
                        "status": "type_error",
                        "expected": feature.dtype,
                        "actual": type(value).__name__
                    })
                    errors.append(feature.name)
                else:
                    validations.append({
                        "feature": feature.name,
                        "status": "valid",
                        "value": str(value)
                    })
        
        return {
            "valid": len(errors) == 0,
            "total_features": len(group.features),
            "validated": len(validations),
            "errors": errors,
            "validations": validations
        }

# Feature Ingestion实例
ingestion_service = FeatureIngestion(None)  # 初始化时设置feature_store
