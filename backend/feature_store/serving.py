"""
特征服务模块
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
import asyncio

class FeatureServing:
    """特征服务"""
    
    def __init__(self, feature_store):
        self.feature_store = feature_store
        self.online_storage: Dict[str, Dict[str, Any]] = {}  # entity_id -> features
        self.cache: Dict[str, Dict] = {}  # cache_key -> (value, expiry)
        self.default_ttl = 300  # 5 minutes
    
    async def get_online_features(
        self,
        feature_group_id: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取在线特征
        
        Args:
            feature_group_id: 特征组ID
            entity_id: 实体ID
            feature_names: 要获取的特征列表 (None表示全部)
        """
        key = f"{feature_group_id}:{entity_id}"
        
        # 检查缓存
        cache_key = f"online:{key}"
        if cache_key in self.cache:
            value, expiry = self.cache[cache_key]
            if datetime.utcnow().timestamp() < expiry:
                cached = value
                if feature_names:
                    cached = {k: v for k, v in cached.items() if k in feature_names}
                return cached
        
        # 从在线存储获取
        if key not in self.online_storage:
            return None
        
        features = self.online_storage[key]
        
        if feature_names:
            features = {k: v for k, v in features.items() if k in feature_names}
        
        # 更新缓存
        cache_key = f"online:{key}"
        self.cache[cache_key] = (
            features,
            datetime.utcnow().timestamp() + self.default_ttl
        )
        
        return features
    
    async def set_online_features(
        self,
        feature_group_id: str,
        entity_id: str,
        features: Dict[str, Any]
    ) -> Dict:
        """
        设置在线特征
        
        Args:
            feature_group_id: 特征组ID
            entity_id: 实体ID
            features: 特征值字典
        """
        key = f"{feature_group_id}:{entity_id}"
        
        # 验证特征
        validation = await self.feature_store.validate_features(
            feature_group_id, features
        )
        
        if not validation["valid"]:
            raise ValueError(f"Invalid features: {validation['errors']}")
        
        # 设置到在线存储
        self.online_storage[key] = features
        
        # 清除缓存
        cache_key = f"online:{key}"
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        return {
            "status": "completed",
            "entity_id": entity_id,
            "features_count": len(features)
        }
    
    async def batch_get_online_features(
        self,
        feature_group_id: str,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        批量获取在线特征
        
        Args:
            feature_group_id: 特征组ID
            entity_ids: 实体ID列表
            feature_names: 要获取的特征列表
        """
        results = {}
        
        for entity_id in entity_ids:
            features = await self.get_online_features(
                feature_group_id=feature_group_id,
                entity_id=entity_id,
                feature_names=feature_names
            )
            results[entity_id] = features
        
        found = sum(1 for v in results.values() if v is not None)
        
        return {
            "total": len(entity_ids),
            "found": found,
            "results": results
        }
    
    async def delete_online_features(
        self,
        feature_group_id: str,
        entity_id: str
    ) -> bool:
        """删除在线特征"""
        key = f"{feature_group_id}:{entity_id}"
        
        if key in self.online_storage:
            del self.online_storage[key]
        
        # 清除缓存
        cache_key = f"online:{key}"
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        return True
    
    async def get_feature_vector(
        self,
        feature_group_id: str,
        entity_id: str,
        feature_names: List[str]
    ) -> List[Any]:
        """获取特征向量 (用于ML)"""
        features = await self.get_online_features(
            feature_group_id=feature_group_id,
            entity_id=entity_id,
            feature_names=feature_names
        )
        
        if not features:
            return [None] * len(feature_names)
        
        return [features.get(name) for name in feature_names]
    
    async def clear_cache(self, feature_group_id: Optional[str] = None):
        """清除缓存"""
        if feature_group_id:
            keys_to_delete = [
                k for k in self.cache.keys()
                if k.startswith(f"online:{feature_group_id}:")
            ]
            for k in keys_to_delete:
                del self.cache[k]
        else:
            self.cache.clear()
        
        return {"cleared": True}
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            "online_entities": len(self.online_storage),
            "cached_items": len(self.cache),
            "cache_ttl_seconds": self.default_ttl
        }

# Feature Serving实例
serving_service = FeatureServing(None)  # 初始化时设置feature_store
