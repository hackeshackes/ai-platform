"""
Knowledge Base - Pattern Storage
知识库 - 模式存储
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import json
import hashlib
import logging

from .models import Pattern, PatternRecord, IntentType

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    知识库实现
    
    存储和管理学习到的交互模式
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化知识库
        
        Args:
            storage_path: 持久化存储路径
        """
        self.storage_path = storage_path
        
        # 模式存储
        self._patterns: Dict[str, Pattern] = {}
        
        # 按类型索引
        self._intent_index: Dict[IntentType, List[str]] = defaultdict(list)
        
        # 频率统计
        self._pattern_frequency: Dict[str, int] = defaultdict(int)
        
        # 成功率统计
        self._success_count: Dict[str, int] = defaultdict(int)
        
        # 奖励统计
        self._reward_sum: Dict[str, float] = defaultdict(float)
        
        # 加载已有数据
        if storage_path:
            self._load()
    
    async def add(self, pattern: Pattern) -> str:
        """
        添加模式到知识库
        
        Args:
            pattern: 模式对象
            
        Returns:
            模式ID
        """
        pattern_id = pattern.id
        
        # 存储模式
        self._patterns[pattern_id] = pattern
        
        # 更新索引
        self._intent_index[pattern.intent].append(pattern_id)
        
        # 更新统计
        self._pattern_frequency[pattern_id] += 1
        if pattern.success:
            self._success_count[pattern_id] += 1
        self._reward_sum[pattern_id] += pattern.reward
        
        # 持久化
        if self.storage_path:
            self._save()
        
        logger.info(f"Pattern added: {pattern_id}, intent: {pattern.intent.value}")
        return pattern_id
    
    async def update(self, pattern: Pattern) -> bool:
        """
        更新已有模式
        
        Args:
            pattern: 模式对象
            
        Returns:
            是否更新成功
        """
        pattern_id = pattern.id
        
        if pattern_id not in self._patterns:
            return await self.add(pattern)
        
        # 更新统计
        self._pattern_frequency[pattern_id] += 1
        if pattern.success:
            self._success_count[pattern_id] += 1
        self._reward_sum[pattern_id] += pattern.reward
        
        # 更新模式
        self._patterns[pattern_id] = pattern
        
        # 持久化
        if self.storage_path:
            self._save()
        
        return True
    
    async def get(self, pattern_id: str) -> Optional[Pattern]:
        """
        获取模式
        
        Args:
            pattern_id: 模式ID
            
        Returns:
            模式对象或None
        """
        return self._patterns.get(pattern_id)
    
    async def find_similar(
        self,
        intent: IntentType,
        entities: List[str],
        limit: int = 5
    ) -> List[Pattern]:
        """
        查找相似模式
        
        Args:
            intent: 意图类型
            entities: 实体列表
            limit: 返回数量限制
            
        Returns:
            相似模式列表
        """
        similar_patterns = []
        
        for pattern_id in self._intent_index.get(intent, []):
            pattern = self._patterns.get(pattern_id)
            if pattern:
                # 计算相似度
                pattern_entities = [e.name for e in pattern.entities]
                overlap = len(set(entities) & set(pattern_entities))
                similarity = overlap / max(len(entities), 1)
                
                if similarity > 0:
                    similar_patterns.append((pattern, similarity))
        
        # 按相似度排序
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in similar_patterns[:limit]]
    
    async def list(
        self,
        intent_type: Optional[IntentType] = None,
        limit: int = 100
    ) -> List[PatternRecord]:
        """
        列出所有模式
        
        Args:
            intent_type: 按意图类型过滤
            limit: 返回数量限制
            
        Returns:
            模式记录列表
        """
        pattern_ids = self._intent_index.get(intent_type) if intent_type else self._patterns.keys()
        
        records = []
        for pattern_id in list(pattern_ids)[:limit]:
            pattern = self._patterns.get(pattern_id)
            if pattern:
                record = PatternRecord(
                    id=pattern_id,
                    pattern_type=pattern.intent.value,
                    frequency=self._pattern_frequency[pattern_id],
                    success_count=self._success_count[pattern_id],
                    avg_reward=self._reward_sum[pattern_id] / max(self._pattern_frequency[pattern_id], 1),
                    last_seen=pattern.created_at,
                    metadata=pattern.metadata
                )
                records.append(record)
        
        return records
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计
        
        Returns:
            统计字典
        """
        total_patterns = len(self._patterns)
        total_interactions = sum(self._pattern_frequency.values())
        total_successes = sum(self._success_count.values())
        
        avg_success_rate = total_successes / max(total_interactions, 1)
        
        return {
            "total_patterns": total_patterns,
            "total_interactions": total_interactions,
            "total_successes": total_successes,
            "success_rate": avg_success_rate,
            "intent_distribution": {
                intent.value: len(ids) for intent, ids in self._intent_index.items()
            }
        }
    
    async def clear(self) -> None:
        """清空知识库"""
        self._patterns.clear()
        self._intent_index.clear()
        self._pattern_frequency.clear()
        self._success_count.clear()
        self._reward_sum.clear()
        
        if self.storage_path:
            self._save()
        
        logger.info("Knowledge base cleared")
    
    def _save(self) -> None:
        """保存到文件"""
        if not self.storage_path:
            return
        
        data = {
            "patterns": {
                pid: pattern.to_dict() for pid, pattern in self._patterns.items()
            },
            "frequency": dict(self._pattern_frequency),
            "success_count": dict(self._success_count),
            "reward_sum": dict(self._reward_sum),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Knowledge base saved to {self.storage_path}")
    
    def _load(self) -> None:
        """从文件加载"""
        if not self.storage_path:
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # 恢复模式
            for pid, pattern_data in data.get("patterns", {}).items():
                # 重建Pattern对象
                intent = IntentType(pattern_data.get("intent", "unknown"))
                self._patterns[pid] = Pattern(
                    id=pid,
                    intent=intent,
                    success=pattern_data.get("success", True),
                    reward=pattern_data.get("reward", 0.0)
                )
                self._intent_index[intent].append(pid)
            
            # 恢复统计
            self._pattern_frequency = defaultdict(int, data.get("frequency", {}))
            self._success_count = defaultdict(int, data.get("success_count", {}))
            self._reward_sum = defaultdict(float, data.get("reward_sum", {}))
            
            logger.info(f"Knowledge base loaded from {self.storage_path}")
        except FileNotFoundError:
            logger.info("No existing knowledge base found")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
