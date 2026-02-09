"""
记忆管理模块
管理Agent的长期记忆和短期记忆
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器 - 管理Agent的各种记忆"""
    
    def __init__(
        self,
        initial_memory: Optional[Dict] = None,
        max_short_term: int = 100,
        max_long_term: int = 1000
    ):
        """
        初始化记忆管理器
        
        Args:
            initial_memory: 初始记忆数据
            max_short_term: 短期记忆最大容量
            max_long_term: 长期记忆最大容量
        """
        # 短期记忆 - 快速访问，容量有限
        self.short_term: deque = deque(maxlen=max_short_term)
        
        # 长期记忆 - 持久化存储
        self.long_term: List[Dict] = []
        
        # 工作记忆 - 当前任务上下文
        self.working_memory: Dict[str, Any] = {}
        
        # 记忆索引
        self._memory_index: Dict[str, List[int]] = {}
        
        # 加载初始记忆
        if initial_memory:
            self._load_initial_memory(initial_memory)
    
    def _load_initial_memory(self, memory: Dict):
        """加载初始记忆"""
        # 设置工作记忆
        if "context" in memory:
            self.working_memory.update(memory["context"])
        
        # 加载历史记忆
        if "memories" in memory:
            for mem in memory["memories"]:
                self.add_memory(mem)
    
    def add_memory(
        self,
        memory_data: Dict[str, Any],
        memory_type: str = "general"
    ) -> int:
        """
        添加记忆
        
        Args:
            memory_data: 记忆数据
            memory_type: 记忆类型 (reasoning, task, context, etc.)
            
        Returns:
            记忆ID
        """
        memory = {
            "id": self._generate_memory_id(),
            "type": memory_type,
            "content": memory_data,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        
        # 根据类型存储到不同区域
        if memory_type == "reasoning":
            self.short_term.append(memory)
        elif memory_type == "task":
            self.long_term.append(memory)
        elif memory_type == "context":
            self.working_memory.update(memory_data)
        else:
            # 默认添加到短期记忆
            self.short_term.append(memory)
        
        # 更新索引
        self._index_memory(memory)
        
        logger.debug(f"Memory added: {memory['id']} ({memory_type})")
        return memory["id"]
    
    def get_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        获取记忆
        
        Args:
            memory_type: 记忆类型过滤
            limit: 返回数量限制
            
        Returns:
            记忆列表
        """
        memories = []
        
        # 收集所有记忆
        if memory_type == "reasoning" or memory_type is None:
            memories.extend(list(self.short_term))
        if memory_type == "task" or memory_type is None:
            memories.extend(self.long_term)
        if memory_type == "context":
            # 工作记忆作为单一记忆返回
            return [{"type": "context", "data": self.working_memory}]
        
        # 限制数量
        if limit and len(memories) > limit:
            memories = memories[-limit:]
        
        # 按时间排序
        memories.sort(key=lambda x: x["timestamp"])
        
        return memories
    
    def get_working_memory(self) -> Dict[str, Any]:
        """获取工作记忆"""
        return self.working_memory.copy()
    
    def update_working_memory(self, key: str, value: Any):
        """更新工作记忆"""
        self.working_memory[key] = value
    
    def clear_working_memory(self):
        """清空工作记忆"""
        self.working_memory.clear()
    
    def clear(self, memory_type: Optional[str] = None):
        """
        清空记忆
        
        Args:
            memory_type: 要清空的记忆类型，None表示清空所有
        """
        if memory_type is None or memory_type == "reasoning":
            self.short_term.clear()
        if memory_type is None or memory_type == "task":
            self.long_term.clear()
        if memory_type is None or memory_type == "context":
            self.working_memory.clear()
        
        logger.info(f"Memory cleared: {memory_type or 'all'}")
    
    def promote_to_long_term(self, memory_id: str) -> bool:
        """
        将短期记忆提升为长期记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功
        """
        for mem in self.short_term:
            if mem["id"] == memory_id:
                mem["promoted_at"] = datetime.now().isoformat()
                self.long_term.append(mem)
                self.short_term.remove(mem)
                return True
        return False
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        搜索记忆
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            匹配的記憶列表
        """
        results = []
        query_lower = query.lower()
        
        for mem in self.short_term:
            if self._match_query(mem, query_lower):
                mem["access_count"] += 1
                results.append(mem)
        
        for mem in self.long_term:
            if self._match_query(mem, query_lower):
                mem["access_count"] += 1
                results.append(mem)
        
        # 按访问频率和相关性排序
        results.sort(key=lambda x: (x["access_count"], x["timestamp"]), reverse=True)
        
        return results[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "working_memory_keys": len(self.working_memory),
            "total_memories": len(self.short_term) + len(self.long_term),
            "memory_types": self._get_memory_types()
        }
    
    def export_memories(self) -> Dict[str, Any]:
        """导出所有记忆"""
        return {
            "short_term": list(self.short_term),
            "long_term": self.long_term,
            "working_memory": self.working_memory,
            "exported_at": datetime.now().isoformat()
        }
    
    def import_memories(self, data: Dict[str, Any]):
        """导入记忆"""
        if "short_term" in data:
            self.short_term.extend(data["short_term"])
        if "long_term" in data:
            self.long_term.extend(data["long_term"])
        if "working_memory" in data:
            self.working_memory.update(data["working_memory"])
    
    def _generate_memory_id(self) -> str:
        """生成唯一记忆ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _index_memory(self, memory: Dict):
        """索引记忆"""
        memory_id = memory["id"]
        memory_type = memory["type"]
        
        if memory_type not in self._memory_index:
            self._memory_index[memory_type] = []
        
        self._memory_index[memory_type].append(memory_id)
    
    def _match_query(self, memory: Dict, query: str) -> bool:
        """检查记忆是否匹配查询"""
        content = str(memory.get("content", ""))
        return query in content.lower()
    
    def _get_memory_types(self) -> List[str]:
        """获取所有记忆类型"""
        return list(self._memory_index.keys())
    
    def consolidate_memories(self, similarity_threshold: float = 0.8):
        """
        记忆整合 - 合并相似的短期记忆
        
        Args:
            similarity_threshold: 相似度阈值
        """
        # 简化实现：合并相同类型的连续记忆
        short_term_list = list(self.short_term)
        
        if len(short_term_list) < 2:
            return
        
        consolidated = []
        current_group = None
        
        for memory in short_term_list:
            if current_group is None:
                current_group = memory
            elif (
                memory["type"] == current_group["type"] and
                memory["timestamp"] > current_group["timestamp"]
            ):
                # 合并内容
                if isinstance(current_group["content"], list):
                    current_group["content"].append(memory["content"])
                else:
                    current_group["content"] = [current_group["content"], memory["content"]]
            else:
                consolidated.append(current_group)
                current_group = memory
        
        if current_group:
            consolidated.append(current_group)
        
        # 更新短期记忆
        self.short_term.clear()
        self.short_term.extend(consolidated)
