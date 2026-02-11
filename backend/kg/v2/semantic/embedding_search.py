"""
向量搜索模块

提供基于向量嵌入的语义搜索功能。
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    source: str  # "vector", "graph", "hybrid"
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata or {}
        }


class VectorStore:
    """向量存储 (简化版)"""
    
    def __init__(self, dimension: int = 768, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.documents: Dict[str, Dict] = {}
        self.vectors: Dict[str, List[float]] = {}
    
    def add(self, 
            doc_id: str,
            content: str,
            vector: List[float],
            metadata: Dict = None):
        """添加文档"""
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {}
        }
        self.vectors[doc_id] = vector
    
    def search(self, 
               query_vector: List[float],
               top_k: int = 10,
               filter_ids: List[str] = None) -> List[Tuple[str, float]]:
        """向量搜索"""
        if not self.vectors:
            return []
        
        query = np.array(query_vector)
        results = []
        
        for doc_id, vector in self.vectors.items():
            if filter_ids and doc_id not in filter_ids:
                continue
            
            doc_vec = np.array(vector)
            
            # 余弦相似度
            similarity = np.dot(query, doc_vec) / (
                np.linalg.norm(query) * np.linalg.norm(doc_vec) + 1e-10
            )
            
            results.append((doc_id, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """删除"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            return True
        return False
    
    def count(self) -> int:
        """计数"""
        return len(self.documents)


class EmbeddingSearch:
    """嵌入搜索服务"""
    
    def __init__(self, 
                 dimension: int = 768,
                 similarity_threshold: float = 0.5):
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.vector_store = VectorStore(dimension=dimension)
        self._entity_cache: Dict[str, Dict] = {}
    
    def index_entity(self,
                    entity_id: str,
                    name: str,
                    entity_type: str,
                    embedding: List[float],
                    properties: Dict = None):
        """索引实体"""
        self.vector_store.add(
            doc_id=entity_id,
            content=name,
            vector=embedding,
            metadata={
                "name": name,
                "type": entity_type,
                "properties": properties or {}
            }
        )
        self._entity_cache[entity_id] = {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "properties": properties or {}
        }
    
    def search(self,
              query_vector: List[float],
              entity_type: str = None,
              top_k: int = 10) -> List[SearchResult]:
        """语义搜索"""
        # 向量搜索
        results = self.vector_store.search(query_vector, top_k=top_k * 2)
        
        search_results = []
        for doc_id, score in results:
            if score < self.similarity_threshold:
                continue
            
            doc = self.vector_store.get_document(doc_id)
            if doc:
                # 类型过滤
                if entity_type and doc["metadata"].get("type") != entity_type:
                    continue
                
                search_results.append(SearchResult(
                    id=doc_id,
                    content=doc["content"],
                    score=score,
                    source="vector",
                    metadata=doc.get("metadata", {})
                ))
        
        return search_results[:top_k]
    
    def search_by_text(self,
                      query_text: str,
                      embedding_model,
                      entity_type: str = None,
                      top_k: int = 10) -> List[SearchResult]:
        """文本搜索 (自动编码)"""
        # 生成查询向量
        query_vector = embedding_model.encode(query_text)
        
        return self.search(query_vector, entity_type, top_k)
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """获取实体"""
        return self._entity_cache.get(entity_id)
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if entity_id in self._entity_cache:
            del self._entity_cache[entity_id]
            return self.vector_store.delete(entity_id)
        return False
    
    def batch_index(self, 
                    entities: List[Dict],
                    embeddings: List[List[float]]):
        """批量索引"""
        for entity, embedding in zip(entities, embeddings):
            self.index_entity(
                entity_id=entity.get("id", ""),
                name=entity.get("name", ""),
                entity_type=entity.get("type", ""),
                embedding=embedding,
                properties=entity.get("properties", {})
            )
    
    def find_similar(self,
                    entity_id: str,
                    top_k: int = 10) -> List[SearchResult]:
        """查找相似实体"""
        if entity_id not in self.vector_store.vectors:
            return []
        
        query_vector = self.vector_store.vectors[entity_id]
        results = self.vector_store.search(query_vector, top_k + 1)
        
        search_results = []
        for doc_id, score in results:
            if doc_id != entity_id:
                doc = self.vector_store.get_document(doc_id)
                if doc:
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=doc["content"],
                        score=score,
                        source="vector",
                        metadata=doc.get("metadata", {})
                    ))
        
        return search_results[:top_k]


# 全局嵌入搜索实例
_embedding_search: Optional[EmbeddingSearch] = None


def get_embedding_search(dimension: int = 768) -> EmbeddingSearch:
    """获取嵌入搜索实例"""
    global _embedding_search
    
    if _embedding_search is None:
        _embedding_search = EmbeddingSearch(dimension=dimension)
    
    return _embedding_search
