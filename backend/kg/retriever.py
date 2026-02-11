"""
混合检索模块 - 结合向量检索和知识图谱检索
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict
import re


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    source: str  # "vector" or "graph" or "hybrid"
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata or {}
        }


@dataclass
class HybridSearchConfig:
    """混合搜索配置"""
    vector_weight: float = 0.5  # 向量检索权重
    graph_weight: float = 0.5   # 图谱检索权重
    vector_top_k: int = 10      # 向量检索返回数量
    graph_top_k: int = 10       # 图谱检索返回数量
    fusion_method: str = "rrf"  # rrf, weighted, reciprocal
    rrf_k: int = 60             # RRF参数


class VectorStore:
    """向量存储 (简化版，支持FAISS/ Milvus)"""
    
    def __init__(self, dimension: int = 768, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.documents: Dict[str, Dict] = {}
        self.vectors: Dict[str, List[float]] = {}
        self._index = None
    
    def add(self, doc_id: str, 
            content: str, 
            vector: List[float],
            metadata: Dict = None):
        """添加文档"""
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {}
        }
        self.vectors[doc_id] = vector
        self._index = None  # 标记需要重建索引
    
    def search(self, query_vector: List[float], 
               top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索"""
        if not self.vectors:
            return []
        
        # 简化的暴力搜索
        query = np.array(query_vector)
        results = []
        
        for doc_id, vector in self.vectors.items():
            doc_vec = np.array(vector)
            
            # 计算余弦相似度
            similarity = np.dot(query, doc_vec) / (
                np.linalg.norm(query) * np.linalg.norm(doc_vec) + 1e-10
            )
            
            results.append((doc_id, float(similarity)))
        
        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            self._index = None
            return True
        return False
    
    def count(self) -> int:
        """获取文档数量"""
        return len(self.documents)
    
    def build_index(self):
        """构建索引 (实际项目中可以使用FAISS/Milvus)"""
        # 这里预留索引构建逻辑
        pass


class HybridRetriever:
    """混合检索器 - 结合向量和图谱检索"""
    
    def __init__(self, 
                 vector_store: VectorStore = None,
                 graph_manager = None):
        self.vector_store = vector_store or VectorStore()
        self.graph_manager = graph_manager
        self.config = HybridSearchConfig()
    
    def set_config(self, config: HybridSearchConfig):
        """设置配置"""
        self.config = config
    
    def search_vector(self, 
                      query_vector: List[float],
                      query_text: str = None) -> List[SearchResult]:
        """向量检索"""
        results = self.vector_store.search(query_vector, self.config.vector_top_k)
        
        search_results = []
        for doc_id, score in results:
            doc = self.vector_store.get_document(doc_id)
            if doc:
                search_results.append(SearchResult(
                    id=doc_id,
                    content=doc["content"],
                    score=score,
                    source="vector",
                    metadata=doc.get("metadata", {})
                ))
        
        return search_results
    
    def search_graph(self, 
                     query: str,
                     entity_type: str = None,
                     max_depth: int = 2) -> List[SearchResult]:
        """图谱检索"""
        if not self.graph_manager:
            return []
        
        results = []
        
        # 1. 按名称搜索实体
        entity = self.graph_manager.get_entity_by_name(query)
        if entity:
            neighbors = self.graph_manager.get_neighbors(entity.id)
            for neighbor, relation in neighbors:
                score = relation.weight * 0.9
                results.append(SearchResult(
                    id=neighbor.id,
                    content=f"{entity.name} --[{relation.relation_type}]--> {neighbor.name}",
                    score=score,
                    source="graph",
                    metadata={
                        "entity_type": neighbor.type,
                        "relation_type": relation.relation_type,
                        "properties": neighbor.properties
                    }
                ))
        
        # 2. 如果没找到精确匹配，搜索相关类型实体
        if not results and entity_type:
            entities = self.graph_manager.list_entities(entity_type, limit=self.config.graph_top_k)
            for entity in entities:
                score = 0.5  # 基础分数
                results.append(SearchResult(
                    id=entity.id,
                    content=entity.properties.get("description", entity.name),
                    score=score,
                    source="graph",
                    metadata={
                        "entity_type": entity.type,
                        "properties": entity.properties
                    }
                ))
        
        return results
    
    def hybrid_search(self, 
                      query: str,
                      query_vector: List[float] = None,
                      entity_type: str = None) -> List[SearchResult]:
        """混合搜索"""
        vector_results = []
        graph_results = []
        
        # 并行执行两种检索
        if query_vector is not None:
            vector_results = self.search_vector(query_vector, query)
        
        if self.graph_manager:
            graph_results = self.search_graph(query, entity_type)
        
        # 结果融合
        fused_results = self._fuse_results(vector_results, graph_results)
        
        return fused_results
    
    def _fuse_results(self, 
                      vector_results: List[SearchResult],
                      graph_results: List[SearchResult]) -> List[SearchResult]:
        """融合检索结果"""
        all_results = vector_results + graph_results
        
        if not all_results:
            return []
        
        if self.config.fusion_method == "rrf":
            return self._rrf_fusion(all_results)
        elif self.config.fusion_method == "weighted":
            return self._weighted_fusion(vector_results, graph_results)
        else:
            return self._reciprocal_fusion(vector_results, graph_results)
    
    def _rrf_fusion(self, results: List[SearchResult]) -> List[SearchResult]:
        """RRF (Reciprocal Rank Fusion)"""
        doc_scores = defaultdict(float)
        
        for rank, result in enumerate(results, 1):
            doc_scores[result.id] += 1.0 / (self.config.rrf_k + rank)
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建结果
        fused = []
        seen = set()
        for doc_id, score in sorted_docs:
            if doc_id not in seen:
                seen.add(doc_id)
                # 找到原始结果
                original = next((r for r in results if r.id == doc_id), None)
                if original:
                    fused.append(SearchResult(
                        id=doc_id,
                        content=original.content,
                        score=score,
                        source="hybrid",
                        metadata=original.metadata
                    ))
        
        return fused
    
    def _weighted_fusion(self, 
                        vector_results: List[SearchResult],
                        graph_results: List[SearchResult]) -> List[SearchResult]:
        """加权融合"""
        # 归一化分数
        def normalize(results):
            if not results:
                return []
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            if max_score == min_score:
                return [SearchResult(
                    r.id, r.content, 0.5, "hybrid", r.metadata
                ) for r in results]
            return [SearchResult(
                r.id, r.content,
                (r.score - min_score) / (max_score - min_score),
                "hybrid", r.metadata
            ) for r in results]
        
        vector_norm = normalize(vector_results)
        graph_norm = normalize(graph_results)
        
        # 合并分数
        doc_scores: Dict[str, Tuple[float, str, str, Dict]] = {}
        
        for r in vector_norm:
            doc_scores[r.id] = (
                r.score * self.config.vector_weight,
                r.content, "hybrid", r.metadata
            )
        
        for r in graph_norm:
            if r.id in doc_scores:
                score, content, _, metadata = doc_scores[r.id]
                doc_scores[r.id] = (
                    score + r.score * self.config.graph_weight,
                    content, "hybrid", metadata
                )
            else:
                doc_scores[r.id] = (
                    r.score * self.config.graph_weight,
                    r.content, "hybrid", r.metadata
                )
        
        # 排序
        sorted_results = sorted(
            doc_scores.items(), 
            key=lambda x: x[1][0], 
            reverse=True
        )
        
        return [SearchResult(
            doc_id, content, score, source, metadata
        ) for doc_id, (score, content, source, metadata) in sorted_results]
    
    def _reciprocal_fusion(self,
                          vector_results: List[SearchResult],
                          graph_results: List[SearchResult]) -> List[SearchResult]:
        """倒数排序融合"""
        doc_scores = defaultdict(lambda: {"vector": None, "graph": None, "content": "", "metadata": {}})
        
        for rank, r in enumerate(vector_results, 1):
            doc_scores[r.id]["vector"] = rank
            doc_scores[r.id]["content"] = r.content
            doc_scores[r.id]["metadata"] = r.metadata
        
        for rank, r in enumerate(graph_results, 1):
            if doc_scores[r.id]["graph"] is None:
                doc_scores[r.id]["graph"] = rank
            doc_scores[r.id]["content"] = r.content
            doc_scores[r.id]["metadata"] = r.metadata
        
        # 计算融合分数
        fusion_scores = []
        for doc_id, info in doc_scores.items():
            vector_rank = info["vector"] or float('inf')
            graph_rank = info["graph"] or float('inf')
            
            fusion_score = (
                1.0 / (self.config.rrf_k + vector_rank) + 
                1.0 / (self.config.rrf_k + graph_rank)
            )
            
            fusion_scores.append((doc_id, fusion_score, info["content"], info["metadata"]))
        
        fusion_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [SearchResult(
            doc_id, content, score, "hybrid", metadata
        ) for doc_id, score, content, metadata in fusion_scores]
    
    def add_to_vector_store(self,
                           doc_id: str,
                           content: str,
                           vector: List[float],
                           metadata: Dict = None):
        """添加文档到向量存储"""
        self.vector_store.add(doc_id, content, vector, metadata)
    
    def set_graph_manager(self, graph_manager):
        """设置图谱管理器"""
        self.graph_manager = graph_manager


# 全局检索器实例
_hybrid_retriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器实例"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def hybrid_search(query: str,
                  query_vector: List[float] = None,
                  entity_type: str = None,
                  config: HybridSearchConfig = None) -> List[SearchResult]:
    """便捷的混合搜索函数"""
    retriever = get_hybrid_retriever()
    if config:
        retriever.set_config(config)
    return retriever.hybrid_search(query, query_vector, entity_type)
