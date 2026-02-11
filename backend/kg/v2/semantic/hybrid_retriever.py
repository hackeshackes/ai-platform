"""
混合检索器

结合向量检索和知识图谱检索的统一接口。
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    id: str
    content: str
    score: float
    source: str  # "vector", "graph", "hybrid"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class HybridSearchConfig:
    """混合搜索配置"""
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    vector_top_k: int = 10
    graph_top_k: int = 10
    fusion_method: str = "rrf"  # rrf, weighted, reciprocal
    rrf_k: int = 60
    similarity_threshold: float = 0.5


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, config: HybridSearchConfig = None):
        self.config = config or HybridSearchConfig()
        self._vector_store = None
        self._graph_builder = None
    
    def initialize(self,
                  vector_store = None,
                  graph_builder = None):
        """初始化"""
        self._vector_store = vector_store
        self._graph_builder = graph_builder
    
    def search(self,
              query: str = None,
              query_vector: List[float] = None,
              entity_type: str = None,
              top_k: int = 10) -> List[HybridSearchResult]:
        """混合搜索"""
        start_time = time.time()
        
        vector_results = []
        graph_results = []
        
        # 向量搜索
        if query_vector is not None and self._vector_store:
            vector_results = self._vector_search(query_vector, entity_type, self.config.vector_top_k)
        
        # 图搜索
        if query and self._graph_builder:
            graph_results = self._graph_search(query, entity_type, self.config.graph_top_k)
        
        # 融合结果
        fused_results = self._fuse_results(vector_results, graph_results)
        
        # 性能日志
        elapsed = (time.time() - start_time) * 1000
        
        return fused_results[:top_k]
    
    def _vector_search(self,
                      query_vector: List[float],
                      entity_type: str = None,
                      top_k: int = 10) -> List[HybridSearchResult]:
        """向量检索"""
        if not self._vector_store:
            return []
        
        results = self._vector_store.search(query_vector, top_k * 2)
        
        search_results = []
        for doc_id, score in results:
            if score < self.config.similarity_threshold:
                continue
            
            doc = self._vector_store.get_document(doc_id)
            if doc:
                if entity_type and doc["metadata"].get("type") != entity_type:
                    continue
                
                search_results.append(HybridSearchResult(
                    id=doc_id,
                    content=doc["content"],
                    score=score,
                    source="vector",
                    metadata=doc.get("metadata", {})
                ))
        
        return search_results[:top_k]
    
    def _graph_search(self,
                     query: str,
                     entity_type: str = None,
                     top_k: int = 10) -> List[HybridSearchResult]:
        """图谱检索"""
        if not self._graph_builder:
            return []
        
        results = []
        
        # 按名称搜索
        entity = self._graph_builder.get_entity_by_name(query)
        if entity:
            neighbors = self._graph_builder.get_neighbors(entity.id)
            for neighbor, relation in neighbors[:top_k]:
                score = relation.weight * 0.9
                
                # 类型过滤
                if entity_type and neighbor.type != entity_type:
                    continue
                
                results.append(HybridSearchResult(
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
        
        # 按类型搜索
        if not results and entity_type:
            entities = self._graph_builder.list_entities(entity_type, limit=top_k)
            for entity in entities:
                results.append(HybridSearchResult(
                    id=entity.id,
                    content=entity.properties.get("description", entity.name),
                    score=0.5,
                    source="graph",
                    metadata={
                        "entity_type": entity.type,
                        "properties": entity.properties
                    }
                ))
        
        return results
    
    def _fuse_results(self,
                     vector_results: List[HybridSearchResult],
                     graph_results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """融合结果"""
        all_results = vector_results + graph_results
        
        if not all_results:
            return []
        
        if self.config.fusion_method == "rrf":
            return self._rrf_fusion(all_results)
        elif self.config.fusion_method == "weighted":
            return self._weighted_fusion(vector_results, graph_results)
        else:
            return self._reciprocal_fusion(vector_results, graph_results)
    
    def _rrf_fusion(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """RRF融合"""
        from collections import defaultdict
        
        doc_scores = defaultdict(float)
        
        for rank, result in enumerate(results, 1):
            doc_scores[result.id] += 1.0 / (self.config.rrf_k + rank)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused = []
        seen = set()
        for doc_id, score in sorted_docs:
            if doc_id not in seen:
                seen.add(doc_id)
                original = next((r for r in results if r.id == doc_id), None)
                if original:
                    fused.append(HybridSearchResult(
                        id=doc_id,
                        content=original.content,
                        score=score,
                        source="hybrid",
                        metadata=original.metadata
                    ))
        
        return fused
    
    def _weighted_fusion(self,
                        vector_results: List[HybridSearchResult],
                        graph_results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """加权融合"""
        def normalize(results):
            if not results:
                return []
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            if max_score == min_score:
                return [HybridSearchResult(r.id, r.content, 0.5, "hybrid", r.metadata) for r in results]
            return [HybridSearchResult(
                r.id, r.content,
                (r.score - min_score) / (max_score - min_score),
                "hybrid", r.metadata
            ) for r in results]
        
        vector_norm = normalize(vector_results)
        graph_norm = normalize(graph_results)
        
        doc_scores = {}
        
        for r in vector_norm:
            doc_scores[r.id] = (
                r.score * self.config.vector_weight,
                r.content, r.metadata
            )
        
        for r in graph_norm:
            if r.id in doc_scores:
                score, content, metadata = doc_scores[r.id]
                doc_scores[r.id] = (
                    score + r.score * self.config.graph_weight,
                    content, metadata
                )
            else:
                doc_scores[r.id] = (
                    r.score * self.config.graph_weight,
                    r.content, r.metadata
                )
        
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)
        
        return [HybridSearchResult(
            doc_id, content, score, "hybrid", metadata
        ) for doc_id, (score, content, metadata) in sorted_results]
    
    def _reciprocal_fusion(self,
                          vector_results: List[HybridSearchResult],
                          graph_results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        """倒数排序融合"""
        from collections import defaultdict
        
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
        
        return [HybridSearchResult(
            doc_id, content, score, "hybrid", metadata
        ) for doc_id, score, content, metadata in fusion_scores]
    
    def set_vector_store(self, vector_store):
        """设置向量存储"""
        self._vector_store = vector_store
    
    def set_graph_builder(self, graph_builder):
        """设置图构建器"""
        self._graph_builder = graph_builder


# 全局混合检索器实例
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(config: HybridSearchConfig = None) -> HybridRetriever:
    """获取混合检索器实例"""
    global _hybrid_retriever
    
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(config)
    
    return _hybrid_retriever
