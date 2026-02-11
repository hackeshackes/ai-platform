"""
神经网络推理引擎

基于神经网络的推理，支持实体嵌入和关系嵌入。
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict


class ReasoningType(str, Enum):
    """推理类型"""
    RULE_BASED = "rule"
    NEURAL = "neural"
    HYBRID = "hybrid"
    PATH_FINDING = "path"


@dataclass
class NeuralInferenceResult:
    """神经网络推理结果"""
    conclusion: str
    confidence: float
    evidence: List[Dict]
    method: str  # similarity, path_reasoning, link_prediction
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "method": self.method,
            "timestamp": self.timestamp.isoformat()
        }


class NeuralReasoner:
    """神经网络推理器"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 similarity_threshold: float = 0.7):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self._entity_embeddings: Dict[str, List[float]] = {}
        self._relation_embeddings: Dict[str, List[float]] = {}
        self._initialized = False
    
    def initialize(self, 
                   graph_builder = None,
                   embeddings: Dict[str, List[float]] = None):
        """初始化推理器"""
        self._initialized = True
        
        if embeddings:
            self._entity_embeddings = embeddings
        
        if graph_builder:
            entities = graph_builder.list_entities(limit=10000)
            for entity in entities:
                if entity.embeddings:
                    self._entity_embeddings[entity.id] = entity.embeddings
    
    def add_entity_embedding(self, 
                            entity_id: str, 
                            embedding: List[float]):
        """添加实体嵌入"""
        self._entity_embeddings[entity_id] = embedding
    
    def _compute_similarity(self, 
                           vec1: List[float], 
                           vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def find_similar_entities(self,
                              entity_id: str,
                              top_k: int = 10) -> List[Tuple[Dict, float]]:
        """查找相似实体"""
        if entity_id not in self._entity_embeddings:
            return []
        
        query_vec = self._entity_embeddings[entity_id]
        similarities = []
        
        for other_id, emb in self._entity_embeddings.items():
            if other_id != entity_id:
                sim = self._compute_similarity(query_vec, emb)
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for entity_id, sim in similarities[:top_k]:
            if sim >= self.similarity_threshold:
                results.append(({"id": entity_id, "similarity": sim}, sim))
        
        return results
    
    def link_prediction(self,
                       source_id: str,
                       target_id: str,
                       relation_type: str = None) -> List[NeuralInferenceResult]:
        """链接预测"""
        results = []
        
        if source_id not in self._entity_embeddings or target_id not in self._entity_embeddings:
            return results
        
        src_emb = self._entity_embeddings[source_id]
        tgt_emb = self._entity_embeddings[target_id]
        similarity = self._compute_similarity(src_emb, tgt_emb)
        
        if similarity >= self.similarity_threshold:
            results.append(NeuralInferenceResult(
                conclusion=f"{source_id} 和 {target_id} 可能具有相似性",
                confidence=similarity,
                evidence=[{"similarity": similarity}],
                method="similarity"
            ))
        
        avg_emb = (np.array(src_emb) + np.array(tgt_emb)) / 2
        
        results.append(NeuralInferenceResult(
            conclusion=f"链接预测: {source_id} -> {target_id}",
            confidence=similarity,
            evidence=[{
                "source_embedding_norm": float(np.linalg.norm(src_emb)),
                "target_embedding_norm": float(np.linalg.norm(tgt_emb))
            }],
            method="link_prediction"
        ))
        
        return results
    
    def path_based_reasoning(self,
                            source_id: str,
                            target_id: str,
                            graph_builder,
                            max_path_length: int = 3) -> List[NeuralInferenceResult]:
        """基于路径的推理"""
        results = []
        
        paths = graph_builder.get_all_paths(
            source_id, target_id, max_depth=max_path_length
        )
        
        if not paths:
            shortest = graph_builder.get_shortest_path(source_id, target_id)
            if shortest and len(shortest) > 1:
                paths = [shortest]
        
        for path in paths:
            if len(path) < 2:
                continue
            
            path_embeddings = []
            for entity in path:
                if entity.id in self._entity_embeddings:
                    path_embeddings.append(self._entity_embeddings[entity.id])
            
            if not path_embeddings:
                continue
            
            path_emb = np.mean(path_embeddings, axis=0)
            confidence = 1.0 / len(path)
            
            results.append(NeuralInferenceResult(
                conclusion=f"{path[0].name} -> ... -> {path[-1].name} (路径长度: {len(path)-1})",
                confidence=confidence,
                evidence=[{
                    "path_length": len(path) - 1,
                    "entities": [e.name for e in path],
                    "path_embedding_norm": float(np.linalg.norm(path_emb))
                }],
                method="path_reasoning"
            ))
        
        return results
    
    def reason_entity(self,
                      entity_id: str,
                      graph_builder,
                      max_depth: int = 2) -> List[NeuralInferenceResult]:
        """对实体进行推理"""
        results = []
        
        entity = graph_builder.get_entity(entity_id)
        if not entity:
            return results
        
        # 相似实体
        similar = self.find_similar_entities(entity_id, top_k=5)
        for similar_entity, sim in similar:
            results.append(NeuralInferenceResult(
                conclusion=f"{entity.name} 与 {similar_entity['id']} 相似",
                confidence=sim,
                evidence=[{"similar_entity_id": similar_entity['id']}],
                method="similarity"
            ))
        
        # 邻居实体的类型推断
        neighbors = graph_builder.get_neighbors(entity_id)
        neighbor_types = defaultdict(list)
        for neighbor, relation in neighbors:
            neighbor_types[neighbor.type].append(neighbor.name)
        
        for neighbor_type, names in neighbor_types.items():
            if len(names) >= 2:
                results.append(NeuralInferenceResult(
                    conclusion=f"{entity.name} 可能与 {neighbor_type} 类型相关",
                    confidence=min(0.5 + len(names) * 0.1, 0.95),
                    evidence=[{"neighbor_type": neighbor_type, "neighbors": names}],
                    method="type_inference"
                ))
        
        # 链接预测
        for neighbor, _ in neighbors[:3]:
            link_results = self.link_prediction(entity_id, neighbor.id)
            results.extend(link_results)
        
        return results
    
    def batch_reason(self,
                     entity_ids: List[str],
                     graph_builder,
                     max_depth: int = 2) -> Dict[str, List[NeuralInferenceResult]]:
        """批量推理"""
        results = {}
        
        for entity_id in entity_ids:
            results[entity_id] = self.reason_entity(
                entity_id, graph_builder, max_depth
            )
        
        return results


# 全局神经网络推理器实例
_neural_reasoner: Optional[NeuralReasoner] = None


def get_neural_reasoner(embedding_dim: int = 768) -> NeuralReasoner:
    """获取神经网络推理器实例"""
    global _neural_reasoner
    
    if _neural_reasoner is None:
        _neural_reasoner = NeuralReasoner(embedding_dim=embedding_dim)
    
    return _neural_reasoner
