"""
混合推理引擎

结合规则推理和神经网络推理的统一接口。
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReasoningType(str, Enum):
    """推理类型"""
    RULE_BASED = "rule"
    NEURAL = "neural"
    HYBRID = "hybrid"
    PATH_FINDING = "path"


@dataclass
class HybridInferenceResult:
    """混合推理结果"""
    conclusion: str
    confidence: float
    evidence: List[Dict]
    reasoning_type: ReasoningType
    rule_id: Optional[str] = None
    method: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reasoning_type": self.reasoning_type.value,
            "rule_id": self.rule_id,
            "method": self.method,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReasoningConfig:
    """推理配置"""
    max_depth: int = 3
    rule_weight: float = 0.6  # 规则推理权重
    neural_weight: float = 0.4  # 神经网络推理权重
    enable_rules: bool = True
    enable_neural: bool = True
    enable_path_finding: bool = True
    similarity_threshold: float = 0.7
    top_k_similar: int = 10


class HybridReasoner:
    """混合推理引擎"""
    
    def __init__(self, config: ReasoningConfig = None):
        self.config = config or ReasoningConfig()
        self._rule_engine = None
        self._neural_reasoner = None
    
    def _get_rule_engine(self):
        """获取规则引擎"""
        if self._rule_engine is None:
            from kg.v2.reasoning.rule_engine import get_rule_engine
            self._rule_engine = get_rule_engine()
        return self._rule_engine
    
    def _get_neural_reasoner(self):
        """获取神经网络推理器"""
        if self._neural_reasoner is None:
            from kg.v2.reasoning.neural_reasoner import get_neural_reasoner
            self._neural_reasoner = get_neural_reasoner()
            self._neural_reasoner.similarity_threshold = self.config.similarity_threshold
        return self._neural_reasoner
    
    def reason(self,
              entity_id: str,
              graph_builder,
              reasoning_type: ReasoningType = ReasoningType.HYBRID) -> List[HybridInferenceResult]:
        """推理"""
        results = []
        
        if reasoning_type in [ReasoningType.RULE_BASED, ReasoningType.HYBRID]:
            if self.config.enable_rules:
                rule_results = self._get_rule_engine().reason(
                    entity_id, graph_builder, self.config.max_depth
                )
                
                for r in rule_results:
                    results.append(HybridInferenceResult(
                        conclusion=r.conclusion,
                        confidence=r.confidence * self.config.rule_weight,
                        evidence=r.evidence,
                        reasoning_type=ReasoningType.RULE_BASED,
                        rule_id=r.rule_id
                    ))
        
        if reasoning_type in [ReasoningType.NEURAL, ReasoningType.HYBRID]:
            if self.config.enable_neural:
                neural_results = self._get_neural_reasoner().reason_entity(
                    entity_id, graph_builder, self.config.max_depth
                )
                
                for r in neural_results:
                    results.append(HybridInferenceResult(
                        conclusion=r.conclusion,
                        confidence=r.confidence * self.config.neural_weight,
                        evidence=r.evidence,
                        reasoning_type=ReasoningType.NEURAL,
                        method=r.method
                    ))
        
        if reasoning_type == ReasoningType.PATH_FINDING or self.config.enable_path_finding:
            # 路径查找推理
            entity = graph_builder.get_entity(entity_id)
            if entity:
                neighbors = graph_builder.get_neighbors(entity_id)
                for neighbor, relation in neighbors:
                    path = graph_builder.get_shortest_path(entity_id, neighbor.id)
                    if path and len(path) > 1:
                        results.append(HybridInferenceResult(
                            conclusion=f"{entity.name} -> {neighbor.name}",
                            confidence=relation.weight,
                            evidence=[{"path_length": len(path)}],
                            reasoning_type=ReasoningType.PATH_FINDING
                        ))
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def reason_with_query(self,
                          query: str,
                          graph_builder,
                          reasoning_type: ReasoningType = ReasoningType.HYBRID) -> List[HybridInferenceResult]:
        """基于查询推理"""
        results = []
        
        if reasoning_type in [ReasoningType.RULE_BASED, ReasoningType.HYBRID]:
            if self.config.enable_rules:
                rule_results = self._get_rule_engine().reason_with_query(
                    query, graph_builder
                )
                
                for r in rule_results:
                    results.append(HybridInferenceResult(
                        conclusion=r.conclusion,
                        confidence=r.confidence * self.config.rule_weight,
                        evidence=r.evidence,
                        reasoning_type=ReasoningType.RULE_BASED,
                        rule_id=r.rule_id
                    ))
        
        return results
    
    def find_path(self,
                 source_id: str,
                 target_id: str,
                 graph_builder) -> List[HybridInferenceResult]:
        """查找路径"""
        results = []
        
        # 最短路径
        path = graph_builder.get_shortest_path(source_id, target_id)
        if path:
            results.append(HybridInferenceResult(
                conclusion=f"找到路径，长度: {len(path)-1}",
                confidence=1.0,
                evidence=[{"entities": [e.name for e in path]}],
                reasoning_type=ReasoningType.PATH_FINDING,
                method="shortest_path"
            ))
        
        # 神经网络推理的路径
        neural_results = self._get_neural_reasoner().path_based_reasoning(
            source_id, target_id, graph_builder
        )
        
        for r in neural_results:
            results.append(HybridInferenceResult(
                conclusion=r.conclusion,
                confidence=r.confidence * self.config.neural_weight,
                evidence=r.evidence,
                reasoning_type=ReasoningType.NEURAL,
                method=r.method
            ))
        
        return results
    
    def reason_batch(self,
                    entity_ids: List[str],
                    graph_builder,
                    reasoning_type: ReasoningType = ReasoningType.HYBRID) -> Dict[str, List[HybridInferenceResult]]:
        """批量推理"""
        results = {}
        
        for entity_id in entity_ids:
            results[entity_id] = self.reason(entity_id, graph_builder, reasoning_type)
        
        return results


# 全局混合推理器实例
_hybrid_reasoner: Optional[HybridReasoner] = None


def get_hybrid_reasoner(config: ReasoningConfig = None) -> HybridReasoner:
    """获取混合推理器实例"""
    global _hybrid_reasoner
    
    if _hybrid_reasoner is None:
        _hybrid_reasoner = HybridReasoner(config)
    
    return _hybrid_reasoner
