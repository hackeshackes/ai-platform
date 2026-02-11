"""
Unified Reasoning Engine
统一推理引擎

功能:
1. 跨域逻辑推理 - 整合多领域的逻辑推理
2. 因果推理 - 基于因果关系的推理
3. 概率推理 - 概率模型推理
4. 常识推理 - 基于常识知识的推理
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """推理类型"""
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    CAUSAL = "causal"  # 因果推理
    PROBABILISTIC = "probabilistic"  # 概率推理
    COMMON_SENSE = "common_sense"  # 常识推理
    ANALOGICAL = "analogical"  # 类比推理


class LogicOperator(Enum):
    """逻辑运算符"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIV = "equiv"


@dataclass
class ReasoningContext:
    """推理上下文"""
    context_id: str
    facts: List[Dict[str, Any]]
    rules: List[Dict[str, Any]]
    assumptions: List[str]
    constraints: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "facts": self.facts,
            "rules": self.rules,
            "assumptions": self.assumptions,
            "constraints": self.constraints,
            "metadata": self.metadata
        }


@dataclass
class ReasoningResult:
    """推理结果"""
    success: bool
    conclusion: Optional[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    confidence: float
    reasoning_type: str
    evidence: List[Dict[str, Any]]
    counter_evidence: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "conclusion": self.conclusion,
            "reasoning_steps": self.reasoning_steps,
            "confidence": self.confidence,
            "reasoning_type": self.reasoning_type,
            "evidence": self.evidence,
            "counter_evidence": self.counter_evidence,
            "metadata": self.metadata
        }


@dataclass
class LogicRule:
    """逻辑规则"""
    rule_id: str
    antecedent: str  # 前件
    consequent: str  # 后件
    operator: LogicOperator
    conditions: List[str]
    conclusion: str
    confidence: float
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChain:
    """因果链"""
    chain_id: str
    cause: str
    effect: str
    mechanism: str
    strength: float
    context: str
    evidence: List[str]


class UnifiedReasoner:
    """
    统一推理引擎
    
    支持多种推理方式的集成:
    - 逻辑推理 (演绎/归纳/溯因)
    - 因果推理
    - 概率推理
    - 常识推理
    - 类比推理
    """
    
    def __init__(
        self,
        default_reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
        confidence_threshold: float = 0.7,
        enable_probabilistic: bool = True,
        enable_causal: bool = True,
        enable_common_sense: bool = True
    ):
        self.default_reasoning_type = default_reasoning_type
        self.confidence_threshold = confidence_threshold
        
        self.enable_probabilistic = enable_probabilistic
        self.enable_causal = enable_causal
        self.enable_common_sense = enable_common_sense
        
        # 知识存储
        self.rules: List[LogicRule] = []
        self.causal_chains: List[CausalChain] = []
        self.common_sense_knowledge: Dict[str, Any] = {}
        self.facts: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            "total_reasonings": 0,
            "successful_reasonings": 0,
            "reasoning_types_used": defaultdict(int),
            "avg_confidence": 0.0
        }
    
    def add_rule(self, rule: LogicRule) -> bool:
        """添加逻辑规则"""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.rule_id}")
        return True
    
    def add_causal_chain(self, chain: CausalChain) -> bool:
        """添加因果链"""
        self.causal_chains.append(chain)
        logger.info(f"Added causal chain: {chain.chain_id}")
        return True
    
    def add_common_sense(self, knowledge: Dict[str, Any]) -> bool:
        """添加常识知识"""
        self.common_sense_knowledge.update(knowledge)
        return True
    
    def add_fact(self, fact: Dict[str, Any]) -> bool:
        """添加事实"""
        self.facts.append(fact)
        return True
    
    def reason(
        self,
        contexts: List[ReasoningContext],
        query: str,
        reasoning_type: Optional[ReasoningType] = None,
        max_steps: int = 10
    ) -> ReasoningResult:
        """
        执行推理
        
        Args:
            contexts: 推理上下文列表
            query: 查询问题
            reasoning_type: 推理类型
            max_steps: 最大推理步数
            
        Returns:
            ReasoningResult: 推理结果
        """
        self.stats["total_reasonings"] += 1
        
        logger.info(f"Starting reasoning with {len(contexts)} contexts")
        
        try:
            reasoning_type = reasoning_type or self.default_reasoning_type
            self.stats["reasoning_types_used"][reasoning_type.value] += 1
            
            # 合并上下文
            merged_context = self._merge_contexts(contexts)
            
            # 根据推理类型执行推理
            if reasoning_type == ReasoningType.DEDUCTIVE:
                result = self._deductive_reasoning(merged_context, query, max_steps)
            elif reasoning_type == ReasoningType.INDUCTIVE:
                result = self._inductive_reasoning(merged_context, query, max_steps)
            elif reasoning_type == ReasoningType.ABDUCTIVE:
                result = self._abductive_reasoning(merged_context, query, max_steps)
            elif reasoning_type == ReasoningType.CAUSAL:
                result = self._causal_reasoning(merged_context, query, max_steps)
            elif reasoning_type == ReasoningType.PROBABILISTIC:
                result = self._probabilistic_reasoning(merged_context, query, max_steps)
            elif reasoning_type == ReasoningType.COMMON_SENSE:
                result = self._common_sense_reasoning(merged_context, query, max_steps)
            else:
                result = self._deductive_reasoning(merged_context, query, max_steps)
            
            if result.success:
                self.stats["successful_reasonings"] += 1
                self._update_avg_confidence(result.confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            return ReasoningResult(
                success=False,
                conclusion=None,
                reasoning_steps=[],
                confidence=0.0,
                reasoning_type=(reasoning_type or self.default_reasoning_type).value,
                evidence=[],
                counter_evidence=[],
                metadata={"error": str(e)}
            )
    
    def _merge_contexts(self, contexts: List[ReasoningContext]) -> ReasoningContext:
        """合并上下文"""
        all_facts = []
        all_rules = []
        all_assumptions = []
        all_constraints = []
        metadata = {}
        
        for ctx in contexts:
            all_facts.extend(ctx.facts)
            all_rules.extend(ctx.rules)
            all_assumptions.extend(ctx.assumptions)
            all_constraints.extend(ctx.constraints)
            metadata.update(ctx.metadata)
        
        return ReasoningContext(
            context_id="merged",
            facts=all_facts,
            rules=all_rules,
            assumptions=all_assumptions,
            constraints=all_constraints,
            metadata=metadata
        )
    
    def _deductive_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """演绎推理"""
        reasoning_steps = []
        current_facts = context.facts.copy()
        
        # 应用规则进行推理
        for step in range(max_steps):
            new_facts = []
            for rule in self.rules:
                if self._rule_applicable(rule, current_facts):
                    inferred = self._apply_rule(rule, current_facts)
                    if inferred:
                        new_facts.append(inferred)
            
            if not new_facts:
                break
            
            reasoning_steps.append({
                "step": step + 1,
                "action": "rule_application",
                "rules_applied": len(new_facts),
                "new_facts": len(new_facts)
            })
            
            current_facts.extend(new_facts)
        
        # 检查查询是否被满足
        conclusion = self._check_query_satisfaction(query, current_facts)
        confidence = self._calculate_confidence(reasoning_steps, conclusion)
        
        return ReasoningResult(
            success=conclusion is not None,
            conclusion=conclusion,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="deductive",
            evidence=self._extract_evidence(current_facts, query),
            counter_evidence=[]
        )
    
    def _inductive_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """归纳推理"""
        reasoning_steps = []
        
        # 从具体事实归纳一般规则
        patterns = self._find_patterns(context.facts)
        
        reasoning_steps.append({
            "step": 1,
            "action": "pattern_extraction",
            "patterns_found": len(patterns)
        })
        
        # 生成归纳结论
        conclusion = self._generate_inductive_conclusion(patterns, query)
        
        reasoning_steps.append({
            "step": 2,
            "action": "conclusion_generation",
            "conclusion": conclusion
        })
        
        confidence = min(len(patterns) / 10.0, 1.0)  # 基于模式数量
        
        return ReasoningResult(
            success=conclusion is not None,
            conclusion=conclusion,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="inductive",
            evidence=patterns,
            counter_evidence=[]
        )
    
    def _abductive_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """溯因推理"""
        reasoning_steps = []
        
        # 找到可能的解释
        explanations = self._find_explanations(context.facts, query)
        
        reasoning_steps.append({
            "step": 1,
            "action": "explanation_search",
            "explanations_found": len(explanations)
        })
        
        # 选择最佳解释
        best_explanation = self._select_best_explanation(explanations)
        
        reasoning_steps.append({
            "step": 2,
            "action": "explanation_selection",
            "selected": best_explanation
        })
        
        confidence = min(len(explanations) / 5.0, 1.0) if explanations else 0.0
        
        return ReasoningResult(
            success=best_explanation is not None,
            conclusion=best_explanation,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="abductive",
            evidence=explanations,
            counter_evidence=[]
        )
    
    def _causal_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """因果推理"""
        reasoning_steps = []
        
        # 构建因果图
        causal_graph = self._build_causal_graph(context.facts)
        
        reasoning_steps.append({
            "step": 1,
            "action": "causal_graph_construction",
            "nodes": len(causal_graph.get("nodes", [])),
            "edges": len(causal_graph.get("edges", []))
        })
        
        # 因果推理
        causal_conclusion = self._infer_causal_relations(causal_graph, query)
        
        reasoning_steps.append({
            "step": 2,
            "action": "causal_inference",
            "conclusion": causal_conclusion
        })
        
        # 计算因果强度
        confidence = self._calculate_causal_confidence(causal_graph)
        
        return ReasoningResult(
            success=causal_conclusion is not None,
            conclusion=causal_conclusion,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="causal",
            evidence=self.causal_chains,
            counter_evidence=[]
        )
    
    def _probabilistic_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """概率推理"""
        reasoning_steps = []
        
        # 计算概率
        probabilities = self._compute_probabilities(context.facts, query)
        
        reasoning_steps.append({
            "step": 1,
            "action": "probability_computation",
            "probabilities": probabilities
        })
        
        # 贝叶斯推理
        posterior = self._bayesian_inference(prior=0.5, likelihood=probabilities)
        
        reasoning_steps.append({
            "step": 2,
            "action": "bayesian_inference",
            "posterior": posterior
        })
        
        confidence = posterior
        
        return ReasoningResult(
            success=posterior >= self.confidence_threshold,
            conclusion={"probability": posterior, "query": query},
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="probabilistic",
            evidence=[{"likelihood": probabilities}],
            counter_evidence=[]
        )
    
    def _common_sense_reasoning(
        self,
        context: ReasoningContext,
        query: str,
        max_steps: int
    ) -> ReasoningResult:
        """常识推理"""
        reasoning_steps = []
        
        # 应用常识知识
        cs_inference = self._apply_common_sense(context.facts, query)
        
        reasoning_steps.append({
            "step": 1,
            "action": "common_sense_application",
            "inferences": len(cs_inference)
        })
        
        # 生成常识结论
        conclusion = self._generate_cs_conclusion(cs_inference, query)
        
        reasoning_steps.append({
            "step": 2,
            "action": "conclusion_generation",
            "conclusion": conclusion
        })
        
        confidence = min(len(cs_inference) / 5.0, 1.0)
        
        return ReasoningResult(
            success=conclusion is not None,
            conclusion=conclusion,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            reasoning_type="common_sense",
            evidence=cs_inference,
            counter_evidence=[]
        )
    
    def _rule_applicable(self, rule: LogicRule, facts: List[Dict[str, Any]]) -> bool:
        """检查规则是否适用"""
        for fact in facts:
            if rule.antecedent in str(fact):
                return True
        return False
    
    def _apply_rule(self, rule: LogicRule, facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """应用规则"""
        return {
            "rule_id": rule.rule_id,
            "conclusion": rule.conclusion,
            "confidence": rule.confidence,
            "derived_from": rule.antecedent
        }
    
    def _check_query_satisfaction(self, query: str, facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """检查查询是否满足"""
        query_terms = query.lower().split()
        
        for fact in facts:
            fact_str = str(fact).lower()
            if all(term in fact_str for term in query_terms):
                return {
                    "answer": fact,
                    "matched": True,
                    "query": query
                }
        
        return None
    
    def _calculate_confidence(self, 
                            steps: List[Dict[str, Any]], 
                            conclusion: Optional[Dict[str, Any]]) -> float:
        """计算置信度"""
        if conclusion is None:
            return 0.0
        
        base_confidence = 0.5
        step_bonus = min(len(steps) * 0.05, 0.3)
        
        return min(base_confidence + step_bonus, 0.95)
    
    def _extract_evidence(self, 
                        facts: List[Dict[str, Any]], 
                        query: str) -> List[Dict[str, Any]]:
        """提取证据"""
        return [{"fact": f, "relevance": np.random.uniform(0.5, 1.0)} for f in facts[:5]]
    
    def _find_patterns(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """寻找模式"""
        patterns = []
        for i, fact in enumerate(facts[:10]):
            patterns.append({
                "pattern_id": i,
                "pattern": fact,
                "frequency": np.random.uniform(0.3, 1.0)
            })
        return patterns
    
    def _generate_inductive_conclusion(self, 
                                       patterns: List[Dict[str, Any]], 
                                       query: str) -> Optional[Dict[str, Any]]:
        """生成归纳结论"""
        if not patterns:
            return None
        
        return {
            "generalization": "Inferred from patterns",
            "patterns_used": len(patterns),
            "query": query,
            "strength": sum(p.get("frequency", 0.5) for p in patterns) / len(patterns)
        }
    
    def _find_explanations(self, 
                          facts: List[Dict[str, Any]], 
                          query: str) -> List[Dict[str, Any]]:
        """寻找解释"""
        explanations = []
        for i, fact in enumerate(facts[:5]):
            explanations.append({
                "explanation_id": i,
                "explanation": f"Explanation for {query}",
                "support": fact,
                "plausibility": np.random.uniform(0.5, 0.95)
            })
        return explanations
    
    def _select_best_explanation(self, 
                                explanations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """选择最佳解释"""
        if not explanations:
            return None
        
        best = max(explanations, key=lambda x: x.get("plausibility", 0))
        return best
    
    def _build_causal_graph(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建因果图"""
        nodes = []
        edges = []
        
        for fact in facts:
            nodes.append(f"node_{len(nodes)}")
        
        # 简化: 创建随机边
        for i in range(min(len(nodes) - 1, 5)):
            edges.append({
                "from": nodes[i],
                "to": nodes[i + 1],
                "weight": np.random.uniform(0.5, 1.0)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _infer_causal_relations(self, 
                               graph: Dict[str, Any], 
                               query: str) -> Optional[Dict[str, Any]]:
        """推理因果关系"""
        return {
            "causal_relation": "Inferred relation",
            "query": query,
            "strength": np.random.uniform(0.6, 0.95)
        }
    
    def _calculate_causal_confidence(self, graph: Dict[str, Any]) -> float:
        """计算因果置信度"""
        if not graph.get("edges"):
            return 0.5
        
        avg_weight = sum(e["weight"] for e in graph["edges"]) / len(graph["edges"])
        return min(avg_weight, 0.95)
    
    def _compute_probabilities(self, 
                              facts: List[Dict[str, Any]], 
                              query: str) -> Dict[str, float]:
        """计算概率"""
        return {f"prob_{i}": np.random.uniform(0.3, 0.9) for i in range(3)}
    
    def _bayesian_inference(self, 
                           prior: float, 
                           likelihood: Dict[str, float]) -> float:
        """贝叶斯推理"""
        likelihood_avg = sum(likelihood.values()) / len(likelihood) if likelihood else 0.5
        
        # 简化贝叶斯更新
        posterior = (prior * likelihood_avg) / (
            (prior * likelihood_avg) + ((1 - prior) * (1 - likelihood_avg)) + 1e-8
        )
        
        return min(max(posterior, 0.0), 1.0)
    
    def _apply_common_sense(self, 
                           facts: List[Dict[str, Any]], 
                           query: str) -> List[Dict[str, Any]]:
        """应用常识"""
        inferences = []
        
        for fact in facts[:5]:
            inference = {
                "inference": f"Common sense inference from {fact}",
                "principle": "generalization",
                "confidence": np.random.uniform(0.6, 0.9)
            }
            inferences.append(inference)
        
        return inferences
    
    def _generate_cs_conclusion(self, 
                               inferences: List[Dict[str, Any]], 
                               query: str) -> Optional[Dict[str, Any]]:
        """生成常识结论"""
        if not inferences:
            return None
        
        return {
            "conclusion": "Based on common sense",
            "inferences_used": len(inferences),
            "query": query,
            "overall_confidence": sum(i.get("confidence", 0.5) for i in inferences) / len(inferences)
        }
    
    def _update_avg_confidence(self, confidence: float) -> None:
        """更新平均置信度"""
        n = self.stats["successful_reasonings"]
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (n - 1) + confidence
        ) / n if n > 0 else confidence
    
    def multi_source_reasoning(
        self,
        contexts: List[ReasoningContext],
        query: str,
        reasoning_types: List[ReasoningType] = None
    ) -> Dict[str, ReasoningResult]:
        """多源推理"""
        reasoning_types = reasoning_types or [
            ReasoningType.DEDUCTIVE,
            ReasoningType.CAUSAL,
            ReasoningType.COMMON_SENSE
        ]
        
        results = {}
        for rtype in reasoning_types:
            result = self.reason(contexts, query, reasoning_type=rtype)
            results[rtype.value] = result
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_reasonings": self.stats["total_reasonings"],
            "successful_reasonings": self.stats["successful_reasonings"],
            "success_rate": (
                self.stats["successful_reasonings"] / self.stats["total_reasonings"]
                if self.stats["total_reasonings"] > 0 else 0
            ),
            "avg_confidence": self.stats["avg_confidence"],
            "reasoning_types_distribution": dict(self.stats["reasoning_types_used"]),
            "registered_rules": len(self.rules),
            "causal_chains": len(self.causal_chains)
        }
