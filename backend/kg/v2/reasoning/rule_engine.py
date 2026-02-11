"""
规则推理引擎

基于预定义规则的推理引擎，支持传递性、逆关系等规则。
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class ReasoningType(str, Enum):
    """推理类型"""
    RULE_BASED = "rule"
    NEURAL = "neural"
    HYBRID = "hybrid"
    PATH_FINDING = "path"


@dataclass
class InferenceRule:
    """推理规则"""
    id: str
    name: str
    premise: str  # 前提条件
    conclusion: str  # 结论
    confidence: float  # 置信度
    rule_type: str  # transitive, inverse, symmetric, custom
    enabled: bool = True
    priority: int = 0  # 优先级

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "premise": self.premise,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "rule_type": self.rule_type,
            "enabled": self.enabled,
            "priority": self.priority
        }


@dataclass
class InferenceResult:
    """推理结果"""
    conclusion: str
    evidence: List[Dict]  # 推理路径/证据
    confidence: float
    rule_id: str
    reasoning_type: ReasoningType = ReasoningType.RULE_BASED
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "conclusion": self.conclusion,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "reasoning_type": self.reasoning_type.value,
            "timestamp": self.timestamp.isoformat()
        }


class RuleEngine:
    """规则推理引擎"""
    
    def __init__(self):
        self.rules: Dict[str, InferenceRule] = {}
        self._add_default_rules()
    
    def _add_default_rules(self):
        """添加默认推理规则"""
        default_rules = [
            InferenceRule(
                id="rule_is_a_transitive",
                name="IsA传递性",
                premise="(?x IS_A ?y) AND (?y IS_A ?z)",
                conclusion="(?x IS_A ?z)",
                confidence=0.9,
                rule_type="transitive",
                priority=10
            ),
            InferenceRule(
                id="rule_part_of_transitive",
                name="PartOf传递性",
                premise="(?x PART_OF ?y) AND (?y PART_OF ?z)",
                conclusion="(?x PART_OF ?z)",
                confidence=0.85,
                rule_type="transitive",
                priority=10
            ),
            InferenceRule(
                id="rule_inverse_parent_child",
                name="父子关系逆",
                premise="(?x PARENT_OF ?y)",
                conclusion="(?y CHILD_OF ?x)",
                confidence=1.0,
                rule_type="inverse",
                priority=15
            ),
            InferenceRule(
                id="rule_inverse_employ",
                name="雇佣关系逆",
                premise="(?x WORKS_AT ?y)",
                conclusion="(?y HAS_EMPLOYEE ?x)",
                confidence=0.95,
                rule_type="inverse",
                priority=10
            ),
            InferenceRule(
                id="rule_located_in_transitive",
                name="LocatedIn传递性",
                premise="(?x LOCATED_IN ?y) AND (?y LOCATED_IN ?z)",
                conclusion="(?x LOCATED_IN ?z)",
                confidence=0.8,
                rule_type="transitive",
                priority=8
            ),
            InferenceRule(
                id="rule_symmetric_spouse",
                name="配偶关系对称",
                premise="(?x SPOUSE_OF ?y)",
                conclusion="(?y SPOUSE_OF ?x)",
                confidence=1.0,
                rule_type="symmetric",
                priority=15
            ),
            InferenceRule(
                id="rule_friend_symmetric",
                name="朋友关系对称",
                premise="(?x FRIEND_OF ?y)",
                conclusion="(?y FRIEND_OF ?x)",
                confidence=0.9,
                rule_type="symmetric",
                priority=5
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: InferenceRule):
        """添加推理规则"""
        self.rules[rule.id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """删除推理规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def get_rules(self, 
                  enabled_only: bool = True,
                  rule_type: str = None) -> List[InferenceRule]:
        """获取规则列表"""
        rules = []
        
        for rule in self.rules.values():
            if enabled_only and not rule.enabled:
                continue
            if rule_type and rule.rule_type != rule_type:
                continue
            rules.append(rule)
        
        # 按优先级排序
        rules.sort(key=lambda r: r.priority, reverse=True)
        return rules
    
    def reason(self,
              entity_id: str,
              graph_builder,
              max_depth: int = 3) -> List[InferenceResult]:
        """基于规则推理"""
        results = []
        
        entity = graph_builder.get_entity(entity_id)
        if not entity:
            return results
        
        # 应用传递性规则
        for rule in self.get_rules(rule_type="transitive"):
            derived = self._apply_transitive_rule(entity, graph_builder, rule, max_depth)
            results.extend(derived)
        
        # 应用逆关系规则
        for rule in self.get_rules(rule_type="inverse"):
            derived = self._apply_inverse_rule(entity, graph_builder, rule)
            results.extend(derived)
        
        # 应用对称关系规则
        for rule in self.get_rules(rule_type="symmetric"):
            derived = self._apply_symmetric_rule(entity, graph_builder, rule)
            results.extend(derived)
        
        return results
    
    def _apply_transitive_rule(self,
                               entity,
                               graph_builder,
                               rule: InferenceRule,
                               max_depth: int) -> List[InferenceResult]:
        """应用传递性规则"""
        results = []
        
        # 解析规则中的关系类型
        match = re.search(r'\?x\s+(\w+)\s+\?y', rule.premise)
        if not match:
            return results
        
        relation_type = match.group(1)
        
        # 获取直接邻居
        neighbors = graph_builder.get_neighbors(entity.id)
        for neighbor, relation in neighbors:
            if relation.relation_type == relation_type:
                # 递归查找
                derived = self._find_transitive_chain(
                    neighbor.id,
                    relation_type,
                    graph_builder,
                    max_depth,
                    [(entity.name, relation.relation_type, neighbor.name)],
                    rule
                )
                results.extend(derived)
        
        return results
    
    def _find_transitive_chain(self,
                              current_id: str,
                              relation_type: str,
                              graph_builder,
                              max_depth: int,
                              path: List[Tuple],
                              rule: InferenceRule,
                              current_depth: int = 1) -> List[InferenceResult]:
        """查找传递链"""
        if current_depth >= max_depth:
            return []
        
        results = []
        
        current_entity = graph_builder.get_entity(current_id)
        if not current_entity:
            return results
        
        neighbors = graph_builder.get_neighbors(current_id)
        
        for neighbor, relation in neighbors:
            if relation.relation_type == relation_type:
                new_path = path + [(current_entity.name, relation.relation_type, neighbor.name)]
                
                # 生成推理结论
                start_entity = graph_builder.get_entity(path[0][0])
                end_entity = neighbor
                
                if start_entity and end_entity:
                    conclusion = f"{start_entity.name} {relation_type} {end_entity.name}"
                    confidence = rule.confidence * (0.9 ** (current_depth - 1))
                    
                    results.append(InferenceResult(
                        conclusion=conclusion,
                        evidence=[{"path": new_path, "rule": rule.name}],
                        confidence=confidence,
                        rule_id=rule.id,
                        reasoning_type=ReasoningType.RULE_BASED
                    ))
                
                # 继续递归
                deeper = self._find_transitive_chain(
                    neighbor.id,
                    relation_type,
                    graph_builder,
                    max_depth,
                    new_path,
                    rule,
                    current_depth + 1
                )
                results.extend(deeper)
        
        return results
    
    def _apply_inverse_rule(self,
                           entity,
                           graph_builder,
                           rule: InferenceRule) -> List[InferenceResult]:
        """应用逆关系规则"""
        results = []
        
        # 解析规则
        premise_match = re.search(r'\?(\w+)\s+(\w+)\s+\?(\w+)', rule.premise)
        conc_match = re.search(r'\?(\w+)\s+(\w+)\s+\?(\w+)', rule.conclusion)
        
        if not premise_match or not conc_match:
            return results
        
        # 获取邻居，生成逆关系
        neighbors = graph_builder.get_neighbors(entity.id)
        
        for neighbor, relation in neighbors:
            if relation.relation_type == premise_match.group(2):
                conclusion = f"{neighbor.name} {conc_match.group(2)} {entity.name}"
                
                results.append(InferenceResult(
                    conclusion=conclusion,
                    evidence=[{
                        "original_relation": relation.relation_type,
                        "rule": rule.name
                    }],
                    confidence=rule.confidence * relation.weight,
                    rule_id=rule.id,
                    reasoning_type=ReasoningType.RULE_BASED
                ))
        
        return results
    
    def _apply_symmetric_rule(self,
                              entity,
                              graph_builder,
                              rule: InferenceRule) -> List[InferenceResult]:
        """应用对称关系规则"""
        results = []
        
        neighbors = graph_builder.get_neighbors(entity.id)
        
        for neighbor, relation in neighbors:
            results.append(InferenceResult(
                conclusion=f"{neighbor.name} {relation.relation_type} {entity.name}",
                evidence=[{
                    "rule": rule.name,
                    "original": f"{entity.name} {relation.relation_type} {neighbor.name}"
                }],
                confidence=rule.confidence * relation.weight,
                rule_id=rule.id,
                reasoning_type=ReasoningType.RULE_BASED
            ))
        
        return results
    
    def reason_with_query(self,
                         query: str,
                         graph_builder) -> List[InferenceResult]:
        """基于查询推理"""
        results = []
        
        # 解析查询类型
        patterns = [
            (r'(.+)的(.+)是什么', 'property'),
            (r'(.+)和(.+)的关系', 'relation'),
            (r'(.+)属于(.+)', 'classification'),
            (r'(.+)的(.+)是谁', 'person'),
            (r'(.+)在哪里', 'location'),
        ]
        
        for pattern, query_type in patterns:
            match = re.match(pattern, query)
            if match:
                if query_type == 'property':
                    results = self._query_property(
                        match.group(1).strip(),
                        match.group(2).strip(),
                        graph_builder
                    )
                elif query_type == 'relation':
                    results = self._query_relation(
                        match.group(1).strip(),
                        match.group(2).strip(),
                        graph_builder
                    )
                elif query_type == 'classification':
                    results = self._query_classification(
                        match.group(1).strip(),
                        match.group(2).strip(),
                        graph_builder
                    )
                break
        
        return results
    
    def _query_property(self,
                       entity_name: str,
                       property_name: str,
                       graph_builder) -> List[InferenceResult]:
        """查询属性"""
        results = []
        
        entity = graph_builder.get_entity_by_name(entity_name)
        if not entity:
            return results
        
        # 直接属性
        if property_name in entity.properties:
            results.append(InferenceResult(
                conclusion=str(entity.properties[property_name]),
                evidence=[{"source": "direct_property", "property": property_name}],
                confidence=1.0,
                rule_id="property_lookup",
                reasoning_type=ReasoningType.RULE_BASED
            ))
        
        # 邻居属性
        neighbors = graph_builder.get_neighbors(entity.id)
        for neighbor, relation in neighbors:
            if property_name in neighbor.properties:
                results.append(InferenceResult(
                    conclusion=f"{neighbor.properties[property_name]} (来自 {neighbor.name})",
                    evidence=[{
                        "source": "neighbor_property",
                        "via_relation": relation.relation_type,
                        "entity": neighbor.name
                    }],
                    confidence=relation.weight,
                    rule_id="neighbor_property",
                    reasoning_type=ReasoningType.RULE_BASED
                ))
        
        return results
    
    def _query_relation(self,
                       entity1_name: str,
                       entity2_name: str,
                       graph_builder) -> List[InferenceResult]:
        """查询关系"""
        results = []
        
        entity1 = graph_builder.get_entity_by_name(entity1_name)
        entity2 = graph_builder.get_entity_by_name(entity2_name)
        
        if not entity1 or not entity2:
            return results
        
        # 直接关系
        neighbors = graph_builder.get_neighbors(entity1.id)
        for neighbor, relation in neighbors:
            if neighbor.id == entity2.id:
                results.append(InferenceResult(
                    conclusion=f"直接关系: {relation.relation_type}",
                    evidence=[{
                        "relation_type": relation.relation_type,
                        "weight": relation.weight
                    }],
                    confidence=relation.weight,
                    rule_id="direct_relation",
                    reasoning_type=ReasoningType.RULE_BASED
                ))
        
        # 路径关系
        path = graph_builder.get_shortest_path(entity1.id, entity2.id)
        if path and len(path) > 2:
            results.append(InferenceResult(
                conclusion=f"间接关系路径 (长度: {len(path)-1})",
                evidence=[{
                    "path_length": len(path) - 1,
                    "path": [e.name for e in path]
                }],
                confidence=0.8,
                rule_id="path_relation",
                reasoning_type=ReasoningType.PATH_FINDING
            ))
        
        return results
    
    def _query_classification(self,
                             entity_name: str,
                             class_name: str,
                             graph_builder) -> List[InferenceResult]:
        """查询分类"""
        results = []
        
        entity = graph_builder.get_entity_by_name(entity_name)
        if not entity:
            return results
        
        # 检查类型是否匹配
        if entity.type.upper() == class_name.upper():
            results.append(InferenceResult(
                conclusion=f"{entity.name} 属于 {class_name} 类型",
                evidence=[{"entity_type": entity.type}],
                confidence=1.0,
                rule_id="type_match",
                reasoning_type=ReasoningType.RULE_BASED
            ))
        
        # 使用规则推理
        for rule in self.get_rules(rule_type="transitive"):
            match = re.search(r'\?x\s+(\w+)\s+\?y', rule.premise)
            if match and match.group(1).upper() == "IS_A":
                # 检查IS_A关系链
                path = graph_builder.get_shortest_path(entity.id, None)
                # 简化处理
                results.append(InferenceResult(
                    conclusion=f"{entity.name} 通过规则推理属于 {class_name}",
                    evidence=[{"rule": rule.name}],
                    confidence=rule.confidence,
                    rule_id=rule.id,
                    reasoning_type=ReasoningType.RULE_BASED
                ))
        
        return results
    
    def export_rules(self) -> List[Dict]:
        """导出规则"""
        return [rule.to_dict() for rule in self.rules.values()]
    
    def import_rules(self, rules_data: List[Dict]):
        """导入规则"""
        for rule_data in rules_data:
            rule = InferenceRule(
                id=rule_data["id"],
                name=rule_data["name"],
                premise=rule_data["premise"],
                conclusion=rule_data["conclusion"],
                confidence=rule_data["confidence"],
                rule_type=rule_data["rule_type"],
                enabled=rule_data.get("enabled", True),
                priority=rule_data.get("priority", 0)
            )
            self.add_rule(rule)


# 全局规则引擎实例
_rule_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    """获取规则引擎实例"""
    global _rule_engine
    
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    
    return _rule_engine
