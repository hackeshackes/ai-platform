"""
知识推理引擎 - 基于规则和图的推理
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import re


@dataclass
class InferenceRule:
    """推理规则"""
    id: str
    name: str
    premise: str        # 前提条件 (SPARQL-like或自定义语法)
    conclusion: str     # 结论
    confidence: float   # 置信度
    rule_type: str      # "transitive", "inverse", "symmetric", "custom"
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "premise": self.premise,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "rule_type": self.rule_type,
            "enabled": self.enabled
        }


@dataclass
class InferenceResult:
    """推理结果"""
    conclusion: str
    evidence: List[Dict]  # 推理路径/证据
    confidence: float
    rule_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "conclusion": self.conclusion,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "rule_id": self.rule_id,
            "timestamp": self.timestamp.isoformat()
        }


class KnowledgeReasoner:
    """知识推理引擎"""
    
    def __init__(self, graph_manager = None):
        self.graph_manager = graph_manager
        self.rules: Dict[str, InferenceRule] = {}
        self.rule_index: Dict[str, List[str]] = defaultdict(list)  # entity_type -> rule_ids
        self._add_default_rules()
    
    def _add_default_rules(self):
        """添加默认推理规则"""
        default_rules = [
            InferenceRule(
                id="rule_1",
                name="IsA传递性",
                premise="(?x IS_A ?y) AND (?y IS_A ?z)",
                conclusion="(?x IS_A ?z)",
                confidence=0.9,
                rule_type="transitive"
            ),
            InferenceRule(
                id="rule_2",
                name="PartOf传递性",
                premise="(?x PART_OF ?y) AND (?y PART_OF ?z)",
                conclusion="(?x PART_OF ?z)",
                confidence=0.85,
                rule_type="transitive"
            ),
            InferenceRule(
                id="rule_3",
                name="Inverse关系",
                premise="(?x PARENT_OF ?y)",
                conclusion="(?y CHILD_OF ?x)",
                confidence=1.0,
                rule_type="inverse"
            ),
            InferenceRule(
                id="rule_4",
                name="雇佣关系",
                premise="(?x WORKS_AT ?y) AND (?y HAS_EMPLOYEE ?x)",
                conclusion="(?x IS_EMPLOYEE_OF ?y)",
                confidence=0.95,
                rule_type="custom"
            ),
            InferenceRule(
                id="rule_5",
                name="LocatedIn传递性",
                premise="(?x LOCATED_IN ?y) AND (?y LOCATED_IN ?z)",
                conclusion="(?x LOCATED_IN ?z)",
                confidence=0.8,
                rule_type="transitive"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: InferenceRule):
        """添加推理规则"""
        self.rules[rule.id] = rule
        # 简单索引：从前提中提取实体类型
        types = self._extract_types_from_rule(rule.premise)
        for entity_type in types:
            self.rule_index[entity_type].append(rule.id)
    
    def _extract_types_from_rule(self, premise: str) -> Set[str]:
        """从规则前提中提取实体类型"""
        types = set()
        # 匹配 ?x TYPE 模式
        pattern = r'\?(\w+)\s+(\w+)'
        matches = re.findall(pattern, premise)
        # 简化处理
        return types
    
    def remove_rule(self, rule_id: str) -> bool:
        """删除推理规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def get_rules(self, entity_type: str = None) -> List[InferenceRule]:
        """获取规则"""
        if entity_type:
            rule_ids = self.rule_index.get(entity_type, [])
            return [self.rules[rid] for rid in rule_ids if rid in self.rules]
        return list(self.rules.values())
    
    def reason(self, 
               entity_id: str = None,
               entity_name: str = None,
               max_depth: int = 3) -> List[InferenceResult]:
        """基于实体进行推理"""
        if not self.graph_manager:
            return []
        
        # 获取目标实体
        target_entity = None
        if entity_id:
            target_entity = self.graph_manager.get_entity(entity_id)
        elif entity_name:
            target_entity = self.graph_manager.get_entity_by_name(entity_name)
        
        if not target_entity:
            return []
        
        results = []
        
        # 1. 基于规则推理
        rule_results = self._apply_rules(target_entity, max_depth)
        results.extend(rule_results)
        
        # 2. 基于路径推理
        path_results = self._path_based_reasoning(target_entity, max_depth)
        results.extend(path_results)
        
        # 3. 基于邻居推理
        neighbor_results = self._neighbor_based_reasoning(target_entity)
        results.extend(neighbor_results)
        
        return results
    
    def _apply_rules(self, entity, max_depth: int) -> List[InferenceResult]:
        """应用推理规则"""
        results = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 简单规则匹配
            if rule.rule_type == "transitive":
                derived = self._transitive_reasoning(entity, rule, max_depth)
                results.extend(derived)
            elif rule.rule_type == "inverse":
                derived = self._inverse_reasoning(entity, rule)
                results.extend(derived)
            elif rule.rule_type == "symmetric":
                derived = self._symmetric_reasoning(entity, rule)
                results.extend(derived)
        
        return results
    
    def _transitive_reasoning(self, 
                              entity, 
                              rule: InferenceRule,
                              max_depth: int) -> List[InferenceResult]:
        """传递性推理"""
        results = []
        
        # 获取实体的直接关系
        neighbors = self.graph_manager.get_neighbors(entity.id)
        
        for neighbor, relation in neighbors:
            # 递归查找
            path = [(entity.name, relation.relation_type, neighbor.name)]
            
            derived = self._find_transitive_paths(
                neighbor.id, 
                relation.relation_type, 
                max_depth, 
                path
            )
            
            for d in derived:
                results.append(InferenceResult(
                    conclusion=f"{entity.name} {rule.conclusion.split(' ')[2]} {d}",
                    evidence=[{
                        "path": d.get("path", []),
                        "rule": rule.name
                    }],
                    confidence=rule.confidence * d.get("confidence", 1.0),
                    rule_id=rule.id
                ))
        
        return results
    
    def _find_transitive_paths(self, 
                               current_id: str, 
                               relation_type: str,
                               max_depth: int,
                               path: List[Tuple],
                               current_depth: int = 1) -> List[Dict]:
        """查找传递路径"""
        if current_depth >= max_depth:
            return []
        
        results = []
        
        neighbors = self.graph_manager.get_neighbors(current_id)
        for neighbor, relation in neighbors:
            if relation.relation_type == relation_type:
                new_path = path + [(neighbor.name, relation.relation_type, neighbor.id)]
                
                results.append({
                    "end": neighbor.name,
                    "path": new_path,
                    "confidence": relation.weight
                })
                
                # 递归
                deeper = self._find_transitive_paths(
                    neighbor.id, relation_type, max_depth, new_path, current_depth + 1
                )
                results.extend(deeper)
        
        return results
    
    def _inverse_reasoning(self, 
                          entity, 
                          rule: InferenceRule) -> List[InferenceResult]:
        """逆关系推理"""
        results = []
        
        # 解析规则
        # premise: (?x PARENT_OF ?y) -> conclusion: (?y CHILD_OF ?x)
        premise_match = re.search(r'\?(\w+)\s+(\w+)\s+\?(\w+)', rule.premise)
        conc_match = re.search(r'\?(\w+)\s+(\w+)\s+\?(\w+)', rule.conclusion)
        
        if not premise_match or not conc_match:
            return results
        
        # 查找实体对应的逆关系
        neighbors = self.graph_manager.get_neighbors(entity.id)
        
        for neighbor, relation in neighbors:
            # 如果关系匹配，生成逆关系
            if relation.relation_type == premise_match.group(2):
                results.append(InferenceResult(
                    conclusion=f"{neighbor.name} {conc_match.group(2)} {entity.name}",
                    evidence=[{
                        "original_relation": relation.relation_type,
                        "rule": rule.name
                    }],
                    confidence=rule.confidence,
                    rule_id=rule.id
                ))
        
        return results
    
    def _symmetric_reasoning(self, 
                            entity, 
                            rule: InferenceRule) -> List[InferenceResult]:
        """对称关系推理"""
        results = []
        
        neighbors = self.graph_manager.get_neighbors(entity.id)
        
        for neighbor, relation in neighbors:
            results.append(InferenceResult(
                conclusion=f"{neighbor.name} {relation.relation_type} {entity.name}",
                evidence=[{
                    "rule": rule.name,
                    "original": f"{entity.name} {relation.relation_type} {neighbor.name}"
                }],
                confidence=rule.confidence,
                rule_id=rule.id
            ))
        
        return results
    
    def _path_based_reasoning(self, entity, max_depth: int) -> List[InferenceResult]:
        """基于路径的推理"""
        results = []
        
        # BFS查找路径
        queue = deque([(entity.id, [entity.id], 0)])
        visited = {entity.id}
        
        while queue:
            current_id, path, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            neighbors = self.graph_manager.get_neighbors(current_id)
            
            for neighbor, relation in neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    new_path = path + [neighbor.id]
                    
                    # 生成推理结果
                    if len(new_path) >= 3:
                        start_entity = self.graph_manager.get_entity(new_path[0])
                        end_entity = self.graph_manager.get_entity(new_path[-1])
                        
                        if start_entity and end_entity:
                            results.append(InferenceResult(
                                conclusion=f"{start_entity.name} 相关于 {end_entity.name}",
                                evidence=[{
                                    "path_length": len(new_path) - 1,
                                    "intermediate_entities": [
                                        self.graph_manager.get_entity(eid).name 
                                        for eid in new_path[1:-1]
                                    ]
                                }],
                                confidence=1.0 / (len(new_path)),
                                rule_id="path_reasoning"
                            ))
                    
                    queue.append((neighbor.id, new_path, depth + 1))
        
        return results
    
    def _neighbor_based_reasoning(self, entity) -> List[InferenceResult]:
        """基于邻居的推理"""
        results = []
        
        neighbors = self.graph_manager.get_neighbors(entity.id)
        
        # 统计邻居类型
        type_counts = defaultdict(int)
        for neighbor, relation in neighbors:
            type_counts[neighbor.type] += 1
        
        # 生成推理结果
        for neighbor_type, count in type_counts.items():
            results.append(InferenceResult(
                conclusion=f"{entity.name} 可能属于 {neighbor_type} 类别",
                evidence=[{
                    "neighbors_of_type": count,
                    "neighbors": [
                        {"name": n.name, "relation": r.relation_type}
                        for n, r in neighbors if n.type == neighbor_type
                    ]
                }],
                confidence=min(0.5 + count * 0.1, 0.95),
                rule_id="neighbor_inference"
            ))
        
        return results
    
    def reason_with_query(self, 
                          query: str) -> List[InferenceResult]:
        """基于查询进行推理"""
        results = []
        
        # 解析查询
        # 格式: "X 的 Y 是什么" 或 "X 和 Y 的关系"
        patterns = [
            (r'(.+)的(.+)是什么', 'property_query'),
            (r'(.+)和(.+)的关系', 'relation_query'),
            (r'(.+)属于(.+)', 'classification_query'),
            (r'(.+)的(.+)是谁', 'person_query')
        ]
        
        for pattern, query_type in patterns:
            match = re.match(pattern, query)
            if match:
                if query_type == 'property_query':
                    results = self._reason_property(
                        match.group(1).strip(), 
                        match.group(2).strip()
                    )
                elif query_type == 'relation_query':
                    results = self._reason_relation(
                        match.group(1).strip(), 
                        match.group(2).strip()
                    )
        
        return results
    
    def _reason_property(self, 
                         entity_name: str, 
                         property_name: str) -> List[InferenceResult]:
        """属性推理"""
        results = []
        
        entity = self.graph_manager.get_entity_by_name(entity_name)
        if not entity:
            return results
        
        # 直接属性
        if property_name in entity.properties:
            results.append(InferenceResult(
                conclusion=str(entity.properties[property_name]),
                evidence=[{
                    "source": "direct_property",
                    "property": property_name
                }],
                confidence=1.0,
                rule_id="property_lookup"
            ))
        
        # 邻居属性
        neighbors = self.graph_manager.get_neighbors(entity.id)
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
                    rule_id="neighbor_property"
                ))
        
        return results
    
    def _reason_relation(self, 
                        entity1_name: str, 
                        entity2_name: str) -> List[InferenceResult]:
        """关系推理"""
        results = []
        
        entity1 = self.graph_manager.get_entity_by_name(entity1_name)
        entity2 = self.graph_manager.get_entity_by_name(entity2_name)
        
        if not entity1 or not entity2:
            return results
        
        # 查找直接关系
        neighbors = self.graph_manager.get_neighbors(entity1.id)
        for neighbor, relation in neighbors:
            if neighbor.id == entity2.id:
                results.append(InferenceResult(
                    conclusion=f"直接关系: {relation.relation_type}",
                    evidence=[{
                        "relation_type": relation.relation_type,
                        "weight": relation.weight
                    }],
                    confidence=relation.weight,
                    rule_id="direct_relation"
                ))
        
        # 查找间接关系
        path = self.graph_manager.get_shortest_path(entity1.id, entity2.id)
        if path:
            relations = []
            for i in range(len(path) - 1):
                rels = self.graph_manager.get_relations(
                    source_id=path[i].id, 
                    target_id=path[i+1].id
                )
                if rels:
                    relations.append({
                        "from": path[i].name,
                        "to": path[i+1].name,
                        "type": rels[0].relation_type
                    })
            
            results.append(InferenceResult(
                conclusion=f"间接关系路径",
                evidence=[{
                    "path_length": len(path) - 1,
                    "path": [e.name for e in path],
                    "relations": relations
                }],
                confidence=0.8,
                rule_id="path_relation"
            ))
        
        return results
    
    def reason_batch(self, 
                     entities: List[str],
                     max_depth: int = 2) -> Dict[str, List[InferenceResult]]:
        """批量推理"""
        results = {}
        
        for entity in entities:
            results[entity] = self.reason(entity_name=entity, max_depth=max_depth)
        
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
                enabled=rule_data.get("enabled", True)
            )
            self.add_rule(rule)


# 全局推理器实例
_reasoner = None


def get_reasoner(graph_manager = None) -> KnowledgeReasoner:
    """获取推理器实例"""
    global _reasoner
    if _reasoner is None:
        _reasoner = KnowledgeReasoner(graph_manager)
    return _reasoner


def set_graph_manager_for_reasoner(graph_manager):
    """为推理器设置图谱管理器"""
    global _reasoner
    if _reasoner:
        _reasoner.graph_manager = graph_manager
