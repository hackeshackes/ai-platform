"""
知识图谱管理器 - 管理实体、关系和图谱操作
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import json
import re
from collections import defaultdict


@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "embeddings": self.embeddings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            type=data["type"],
            properties=data.get("properties", {}),
            embeddings=data.get("embeddings"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )


@dataclass
class Relation:
    """关系类"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Relation':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


class KnowledgeGraphManager:
    """知识图谱管理器"""
    
    def __init__(self, graph_id: str = "default"):
        self.graph_id = graph_id
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        # 索引加速查询
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # type -> entity_ids
        self.name_index: Dict[str, str] = {}  # name -> entity_id
        self.relation_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)  # (source, target) -> relation_ids
        
    def add_entity(self, name: str, entity_type: str, 
                   properties: Dict = None, 
                   embeddings: List[float] = None) -> Entity:
        """添加实体"""
        entity = Entity(
            id=str(uuid.uuid4()),
            name=name,
            type=entity_type,
            properties=properties or {},
            embeddings=embeddings
        )
        self.entities[entity.id] = entity
        self.entity_index[entity_type].append(entity.id)
        self.name_index[name] = entity.id
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """根据名称获取实体"""
        entity_id = self.name_index.get(name)
        return self.entities.get(entity_id) if entity_id else None
    
    def list_entities(self, entity_type: str = None, 
                      limit: int = 100, 
                      offset: int = 0) -> List[Entity]:
        """列出实体"""
        if entity_type:
            entity_ids = self.entity_index.get(entity_type, [])
        else:
            entity_ids = list(self.entities.keys())
        
        return [
            self.entities[eid] 
            for eid in entity_ids[offset:offset+limit]
            if eid in self.entities
        ]
    
    def update_entity(self, entity_id: str, 
                      name: str = None, 
                      properties: Dict = None) -> Optional[Entity]:
        """更新实体"""
        entity = self.entities.get(entity_id)
        if not entity:
            return None
        
        if name:
            # 更新名称索引
            del self.name_index[entity.name]
            entity.name = name
            self.name_index[name] = entity_id
        
        if properties:
            entity.properties.update(properties)
        
        entity.updated_at = datetime.now()
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体及其关联关系"""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        
        # 从索引中移除
        if entity.type in self.entity_index:
            self.entity_index[entity.type] = [
                eid for eid in self.entity_index[entity.type] 
                if eid != entity_id
            ]
        
        if entity.name in self.name_index:
            del self.name_index[entity.name]
        
        # 删除关联关系
        relations_to_delete = [
            rid for rid, rel in self.relations.items()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
        for rid in relations_to_delete:
            del self.relations[rid]
        
        del self.entities[entity_id]
        return True
    
    def add_relation(self, source_id: str, 
                     target_id: str, 
                     relation_type: str,
                     properties: Dict = None,
                     weight: float = 1.0) -> Optional[Relation]:
        """添加关系"""
        # 验证实体存在
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        relation = Relation(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight
        )
        self.relations[relation.id] = relation
        
        # 更新索引
        key = (source_id, target_id)
        self.relation_index[key].append(relation.id)
        
        # 按关系类型索引
        self.entity_index[f"relation_{relation_type}"].append(relation.id)
        
        return relation
    
    def get_relations(self, source_id: str = None, 
                      target_id: str = None,
                      relation_type: str = None) -> List[Relation]:
        """获取关系"""
        results = []
        for rel in self.relations.values():
            if source_id and rel.source_id != source_id:
                continue
            if target_id and rel.target_id != target_id:
                continue
            if relation_type and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return results
    
    def delete_relation(self, relation_id: str) -> bool:
        """删除关系"""
        if relation_id not in self.relations:
            return False
        
        rel = self.relations[relation_id]
        key = (rel.source_id, rel.target_id)
        
        if key in self.relation_index:
            self.relation_index[key] = [
                rid for rid in self.relation_index[key] 
                if rid != relation_id
            ]
        
        del self.relations[relation_id]
        return True
    
    def get_neighbors(self, entity_id: str, 
                      relation_type: str = None) -> List[Tuple[Entity, Relation]]:
        """获取实体的邻居节点"""
        neighbors = []
        for rel in self.relations.values():
            if rel.source_id == entity_id:
                target = self.entities.get(rel.target_id)
                if target and (not relation_type or rel.relation_type == relation_type):
                    neighbors.append((target, rel))
            elif rel.target_id == entity_id:
                source = self.entities.get(rel.source_id)
                if source and (not relation_type or rel.relation_type == relation_type):
                    neighbors.append((source, rel))
        return neighbors
    
    def get_shortest_path(self, source_id: str, 
                          target_id: str) -> Optional[List[Entity]]:
        """获取最短路径 (BFS)"""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        from collections import deque
        
        visited = {source_id}
        queue = deque([(source_id, [])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                entities = [self.entities[pid] for pid in path + [target_id]]
                return entities
            
            for rel in self.relations.values():
                if rel.source_id == current_id and rel.target_id not in visited:
                    visited.add(rel.target_id)
                    queue.append((rel.target_id, path + [rel.target_id]))
                elif rel.target_id == current_id and rel.source_id not in visited:
                    visited.add(rel.source_id)
                    queue.append((rel.source_id, path + [rel.source_id]))
        
        return None
    
    def get_subgraph(self, entity_ids: List[str], 
                     depth: int = 1) -> Dict:
        """获取子图"""
        subgraph_entities = set(entity_ids)
        subgraph_relations = {}
        
        current_layer = set(entity_ids)
        for _ in range(depth):
            next_layer = set()
            for rel in self.relations.values():
                if rel.source_id in current_layer:
                    next_layer.add(rel.target_id)
                    subgraph_relations[rel.id] = rel
                elif rel.target_id in current_layer:
                    next_layer.add(rel.source_id)
                    subgraph_relations[rel.id] = rel
            
            subgraph_entities.update(next_layer)
            current_layer = next_layer
        
        return {
            "entities": [self.entities[eid].to_dict() for eid in subgraph_entities if eid in self.entities],
            "relations": [rel.to_dict() for rel in subgraph_relations.values()]
        }
    
    def export_graph(self) -> Dict:
        """导出整个图谱"""
        return {
            "graph_id": self.graph_id,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_graph(self, data: Dict, 
                     merge: bool = False) -> Tuple[int, int]:
        """导入图谱"""
        imported_entities = 0
        imported_relations = 0
        
        if not merge:
            self.entities.clear()
            self.relations.clear()
            self.entity_index.clear()
            self.name_index.clear()
            self.relation_index.clear()
        
        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            self.entities[entity.id] = entity
            self.entity_index[entity.type].append(entity.id)
            self.name_index[entity.name] = entity.id
            imported_entities += 1
        
        for relation_data in data.get("relations", []):
            relation = Relation.from_dict(relation_data)
            self.relations[relation.id] = relation
            key = (relation.source_id, relation.target_id)
            self.relation_index[key].append(relation.id)
            imported_relations += 1
        
        return imported_entities, imported_relations
    
    def stats(self) -> Dict:
        """获取图谱统计信息"""
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.type] += 1
        
        relation_counts = defaultdict(int)
        for relation in self.relations.values():
            relation_counts[relation.relation_type] += 1
        
        return {
            "graph_id": self.graph_id,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_types": dict(type_counts),
            "relation_types": dict(relation_counts)
        }


# 全局图谱实例
_graph_managers: Dict[str, KnowledgeGraphManager] = {}


def get_graph_manager(graph_id: str = "default") -> KnowledgeGraphManager:
    """获取图谱管理器实例"""
    if graph_id not in _graph_managers:
        _graph_managers[graph_id] = KnowledgeGraphManager(graph_id)
    return _graph_managers[graph_id]


def list_graphs() -> List[str]:
    """列出所有图谱"""
    return list(_graph_managers.keys())


def delete_graph(graph_id: str) -> bool:
    """删除图谱"""
    if graph_id in _graph_managers:
        del _graph_managers[graph_id]
        return True
    return False
