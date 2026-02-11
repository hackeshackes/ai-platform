"""
NetworkX图构建器

提供基于NetworkX的内存图构建和操作。
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import networkx as nx
import numpy as np


@dataclass
class KGEntityV2:
    """知识图谱实体 v2"""
    id: str
    name: str
    type: str  # Person, Organization, Concept, Event, Location...
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
    def from_dict(cls, data: Dict) -> 'KGEntityV2':
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
class KGRelationV2:
    """知识图谱关系 v2"""
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
    def from_dict(cls, data: Dict) -> 'KGRelationV2':
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


class NetworkXGraphBuilder:
    """NetworkX图构建器"""
    
    def __init__(self, directed: bool = True):
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.entity_index: Dict[str, KGEntityV2] = {}
        self.relation_index: Dict[str, KGRelationV2] = {}
        self.type_index: Dict[str, Set[str]] = {}  # type -> entity_ids
        self.name_index: Dict[str, str] = {}  # name -> entity_id
    
    def add_entity(self, 
                   name: str,
                   entity_type: str,
                   properties: Dict = None,
                   embeddings: List[float] = None,
                   entity_id: str = None) -> KGEntityV2:
        """添加实体"""
        entity = KGEntityV2(
            id=entity_id or str(uuid.uuid4()),
            name=name,
            type=entity_type,
            properties=properties or {},
            embeddings=embeddings
        )
        
        # 添加到图
        self.graph.add_node(entity.id, 
                           name=entity.name,
                           type=entity.type,
                           properties=entity.properties)
        
        # 更新索引
        self.entity_index[entity.id] = entity
        self.name_index[entity.name] = entity.id
        
        if entity_type not in self.type_index:
            self.type_index[entity_type] = set()
        self.type_index[entity_type].add(entity.id)
        
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[KGEntityV2]:
        """获取实体"""
        return self.entity_index.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[KGEntityV2]:
        """根据名称获取实体"""
        entity_id = self.name_index.get(name)
        return self.entity_index.get(entity_id) if entity_id else None
    
    def update_entity(self,
                     entity_id: str,
                     name: str = None,
                     properties: Dict = None,
                     embeddings: List[float] = None) -> Optional[KGEntityV2]:
        """更新实体"""
        entity = self.entity_index.get(entity_id)
        if not entity:
            return None
        
        if name:
            del self.name_index[entity.name]
            entity.name = name
            self.name_index[name] = entity_id
            self.graph.nodes[entity_id]["name"] = name
        
        if properties:
            entity.properties.update(properties)
            self.graph.nodes[entity_id]["properties"] = entity.properties
        
        if embeddings is not None:
            entity.embeddings = embeddings
        
        entity.updated_at = datetime.now()
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if entity_id not in self.entity_index:
            return False
        
        entity = self.entity_index[entity_id]
        
        # 从索引中移除
        del self.name_index[entity.name]
        if entity.type in self.type_index:
            self.type_index[entity.type].discard(entity_id)
        
        # 删除关联的边
        edges_to_remove = [
            (s, t) for s, t in self.graph.edges()
            if s == entity_id or t == entity_id
        ]
        self.graph.remove_edges_from(edges_to_remove)
        
        # 从图中移除
        self.graph.remove_node(entity_id)
        del self.entity_index[entity_id]
        
        return True
    
    def add_relation(self,
                     source_id: str,
                     target_id: str,
                     relation_type: str,
                     properties: Dict = None,
                     weight: float = 1.0,
                     relation_id: str = None) -> Optional[KGRelationV2]:
        """添加关系"""
        # 验证实体存在
        if source_id not in self.entity_index or target_id not in self.entity_index:
            return None
        
        relation = KGRelationV2(
            id=relation_id or str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight
        )
        
        # 添加到图
        self.graph.add_edge(source_id, target_id,
                          id=relation.id,
                          type=relation_type,
                          properties=relation.properties,
                          weight=weight)
        
        # 更新索引
        self.relation_index[relation.id] = relation
        
        return relation
    
    def get_relation(self, relation_id: str) -> Optional[KGRelationV2]:
        """获取关系"""
        return self.relation_index.get(relation_id)
    
    def get_relations(self,
                      source_id: str = None,
                      target_id: str = None,
                      relation_type: str = None) -> List[KGRelationV2]:
        """获取关系"""
        results = []
        
        for relation in self.relation_index.values():
            if source_id and relation.source_id != source_id:
                continue
            if target_id and relation.target_id != target_id:
                continue
            if relation_type and relation.relation_type != relation_type:
                continue
            results.append(relation)
        
        return results
    
    def delete_relation(self, relation_id: str) -> bool:
        """删除关系"""
        if relation_id not in self.relation_index:
            return False
        
        relation = self.relation_index[relation_id]
        
        # 从图中移除
        if self.graph.has_edge(relation.source_id, relation.target_id):
            self.graph.remove_edge(relation.source_id, relation.target_id)
        
        del self.relation_index[relation_id]
        return True
    
    def get_neighbors(self, 
                     entity_id: str,
                     relation_type: str = None) -> List[Tuple[KGEntityV2, KGRelationV2]]:
        """获取实体的邻居"""
        if entity_id not in self.entity_index:
            return []
        
        neighbors = []
        
        # 出边
        for successor in self.graph.successors(entity_id):
            edge_data = self.graph.edges[entity_id, successor]
            if edge_data["id"] in self.relation_index:
                relation = self.relation_index[edge_data["id"]]
                if successor in self.entity_index:
                    if relation_type is None or relation.relation_type == relation_type:
                        neighbors.append((self.entity_index[successor], relation))
        
        # 入边
        for predecessor in self.graph.predecessors(entity_id):
            edge_data = self.graph.edges[predecessor, entity_id]
            if edge_data["id"] in self.relation_index:
                relation = self.relation_index[edge_data["id"]]
                if predecessor in self.entity_index:
                    if relation_type is None or relation.relation_type == relation_type:
                        neighbors.append((self.entity_index[predecessor], relation))
        
        return neighbors
    
    def get_shortest_path(self, 
                          source_id: str, 
                          target_id: str) -> Optional[List[KGEntityV2]]:
        """获取最短路径"""
        if source_id not in self.entity_index or target_id not in self.entity_index:
            return None
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return [self.entity_index[node_id] for node_id in path]
        except nx.NetworkXNoPath:
            return None
    
    def get_all_paths(self,
                     source_id: str,
                     target_id: str,
                     max_depth: int = 5) -> List[List[KGEntityV2]]:
        """获取所有路径"""
        if source_id not in self.entity_index or target_id not in self.entity_index:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
            return [[self.entity_index[node_id] for node_id in path] for path in paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(self,
                    entity_ids: List[str],
                    depth: int = 1) -> 'NetworkXGraphBuilder':
        """获取子图"""
        subgraph = NetworkXGraphBuilder()
        
        # 添加种子实体
        for eid in entity_ids:
            if eid in self.entity_index:
                entity = self.entity_index[eid]
                subgraph.add_entity(
                    name=entity.name,
                    entity_type=entity.type,
                    properties=entity.properties,
                    embeddings=entity.embeddings,
                    entity_id=entity.id
                )
        
        # BFS扩展
        current_layer = set(entity_ids)
        for _ in range(depth):
            next_layer = set()
            
            for eid in current_layer:
                entity = self.entity_index.get(eid)
                if entity:
                    neighbors, relations = zip(*self.get_neighbors(eid)) if self.get_neighbors(eid) else ([], [])
                    for neighbor in neighbors:
                        if neighbor.id not in subgraph.entity_index:
                            subgraph.add_entity(
                                name=neighbor.name,
                                entity_type=neighbor.type,
                                properties=neighbor.properties,
                                embeddings=neighbor.embeddings,
                                entity_id=neighbor.id
                            )
                            next_layer.add(neighbor.id)
                        
                        # 添加关系
                        for neighbor, relation in self.get_neighbors(eid):
                            if neighbor.id in subgraph.entity_index:
                                subgraph.add_relation(
                                    source_id=eid,
                                    target_id=neighbor.id,
                                    relation_type=relation.relation_type,
                                    properties=relation.properties,
                                    weight=relation.weight,
                                    relation_id=relation.id
                                )
            
            current_layer = next_layer
        
        return subgraph
    
    def list_entities(self,
                     entity_type: str = None,
                     limit: int = 100,
                     offset: int = 0) -> List[KGEntityV2]:
        """列出实体"""
        if entity_type:
            entity_ids = list(self.type_index.get(entity_type, set()))[offset:offset+limit]
        else:
            entity_ids = list(self.entity_index.keys())[offset:offset+limit]
        
        return [self.entity_index[eid] for eid in entity_ids if eid in self.entity_index]
    
    def stats(self) -> Dict:
        """获取统计信息"""
        type_counts = {t: len(ids) for t, ids in self.type_index.items()}
        relation_counts = {}
        for relation in self.relation_index.values():
            relation_counts[relation.relation_type] = relation_counts.get(relation.relation_type, 0) + 1
        
        return {
            "entity_count": len(self.entity_index),
            "relation_count": len(self.relation_index),
            "entity_types": type_counts,
            "relation_types": relation_counts,
            "is_directed": self.graph.is_directed()
        }
    
    def export_graph(self) -> Dict:
        """导出图"""
        return {
            "entities": [e.to_dict() for e in self.entity_index.values()],
            "relations": [r.to_dict() for r in self.relation_index.values()],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_graph(self, data: Dict, merge: bool = False) -> Tuple[int, int]:
        """导入图"""
        if not merge:
            self.graph.clear()
            self.entity_index.clear()
            self.relation_index.clear()
            self.type_index.clear()
            self.name_index.clear()
        
        imported_entities = 0
        imported_relations = 0
        
        for entity_data in data.get("entities", []):
            entity = KGEntityV2.from_dict(entity_data)
            self.add_entity(
                name=entity.name,
                entity_type=entity.type,
                properties=entity.properties,
                embeddings=entity.embeddings,
                entity_id=entity.id
            )
            imported_entities += 1
        
        for relation_data in data.get("relations", []):
            relation = KGRelationV2.from_dict(relation_data)
            self.add_relation(
                source_id=relation.source_id,
                target_id=relation.target_id,
                relation_type=relation.relation_type,
                properties=relation.properties,
                weight=relation.weight,
                relation_id=relation.id
            )
            imported_relations += 1
        
        return imported_entities, imported_relations
    
    def compute_similarity(self, 
                          entity_id1: str, 
                          entity_id2: str) -> float:
        """计算实体相似度 (基于属性和类型)"""
        entity1 = self.entity_index.get(entity_id1)
        entity2 = self.entity_index.get(entity_id2)
        
        if not entity1 or not entity2:
            return 0.0
        
        # 类型相似度
        type_sim = 1.0 if entity1.type == entity2.type else 0.0
        
        # 属性相似度
        common_keys = set(entity1.properties.keys()) & set(entity2.properties.keys())
        if common_keys:
            prop_sim = sum(
                1.0 for k in common_keys 
                if entity1.properties.get(k) == entity2.properties.get(k)
            ) / len(common_keys)
        else:
            prop_sim = 0.0
        
        # 向量相似度
        if entity1.embeddings and entity2.embeddings:
            vec1 = np.array(entity1.embeddings)
            vec2 = np.array(entity2.embeddings)
            embedding_sim = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
            )
        else:
            embedding_sim = 0.0
        
        return (type_sim + prop_sim + embedding_sim) / 3.0
    
    def find_similar_entities(self, 
                             entity_id: str,
                             top_k: int = 10) -> List[Tuple[KGEntityV2, float]]:
        """查找相似实体"""
        entity = self.entity_index.get(entity_id)
        if not entity:
            return []
        
        similarities = []
        for other_id, other_entity in self.entity_index.items():
            if other_id != entity_id:
                sim = self.compute_similarity(entity_id, other_id)
                similarities.append((other_entity, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# 全局图实例
_networkx_graphs: Dict[str, NetworkXGraphBuilder] = {}


def get_networkx_graph(graph_id: str = "default") -> NetworkXGraphBuilder:
    """获取NetworkX图实例"""
    if graph_id not in _networkx_graphs:
        _networkx_graphs[graph_id] = NetworkXGraphBuilder()
    return _networkx_graphs[graph_id]


def list_networkx_graphs() -> List[str]:
    """列出所有图"""
    return list(_networkx_graphs.keys())


def delete_networkx_graph(graph_id: str) -> bool:
    """删除图"""
    if graph_id in _networkx_graphs:
        del _networkx_graphs[graph_id]
        return True
    return False
