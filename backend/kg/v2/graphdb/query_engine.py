"""
图查询引擎

提供统一的图数据库查询接口，支持Neo4j和NetworkX。
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class GraphDBType(str, Enum):
    """图数据库类型"""
    NEO4J = "neo4j"
    NETWORKX = "networkx"
    AUTO = "auto"  # 自动选择


class QueryEngine:
    """图查询引擎 - 统一接口"""
    
    def __init__(self, db_type: GraphDBType = GraphDBType.AUTO):
        self.db_type = db_type
        self._neo4j_adapter = None
        self._networkx_graph = None
        self._use_neo4j = False
        self._graph_cache: Dict[str, Any] = {}
    
    def initialize(self, 
                   neo4j_uri: str = None,
                   neo4j_user: str = None,
                   neo4j_password: str = None) -> bool:
        """初始化查询引擎"""
        # 尝试连接Neo4j
        if neo4j_uri or self.db_type == GraphDBType.NEO4J:
            try:
                from kg.v2.graphdb.neo4j_adapter import get_neo4j_adapter
                self._neo4j_adapter = get_neo4j_adapter(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                if self._neo4j_adapter.is_connected():
                    self._use_neo4j = True
                    return True
            except Exception:
                pass
        
        # 回退到NetworkX
        from kg.v2.graphdb.networkx_builder import get_networkx_graph
        self._networkx_graph = get_networkx_graph()
        self._use_neo4j = False
        return True
    
    def is_using_neo4j(self) -> bool:
        """是否使用Neo4j"""
        return self._use_neo4j
    
    def _get_networkx_graph(self, graph_id: str = "default"):
        """获取NetworkX图"""
        from kg.v2.graphdb.networkx_builder import get_networkx_graph
        return get_networkx_graph(graph_id)
    
    # ========== 实体操作 ==========
    
    def create_entity(self,
                     name: str,
                     entity_type: str,
                     properties: Dict = None,
                     embeddings: List[float] = None,
                     graph_id: str = "default") -> Optional[str]:
        """创建实体"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.create_entity(
                name, entity_type, properties, embeddings
            )
        else:
            graph = self._get_networkx_graph(graph_id)
            entity = graph.add_entity(name, entity_type, properties, embeddings)
            return entity.id if entity else None
    
    def get_entity(self, 
                   entity_id: str,
                   graph_id: str = "default") -> Optional[Dict]:
        """获取实体"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.get_entity(entity_id)
        else:
            graph = self._get_networkx_graph(graph_id)
            entity = graph.get_entity(entity_id)
            return entity.to_dict() if entity else None
    
    def get_entity_by_name(self,
                          name: str,
                          graph_id: str = "default") -> Optional[Dict]:
        """根据名称获取实体"""
        if self._use_neo4j and self._neo4j_adapter:
            results = self._neo4j_adapter.execute_cypher(
                "MATCH (n:Entity {name: $name}) RETURN n",
                {"name": name}
            )
            if results:
                return results[0]["n"]
            return None
        else:
            graph = self._get_networkx_graph(graph_id)
            entity = graph.get_entity_by_name(name)
            return entity.to_dict() if entity else None
    
    def update_entity(self,
                     entity_id: str,
                     name: str = None,
                     properties: Dict = None,
                     embeddings: List[float] = None,
                     graph_id: str = "default") -> bool:
        """更新实体"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.update_entity(entity_id, name, properties)
        else:
            graph = self._get_networkx_graph(graph_id)
            entity = graph.update_entity(entity_id, name, properties, embeddings)
            return entity is not None
    
    def delete_entity(self, 
                      entity_id: str,
                      graph_id: str = "default") -> bool:
        """删除实体"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.delete_entity(entity_id)
        else:
            graph = self._get_networkx_graph(graph_id)
            return graph.delete_entity(entity_id)
    
    def list_entities(self,
                      entity_type: str = None,
                      limit: int = 100,
                      offset: int = 0,
                      graph_id: str = "default") -> List[Dict]:
        """列出实体"""
        if self._use_neo4j and self._neo4j_adapter:
            query = "MATCH (n:Entity) RETURN n"
            if entity_type:
                query = f"MATCH (n:{entity_type}) RETURN n"
            
            results = self._neo4j_adapter.execute_cypher(query)
            entities = [r["n"] for r in results][offset:offset+limit]
            return entities
        else:
            graph = self._get_networkx_graph(graph_id)
            entities = graph.list_entities(entity_type, limit, offset)
            return [e.to_dict() for e in entities]
    
    # ========== 关系操作 ==========
    
    def create_relation(self,
                       source_id: str,
                       target_id: str,
                       relation_type: str,
                       properties: Dict = None,
                       weight: float = 1.0,
                       graph_id: str = "default") -> Optional[str]:
        """创建关系"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.create_relation(
                source_id, target_id, relation_type, properties, weight
            )
        else:
            graph = self._get_networkx_graph(graph_id)
            relation = graph.add_relation(
                source_id, target_id, relation_type, properties, weight
            )
            return relation.id if relation else None
    
    def get_relations(self,
                      source_id: str = None,
                      target_id: str = None,
                      relation_type: str = None,
                      graph_id: str = "default") -> List[Dict]:
        """获取关系"""
        if self._use_neo4j and self._neo4j_adapter:
            results = self._neo4j_adapter.get_relations(source_id, target_id, relation_type)
            return [r["relation"] for r in results]
        else:
            graph = self._get_networkx_graph(graph_id)
            relations = graph.get_relations(source_id, target_id, relation_type)
            return [r.to_dict() for r in relations]
    
    def delete_relation(self, 
                       relation_id: str,
                       graph_id: str = "default") -> bool:
        """删除关系"""
        if self._use_neo4j and self._neo4j_adapter:
            return self._neo4j_adapter.delete_relation(relation_id)
        else:
            graph = self._get_networkx_graph(graph_id)
            return graph.delete_relation(relation_id)
    
    # ========== 图遍历 ==========
    
    def get_neighbors(self,
                     entity_id: str,
                     relation_type: str = None,
                     graph_id: str = "default") -> List[Dict]:
        """获取邻居"""
        if self._use_neo4j and self._neo4j_adapter:
            query = """
            MATCH (n {id: $id})-[r]->(m)
            WHERE $relation_type IS NULL OR r.type = $relation_type
            RETURN m, r
            """
            results = self._neo4j_adapter.execute_cypher(query, {
                "id": entity_id,
                "relation_type": relation_type
            })
            return [{"entity": r["m"], "relation": r["r"]} for r in results]
        else:
            graph = self._get_networkx_graph(graph_id)
            neighbors = graph.get_neighbors(entity_id, relation_type)
            return [
                {"entity": n.to_dict(), "relation": rel.to_dict()}
                for n, rel in neighbors
            ]
    
    def get_shortest_path(self,
                         source_id: str,
                         target_id: str,
                         graph_id: str = "default") -> Optional[List[Dict]]:
        """获取最短路径"""
        if self._use_neo4j:
            results = self._neo4j_adapter.find_shortest_path(source_id, target_id)
            return results
        else:
            graph = self._get_networkx_graph(graph_id)
            path = graph.get_shortest_path(source_id, target_id)
            return [e.to_dict() for e in path] if path else None
    
    def get_all_paths(self,
                     source_id: str,
                     target_id: str,
                     max_depth: int = 5,
                     graph_id: str = "default") -> List[List[Dict]]:
        """获取所有路径"""
        if self._use_neo4j:
            pass
        
        graph = self._get_networkx_graph(graph_id)
        paths = graph.get_all_paths(source_id, target_id, max_depth)
        return [[e.to_dict() for e in path] for path in paths]
    
    def get_subgraph(self,
                    entity_ids: List[str],
                    depth: int = 1,
                    graph_id: str = "default") -> Dict:
        """获取子图"""
        if self._use_neo4j:
            return {"entities": [], "relations": []}
        else:
            graph = self._get_networkx_graph(graph_id)
            subgraph = graph.get_subgraph(entity_ids, depth)
            return subgraph.export_graph()
    
    # ========== 统计和导出 ==========
    
    def stats(self, graph_id: str = "default") -> Dict:
        """获取统计信息"""
        if self._use_neo4j and self._neo4j_adapter:
            entities = self._neo4j_adapter.execute_cypher("MATCH (n:Entity) RETURN n")
            relations = self._neo4j_adapter.execute_cypher("MATCH ()-[r]->() RETURN r")
            return {
                "entity_count": len(entities),
                "relation_count": len(relations),
                "db_type": "neo4j"
            }
        else:
            graph = self._get_networkx_graph(graph_id)
            stats = graph.stats()
            stats["db_type"] = "networkx"
            return stats
    
    def export_graph(self, graph_id: str = "default") -> Dict:
        """导出图"""
        if self._use_neo4j:
            entities = self._neo4j_adapter.execute_cypher("MATCH (n:Entity) RETURN n")
            relations = self._neo4j_adapter.execute_cypher("MATCH ()-[r]->() RETURN r")
            return {
                "entities": [e["n"] for e in entities],
                "relations": [r["r"] for r in relations],
                "exported_at": datetime.now().isoformat()
            }
        else:
            graph = self._get_networkx_graph(graph_id)
            return graph.export_graph()
    
    def import_graph(self,
                     data: Dict,
                     merge: bool = False,
                     graph_id: str = "default") -> Tuple[int, int]:
        """导入图"""
        if self._use_neo4j:
            # Neo4j批量导入
            entity_count = 0
            relation_count = 0
            
            for entity_data in data.get("entities", []):
                self._neo4j_adapter.create_entity(
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("type", ""),
                    properties=entity_data.get("properties", {}),
                    embeddings=entity_data.get("embeddings")
                )
                entity_count += 1
            
            for relation_data in data.get("relations", []):
                self._neo4j_adapter.create_relation(
                    source_id=relation_data.get("source_id", ""),
                    target_id=relation_data.get("target_id", ""),
                    relation_type=relation_data.get("relation_type", ""),
                    properties=relation_data.get("properties", {}),
                    weight=relation_data.get("weight", 1.0)
                )
                relation_count += 1
            
            return entity_count, relation_count
        else:
            graph = self._get_networkx_graph(graph_id)
            return graph.import_graph(data, merge)


# 全局查询引擎实例
_query_engine: Optional[QueryEngine] = None


def get_query_engine(db_type: GraphDBType = GraphDBType.AUTO) -> QueryEngine:
    """获取查询引擎实例"""
    global _query_engine
    
    if _query_engine is None:
        _query_engine = QueryEngine(db_type)
        _query_engine.initialize()
    
    return _query_engine


def reset_query_engine():
    """重置查询引擎"""
    global _query_engine
    _query_engine = None
