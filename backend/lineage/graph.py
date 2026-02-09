"""
Model Lineage模块
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class LineageNode:
    """血缘节点"""
    node_id: str
    node_type: str  # dataset, run, model, artifact
    external_id: str  # 外部ID
    name: str
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LineageEdge:
    """血缘边"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # data_flow, execution_flow
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class LineageGraph:
    """血缘图"""
    
    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self.node_index: Dict[str, str] = {}  # (node_type, external_id) -> node_id
    
    async def create_node(
        self,
        node_type: str,
        external_id: str,
        name: str,
        metadata: Optional[Dict] = None
    ) -> LineageNode:
        """创建节点"""
        # 检查是否存在
        key = (node_type, external_id)
        if key in self.node_index:
            return self.nodes[self.node_index[key]]
        
        node = LineageNode(
            node_id=str(uuid4()),
            node_type=node_type,
            external_id=external_id,
            name=name,
            metadata=metadata or {}
        )
        
        self.nodes[node.node_id] = node
        self.node_index[key] = node.node_id
        
        return node
    
    async def get_node(self, node_id: str) -> Optional[LineageNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    async def get_node_by_external_id(
        self,
        node_type: str,
        external_id: str
    ) -> Optional[LineageNode]:
        """根据外部ID获取节点"""
        key = (node_type, external_id)
        node_id = self.node_index.get(key)
        if node_id:
            return self.nodes.get(node_id)
        return None
    
    async def create_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: str,
        metadata: Optional[Dict] = None
    ) -> LineageEdge:
        """创建边"""
        # 验证节点存在
        if source_node_id not in self.nodes:
            raise ValueError(f"Source node {source_node_id} not found")
        if target_node_id not in self.nodes:
            raise ValueError(f"Target node {target_node_id} not found")
        
        edge = LineageEdge(
            edge_id=str(uuid4()),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            metadata=metadata or {}
        )
        
        self.edges.append(edge)
        return edge
    
    async def connect(
        self,
        source: tuple,  # (node_type, external_id)
        target: tuple,  # (node_type, external_id)
        edge_type: str,
        metadata: Optional[Dict] = None
    ) -> LineageEdge:
        """连接两个节点"""
        # 获取或创建节点
        source_node = await self.get_node_by_external_id(*source)
        if not source_node:
            source_node = await self.create_node(
                node_type=source[0],
                external_id=source[1],
                name=source[1]
            )
        
        target_node = await self.get_node_by_external_id(*target)
        if not target_node:
            target_node = await self.create_node(
                node_type=target[0],
                external_id=target[1],
                name=target[1]
            )
        
        return await self.create_edge(
            source_node_id=source_node.node_id,
            target_node_id=target_node.node_id,
            edge_type=edge_type,
            metadata=metadata
        )
    
    async def get_lineage(
        self,
        node_id: str,
        direction: str = "upstream",  # upstream, downstream, both
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """获取血缘路径"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        
        # BFS遍历
        visited = {node_id}
        queue = [(node_id, 0)]
        upstream_nodes = []
        downstream_nodes = []
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for edge in self.edges:
                if depth == 0:
                    # 找直接关联
                    if edge.source_node_id == current_id and direction in ["downstream", "both"]:
                        if edge.target_node_id not in visited:
                            visited.add(edge.target_node_id)
                            downstream_nodes.append({
                                "node_id": edge.target_node_id,
                                "edge_type": edge.edge_type
                            })
                            queue.append((edge.target_node_id, depth + 1))
                    
                    if edge.target_node_id == current_id and direction in ["upstream", "both"]:
                        if edge.source_node_id not in visited:
                            visited.add(edge.source_node_id)
                            upstream_nodes.append({
                                "node_id": edge.source_node_id,
                                "edge_type": edge.edge_type
                            })
                            queue.append((edge.source_node_id, depth + 1))
        
        return {
            "node": {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "name": node.name,
                "external_id": node.external_id
            },
            "upstream": upstream_nodes,
            "downstream": downstream_nodes
        }
    
    async def get_execution_lineage(self, run_id: str) -> Dict[str, Any]:
        """获取运行血缘 (Dataset -> Run -> Model)"""
        # 获取Run节点
        run_node = await self.get_node_by_external_id("run", run_id)
        if not run_node:
            raise ValueError(f"Run {run_id} not found")
        
        # 找输入Dataset
        upstream = []
        downstream = []
        
        for edge in self.edges:
            if edge.source_node_id == run_node.node_id and edge.edge_type == "data_flow":
                target = self.nodes.get(edge.target_node_id)
                if target:
                    downstream.append({
                        "node_id": target.node_id,
                        "node_type": target.node_type,
                        "name": target.name
                    })
            
            if edge.target_node_id == run_node.node_id and edge.edge_type == "execution_flow":
                source = self.nodes.get(edge.source_node_id)
                if source:
                    upstream.append({
                        "node_id": source.node_id,
                        "node_type": source.node_type,
                        "name": source.name
                    })
        
        return {
            "run_id": run_id,
            "input_datasets": upstream,
            "output_models": downstream
        }
    
    async def trace_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """追踪模型血缘"""
        model_node = await self.get_node_by_external_id("model", model_id)
        if not model_node:
            raise ValueError(f"Model {model_id} not found")
        
        # 获取完整血缘链
        full_lineage = await self.get_lineage(model_node.node_id, "both")
        
        # 构建血缘链
        chain = []
        
        # 上游: Dataset -> Run
        for upstream in full_lineage["upstream"]:
            node = self.nodes.get(upstream["node_id"])
            if node:
                chain.append({
                    "type": node.node_type,
                    "id": node.external_id,
                    "name": node.name
                })
        
        # 当前: Model
        chain.append({
            "type": "model",
            "id": model_id,
            "name": model_node.name
        })
        
        # 下游: Model -> ...
        for downstream in full_lineage["downstream"]:
            node = self.nodes.get(downstream["node_id"])
            if node:
                chain.append({
                    "type": node.node_type,
                    "id": node.external_id,
                    "name": node.name
                })
        
        return {
            "model_id": model_id,
            "lineage_chain": chain
        }
    
    async def get_graph(self) -> Dict[str, Any]:
        """获取完整图"""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "name": n.name
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source": e.source_node_id,
                    "target": e.target_node_id,
                    "edge_type": e.edge_type
                }
                for e in self.edges
            ],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges)
        }
    
    async def delete_node(self, node_id: str) -> bool:
        """删除节点及其关联边"""
        if node_id not in self.nodes:
            return False
        
        # 删除节点
        node = self.nodes.pop(node_id)
        
        # 删除索引
        key = (node.node_type, node.external_id)
        if key in self.node_index:
            del self.node_index[key]
        
        # 删除关联边
        self.edges = [
            e for e in self.edges
            if e.source_node_id != node_id and e.target_node_id != node_id
        ]
        
        return True

# Lineage Graph实例
lineage_graph = LineageGraph()
