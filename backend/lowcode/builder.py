"""
Agent构建器引擎 - 处理流程图的创建、编辑和执行
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import uuid
import json
from datetime import datetime
from .nodes import (
    BaseNode, NodeRegistry, NodePort, NodeType, NodeCategory
)


class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Edge:
    """边（连接线）定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""  # 源节点ID
    source_port: str = ""  # 源端口名
    target: str = ""  # 目标节点ID
    target_port: str = ""  # 目标端口名
    label: str = ""  # 可选标签


@dataclass
class AgentFlow:
    """Agent流程定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # 节点和边
    nodes: List[Dict] = field(default_factory=list)  # 节点数据
    edges: List[Dict] = field(default_factory=list)  # 边数据
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    
    # 状态
    status: str = "draft"  # draft, published, archived
    tags: List[str] = field(default_factory=list)
    
    # 运行时配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": [dict(n) for n in self.nodes],
            "edges": [dict(e) if isinstance(e, dict) else {
                "id": e.id,
                "source": e.source,
                "source_port": e.source_port,
                "target": e.target,
                "target_port": e.target_port,
                "label": e.label
            } for e in self.edges],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "tags": self.tags,
            "config": self.config
        }


class FlowBuilder:
    """流程构建器"""
    
    def __init__(self):
        self.flow = AgentFlow()
        self.node_map: Dict[str, Dict] = {}  # 节点ID -> 节点数据
        self.edge_map: Dict[str, Edge] = {}  # 边ID -> Edge
        self._build_node_map()
    
    def _build_node_map(self):
        """构建节点映射"""
        self.node_map = {node["id"]: node for node in self.flow.nodes}
    
    def add_node(self, node_type: str, name: str = "", 
                 position: Dict[str, float] = None, 
                 config: Dict[str, Any] = None) -> str:
        """添加节点"""
        node_id = str(uuid.uuid4())
        node_data = {
            "id": node_id,
            "type": node_type,
            "name": name or node_type,
            "position": position or {"x": 0, "y": 0},
            "config": config or {}
        }
        self.flow.nodes.append(node_data)
        self.node_map[node_id] = node_data
        return node_id
    
    def remove_node(self, node_id: str):
        """删除节点"""
        if node_id in self.node_map:
            del self.node_map[node_id]
            self.flow.nodes = [n for n in self.flow.nodes if n["id"] != node_id]
            # 删除相关的边
            self.flow.edges = [
                e for e in self.flow.edges 
                if e["source"] != node_id and e["target"] != node_id
            ]
    
    def update_node(self, node_id: str, updates: Dict[str, Any]):
        """更新节点"""
        if node_id in self.node_map:
            self.node_map[node_id].update(updates)
            self.flow.updated_at = datetime.now().isoformat()
    
    def add_edge(self, source: str, source_port: str,
                 target: str, target_port: str, 
                 label: str = "") -> str:
        """添加边"""
        edge_id = str(uuid.uuid4())
        edge_data = {
            "id": edge_id,
            "source": source,
            "source_port": source_port,
            "target": target,
            "target_port": target_port,
            "label": label
        }
        self.flow.edges.append(edge_data)
        self.edge_map[edge_id] = Edge(**edge_data)
        return edge_id
    
    def remove_edge(self, edge_id: str):
        """删除边"""
        if edge_id in self.edge_map:
            del self.edge_map[edge_id]
            self.flow.edges = [e for e in self.flow.edges if e["id"] != edge_id]
    
    def get_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """获取节点的输入值"""
        node = self.node_map.get(node_id)
        if not node:
            return {}
        
        inputs = {}
        # 查找所有指向该节点的边
        for edge in self.flow.edges:
            if edge["target"] == node_id:
                source_node = self.node_map.get(edge["source"])
                if source_node:
                    # 获取源节点的执行结果
                    # 这里需要在运行时获取实际输出值
                    source_output = self._get_node_output(edge["source"])
                    if source_output:
                        inputs[edge["target_port"]] = source_output.get(
                            edge["source_port"], source_output
                        )
        return inputs
    
    def _get_node_output(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点输出（运行时使用）"""
        # 实际执行时由执行引擎填充
        return None
    
    def validate_flow(self) -> List[str]:
        """验证流程"""
        errors = []
        
        # 检查节点数量
        if len(self.flow.nodes) == 0:
            errors.append("流程必须包含至少一个节点")
        
        # 检查是否有起始节点
        has_start = any(
            n.get("type") == NodeType.START.value for n in self.flow.nodes
        )
        if not has_start and len(self.flow.nodes) > 0:
            errors.append("建议添加START节点作为流程开始")
        
        # 检查边连接的合法性
        for edge in self.flow.edges:
            if edge["source"] not in self.node_map:
                errors.append(f"边引用了不存在的源节点: {edge['source']}")
            if edge["target"] not in self.node_map:
                errors.append(f"边引用了不存在的目标节点: {edge['target']}")
        
        # 检查孤岛节点
        connected_nodes = set()
        for edge in self.flow.edges:
            connected_nodes.add(edge["source"])
            connected_nodes.add(edge["target"])
        
        for node in self.flow.nodes:
            if node["id"] not in connected_nodes and len(self.flow.edges) > 0:
                # 警告：孤立节点
                pass
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "id": self.flow.id,
            "name": self.flow.name,
            "description": self.flow.description,
            "version": self.flow.version,
            "nodes": self.flow.nodes,
            "edges": self.flow.edges,
            "created_at": self.flow.created_at,
            "updated_at": self.flow.updated_at,
            "status": self.flow.status,
            "tags": self.flow.tags,
            "config": self.flow.config
        }
    
    def load_from_dict(self, data: Dict[str, Any]):
        """从字典加载"""
        self.flow = AgentFlow(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            status=data.get("status", "draft"),
            tags=data.get("tags", []),
            config=data.get("config", {})
        )
        self._build_node_map()


class FlowExecutor:
    """流程执行器"""
    
    def __init__(self, registry: NodeRegistry = None):
        self.registry = registry or NodeRegistry()
        self.node_instances: Dict[str, BaseNode] = {}
        self.execution_history: List[Dict] = []
    
    async def execute(self, flow: AgentFlow, 
                      initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行流程"""
        if initial_context is None:
            initial_context = {}
        
        context = {
            **initial_context,
            "_flow_id": flow.id,
            "_execution_id": str(uuid.uuid4()),
            "_start_time": datetime.now().isoformat()
        }
        
        # 构建执行图
        execution_order = self._topological_sort(flow)
        
        results = {}
        errors = []
        
        for node_data in execution_order:
            node_id = node_data["id"]
            try:
                # 创建节点实例
                node = self.registry.create_node(
                    node_data["type"],
                    position=node_data.get("position"),
                    **node_data.get("config", {})
                )
                
                # 获取输入
                inputs = self._gather_inputs(flow, node_id, results)
                node_context = {
                    **context,
                    "config": node_data.get("config", {}),
                    "inputs": inputs
                }
                
                # 执行节点
                output = await node.execute(node_context)
                results[node_id] = output
                
                # 记录执行历史
                self.execution_history.append({
                    "node_id": node_id,
                    "node_type": node_data["type"],
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                errors.append({
                    "node_id": node_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                results[node_id] = {"error": str(e)}
                
                if flow.config.get("stop_on_error", True):
                    break
        
        return {
            "results": results,
            "errors": errors,
            "execution_time": (
                datetime.now() - datetime.fromisoformat(context["_start_time"])
            ).total_seconds(),
            "history": self.execution_history
        }
    
    def _topological_sort(self, flow: AgentFlow) -> List[Dict]:
        """拓扑排序获取执行顺序"""
        # 简单实现：返回所有节点（后续可以优化为真正的拓扑排序）
        return flow.nodes.copy()
    
    def _gather_inputs(self, flow: AgentFlow, node_id: str,
                       previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """收集节点输入"""
        inputs = {}
        
        for edge in flow.edges:
            if edge["target"] == node_id:
                source_output = previous_results.get(edge["source"], {})
                if isinstance(source_output, dict):
                    inputs[edge["target_port"]] = source_output.get(
                        edge["source_port"]
                    )
                else:
                    inputs[edge["target_port"]] = source_output
        
        return inputs
    
    def validate_and_execute(self, flow: AgentFlow,
                              initial_context: Dict = None) -> Dict[str, Any]:
        """验证并执行流程"""
        builder = FlowBuilder()
        builder.flow = flow
        builder.node_map = {n["id"]: n for n in flow.nodes}
        
        errors = builder.validate_flow()
        if errors:
            return {
                "success": False,
                "errors": errors
            }
        
        import asyncio
        return asyncio.run(self.execute(flow, initial_context))


class FlowTemplate:
    """流程模板"""
    
    TEMPLATES = {
        "chat_agent": {
            "name": "对话Agent",
            "description": "基础的对话型AI Agent",
            "nodes": [
                {"type": NodeType.TEXT_INPUT.value, "name": "用户输入", "position": {"x": 100, "y": 100}},
                {"type": NodeType.LLM_COMPLETION.value, "name": "LLM处理", "position": {"x": 300, "y": 100}},
                {"type": NodeType.TEXT_OUTPUT.value, "name": "返回结果", "position": {"x": 500, "y": 100}},
            ],
            "edges": [
                {"source_port": "output", "target_port": "prompt"},
            ]
        },
        "research_agent": {
            "name": "研究Agent",
            "description": "能够搜索和整理信息的Agent",
            "nodes": [
                {"type": NodeType.TEXT_INPUT.value, "name": "研究主题", "position": {"x": 100, "y": 100}},
                {"type": NodeType.WEB_SEARCH.value, "name": "网络搜索", "position": {"x": 300, "y": 100}},
                {"type": NodeType.LLM_COMPLETION.value, "name": "整理结果", "position": {"x": 500, "y": 100}},
                {"type": NodeType.TEXT_OUTPUT.value, "name": "输出报告", "position": {"x": 700, "y": 100}},
            ],
            "edges": [
                {"source_port": "output", "target_port": "query"},
            ]
        },
        "conditional_workflow": {
            "name": "条件分支流程",
            "description": "根据条件执行不同分支",
            "nodes": [
                {"type": NodeType.TEXT_INPUT.value, "name": "输入", "position": {"x": 100, "y": 200}},
                {"type": NodeType.IF.value, "name": "条件判断", "position": {"x": 300, "y": 200}},
                {"type": NodeType.LLM_COMPLETION.value, "name": "真分支", "position": {"x": 500, "y": 100}},
                {"type": NodeType.TEXT_OUTPUT.value, "name": "假分支输出", "position": {"x": 500, "y": 300}},
                {"type": NodeType.TEXT_OUTPUT.value, "name": "最终输出", "position": {"x": 700, "y": 200}},
            ],
            "edges": []
        }
    }
    
    @classmethod
    def get_template(cls, template_id: str) -> Optional[Dict]:
        """获取模板"""
        return cls.TEMPLATES.get(template_id)
    
    @classmethod
    def list_templates(cls) -> List[Dict]:
        """列出所有模板"""
        return [
            {"id": k, **v} for k, v in cls.TEMPLATES.items()
        ]
    
    @classmethod
    def create_from_template(cls, template_id: str, 
                              name: str = "") -> AgentFlow:
        """从模板创建流程"""
        template = cls.get_template(template_id)
        if not template:
            raise ValueError(f"Unknown template: {template_id}")
        
        builder = FlowBuilder()
        node_ids = []
        
        # 创建节点
        for node_data in template["nodes"]:
            node_id = builder.add_node(
                node_type=node_data["type"],
                name=node_data.get("name", ""),
                position=node_data.get("position")
            )
            node_ids.append(node_id)
        
        # 创建边
        edges = template.get("edges", [])
        if edges:
            for i, edge_def in enumerate(edges):
                if i < len(node_ids) - 1:
                    builder.add_edge(
                        source=node_ids[i],
                        source_port=edge_def.get("source_port", "output"),
                        target=node_ids[i + 1],
                        target_port=edge_def.get("target_port", "input")
                    )
        
        flow = builder.flow
        flow.name = name or template["name"]
        flow.description = template.get("description", "")
        
        return flow
