"""
Pipeline生成器模块 - Pipeline Generator

负责从理解结果生成Pipeline结构
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from .nl_understand import NLUnderstand, UnderstandingResult, IntentType, EntityType


class NodeType(Enum):
    """Pipeline节点类型"""
    INPUT = "input"
    OUTPUT = "output"
    PROCESSOR = "processor"
    LLM = "llm"
    CONDITION = "condition"
    FILTER = "filter"
    TRANSFORM = "transform"
    MAP = "map"
    REDUCE = "reduce"
    PARALLEL = "parallel"
    LOOP = "loop"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"


class ConnectionType(Enum):
    """连接类型"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


@dataclass
class Node:
    """Pipeline节点"""
    id: str
    name: str
    type: NodeType
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    position: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "position": self.position
        }


@dataclass
class Connection:
    """Pipeline连接"""
    id: str
    source: str
    target: str
    type: ConnectionType = ConnectionType.SEQUENTIAL
    condition: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "condition": self.condition,
            "config": self.config
        }


@dataclass
class Pipeline:
    """Pipeline定义"""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: List[Node] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "config": self.config,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class PipelineGenerator:
    """Pipeline生成器"""
    
    def __init__(self):
        self.node_counter = 0
        self.connection_counter = 0
    
    def generate(self, understanding: UnderstandingResult) -> Pipeline:
        """
        从理解结果生成Pipeline
        
        Args:
            understanding: 自然语言理解结果
            
        Returns:
            Pipeline: 生成的Pipeline
        """
        # 1. 确定Pipeline基本信息
        pipeline = self._create_base_pipeline(understanding)
        
        # 2. 根据意图生成节点
        if understanding.intent == IntentType.CREATE_PIPELINE:
            nodes = self._generate_nodes_for_create(understanding)
        elif understanding.intent == IntentType.UPDATE_PIPELINE:
            nodes = self._generate_nodes_for_update(understanding)
        else:
            nodes = self._generate_default_nodes(understanding)
        
        pipeline.nodes = nodes
        
        # 3. 生成连接
        connections = self._generate_connections(nodes)
        pipeline.connections = connections
        
        # 4. 配置参数
        pipeline.config = self._extract_config(understanding)
        
        return pipeline
    
    def _create_base_pipeline(self, understanding: UnderstandingResult) -> Pipeline:
        """创建基础Pipeline"""
        # 提取名称
        name = "unnamed_pipeline"
        for entity in understanding.entities:
            if entity.type == EntityType.PIPELINE_NAME:
                name = entity.value
                break
        
        return Pipeline(
            id=self._generate_id("pipeline"),
            name=name,
            description=f"Generated from: {understanding.raw_query}",
            metadata={
                "source": "nl_generator",
                "confidence": understanding.confidence,
                "generated_at": self._get_timestamp()
            }
        )
    
    def _generate_nodes_for_create(self, understanding: UnderstandingResult) -> List[Node]:
        """为创建Pipeline生成节点"""
        nodes = []
        
        # 检查是否指定了节点类型
        node_entities = [e for e in understanding.entities 
                        if e.type == EntityType.NODE_TYPE]
        
        if node_entities:
            # 用户指定了节点类型
            for entity in node_entities:
                node = self._create_node_from_type(entity.value)
                if node:
                    nodes.append(node)
        else:
            # 默认生成输入、处理、输出节点
            input_node = Node(
                id=self._generate_id("node"),
                name="Input",
                type=NodeType.INPUT,
                config={"source": "user_input"}
            )
            processor_node = Node(
                id=self._generate_id("node"),
                name="Processor",
                type=NodeType.PROCESSOR,
                config={"operation": "default"}
            )
            output_node = Node(
                id=self._generate_id("node"),
                name="Output",
                type=NodeType.OUTPUT,
                config={"format": "json"}
            )
            nodes = [input_node, processor_node, output_node]
        
        return nodes
    
    def _generate_nodes_for_update(self, understanding: UnderstandingResult) -> List[Node]:
        """为更新Pipeline生成节点"""
        # 更新场景通常只需要返回修改说明，不生成完整节点
        nodes = []
        
        return nodes
    
    def _generate_default_nodes(self, understanding: UnderstandingResult) -> List[Node]:
        """生成默认节点"""
        nodes = []
        
        # 默认输入节点
        input_node = Node(
            id=self._generate_id("node"),
            name="Input",
            type=NodeType.INPUT,
            config={"source": "default"}
        )
        nodes.append(input_node)
        
        # 默认处理节点
        processor_node = Node(
            id=self._generate_id("node"),
            name="Processor",
            type=NodeType.PROCESSOR,
            config={"operation": "process"}
        )
        nodes.append(processor_node)
        
        # 默认输出节点
        output_node = Node(
            id=self._generate_id("node"),
            name="Output",
            type=NodeType.OUTPUT,
            config={"format": "json"}
        )
        nodes.append(output_node)
        
        return nodes
    
    def _create_node_from_type(self, node_type: str) -> Optional[Node]:
        """根据类型创建节点"""
        type_map = {
            "input": NodeType.INPUT,
            "output": NodeType.OUTPUT,
            "processor": NodeType.PROCESSOR,
            "llm": NodeType.LLM,
            "condition": NodeType.CONDITION,
            "filter": NodeType.FILTER,
            "transform": NodeType.TRANSFORM,
            "map": NodeType.MAP,
            "reduce": NodeType.REDUCE,
            "parallel": NodeType.PARALLEL,
            "loop": NodeType.LOOP,
            "api_call": NodeType.API_CALL,
            "database_query": NodeType.DATABASE_QUERY,
            "file_read": NodeType.FILE_READ,
            "file_write": NodeType.FILE_WRITE,
        }
        
        node_enum_type = type_map.get(node_type.lower())
        if not node_enum_type:
            return None
        
        return Node(
            id=self._generate_id("node"),
            name=f"{node_type.capitalize()}_Node",
            type=node_enum_type,
            config=self._get_default_config(node_enum_type)
        )
    
    def _get_default_config(self, node_type: NodeType) -> Dict[str, Any]:
        """获取节点默认配置"""
        config_maps = {
            NodeType.INPUT: {"source": "default", "format": "json"},
            NodeType.OUTPUT: {"format": "json"},
            NodeType.PROCESSOR: {"operation": "default"},
            NodeType.LLM: {"model": "default", "temperature": 0.7},
            NodeType.CONDITION: {"expression": "true"},
            NodeType.FILTER: {"condition": "always"},
            NodeType.TRANSFORM: {"mapping": {}},
            NodeType.MAP: {"function": "identity"},
            NodeType.REDUCE: {"function": "sum"},
            NodeType.PARALLEL: {"max_workers": 4},
            NodeType.LOOP: {"max_iterations": 10, "condition": "true"},
            NodeType.API_CALL: {"method": "GET", "url": ""},
            NodeType.DATABASE_QUERY: {"query": "", "connection": ""},
            NodeType.FILE_READ: {"path": "", "format": "text"},
            NodeType.FILE_WRITE: {"path": "", "format": "json"},
        }
        
        return config_maps.get(node_type, {})
    
    def _generate_connections(self, nodes: List[Node]) -> List[Connection]:
        """生成节点连接"""
        connections = []
        
        if len(nodes) < 2:
            return connections
        
        # 默认顺序连接
        for i in range(len(nodes) - 1):
            connection = Connection(
                id=self._generate_id("connection"),
                source=nodes[i].id,
                target=nodes[i + 1].id,
                type=ConnectionType.SEQUENTIAL
            )
            connections.append(connection)
        
        return connections
    
    def _extract_config(self, understanding: UnderstandingResult) -> Dict[str, Any]:
        """提取配置参数"""
        config = {}
        
        for entity in understanding.entities:
            if entity.type == EntityType.PARAMETER:
                # 解析参数名和值
                parts = entity.value.split("=")
                if len(parts) == 2:
                    config[parts[0].strip()] = parts[1].strip()
                else:
                    config[f"param_{len(config)}"] = entity.value
            
            elif entity.type == EntityType.DATA_SOURCE:
                config["data_source"] = entity.value
            
            elif entity.type == EntityType.OUTPUT_FORMAT:
                config["output_format"] = entity.value
        
        return config
    
    def _generate_id(self, prefix: str) -> str:
        """生成唯一ID"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def set_node_position(self, node_id: str, x: int, y: int):
        """设置节点位置"""
        for node in self.nodes:
            if node.id == node_id:
                node.position = {"x": x, "y": y}
                break
    
    def add_parallel_branch(self, pipeline: Pipeline, branch_nodes: List[Node]):
        """添加并行分支"""
        # 找到合并点
        if not pipeline.connections:
            return
        
        last_connection = pipeline.connections[-1]
        
        # 为每个分支节点创建并行连接
        for i, node in enumerate(branch_nodes):
            # 创建并行连接
            parallel_connection = Connection(
                id=self._generate_id("connection"),
                source=last_connection.source,
                target=node.id,
                type=ConnectionType.PARALLEL
            )
            pipeline.connections.append(parallel_connection)
            pipeline.nodes.append(node)
    
    def validate_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """
        验证Pipeline结构
        
        Args:
            pipeline: Pipeline实例
            
        Returns:
            Dict: 验证结果
        """
        errors = []
        warnings = []
        
        # 检查节点数量
        if len(pipeline.nodes) < 2:
            errors.append("Pipeline至少需要2个节点")
        
        # 检查输入输出节点
        has_input = any(n.type == NodeType.INPUT for n in pipeline.nodes)
        has_output = any(n.type == NodeType.OUTPUT for n in pipeline.nodes)
        
        if not has_input:
            warnings.append("缺少输入节点")
        if not has_output:
            warnings.append("缺少输出节点")
        
        # 检查连接
        node_ids = {n.id for n in pipeline.nodes}
        for conn in pipeline.connections:
            if conn.source not in node_ids:
                errors.append(f"连接源节点不存在: {conn.source}")
            if conn.target not in node_ids:
                errors.append(f"连接目标节点不存在: {conn.target}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def optimize_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """
        优化Pipeline结构
        
        Args:
            pipeline: Pipeline实例
            
        Returns:
            Pipeline: 优化后的Pipeline
        """
        # 移除孤立节点
        connected_nodes = set()
        for conn in pipeline.connections:
            connected_nodes.add(conn.source)
            connected_nodes.add(conn.target)
        
        pipeline.nodes = [n for n in pipeline.nodes 
                         if n.id in connected_nodes or n.type in [NodeType.INPUT, NodeType.OUTPUT]]
        
        # 合并连续的同类型节点
        # （简化实现，实际应该更复杂）
        
        return pipeline
