"""
流程编译器 - 将流程定义编译为可执行代码
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from .builder import AgentFlow, FlowBuilder
from .nodes import NodeRegistry


class FlowCompiler:
    """流程编译器"""
    
    def __init__(self, registry: NodeRegistry = None):
        self.registry = registry or NodeRegistry()
        self.code_generators: Dict[str, callable] = {}
        self._register_default_generators()
    
    def _register_default_generators(self):
        """注册默认代码生成器"""
        self.code_generators = {
            "python": self._generate_python,
            "json": self._generate_json,
            "dag": self._generate_dag,
        }
    
    def compile(self, flow: AgentFlow, 
                target: str = "python") -> Dict[str, Any]:
        """编译流程"""
        if target not in self.code_generators:
            raise ValueError(f"Unsupported target: {target}")
        
        return self.code_generators[target](flow)
    
    def _generate_python(self, flow: AgentFlow) -> Dict[str, Any]:
        """生成Python代码"""
        code_lines = [
            "#!/usr/bin/env python3",
            f"# Auto-generated Agent Flow: {flow.name}",
            f"# Flow ID: {flow.id}",
            f"# Generated at: {datetime.now().isoformat()}",
            "",
            "import asyncio",
            "import json",
            "from typing import Any, Dict",
            "",
            "",
            "class AgentFlow:",
            f"    flow_id = '{flow.id}'",
            f"    name = '{flow.name}'",
            f"    version = '{flow.version}'",
            "",
            "    def __",
            "       init__(self): self.results = {}",
            "        self.context = {}",
            "",
            "    async def execute(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:",
            "        if initial_context is None:",
            "            initial_context = {}",
            "        self.context = {**initial_context, '_flow_id': self.flow_id}",
            "",
        ]
        
        # 为每个节点生成执行代码
        node_map = {node["id"]: node for node in flow.nodes}
        
        for node in flow.nodes:
            node_code = self._generate_node_code(node, node_map, flow.edges)
            code_lines.extend(node_code)
        
        code_lines.extend([
            "        return self.results",
            "",
            "",
            "if __name__ == '__main__':",
            "    async def main():",
            "        flow = AgentFlow()",
            "        results = await flow.execute({})",
            "        print(json.dumps(results, indent=2, ensure_ascii=False))",
            "    asyncio.run(main())"
        ])
        
        return {
            "language": "python",
            "code": "\n".join(code_lines),
            "flow_id": flow.id,
            "node_count": len(flow.nodes),
            "edge_count": len(flow.edges)
        }
    
    def _generate_node_code(self, node: Dict, 
                            node_map: Dict[str, Dict],
                            edges: List[Dict]) -> List[str]:
        """生成单个节点的代码"""
        node_type = node.get("type", "")
        node_id = node.get("id", "")[:8]  # 短ID
        node_name = node.get("name", node_type)
        
        code_lines = []
        
        # 生成注释
        code_lines.append(f"        # Node: {node_name} ({node_type})")
        
        # 生成节点ID变量
        code_lines.append(f"        node_{node_id} = '{node['id']}'")
        
        # 获取输入边
        input_edges = [e for e in edges if e["target"] == node["id"]]
        
        if input_edges:
            # 有输入依赖
            code_lines.append(f"        # Inputs from {len(input_edges)} source(s)")
            for edge in input_edges:
                source_node = node_map.get(edge["source"], {})
                source_type = source_node.get("type", "unknown")
                code_lines.append(f"        #   From {source_type}: {edge['source_port']} -> {edge['target_port']}")
        
        # 根据节点类型生成代码
        if "llm" in node_type.lower() or "completion" in node_type.lower():
            code_lines.extend([
                f"        # LLM Processing: {node_name}",
                f"        prompt = self.context.get('prompt', '')",
                f"        config = {json.dumps(node.get('config', {}))}",
                "        # TODO: Call LLM service",
                f"        self.results['{node['id']}'] = {{",
                f"            'output': f'Generated: {{prompt[:100]}}...',",
                f"            'usage': {{'total_tokens': 100}}",
                "        }"
            ])
        elif "search" in node_type.lower():
            code_lines.extend([
                f"        # Web Search: {node_name}",
                f"        query = self.context.get('query', '')",
                f"        config = {json.dumps(node.get('config', {}))}",
                "        # TODO: Call search API",
                f"        self.results['{node['id']}'] = {{",
                "            'results': [{'title': query, 'url': '...'}]",
                "        }"
            ])
        elif "input" in node_type.lower():
            code_lines.extend([
                f"        # Input: {node_name}",
                f"        self.results['{node['id']}'] = {{",
                "            'output': self.context.get('input', '')",
                "        }"
            ])
        elif "output" in node_type.lower():
            code_lines.extend([
                f"        # Output: {node_name}",
                f"        input_data = self.context.get('input', {{}})",
                f"        self.results['{node['id']}'] = {{",
                "            'result': input_data",
                "        }"
            ])
        elif node_type == "if":
            code_lines.extend([
                f"        # Conditional: {node_name}",
                f"        condition = self.context.get('condition', '')",
                f"        config = {json.dumps(node.get('config', {}))}",
                "        # TODO: Evaluate condition",
                f"        self.results['{node['id']}'] = {{",
                "            'true_output': None,",
                "            'false_output': None",
                "        }"
            ])
        else:
            # 默认处理
            code_lines.extend([
                f"        # Process: {node_name}",
                f"        self.results['{node['id']}'] = {{",
                f"            'output': None,",
                f"            'status': 'processed'",
                "        }"
            ])
        
        code_lines.append("")  # 空行
        
        return code_lines
    
    def _generate_json(self, flow: AgentFlow) -> Dict[str, Any]:
        """生成JSON格式的编译结果"""
        node_map = {node["id"]: node for node in flow.nodes}
        
        compiled_nodes = []
        for node in flow.nodes:
            # 解析依赖
            input_edges = [e for e in flow.edges if e["target"] == node["id"]]
            output_edges = [e for e in flow.edges if e["source"] == node["id"]]
            
            compiled_node = {
                "id": node["id"],
                "type": node["type"],
                "name": node.get("name", ""),
                "config": node.get("config", {}),
                "dependencies": [e["source"] for e in input_edges],
                "dependents": [e["target"] for e in output_edges],
                "ports": {
                    "inputs": {e["target_port"]: e["source"] for e in input_edges},
                    "outputs": {e["source_port"]: e["target"] for e in output_edges}
                },
                "position": node.get("position", {})
            }
            compiled_nodes.append(compiled_node)
        
        return {
            "format": "compiled_flow",
            "flow": {
                "id": flow.id,
                "name": flow.name,
                "version": flow.version
            },
            "nodes": compiled_nodes,
            "metadata": {
                "node_count": len(flow.nodes),
                "edge_count": len(flow.edges),
                "compiled_at": datetime.now().isoformat()
            }
        }
    
    def _generate_dag(self, flow: AgentFlow) -> Dict[str, Any]:
        """生成DAG（有向无环图）格式"""
        from collections import defaultdict
        
        # 构建邻接表
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 初始化
        for node in flow.nodes:
            in_degree[node["id"]] = 0
        
        # 处理边
        for edge in flow.edges:
            adjacency[edge["source"]].append(edge["target"])
            in_degree[edge["target"]] += 1
        
        # 拓扑排序
        topo_order = []
        queue = [n for n in flow.nodes if in_degree[n["id"]] == 0]
        
        while queue:
            node = queue.pop(0)
            topo_order.append(node["id"])
            for neighbor in adjacency[node["id"]]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有环
        if len(topo_order) != len(flow.nodes):
            # 存在环，返回部分排序
            pass
        
        return {
            "format": "dag",
            "execution_order": topo_order,
            "adjacency": dict(adjacency),
            "in_degree": dict(in_degree),
            "is_valid_dag": len(topo_order) == len(flow.nodes)
        }


class Optimizer:
    """流程优化器"""
    
    @staticmethod
    def optimize(flow: AgentFlow) -> AgentFlow:
        """优化流程"""
        # 移除孤岛节点
        flow.nodes = Optimizer._remove_isolated_nodes(flow)
        
        # 合并相邻的同类节点（如果有意义）
        flow = Optimizer._merge_nodes(flow)
        
        return flow
    
    @staticmethod
    def _remove_isolated_nodes(flow: AgentFlow) -> List[Dict]:
        """移除孤立节点"""
        if len(flow.edges) == 0:
            return flow.nodes
        
        connected = set()
        for edge in flow.edges:
            connected.add(edge["source"])
            connected.add(edge["target"])
        
        return [n for n in flow.nodes if n["id"] in connected]
    
    @staticmethod
    def _merge_nodes(flow: AgentFlow) -> AgentFlow:
        """合并可合并的节点"""
        # TODO: 实现节点合并逻辑
        return flow


class Serializer:
    """流程序列化器"""
    
    @staticmethod
    def to_json(flow: AgentFlow, pretty: bool = True) -> str:
        """序列化为JSON"""
        import json
        from .builder import AgentFlow as FlowDataClass
        
        data = {
            "id": flow.id,
            "name": flow.name,
            "description": flow.description,
            "version": flow.version,
            "nodes": flow.nodes,
            "edges": flow.edges,
            "created_at": flow.created_at,
            "updated_at": flow.updated_at,
            "status": flow.status,
            "tags": flow.tags,
            "config": flow.config
        }
        
        indent = 2 if pretty else None
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str) -> AgentFlow:
        """从JSON反序列化"""
        import json
        data = json.loads(json_str)
        
        return AgentFlow(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            status=data.get("status", "draft"),
            tags=data.get("tags", []),
            config=data.get("config", {})
        )
    
    @staticmethod
    def to_yaml(flow: AgentFlow) -> str:
        """序列化为YAML"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML serialization")
        
        data = {
            "id": flow.id,
            "name": flow.name,
            "description": flow.description,
            "version": flow.version,
            "nodes": flow.nodes,
            "edges": flow.edges,
            "created_at": flow.created_at,
            "updated_at": flow.updated_at,
            "status": flow.status,
            "tags": flow.tags,
            "config": flow.config
        }
        
        return yaml.dump(data, allow_unicode=True, sort_keys=False)


# 导入datetime
from datetime import datetime
