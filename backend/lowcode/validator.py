"""
流程验证器 - 验证流程定义的合法性
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from .nodes import NodeRegistry, NodeType, NodeCategory


class ValidationSeverity(str, Enum):
    """验证严重程度"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """验证问题"""
    severity: ValidationSeverity
    code: str
    message: str
    path: str = ""  # 问题所在位置，如节点ID、边ID等
    suggestion: str = ""  # 修复建议
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.valid = False
    
    def add_issues(self, issues: List[ValidationIssue]):
        for issue in issues:
            self.add_issue(issue)
    
    def summary(self) -> Dict[str, Any]:
        """返回验证摘要"""
        counts = defaultdict(int)
        for issue in self.issues:
            counts[issue.severity.value] += 1
        
        return {
            "valid": self.valid,
            "total": len(self.issues),
            "errors": counts[ValidationSeverity.ERROR.value],
            "warnings": counts[ValidationSeverity.WARNING.value],
            "info": counts[ValidationSeverity.INFO.value],
            "issues": [i.to_dict() for i in self.issues]
        }


class FlowValidator:
    """流程验证器"""
    
    def __init__(self, registry: NodeRegistry = None):
        self.registry = registry or NodeRegistry()
        self._rules: List[callable] = [
            self._check_required_fields,
            self._check_node_existence,
            self._check_edge_validity,
            self._check_port_compatibility,
            self._check_cycles,
            self._check_start_end_nodes,
            self._check_node_config,
            self._check_orphaned_nodes,
        ]
    
    def validate(self, flow: Dict[str, Any]) -> ValidationResult:
        """验证流程"""
        result = ValidationResult(valid=True)
        
        for rule in self._rules:
            try:
                rule(flow, result)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="VALIDATION_ERROR",
                    message=f"验证时发生错误: {str(e)}",
                    suggestion="请检查流程定义格式是否正确"
                ))
        
        return result
    
    def _check_required_fields(self, flow: Dict, result: ValidationResult):
        """检查必需字段"""
        required_fields = ["nodes"]
        
        for field_name in required_fields:
            if field_name not in flow:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_REQUIRED_FIELD",
                    message=f"缺少必需字段: {field_name}",
                    path=f"/{field_name}",
                    suggestion=f"请添加 {field_name} 字段"
                ))
        
        # 检查nodes是否为列表
        if "nodes" in flow and not isinstance(flow["nodes"], list):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_NODES_TYPE",
                message="nodes必须是数组类型",
                path="/nodes",
                suggestion="请将nodes设置为数组格式"
            ))
    
    def _check_node_existence(self, flow: Dict, result: ValidationResult):
        """检查节点是否存在且有效"""
        nodes = flow.get("nodes", [])
        
        for i, node in enumerate(nodes):
            node_id = node.get("id", f"index_{i}")
            path = f"/nodes[{i}]"
            
            # 检查节点类型
            node_type = node.get("type")
            if not node_type:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_NODE_TYPE",
                    message="节点缺少type字段",
                    path=f"{path}/type",
                    suggestion="请为节点指定type"
                ))
            elif not self.registry.get(node_type):
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="UNKNOWN_NODE_TYPE",
                    message=f"未知节点类型: {node_type}",
                    path=f"{path}/type",
                    suggestion="请使用已注册的节点类型"
                ))
            
            # 检查节点ID
            if "id" not in node:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_NODE_ID",
                    message="节点缺少id字段，将自动生成",
                    path=path,
                    suggestion="建议显式指定节点ID"
                ))
    
    def _check_edge_validity(self, flow: Dict, result: ValidationResult):
        """检查边的有效性"""
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        # 构建节点ID集合
        node_ids = {node.get("id") for node in nodes if node.get("id")}
        
        for i, edge in enumerate(edges):
            path = f"/edges[{i}]"
            
            # 检查源节点
            source = edge.get("source")
            if not source:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_EDGE_SOURCE",
                    message="边缺少source字段",
                    path=f"{path}/source",
                    suggestion="请指定边的源节点ID"
                ))
            elif source not in node_ids:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_EDGE_SOURCE",
                    message=f"边引用了不存在的源节点: {source}",
                    path=f"{path}/source",
                    suggestion="请检查源节点ID是否正确"
                ))
            
            # 检查目标节点
            target = edge.get("target")
            if not target:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_EDGE_TARGET",
                    message="边缺少target字段",
                    path=f"{path}/target",
                    suggestion="请指定边的目标节点ID"
                ))
            elif target not in node_ids:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_EDGE_TARGET",
                    message=f"边引用了不存在的目标节点: {target}",
                    path=f"{path}/target",
                    suggestion="请检查目标节点ID是否正确"
                ))
            
            # 检查是否有自环
            if source == target:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="SELF_LOOP_EDGE",
                    message="检测到自环边，可能导致无限循环",
                    path=path,
                    suggestion="请检查是否需要自环"
                ))
    
    def _check_port_compatibility(self, flow: Dict, result: ValidationResult):
        """检查端口兼容性"""
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        # 构建节点端口映射
        node_ports = {}
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            
            node_info = self.registry.get(node.get("type", ""))
            if node_info:
                # 获取节点定义的端口
                input_ports = {p.id: p for p in node_info.get("input_ports", [])}
                output_ports = {p.id: p for p in node_info.get("output_ports", [])}
                node_ports[node_id] = {
                    "inputs": input_ports,
                    "outputs": output_ports,
                    "type": node.get("type")
                }
        
        # 检查每条边
        for i, edge in enumerate(edges):
            source = edge.get("source")
            target = edge.get("target")
            source_port = edge.get("source_port", "")
            target_port = edge.get("target_port", "")
            
            # 检查源端口
            if source in node_ports:
                source_info = node_ports[source]
                if source_info["outputs"]:
                    if source_port and source_port not in source_info["outputs"]:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="INVALID_SOURCE_PORT",
                            message=f"节点 {source_info['type']} 没有名为 {source_port} 的输出端口",
                            path=f"/edges[{i}]/source_port",
                            suggestion=f"可用的输出端口: {list(source_info['outputs'].keys())}"
                        ))
            
            # 检查目标端口
            if target in node_ports:
                target_info = node_ports[target]
                if target_info["inputs"]:
                    if target_port and target_port not in target_info["inputs"]:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="INVALID_TARGET_PORT",
                            message=f"节点 {target_info['type']} 没有名为 {target_port} 的输入端口",
                            path=f"/edges[{i}]/target_port",
                            suggestion=f"可用的输入端口: {list(target_info['inputs'].keys())}"
                        ))
    
    def _check_cycles(self, flow: Dict, result: ValidationResult):
        """检查循环"""
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        if not nodes:
            return
        
        # 构建邻接表
        adjacency = defaultdict(list)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)
        
        # DFS检测循环
        visited = set()
        recursion_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False
        
        # 检查每个节点
        for node in nodes:
            node_id = node.get("id")
            if node_id and node_id not in visited:
                if has_cycle(node_id):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="CYCLE_DETECTED",
                        message="流程中存在循环引用，可能导致无限执行",
                        suggestion="请检查并移除循环"
                    ))
                    break
    
    def _check_start_end_nodes(self, flow: Dict, result: ValidationResult):
        """检查起始和结束节点"""
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        if not nodes:
            return
        
        # 收集所有引用到的节点
        referenced_nodes = set()
        for edge in edges:
            if edge.get("source"):
                referenced_nodes.add(edge["source"])
            if edge.get("target"):
                referenced_nodes.add(edge["target"])
        
        # 检查起始节点
        has_start = any(
            node.get("type") == NodeType.START.value 
            for node in nodes 
            if node.get("id") in referenced_nodes or not edges
        )
        
        if not has_start and edges:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MISSING_START_NODE",
                message="流程没有明确的起始节点",
                suggestion="建议添加START节点作为流程入口"
            ))
        
        # 检查结束节点
        has_end = any(
            node.get("type") in [NodeType.END.value, NodeType.ERROR.value]
            for node in nodes
        )
        
        if not has_end and edges:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MISSING_END_NODE",
                message="流程没有明确的结束节点",
                suggestion="建议添加END或ERROR节点作为流程出口"
            ))
    
    def _check_node_config(self, flow: Dict, result: ValidationResult):
        """检查节点配置"""
        nodes = flow.get("nodes", [])
        
        for i, node in enumerate(nodes):
            node_type = node.get("type")
            config = node.get("config", {})
            node_id = node.get("id", f"index_{i}")
            
            node_info = self.registry.get(node_type)
            if not node_info:
                continue
            
            # 检查必需的配置项
            for cfg in node_info.get("config", []):
                if cfg.required and cfg.key not in config:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_REQUIRED_CONFIG",
                        message=f"节点缺少必需的配置项: {cfg.name}",
                        path=f"/nodes[{i}]/config/{cfg.key}",
                        suggestion=f"请添加 {cfg.key} 配置项"
                    ))
    
    def _check_orphaned_nodes(self, flow: Dict, result: ValidationResult):
        """检查孤立节点"""
        nodes = flow.get("nodes", [])
        edges = flow.get("edges", [])
        
        if not edges or not nodes:
            return
        
        # 收集所有连接的节点
        connected = set()
        for edge in edges:
            if edge.get("source"):
                connected.add(edge["source"])
            if edge.get("target"):
                connected.add(edge["target"])
        
        # 检查孤立节点
        for i, node in enumerate(nodes):
            node_id = node.get("id")
            if node_id and node_id not in connected:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="ORPHANED_NODE",
                    message=f"节点 '{node.get('name', node_id)}' 没有连接到任何其他节点",
                    path=f"/nodes[{i}]",
                    suggestion="考虑删除该节点或添加连接"
                ))


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_node_config(node_type: str, config: Dict,
                              registry: NodeRegistry) -> ValidationResult:
        """验证单个节点配置"""
        result = ValidationResult(valid=True)
        
        node_info = registry.get(node_type)
        if not node_info:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="UNKNOWN_NODE_TYPE",
                message=f"未知节点类型: {node_type}"
            ))
            return result
        
        # 验证每个配置项
        for cfg in node_info.get("config", []):
            if cfg.required and cfg.key not in config:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_REQUIRED_CONFIG",
                    message=f"缺少必需配置: {cfg.name}"
                ))
                continue
            
            if cfg.key in config:
                value = config[cfg.key]
                
                # 类型检查
                if cfg.type == "number" and not isinstance(value, (int, float)):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_CONFIG_TYPE",
                        message=f"{cfg.name} 必须是数字类型"
                    ))
                elif cfg.type == "boolean" and not isinstance(value, bool):
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_CONFIG_TYPE",
                        message=f"{cfg.name} 必须是布尔类型"
                    ))
                elif cfg.type == "json":
                    if isinstance(value, str):
                        try:
                            import json
                            json.loads(value)
                        except json.JSONDecodeError:
                            result.add_issue(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                code="INVALID_JSON_CONFIG",
                                message=f"{cfg.name} 不是有效的JSON格式"
                            ))
        
        return result
