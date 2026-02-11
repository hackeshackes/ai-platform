"""
验证器模块 - Validator

负责验证Pipeline和Agent的正确性
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from .pipeline_generator import Pipeline, Node, Connection
from .agent_generator import Agent, Skill, MemoryConfig
from .nl_understand import UnderstandingResult, IntentType


class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class Validator:
    """验证器"""
    
    def __init__(self):
        self.rules = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        # Pipeline验证规则
        self.rules.extend([
            ("pipeline_name_required", self._check_pipeline_name_required),
            ("pipeline_nodes_min", self._check_pipeline_nodes_min),
            ("pipeline_nodes_connected", self._check_pipeline_nodes_connected),
            ("pipeline_input_output", self._check_pipeline_input_output),
            ("pipeline_connection_valid", self._check_pipeline_connection_valid),
        ])
        
        # Agent验证规则
        self.rules.extend([
            ("agent_name_required", self._check_agent_name_required),
            ("agent_skills_min", self._check_agent_skills_min),
            ("agent_skill_unique", self._check_agent_skill_unique),
            ("agent_memory_valid", self._check_agent_memory_valid),
        ])
        
        # 代码验证规则
        self.rules.extend([
            ("code_syntax_valid", self._check_code_syntax_valid),
            ("code_imports_valid", self._check_code_imports_valid),
        ])
    
    def validate_pipeline(self, pipeline: Pipeline) -> List[ValidationResult]:
        """
        验证Pipeline
        
        Args:
            pipeline: Pipeline实例
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []
        
        for rule_name, rule_func in self.rules:
            if rule_name.startswith("pipeline_"):
                result = rule_func(pipeline)
                if result:
                    results.append(result)
        
        # 运行额外的Pipeline验证
        results.extend(self._validate_pipeline_structure(pipeline))
        results.extend(self._validate_pipeline_config(pipeline))
        
        return results
    
    def validate_agent(self, agent: Agent) -> List[ValidationResult]:
        """
        验证Agent
        
        Args:
            agent: Agent实例
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []
        
        for rule_name, rule_func in self.rules:
            if rule_name.startswith("agent_"):
                result = rule_func(agent)
                if result:
                    results.append(result)
        
        # 运行额外的Agent验证
        results.extend(self._validate_agent_skills(agent))
        results.extend(self._validate_agent_personality(agent))
        
        return results
    
    def validate_code(self, code: str) -> List[ValidationResult]:
        """
        验证代码
        
        Args:
            code: 代码字符串
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []
        
        # 语法检查
        syntax_result = self._check_code_syntax_valid(code)
        if syntax_result:
            results.append(syntax_result)
        
        # 导入检查
        import_result = self._check_code_imports_valid(code)
        if import_result:
            results.append(import_result)
        
        # 编码规范检查
        results.extend(self._validate_code_style(code))
        
        return results
    
    def validate_understanding(self, understanding: UnderstandingResult) -> List[ValidationResult]:
        """
        验证理解结果
        
        Args:
            understanding: 理解结果
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []
        
        # 检查意图
        if understanding.intent == IntentType.UNKNOWN:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="无法识别的意图",
                suggestions=["请使用更明确的指令，如'创建一个pipeline'"]
            ))
        
        # 检查置信度
        if understanding.confidence < 0.5:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"理解置信度较低: {understanding.confidence:.2%}",
                suggestions=["请提供更详细的信息"]
            ))
        
        # 检查槽位
        unfilled_required = [s for s in understanding.slots.values() 
                           if s.required and not s.filled]
        if unfilled_required:
            slot_names = ", ".join([s.name for s in unfilled_required])
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"缺少必要槽位: {slot_names}",
                suggestions=[f"请提供{slot_names}信息"]
            ))
        
        return results
    
    # Pipeline验证规则
    def _check_pipeline_name_required(self, pipeline: Pipeline) -> Optional[ValidationResult]:
        """检查Pipeline名称是否必填"""
        if not pipeline.name or pipeline.name == "unnamed_pipeline":
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Pipeline缺少名称",
                suggestions=["请指定Pipeline名称"]
            )
        return None
    
    def _check_pipeline_nodes_min(self, pipeline: Pipeline) -> Optional[ValidationResult]:
        """检查Pipeline节点数量"""
        if len(pipeline.nodes) < 2:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Pipeline节点数量不足: {len(pipeline.nodes)}",
                suggestions=["Pipeline至少需要2个节点"]
            )
        return None
    
    def _check_pipeline_nodes_connected(self, pipeline: Pipeline) -> Optional[ValidationResult]:
        """检查节点是否连接"""
        if not pipeline.connections:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="Pipeline没有定义连接",
                suggestions=["请添加节点之间的连接"]
            )
        return None
    
    def _check_pipeline_input_output(self, pipeline: Pipeline) -> Optional[ValidationResult]:
        """检查输入输出节点"""
        has_input = any(n.type.value == "input" for n in pipeline.nodes)
        has_output = any(n.type.value == "output" for n in pipeline.nodes)
        
        if not has_input or not has_output:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="Pipeline缺少输入或输出节点",
                suggestions=["建议添加input和output节点以完善流程"]
            )
        return None
    
    def _check_pipeline_connection_valid(self, pipeline: Pipeline) -> Optional[ValidationResult]:
        """检查连接有效性"""
        node_ids = {n.id for n in pipeline.nodes}
        errors = []
        
        for conn in pipeline.connections:
            if conn.source not in node_ids:
                errors.append(f"连接源节点不存在: {conn.source}")
            if conn.target not in node_ids:
                errors.append(f"连接目标节点不存在: {conn.target}")
        
        if errors:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Pipeline连接无效",
                details={"errors": errors}
            )
        return None
    
    def _validate_pipeline_structure(self, pipeline: Pipeline) -> List[ValidationResult]:
        """验证Pipeline结构"""
        results = []
        
        # 检查循环依赖
        if self._has_cycle(pipeline):
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Pipeline存在循环依赖",
                suggestions=["请移除循环依赖以确保Pipeline可以执行"]
            ))
        
        # 检查孤立节点
        connected_nodes = set()
        for conn in pipeline.connections:
            connected_nodes.add(conn.source)
            connected_nodes.add(conn.target)
        
        isolated_nodes = [n for n in pipeline.nodes if n.id not in connected_nodes]
        if isolated_nodes:
            results.append(ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"存在{len(isolated_nodes)}个孤立节点",
                details={"nodes": [n.name for n in isolated_nodes]},
                suggestions=["请连接孤立节点或将其移除"]
            ))
        
        return results
    
    def _validate_pipeline_config(self, pipeline: Pipeline) -> List[ValidationResult]:
        """验证Pipeline配置"""
        results = []
        
        # 检查配置参数
        for node in pipeline.nodes:
            if node.type.value == "llm":
                if "model" not in node.config:
                    results.append(ValidationResult(
                        valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"LLM节点'{node.name}'缺少model配置",
                        suggestions=["请为LLM节点指定model参数"]
                    ))
        
        return results
    
    # Agent验证规则
    def _check_agent_name_required(self, agent: Agent) -> Optional[ValidationResult]:
        """检查Agent名称是否必填"""
        if not agent.name or agent.name == "unnamed_agent":
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Agent缺少名称",
                suggestions=["请指定Agent名称"]
            )
        return None
    
    def _check_agent_skills_min(self, agent: Agent) -> Optional[ValidationResult]:
        """检查Agent技能数量"""
        if len(agent.skills) < 1:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message="Agent没有绑定任何技能",
                suggestions=["建议为Agent添加至少一个技能"]
            )
        return None
    
    def _check_agent_skill_unique(self, agent: Agent) -> Optional[ValidationResult]:
        """检查Agent技能唯一性"""
        skill_ids = [s.id for s in agent.skills]
        if len(skill_ids) != len(set(skill_ids)):
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message="Agent存在重复的技能ID"
            )
        return None
    
    def _check_agent_memory_valid(self, agent: Agent) -> Optional[ValidationResult]:
        """检查Agent记忆配置"""
        if agent.memory.type.value == "none":
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message="Agent未配置记忆功能",
                suggestions=["如需上下文理解，可考虑添加记忆功能"]
            )
        return None
    
    def _validate_agent_skills(self, agent: Agent) -> List[ValidationResult]:
        """验证Agent技能"""
        results = []
        
        for skill in agent.skills:
            # 检查技能配置
            if skill.type.value == "llm" and "model" not in skill.config:
                results.append(ValidationResult(
                    valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"技能'{skill.name}'可能需要model配置"
                ))
        
        return results
    
    def _validate_agent_personality(self, agent: Agent) -> List[ValidationResult]:
        """验证Agent人格配置"""
        results = []
        
        if not agent.personality.traits:
            results.append(ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message="Agent未配置人格特质，将使用默认设置"
            ))
        
        return results
    
    # 代码验证规则
    def _check_code_syntax_valid(self, code: str) -> Optional[ValidationResult]:
        """检查代码语法"""
        try:
            compile(code, "<string>", "exec")
            return None
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"代码语法错误: {str(e)}",
                details={"line": e.lineno, "offset": e.offset}
            )
    
    def _check_code_imports_valid(self, code: str) -> Optional[ValidationResult]:
        """检查导入语句"""
        # 提取import语句
        import_pattern = r"^(import\s+\w+|from\s+\w+\s+import)"
        lines = code.split('\n')
        
        invalid_imports = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                try:
                    # 简单检查导入模块是否存在
                    if line.startswith("from ") and " import " in line:
                        module = line.split(" import ")[0].replace("from ", "").strip()
                        # 检查模块名格式
                        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', module):
                            invalid_imports.append((i, line))
                    elif line.startswith("import "):
                        modules = line.replace("import ", "").split(",")
                        for mod in modules:
                            mod = mod.strip()
                            if mod and not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', mod):
                                invalid_imports.append((i, line))
                                break
                except Exception:
                    pass
        
        if invalid_imports:
            return ValidationResult(
                valid=False,
                level=ValidationLevel.WARNING,
                message=f"存在无效的导入语句",
                details={"invalid_imports": invalid_imports}
            )
        return None
    
    def _validate_code_style(self, code: str) -> List[ValidationResult]:
        """验证代码风格"""
        results = []
        
        lines = code.split('\n')
        
        # 检查行长度
        long_lines = [(i+1, len(line)) for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            results.append(ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"存在{len(long_lines)}行超过120字符",
                details={"lines": long_lines}
            ))
        
        # 检查函数文档
        function_pattern = r"^\s*def\s+(\w+)"
        docstring_pattern = r'"""[\s\S]*?"""'
        
        for i, line in enumerate(lines):
            match = re.match(function_pattern, line)
            if match:
                func_name = match.group(1)
                # 检查下一个非空行是否有文档字符串
                has_docstring = False
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        has_docstring = bool(re.search(docstring_pattern, lines[j]))
                        break
                
                if not has_docstring:
                    results.append(ValidationResult(
                        valid=True,
                        level=ValidationLevel.INFO,
                        message=f"函数'{func_name}'缺少文档字符串"
                    ))
        
        return results
    
    # 辅助方法
    def _has_cycle(self, pipeline: Pipeline) -> bool:
        """检查是否存在循环依赖"""
        from collections import defaultdict, deque
        
        # 构建图
        graph = defaultdict(list)
        for conn in pipeline.connections:
            graph[conn.source].append(conn.target)
        
        # 拓扑排序检测环
        in_degree = defaultdict(int)
        for node in pipeline.nodes:
            for neighbor in graph[node.id]:
                in_degree[neighbor] += 1
        
        queue = deque([n.id for n in pipeline.nodes if in_degree[n.id] == 0])
        count = 0
        
        while queue:
            node_id = queue.popleft()
            count += 1
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return count != len(pipeline.nodes)
    
    def run_all_checks(self, understanding: UnderstandingResult,
                       pipeline: Pipeline = None,
                       agent: Agent = None,
                       code: str = None) -> Dict[str, List[ValidationResult]]:
        """
        运行所有检查
        
        Args:
            understanding: 理解结果
            pipeline: Pipeline实例
            agent: Agent实例
            code: 代码字符串
            
        Returns:
            Dict: 各部分验证结果
        """
        results = {
            "understanding": [],
            "pipeline": [],
            "agent": [],
            "code": []
        }
        
        # 验证理解结果
        if understanding:
            results["understanding"] = self.validate_understanding(understanding)
        
        # 验证Pipeline
        if pipeline:
            results["pipeline"] = self.validate_pipeline(pipeline)
        
        # 验证Agent
        if agent:
            results["agent"] = self.validate_agent(agent)
        
        # 验证代码
        if code:
            results["code"] = self.validate_code(code)
        
        return results
    
    def get_summary(self, results: Dict[str, List[ValidationResult]]) -> Dict[str, Any]:
        """获取验证结果摘要"""
        summary = {
            "total_errors": 0,
            "total_warnings": 0,
            "total_infos": 0,
            "by_category": {}
        }
        
        for category, category_results in results.items():
            errors = sum(1 for r in category_results if r.level == ValidationLevel.ERROR)
            warnings = sum(1 for r in category_results if r.level == ValidationLevel.WARNING)
            infos = sum(1 for r in category_results if r.level == ValidationLevel.INFO)
            
            summary["by_category"][category] = {
                "errors": errors,
                "warnings": warnings,
                "infos": infos,
                "valid": errors == 0
            }
            
            summary["total_errors"] += errors
            summary["total_warnings"] += warnings
            summary["total_infos"] += infos
        
        return summary
