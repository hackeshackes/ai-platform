"""
任务分解器
将复杂任务分解为可执行的子任务
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .models import TaskInput, TaskDefinition, AgentRole

logger = logging.getLogger(__name__)


class DecompositionStrategy(str, Enum):
    """分解策略"""
    SEQUENTIAL = "sequential"      # 顺序分解
    PARALLEL = "parallel"           # 并行分解
    HIERARCHICAL = "hierarchical"   # 层级分解
    LINEAR = "linear"               # 线性分解


@dataclass
class SubTask:
    """子任务"""
    name: str
    description: str
    dependencies: List[str]
    estimated_complexity: float
    required_capabilities: List[str]
    order: int


class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self):
        self.built_patterns = []
        self._load_builtin_patterns()
    
    def _load_builtin_patterns(self) -> None:
        """加载内置分解模式"""
        self.built_patterns = [
            {
                "name": "research",
                "pattern": r"(research|investigate|analyze|study|explore)\s+(.+?)(?:\s+for|\s+and|\s+with|\.|$)",
                "decomposition": ["gather_information", "synthesize_findings", "validate_results"]
            },
            {
                "name": "generate",
                "pattern": r"(generate|create|produce|write|make)\s+(.+?)(?:\.|$)",
                "decomposition": ["plan_content", "draft_content", "review_and_refine"]
            },
            {
                "name": "compare",
                "pattern": r"(compare|contrast|evaluate|assess)\s+(.+?)(?:\s+with|\s+and|\.|$)",
                "decomposition": ["analyze_first", "analyze_second", "compare_results"]
            },
            {
                "name": "transform",
                "pattern": r"(convert|transform|translate|convert)\s+(.+?)(?:\s+to|\s+into|\.|$)",
                "decomposition": ["parse_input", "transform_data", "validate_output"]
            }
        ]
    
    async def decompose(
        self,
        task: TaskInput,
        strategy: DecompositionStrategy = DecompositionStrategy.HIERARCHICAL,
        available_agents: Optional[List[str]] = None
    ) -> List[TaskInput]:
        """
        分解任务
        
        Args:
            task: 输入任务
            strategy: 分解策略
            available_agents: 可用Agent列表
            
        Returns:
            子任务列表
        """
        logger.info(f"Decomposing task: {task.name}")
        
        # 分析任务类型
        task_type = self._analyze_task_type(task.name, task.description)
        
        # 根据策略进行分解
        if strategy == DecompositionStrategy.SEQUENTIAL:
            subtasks = await self._sequential_decompose(task)
        elif strategy == DecompositionStrategy.PARALLEL:
            subtasks = await self._parallel_decompose(task)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            subtasks = await self._hierarchical_decompose(task)
        else:
            subtasks = await self._linear_decompose(task)
        
        logger.info(f"Decomposed into {len(subtasks)} subtasks")
        return subtasks
    
    def _analyze_task_type(self, name: str, description: str) -> str:
        """分析任务类型"""
        text = f"{name} {description}".lower()
        
        for pattern_info in self.built_patterns:
            if re.search(pattern_info["pattern"], text):
                return pattern_info["name"]
        
        return "generic"
    
    async def _sequential_decompose(self, task: TaskInput) -> List[TaskInput]:
        """顺序分解"""
        subtasks = []
        base_order = 0
        
        # 基于任务类型选择分解步骤
        decomp_map = {
            "research": ["收集信息", "整理发现", "验证结果"],
            "generate": ["规划内容", "起草内容", "审核完善"],
            "compare": ["分析第一个对象", "分析第二个对象", "比较结果"],
            "transform": ["解析输入", "转换数据", "验证输出"],
            "generic": ["理解任务", "执行核心步骤", "总结结果"]
        }
        
        steps = decomp_map.get(self._analyze_task_type(task.name, task.description), decomp_map["generic"])
        
        for i, step_name in enumerate(steps):
            subtask = TaskInput(
                name=f"{task.name} - Step {i+1}: {step_name}",
                description=f"第{i+1}步: {step_name}",
                payload={"parent_task_id": task.task_id, "step": i+1, "total_steps": len(steps)},
                priority=task.priority + i,
                dependencies=[subtasks[-1].task_id] if i > 0 else [],
                metadata={"step_order": i}
            )
            subtasks.append(subtask)
            base_order += 1
        
        return subtasks
    
    async def _parallel_decompose(self, task: TaskInput) -> List[TaskInput]:
        """并行分解"""
        subtasks = []
        
        # 识别可以并行执行的部分
        task_type = self._analyze_task_type(task.name, task.description)
        
        if task_type == "research":
            aspects = ["收集背景信息", "获取最新数据", "分析相关案例"]
        elif task_type == "compare":
            aspects = ["分析第一个选项", "分析第二个选项"]
        else:
            aspects = ["选项A", "选项B", "选项C"]
        
        for i, aspect in enumerate(aspects):
            subtask = TaskInput(
                name=f"{task.name} - {aspect}",
                description=f"并行处理: {aspect}",
                payload={"parent_task_id": task.task_id, "aspect": aspect},
                priority=task.priority + i,
                dependencies=[],
                metadata={"parallel_group": task.task_id, "parallel_index": i}
            )
            subtasks.append(subtask)
        
        # 添加聚合任务
        aggregate_task = TaskInput(
            name=f"{task.name} - 汇总结果",
            description="聚合所有并行任务的结果",
            payload={"parent_task_id": task.task_id, "aggregation": True},
            priority=task.priority + len(aspects),
            dependencies=[s.task_id for s in subtasks],
            metadata={"is_aggregation": True}
        )
        subtasks.append(aggregate_task)
        
        return subtasks
    
    async def _hierarchical_decompose(self, task: TaskInput) -> List[TaskInput]:
        """层级分解"""
        subtasks = []
        
        # 主任务分解为阶段
        phases = [
            ("planning", "规划阶段", 0),
            ("execution", "执行阶段", 1),
            ("review", "审核阶段", 2)
        ]
        
        for phase_key, phase_name, phase_order in phases:
            # 每个阶段可能有子任务
            subtask = TaskInput(
                name=f"{task.name} - {phase_name}",
                description=f"执行{phase_name}",
                payload={
                    "parent_task_id": task.task_id,
                    "phase": phase_key
                },
                priority=task.priority + phase_order,
                dependencies=[subtasks[-1].task_id] if phase_order > 0 else [],
                metadata={"phase": phase_key, "phase_order": phase_order}
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _linear_decompose(self, task: TaskInput) -> List[TaskInput]:
        """线性分解"""
        # 简单的线性步骤分解
        return await self._sequential_decompose(task)
    
    def estimate_complexity(self, task: TaskInput) -> float:
        """估计任务复杂度"""
        score = 0.0
        
        # 基于名称长度
        name_length = len(task.name)
        score += min(name_length / 100, 0.3)
        
        # 基于描述详细程度
        desc_length = len(task.description)
        score += min(desc_length / 500, 0.3)
        
        # 基于依赖数量
        dep_count = len(task.dependencies)
        score += min(dep_count * 0.1, 0.2)
        
        # 基于优先级
        if task.priority > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def suggest_agent_role(self, task: TaskInput) -> AgentRole:
        """建议Agent角色"""
        complexity = self.estimate_complexity(task)
        
        if complexity > 0.7:
            return AgentRole.SUPERVISOR
        elif complexity > 0.4:
            return AgentRole.COORDINATOR
        else:
            return AgentRole.WORKER
    
    async def create_workflow_from_tasks(
        self,
        session_id: str,
        name: str,
        tasks: List[TaskInput],
        start_agent: str,
        mode: str = "sequential"
    ) -> List[TaskDefinition]:
        """从任务列表创建工作流定义"""
        definitions = []
        
        for i, task in enumerate(tasks):
            # 确定前置依赖
            dependencies = task.dependencies
            
            # 查找依赖任务的信息
            dep_names = []
            for dep_id in dependencies:
                for t in tasks:
                    if t.task_id == dep_id:
                        dep_names.append(t.name)
                        break
            
            # 创建任务定义
            task_def = TaskDefinition(
                task_id=task.task_id,
                name=task.name,
                description=task.description,
                source=start_agent if i == 0 else "",
                target=task.assigned_agent or "next_agent",
                condition=",".join(dep_names) if dep_names else "",
                payload_template=task.payload,
                timeout_ms=300000,
                retry_count=3,
                metadata={
                    **task.metadata,
                    "workflow_position": i,
                    "mode": mode
                }
            )
            definitions.append(task_def)
        
        return definitions


class SmartTaskAnalyzer:
    """智能任务分析器"""
    
    def __init__(self):
        self.keywords = {
            "research": ["研究", "调查", "分析", "探索", "研究"],
            "create": ["创建", "生成", "编写", "制作", "设计"],
            "compare": ["比较", "对比", "评估", "权衡"],
            "optimize": ["优化", "改进", "提升", "改善"],
            "debug": ["调试", "排查", "修复", "解决"],
            "plan": ["规划", "计划", "安排", "制定"]
        }
    
    def analyze(self, task_description: str) -> Dict[str, Any]:
        """分析任务"""
        result = {
            "type": "generic",
            "complexity": "medium",
            "estimated_duration": "unknown",
            "required_capabilities": [],
            "suggested_approach": "standard"
        }
        
        desc_lower = task_description.lower()
        
        # 检测任务类型
        for task_type, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    result["type"] = task_type
                    break
        
        # 估计复杂度
        word_count = len(task_description.split())
        if word_count < 10:
            result["complexity"] = "low"
        elif word_count < 50:
            result["complexity"] = "medium"
        else:
            result["complexity"] = "high"
        
        # 估计持续时间
        if result["complexity"] == "low":
            result["estimated_duration"] = "short"
        elif result["complexity"] == "medium":
            result["estimated_duration"] = "medium"
        else:
            result["estimated_duration"] = "long"
        
        return result


def create_task_decomposer() -> TaskDecomposer:
    """创建任务分解器"""
    return TaskDecomposer()


def create_smart_analyzer() -> SmartTaskAnalyzer:
    """创建智能分析器"""
    return SmartTaskAnalyzer()
