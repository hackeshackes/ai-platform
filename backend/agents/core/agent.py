"""
Agent核心模块
实现基础的ReAct Agent
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging

from .reasoning import ReActReasoningEngine
from ..tools.registry import ToolRegistry
from ..memory.memory import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    """ReAct Agent核心类"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        max_steps: int = 10,
        llm_provider: Optional[Callable] = None,
        initial_memory: Optional[Dict] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.created_at = datetime.now()
        
        # 初始化组件
        self.tool_registry = ToolRegistry()
        self.memory_manager = MemoryManager(initial_memory)
        self.reasoning_engine = ReActReasoningEngine(
            llm_provider=llm_provider,
            max_steps=max_steps
        )
        
        # 工具调用历史
        self.execution_history: List[Dict] = []
        
        # 注册内置工具
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        from ..tools.builtin_tools import get_all_tools
        for tool in get_all_tools():
            self.tool_registry.register(tool)
    
    def register_tool(self, name: str, func: Callable, description: str = "", 
                      parameters: Dict[str, Any] = None):
        """注册自定义工具"""
        self.tool_registry.register_tool(name, func, description, parameters)
    
    def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """
        执行任务
        
        Args:
            task: 用户任务描述
            context: 额外上下文信息
            
        Returns:
            执行结果
        """
        logger.info(f"Agent '{self.name}' executing task: {task}")
        
        # 初始化记忆
        self.memory_manager.add_memory({
            "type": "task_start",
            "content": f"开始执行任务: {task}",
            "timestamp": datetime.now().isoformat()
        })
        
        # 添加上下文
        if context:
            for key, value in context.items():
                self.memory_manager.add_memory({
                    "type": "context",
                    "key": key,
                    "value": str(value),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 执行ReAct推理
        result = self.reasoning_engine.execute(
            task=task,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager
        )
        
        # 记录执行历史
        self.execution_history.append({
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        # 添加完成记忆
        self.memory_manager.add_memory({
            "type": "task_complete",
            "content": f"任务完成: {task}",
            "result_summary": result.get("summary", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_memory(self, memory_type: Optional[str] = None) -> List[Dict]:
        """获取记忆"""
        return self.memory_manager.get_memories(memory_type)
    
    def get_tools(self) -> List[Dict]:
        """获取已注册工具列表"""
        return self.tool_registry.list_tools()
    
    def clear_memory(self, memory_type: Optional[str] = None):
        """清空记忆"""
        self.memory_manager.clear(memory_type)
    
    def get_state(self) -> Dict:
        """获取Agent当前状态"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "memory_count": len(self.memory_manager.get_memories()),
            "tool_count": len(self.tool_registry.list_tools()),
            "execution_count": len(self.execution_history)
        }
    
    def to_dict(self) -> Dict:
        """序列化Agent配置"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "max_steps": self.max_steps,
            "created_at": self.created_at.isoformat(),
            "tools": [t["name"] for t in self.tool_registry.list_tools()]
        }
