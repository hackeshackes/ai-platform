"""
AI Platform v3 - Agent编排框架
"""

# Core模块
from .core.agent import Agent
from .core.reasoning import ReActReasoningEngine

# Tools模块
from .tools.registry import ToolRegistry
from .tools.builtin_tools import builtin_tools, BuiltinTools

# Memory模块
from .memory.memory import MemoryManager

# API模块
from .api.endpoints import get_agent_blueprint, create_collaboration_blueprint

__version__ = "1.0.0"
__all__ = [
    # Core
    'Agent',
    'ReActReasoningEngine',
    
    # Tools
    'ToolRegistry',
    'builtin_tools',
    'BuiltinTools',
    
    # Memory
    'MemoryManager',
    
    # API
    'get_agent_blueprint',
    'create_collaboration_blueprint'
]
