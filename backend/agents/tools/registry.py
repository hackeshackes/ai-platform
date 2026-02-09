"""
工具注册表模块
管理Agent可用的工具
"""

from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表 - 管理所有可用工具"""
    
    def __init__(self):
        self._tools: Dict[str, Dict] = {}
    
    def register(
        self,
        tool: Dict[str, Any]
    ) -> bool:
        """
        注册一个工具
        
        Args:
            tool: 工具定义字典，包含:
                - name: 工具名称
                - func: 可调用函数
                - description: 工具描述
                - parameters: 参数定义
        """
        if not self._validate_tool(tool):
            logger.error(f"Invalid tool definition: {tool}")
            return False
        
        name = tool["name"]
        if name in self._tools:
            logger.warning(f"Tool '{name}' already exists, overwriting")
        
        self._tools[name] = {
            "name": name,
            "func": tool.get("func"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
            "category": tool.get("category", "general"),
            "registered_at": tool.get("registered_at")
        }
        
        logger.info(f"Tool '{name}' registered successfully")
        return True
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Dict[str, Any] = None
    ) -> bool:
        """便捷方法：注册工具"""
        return self.register({
            "name": name,
            "func": func,
            "description": description,
            "parameters": parameters or {}
        })
    
    def unregister(self, name: str) -> bool:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Tool '{name}' unregistered")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """获取工具定义"""
        return self._tools.get(name)
    
    def call_tool(self, name: str, **kwargs) -> Any:
        """
        调用工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        func = tool.get("func")
        if not func:
            raise ValueError(f"Tool '{name}' has no executable function")
        
        # 验证参数
        params_def = tool.get("parameters", {})
        if params_def.get("required"):
            for required_param in params_def["required"]:
                if required_param not in kwargs:
                    raise ValueError(f"Missing required parameter: {required_param}")
        
        # 执行函数
        try:
            result = func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise
    
    def list_tools(self) -> List[Dict]:
        """列出所有已注册工具"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"],
                "category": tool["category"]
            }
            for name, tool in self._tools.items()
        ]
    
    def list_tools_by_category(self, category: str) -> List[Dict]:
        """按类别列出工具"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in self._tools.items()
            if tool.get("category") == category
        ]
    
    def search_tools(self, query: str) -> List[Dict]:
        """搜索工具"""
        query_lower = query.lower()
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in self._tools.items()
            if query_lower in name.lower() or query_lower in tool["description"].lower()
        ]
    
    def get_categories(self) -> List[str]:
        """获取所有工具类别"""
        categories = set()
        for tool in self._tools.values():
            categories.add(tool.get("category", "general"))
        return list(categories)
    
    def count(self) -> int:
        """获取工具数量"""
        return len(self._tools)
    
    def clear(self):
        """清空所有工具"""
        self._tools.clear()
    
    def _validate_tool(self, tool: Dict) -> bool:
        """验证工具定义"""
        required_fields = ["name"]
        
        for field in required_fields:
            if field not in tool:
                logger.error(f"Missing required field: {field}")
                return False
        
        # 如果提供了函数，验证是否为可调用对象
        if "func" in tool and tool["func"] is not None:
            if not callable(tool["func"]):
                logger.error(f"Tool 'func' must be callable")
                return False
        
        return True
    
    def export_config(self) -> Dict:
        """导出工具配置（不包含函数）"""
        return {
            "tools": [
                {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "category": tool["category"]
                }
                for name, tool in self._tools.items()
            ],
            "total": len(self._tools)
        }
