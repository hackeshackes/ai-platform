"""
Agent节点库 - 定义低代码构建器中可用的节点类型
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import uuid
import json


class NodeCategory(str, Enum):
    """节点分类"""
    INPUT = "input"          # 输入节点
    OUTPUT = "output"        # 输出节点
    PROCESS = "process"      # 处理节点
    LOGIC = "logic"          # 逻辑节点
    AI = "ai"                # AI能力节点
    TOOL = "tool"            # 工具节点
    DATA = "data"            # 数据处理节点
    FLOW = "flow"            # 流程控制节点


class NodeType(str, Enum):
    """具体节点类型"""
    # 输入节点
    TEXT_INPUT = "text_input"
    FILE_INPUT = "file_input"
    API_INPUT = "api_input"
    
    # 输出节点
    TEXT_OUTPUT = "text_output"
    JSON_OUTPUT = "json_output"
    FILE_OUTPUT = "file_output"
    
    # AI节点
    LLM_COMPLETION = "llm_completion"
    LLM_CHAT = "llm_chat"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SENTIMENT = "sentiment"
    NER = "ner"
    
    # 工具节点
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    CODE_EXEC = "code_exec"
    HTTP_REQUEST = "http_request"
    
    # 逻辑节点
    IF = "if"
    SWITCH = "switch"
    LOOP = "loop"
    PARALLEL = "parallel"
    
    # 数据节点
    FILTER = "filter"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    SORT = "sort"
    
    # 流程控制
    START = "start"
    END = "end"
    ERROR = "error"


@dataclass
class NodePort:
    """节点端口定义"""
    id: str
    name: str
    type: str = "any"  # "any", "string", "number", "array", "object"
    required: bool = False
    description: str = ""


@dataclass
class NodeConfig:
    """节点配置项"""
    key: str
    name: str
    type: str  # "string", "number", "boolean", "select", "json"
    default: Any = None
    options: List[Dict] = field(default_factory=list)  # for select type
    required: bool = False
    description: str = ""


@dataclass
class BaseNode(ABC):
    """基础节点类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    name: str = ""
    category: NodeCategory = NodeCategory.PROCESS
    description: str = ""
    
    # 端口定义
    input_ports: List[NodePort] = field(default_factory=list)
    output_ports: List[NodePort] = field(default_factory=list)
    
    # 配置项
    config: List[NodeConfig] = field(default_factory=list)
    
    # 运行时数据
    position: Dict[str, float] = field(default_factory=dict)  # x, y
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """验证配置"""
        errors = []
        for cfg in self.config:
            if cfg.required and cfg.key not in config:
                errors.append(f"配置项 {cfg.name} 是必需的")
            if cfg.key in config:
                value = config[cfg.key]
                if cfg.type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"{cfg.name} 必须是数字类型")
                elif cfg.type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{cfg.name} 必须是布尔类型")
        return errors


class InputNode(BaseNode):
    """输入节点基类"""
    pass


class OutputNode(BaseNode):
    """输出节点基类"""
    pass


class LLMNode(BaseNode):
    """LLM节点基类"""
    pass


class ToolNode(BaseNode):
    """工具节点基类"""
    pass


class LogicNode(BaseNode):
    """逻辑节点基类"""
    pass


# ============ 具体节点实现 ============

class TextInputNode(InputNode):
    """文本输入节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.TEXT_INPUT.value,
            name="文本输入",
            category=NodeCategory.INPUT,
            description="提供文本输入",
            input_ports=[],
            output_ports=[
                NodePort("output", "文本", "string")
            ],
            config=[
                NodeConfig("placeholder", "占位符", "string", "请输入文本..."),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        text = context.get("input", "")
        return {"output": text}


class APILnputNode(InputNode):
    """API输入节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.API_INPUT.value,
            name="API输入",
            category=NodeCategory.INPUT,
            description="从API获取数据",
            output_ports=[
                NodePort("output", "数据", "any")
            ],
            config=[
                NodeConfig("url", "API地址", "string", ""),
                NodeConfig("method", "请求方法", "select", "GET", 
                          options=[{"value": "GET"}, {"value": "POST"}, {"value": "PUT"}]),
                NodeConfig("headers", "请求头", "json", {}),
                NodeConfig("timeout", "超时时间(秒)", "number", 30),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        config = context.get("config", {})
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=config.get("method", "GET"),
                url=config.get("url", ""),
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30)
            )
            response.raise_for_status()
            return {"output": response.json()}


class LLMCompletionNode(LLMNode):
    """LLM文本补全节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.LLM_COMPLETION.value,
            name="LLM补全",
            category=NodeCategory.AI,
            description="使用大语言模型生成文本",
            input_ports=[
                NodePort("prompt", "提示词", "string", True)
            ],
            output_ports=[
                NodePort("output", "生成结果", "string"),
                NodePort("usage", "token使用量", "object")
            ],
            config=[
                NodeConfig("model", "模型", "select", "gpt-4",
                          options=[
                              {"value": "gpt-4", "label": "GPT-4"},
                              {"value": "gpt-4o", "label": "GPT-4o"},
                              {"value": "claude-3-opus", "label": "Claude 3 Opus"},
                              {"value": "claude-3-sonnet", "label": "Claude 3 Sonnet"},
                          ]),
                NodeConfig("temperature", "温度", "number", 0.7),
                NodeConfig("max_tokens", "最大token数", "number", 1000),
                NodeConfig("system_prompt", "系统提示词", "string", ""),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # 实际调用LLM服务
        prompt = context.get("prompt", "")
        config = context.get("config", {})
        
        # TODO: 集成实际的LLM服务
        result = {
            "output": f"Generated response for: {prompt[:100]}...",
            "usage": {
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 100,
                "total_tokens": len(prompt) // 4 + 100
            }
        }
        return result


class WebSearchNode(ToolNode):
    """网络搜索节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.WEB_SEARCH.value,
            name="网络搜索",
            category=NodeCategory.TOOL,
            description="执行网络搜索",
            input_ports=[
                NodePort("query", "搜索查询", "string", True)
            ],
            output_ports=[
                NodePort("results", "搜索结果", "array")
            ],
            config=[
                NodeConfig("engine", "搜索引擎", "select", "brave",
                          options=[
                              {"value": "brave", "label": "Brave"},
                              {"value": "google", "label": "Google"},
                              {"value": "bing", "label": "Bing"},
                          ]),
                NodeConfig("max_results", "最大结果数", "number", 5),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        # TODO: 集成实际的搜索服务
        return {"results": [{"title": f"Result for {query}", "url": "https://example.com"}]}


class IfNode(LogicNode):
    """条件判断节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.IF.value,
            name="条件判断",
            category=NodeCategory.LOGIC,
            description="根据条件分支执行",
            input_ports=[
                NodePort("condition", "条件", "any", True),
                NodePort("if_true", "条件为真时", "any", False),
            ],
            output_ports=[
                NodePort("true_output", "真分支输出", "any"),
                NodePort("false_output", "假分支输出", "any")
            ],
            config=[
                NodeConfig("operator", "比较操作符", "select", "==",
                          options=[
                              {"value": "==", "label": "等于"},
                              {"value": "!=", "label": "不等于"},
                              {"value": ">", "label": "大于"},
                              {"value": "<", "label": "小于"},
                              {"value": ">=", "label": "大于等于"},
                              {"value": "<=", "label": "小于等于"},
                              {"value": "contains", "label": "包含"},
                              {"value": "not_contains", "label": "不包含"},
                          ]),
                NodeConfig("compare_value", "比较值", "string", ""),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        condition = context.get("condition", "")
        config = context.get("config", {})
        operator = config.get("operator", "==")
        compare_value = config.get("compare_value", "")
        
        result = False
        if operator == "==":
            result = (condition == compare_value)
        elif operator == "!=":
            result = (condition != compare_value)
        elif operator == ">":
            result = float(condition) > float(compare_value)
        elif operator == "<":
            result = float(condition) < float(compare_value)
        elif operator == ">=":
            result = float(condition) >= float(compare_value)
        elif operator == "<=":
            result = float(condition) <= float(compare_value)
        elif operator == "contains":
            result = compare_value in str(condition)
        elif operator == "not_contains":
            result = compare_value not in str(condition)
        
        return {
            "true_output": context.get("if_true") if result else None,
            "false_output": None if result else context.get("if_true")
        }


class TextOutputNode(OutputNode):
    """文本输出节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.TEXT_OUTPUT.value,
            name="文本输出",
            category=NodeCategory.OUTPUT,
            description="输出文本结果",
            input_ports=[
                NodePort("input", "输入文本", "string", True)
            ],
            output_ports=[],
            config=[
                NodeConfig("format", "输出格式", "select", "text",
                          options=[
                              {"value": "text", "label": "纯文本"},
                              {"value": "markdown", "label": "Markdown"},
                          ]),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": context.get("input", "")}


class HTTPRequestNode(ToolNode):
    """HTTP请求节点"""
    
    def __init__(self, **kwargs):
        super().__init__(
            type=NodeType.HTTP_REQUEST.value,
            name="HTTP请求",
            category=NodeCategory.TOOL,
            description="发送HTTP请求",
            input_ports=[
                NodePort("url", "URL", "string", True),
                NodePort("body", "请求体", "object", False),
            ],
            output_ports=[
                NodePort("response", "响应", "object"),
                NodePort("status", "状态码", "number")
            ],
            config=[
                NodeConfig("method", "方法", "select", "POST",
                          options=[
                              {"value": "GET"}, {"value": "POST"}, 
                              {"value": "PUT"}, {"value": "DELETE"}, {"value": "PATCH"}
                          ]),
                NodeConfig("headers", "请求头", "json", {}),
                NodeConfig("timeout", "超时", "number", 30),
            ]
        )
        self.position = kwargs.get("position", {})
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        config = context.get("config", {})
        url = context.get("url", "")
        body = context.get("body", {})
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=config.get("method", "POST"),
                url=url,
                json=body if body else None,
                headers=config.get("headers", {}),
                timeout=config.get("timeout", 30)
            )
            return {
                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                "status": response.status_code
            }


# ============ 节点注册表 ============

class NodeRegistry:
    """节点注册表"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nodes = {}
            cls._instance._register_default_nodes()
        return cls._instance
    
    def _register_default_nodes(self):
        """注册默认节点"""
        default_nodes = [
            TextInputNode,
            APILnputNode,
            TextOutputNode,
            LLMCompletionNode,
            WebSearchNode,
            HTTPRequestNode,
            IfNode,
        ]
        for node_class in default_nodes:
            self.register(node_class)
    
    def register(self, node_class: type):
        """注册节点类"""
        instance = node_class()
        self._nodes[instance.type] = {
            "class": node_class,
            "name": instance.name,
            "category": instance.category,
            "description": instance.description,
            "config": instance.config,
            "input_ports": instance.input_ports,
            "output_ports": instance.output_ports,
        }
    
    def get(self, node_type: str) -> Optional[Dict]:
        """获取节点定义"""
        return self._nodes.get(node_type)
    
    def list_by_category(self, category: NodeCategory) -> List[Dict]:
        """按分类列出节点"""
        return [
            {"type": k, **v} for k, v in self._nodes.items() 
            if v["category"] == category
        ]
    
    def list_all(self) -> List[Dict]:
        """列出所有节点"""
        return [
            {"type": k, **v} for k, v in self._nodes.items()
        ]
    
    def create_node(self, node_type: str, **kwargs) -> BaseNode:
        """创建节点实例"""
        node_info = self._nodes.get(node_type)
        if not node_info:
            raise ValueError(f"Unknown node type: {node_type}")
        return node_info["class"](**kwargs)


# 全局节点注册表实例
node_registry = NodeRegistry()
