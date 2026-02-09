"""
内置工具模块
提供Agent的默认工具集
"""

from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class BuiltinTools:
    """内置工具集合"""
    
    @staticmethod
    def search_web(query: str, **kwargs) -> Dict[str, Any]:
        """
        网络搜索工具
        
        Args:
            query: 搜索关键词
            
        Returns:
            搜索结果
        """
        logger.info(f"Searching web for: {query}")
        
        # 简化实现 - 实际应该调用真实搜索引擎
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result 1 for {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"This is a sample search result for {query}..."
                },
                {
                    "title": f"Result 2 for {query}",
                    "url": f"https://example.com/info/{query}",
                    "snippet": f"More information about {query}..."
                }
            ],
            "count": 2
        }
    
    @staticmethod
    def search_docs(query: str, **kwargs) -> Dict[str, Any]:
        """
        文档搜索工具
        
        Args:
            query: 搜索查询
            
        Returns:
            文档搜索结果
        """
        logger.info(f"Searching docs for: {query}")
        
        return {
            "query": query,
            "documents": [
                {
                    "id": "doc1",
                    "title": "Documentation Page 1",
                    "content": f"Documentation content about {query}...",
                    "relevance": 0.95
                },
                {
                    "id": "doc2",
                    "title": "API Reference",
                    "content": f"API details related to {query}...",
                    "relevance": 0.85
                }
            ],
            "total": 2
        }
    
    @staticmethod
    def execute_code(code: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        """
        代码执行工具
        
        Args:
            code: 要执行的代码
            language: 编程语言
            
        Returns:
            执行结果
        """
        logger.info(f"Executing {language} code")
        
        # 简化实现 - 实际应该使用安全的代码沙箱
        output = {
            "language": language,
            "output": f"[{language}] Code execution simulated for: {code[:100]}...",
            "success": True
        }
        
        return output
    
    @staticmethod
    def call_llm(prompt: str, model: str = "default", **kwargs) -> Dict[str, Any]:
        """
        LLM调用工具
        
        Args:
            prompt: 提示词
            model: 模型名称
            
        Returns:
            LLM响应
        """
        logger.info(f"Calling LLM with prompt: {prompt[:100]}...")
        
        # 简化实现 - 实际应该调用真实LLM API
        return {
            "model": model,
            "response": f"LLM response to: {prompt[:50]}...",
            "success": True
        }
    
    @staticmethod
    def database_query(query: str, database: str = "default", **kwargs) -> Dict[str, Any]:
        """
        数据库查询工具
        
        Args:
            query: SQL查询语句
            database: 数据库名称
            
        Returns:
            查询结果
        """
        logger.info(f"Querying database: {query}")
        
        # 简化实现 - 实际应该连接真实数据库
        return {
            "query": query,
            "database": database,
            "results": [],
            "row_count": 0,
            "success": True
        }
    
    @staticmethod
    def file_read(path: str, **kwargs) -> Dict[str, Any]:
        """
        文件读取工具
        
        Args:
            path: 文件路径
            
        Returns:
            文件内容
        """
        try:
            with open(path, 'r') as f:
                content = f.read()
            return {
                "path": path,
                "content": content,
                "success": True
            }
        except Exception as e:
            return {
                "path": path,
                "error": str(e),
                "success": False
            }
    
    @staticmethod
    def file_write(path: str, content: str, **kwargs) -> Dict[str, Any]:
        """
        文件写入工具
        
        Args:
            path: 文件路径
            content: 文件内容
            
        Returns:
            写入结果
        """
        try:
            with open(path, 'w') as f:
                f.write(content)
            return {
                "path": path,
                "success": True
            }
        except Exception as e:
            return {
                "path": path,
                "error": str(e),
                "success": False
            }
    
    @staticmethod
    def calculate(expression: str, **kwargs) -> Dict[str, Any]:
        """
        计算工具
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 注意：eval有安全风险，生产环境应使用更安全的方式
            result = eval(expression)
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
    
    @staticmethod
    def get_current_time(**kwargs) -> Dict[str, Any]:
        """
        获取当前时间
        
        Returns:
            当前时间信息
        """
        from datetime import datetime
        now = datetime.now()
        return {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "UTC"
        }
    
    @staticmethod
    def format_text(text: str, format: str = "markdown", **kwargs) -> Dict[str, Any]:
        """
        文本格式化工具
        
        Args:
            text: 原始文本
            format: 目标格式
            
        Returns:
            格式化后的文本
        """
        from datetime import datetime
        
        if format == "json":
            try:
                formatted = json.dumps(json.loads(text), indent=2)
            except:
                formatted = f"Invalid JSON: {text}"
        elif format == "uppercase":
            formatted = text.upper()
        elif format == "lowercase":
            formatted = text.lower()
        elif format == "timestamp":
            formatted = datetime.now().isoformat()
        else:
            formatted = text
        
        return {
            "original": text,
            "formatted": formatted,
            "format": format
        }
    
    @staticmethod
    def summarize(text: str, max_length: int = 100, **kwargs) -> Dict[str, Any]:
        """
        文本摘要工具
        
        Args:
            text: 要摘要的文本
            max_length: 最大长度
            
        Returns:
            摘要结果
        """
        # 简化实现 - 实际应该使用LLM进行摘要
        words = text.split()
        if len(words) > max_length:
            summary = " ".join(words[:max_length]) + "..."
        else:
            summary = text
        
        return {
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary
        }


def get_all_tools() -> List[Dict]:
    """获取所有内置工具定义"""
    tools = []
    
    # 定义每个工具的元数据
    tool_definitions = [
        {
            "name": "search_web",
            "func": BuiltinTools.search_web,
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            "category": "search"
        },
        {
            "name": "search_docs",
            "func": BuiltinTools.search_docs,
            "description": "Search documentation for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            "category": "search"
        },
        {
            "name": "execute_code",
            "func": BuiltinTools.execute_code,
            "description": "Execute code in a specified language",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to execute"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                        "default": "python"
                    }
                },
                "required": ["code"]
            },
            "category": "utility"
        },
        {
            "name": "call_llm",
            "func": BuiltinTools.call_llm,
            "description": "Call a language model",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt for the LLM"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name",
                        "default": "default"
                    }
                },
                "required": ["prompt"]
            },
            "category": "llm"
        },
        {
            "name": "database_query",
            "func": BuiltinTools.database_query,
            "description": "Execute a database query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name",
                        "default": "default"
                    }
                },
                "required": ["query"]
            },
            "category": "database"
        },
        {
            "name": "file_read",
            "func": BuiltinTools.file_read,
            "description": "Read file content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    }
                },
                "required": ["path"]
            },
            "category": "file"
        },
        {
            "name": "file_write",
            "func": BuiltinTools.file_write,
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["path", "content"]
            },
            "category": "file"
        },
        {
            "name": "calculate",
            "func": BuiltinTools.calculate,
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression"
                    }
                },
                "required": ["expression"]
            },
            "category": "utility"
        },
        {
            "name": "get_current_time",
            "func": BuiltinTools.get_current_time,
            "description": "Get current timestamp",
            "parameters": {
                "type": "object",
                "properties": {}
            },
            "category": "utility"
        },
        {
            "name": "format_text",
            "func": BuiltinTools.format_text,
            "description": "Format text in various ways",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to format"
                    },
                    "format": {
                        "type": "string",
                        "description": "Target format (json, uppercase, lowercase, timestamp)",
                        "default": "markdown"
                    }
                },
                "required": ["text"]
            },
            "category": "utility"
        },
        {
            "name": "summarize",
            "func": BuiltinTools.summarize,
            "description": "Summarize text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length",
                        "default": 100
                    }
                },
                "required": ["text"]
            },
            "category": "utility"
        }
    ]
    
    for tool_def in tool_definitions:
        tool_def["registered_at"] = "builtin"
        tools.append(tool_def)
    
    return tools


# 创建内置工具模块的单例
builtin_tools = type('BuiltinToolsModule', (), {
    'get_all_tools': get_all_tools
})()
