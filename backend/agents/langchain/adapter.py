"""
LangChain适配器 - LangChain Integration Adapter
支持导入和执行LangChain Agents
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class LangChainAgentConfig:
    """LangChain Agent配置"""
    id: str
    name: str
    description: str
    agent_class: str
    llm_type: str
    llm_config: Dict[str, Any]
    tools: List[str] = None
    prompt_template: str = None
    memory_config: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class LangChainAdapter:
    """LangChain适配器 - 将LangChain Agents集成到协作平台"""
    
    def __init__(self):
        self.imported_agents: Dict[str, LangChainAgentConfig] = {}
        self._agent_instances: Dict[str, Any] = {}
        self._setup_builtins()
    
    def _setup_builtins(self):
        """设置内置Agent类型映射"""
        self._builtin_agents = {
            "zero_shot_react": "langchain.agents.ZeroShotAgent",
            "react_docstore": "langchain.agents.ReactDocstoreAgent",
            "self_ask_with_search": "langchain.agents.SelfAskWithSearchAgent",
            "conversational": "langchain.agents.ConversationalAgent",
            "structured_chat": "langchain.agents.StructuredChatAgent",
            "openai_functions": "langchain.agents.OpenAIFunctionsAgent",
            "openai_multi_functions": "langchain.agents.OpenAIMultiFunctionsAgent",
        }
    
    def import_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入LangChain Agent
        
        Args:
            config: Agent配置字典
                - name: Agent名称
                - description: Agent描述
                - agent_class: Agent类名
                - llm_type: LLM类型 (openai, anthropic, local等)
                - llm_config: LLM配置 (api_key, model等)
                - tools: 工具列表
                - prompt_template: 提示模板
                - memory_config: 记忆配置
        
        Returns:
            导入结果
        """
        agent_id = str(uuid.uuid4())
        
        # 验证必要字段
        required_fields = ["name", "agent_class", "llm_type", "llm_config"]
        missing = [f for f in required_fields if f not in config]
        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {missing}"
            }
        
        # 创建配置
        agent_config = LangChainAgentConfig(
            id=agent_id,
            name=config["name"],
            description=config.get("description", ""),
            agent_class=config["agent_class"],
            llm_type=config["llm_type"],
            llm_config=config["llm_config"],
            tools=config.get("tools", []),
            prompt_template=config.get("prompt_template"),
            memory_config=config.get("memory_config")
        )
        
        self.imported_agents[agent_id] = agent_config
        
        return {
            "success": True,
            "agent_id": agent_id,
            "name": agent_config.name,
            "message": f"Agent '{agent_config.name}' imported successfully"
        }
    
    def create_agent_instance(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        创建LangChain Agent实例
        
        Args:
            agent_id: 已导入的Agent ID
            **kwargs: 额外的初始化参数
        
        Returns:
            创建结果
        """
        if agent_id not in self.imported_agents:
            return {
                "success": False,
                "error": f"Agent '{agent_id}' not found"
            }
        
        config = self.imported_agents[agent_id]
        
        try:
            # 动态导入并创建LLM
            llm = self._create_llm(config.llm_type, config.llm_config)
            
            # 创建Agent
            agent_class = self._get_agent_class(config.agent_class)
            if not agent_class:
                return {
                    "success": False,
                    "error": f"Agent class '{config.agent_class}' not available"
                }
            
            # 创建提示模板
            prompt_template = self._create_prompt_template(config)
            
            # 创建工具
            tools = self._create_tools(config.tools) if config.tools else []
            
            # 创建记忆
            memory = self._create_memory(config.memory_config) if config.memory_config else None
            
            # 实例化Agent
            agent = agent_class.from_llm_and_tools(llm, tools, prompt_template=prompt_template)
            
            if memory:
                agent.memory = memory
            
            self._agent_instances[agent_id] = agent
            
            return {
                "success": True,
                "agent_id": agent_id,
                "instance": agent,
                "message": "Agent instance created"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create agent: {str(e)}"
            }
    
    def execute_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LangChain Agent
        
        Args:
            agent_id: Agent ID
            input_data: 输入数据
        
        Returns:
            执行结果
        """
        # 获取或创建实例
        if agent_id not in self._agent_instances:
            result = self.create_agent_instance(agent_id)
            if not result["success"]:
                return result
            agent = result["instance"]
        else:
            agent = self._agent_instances[agent_id]
        
        try:
            # 提取输入
            query = input_data.get("query", "")
            if isinstance(query, str):
                query = [query]
            
            # 执行Agent
            output = agent.run(query)
            
            return {
                "success": True,
                "output": output,
                "agent_id": agent_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    def get_agent_config(self, agent_id: str) -> Optional[LangChainAgentConfig]:
        """获取Agent配置"""
        return self.imported_agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有导入的Agent"""
        return [
            {
                "id": aid,
                "name": config.name,
                "description": config.description,
                "agent_class": config.agent_class,
                "llm_type": config.llm_type,
                "tools": config.tools,
                "created_at": config.created_at
            }
            for aid, config in self.imported_agents.items()
        ]
    
    def delete_agent(self, agent_id: str) -> bool:
        """删除导入的Agent"""
        if agent_id in self.imported_agents:
            del self.imported_agents[agent_id]
            if agent_id in self._agent_instances:
                del self._agent_instances[agent_id]
            return True
        return False
    
    def _create_llm(self, llm_type: str, config: Dict[str, Any]):
        """创建LLM实例"""
        llm_type = llm_type.lower()
        
        if llm_type == "openai":
            try:
                from langchain_openai import OpenAI
                return OpenAI(
                    model=config.get("model", "gpt-3.5-turbo"),
                    api_key=config.get("api_key"),
                    temperature=config.get("temperature", 0.0),
                    max_tokens=config.get("max_tokens")
                )
            except ImportError:
                pass
            
        if llm_type == "anthropic":
            try:
                from langchain_anthropic import AnthropicLLM
                return AnthropicLLM(
                    model=config.get("model", "claude-2"),
                    api_key=config.get("api_key")
                )
            except ImportError:
                pass
        
        if llm_type == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceHub
                return HuggingFaceHub(
                    repo_id=config.get("repo_id", "google/flan-t5-large"),
                    huggingfacehub_api_token=config.get("api_key")
                )
            except ImportError:
                pass
        
        if llm_type == "local":
            try:
                from langchain_community.llms import Ollama
                return Ollama(
                    model=config.get("model", "llama2"),
                    base_url=config.get("base_url", "http://localhost:11434")
                )
            except ImportError:
                pass
        
        # 默认返回虚拟LLM用于测试
        return self._create_dummy_llm()
    
    def _create_dummy_llm(self):
        """创建虚拟LLM用于测试"""
        class DummyLLM:
            def __call__(self, prompt, **kwargs):
                return f"Dummy response to: {prompt[:50]}..."
            
            def get_num_tokens(self, text):
                return len(text)
        
        return DummyLLM()
    
    def _get_agent_class(self, agent_class: str):
        """获取Agent类"""
        # 检查内置类型
        if agent_class in self._builtin_agents:
            try:
                import importlib
                module_path, class_name = self._builtin_agents[agent_class].rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass
        
        # 尝试直接导入
        try:
            import importlib
            module_path, class_name = agent_class.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError):
            pass
        
        return None
    
    def _create_prompt_template(self, config: LangChainAgentConfig):
        """创建提示模板"""
        if config.prompt_template:
            from langchain.prompts import PromptTemplate
            return PromptTemplate(
                template=config.prompt_template,
                input_variables=["input", "agent_scratchpad", "tools"]
            )
        return None
    
    def _create_tools(self, tools_config: List[str]):
        """创建工具列表"""
        tools = []
        
        for tool_name in tools_config:
            tool = self._create_tool(tool_name)
            if tool:
                tools.append(tool)
        
        return tools
    
    def _create_tool(self, tool_name: str):
        """创建单个工具"""
        tool_name = tool_name.lower()
        
        try:
            if tool_name == "serpapi":
                from langchain_community.utilities import SerpAPIWrapper
                return SerpAPIWrapper()
            elif tool_name == "wikipedia":
                from langchain_community.tools import WikipediaQueryRun
                return WikipediaQueryRun()
            elif tool_name == "python":
                from langchain_experimental.tools import PythonREPLTool
                return PythonREPLTool()
            elif tool_name == "arxiv":
                from langchain_community.tools import ArxivQueryRun
                return ArxivQueryRun()
            elif tool_name == "terminal":
                from langchain_community.agent_toolkits import TerminalTool
                return TerminalTool()
            elif tool_name == "file":
                from langchain_community.agent_toolkits import FileSearchTool
                return FileSearchTool()
        except ImportError:
            pass
        
        return None
    
    def _create_memory(self, memory_config: Dict[str, Any]):
        """创建记忆组件"""
        memory_type = memory_config.get("type", "buffer").lower()
        
        try:
            if memory_type == "buffer":
                from langchain.memory import ConversationBufferMemory
                return ConversationBufferMemory()
            elif memory_type == "entity":
                from langchain.memory import ConversationEntityMemory
                return ConversationEntityMemory()
            elif memory_type == "kg":
                from langchain.memory import ConversationKGMemory
                return ConversationKGMemory()
        except ImportError:
            pass
        
        return None
    
    def export_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """导出Agent配置"""
        config = self.imported_agents.get(agent_id)
        if config:
            return {
                "id": config.id,
                "name": config.name,
                "description": config.description,
                "agent_class": config.agent_class,
                "llm_type": config.llm_type,
                "llm_config": config.llm_config,
                "tools": config.tools,
                "prompt_template": config.prompt_template,
                "memory_config": config.memory_config,
                "created_at": config.created_at,
                "updated_at": config.updated_at
            }
        return None


# 全局LangChain适配器实例
langchain_adapter = LangChainAdapter()
