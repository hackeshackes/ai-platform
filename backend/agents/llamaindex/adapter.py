"""
LlamaIndex适配器 - LlamaIndex Integration Adapter
支持导入和执行LlamaIndex Agents/QueryEngines
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class LlamaIndexAgentConfig:
    """LlamaIndex Agent配置"""
    id: str
    name: str
    description: str
    agent_type: str  # "query_engine", "chat_engine", "agent"
    llm_type: str
    llm_config: Dict[str, Any]
    index_type: str = None
    index_config: Dict[str, Any] = None
    prompt_template: str = None
    system_prompt: str = None
    tools: List[str] = None
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


class LlamaIndexAdapter:
    """LlamaIndex适配器 - 将LlamaIndex Agents/Engines集成到协作平台"""
    
    def __init__(self):
        self.imported_agents: Dict[str, LlamaIndexAgentConfig] = {}
        self._agent_instances: Dict[str, Any] = {}
        self._indices: Dict[str, Any] = {}
        self._setup_builtins()
    
    def _setup_builtins(self):
        """设置内置类型"""
        self._builtin_index_types = {
            "vector": "langchain.vectorstores",
            "list": "llama_index.core.node_parser",
            "tree": "llama_index.core",
            "keyword": "llama_index.core",
            "ferret": "llama_index.core",
        }
    
    def import_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        导入LlamaIndex Agent
        
        Args:
            config: Agent配置字典
                - name: Agent名称
                - description: Agent描述
                - agent_type: Agent类型 (query_engine, chat_engine, agent)
                - llm_type: LLM类型 (openai, anthropic, local等)
                - llm_config: LLM配置
                - index_type: 索引类型 (可选，用于query_engine)
                - index_config: 索引配置
                - prompt_template: 提示模板
                - system_prompt: 系统提示
                - tools: 工具列表
        
        Returns:
            导入结果
        """
        agent_id = str(uuid.uuid4())
        
        # 验证必要字段
        required_fields = ["name", "agent_type", "llm_type", "llm_config"]
        missing = [f for f in required_fields if f not in config]
        if missing:
            return {
                "success": False,
                "error": f"Missing required fields: {missing}"
            }
        
        # 创建配置
        agent_config = LlamaIndexAgentConfig(
            id=agent_id,
            name=config["name"],
            description=config.get("description", ""),
            agent_type=config["agent_type"],
            llm_type=config["llm_type"],
            llm_config=config["llm_config"],
            index_type=config.get("index_type"),
            index_config=config.get("index_config"),
            prompt_template=config.get("prompt_template"),
            system_prompt=config.get("system_prompt"),
            tools=config.get("tools", [])
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
        创建LlamaIndex Agent实例
        
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
            # 创建LLM
            llm = self._create_llm(config.llm_type, config.llm_config)
            
            # 创建索引（如果需要）
            index = None
            if config.index_type and config.agent_type in ["query_engine", "chat_engine"]:
                index = self._create_index(config)
                if not index:
                    return {
                        "success": False,
                        "error": f"Failed to create index of type '{config.index_type}'"
                    }
                self._indices[agent_id] = index
            
            # 创建Agent/Engine
            agent = self._create_engine(config, llm, index)
            
            self._agent_instances[agent_id] = agent
            
            return {
                "success": True,
                "agent_id": agent_id,
                "instance": agent,
                "message": f"{config.agent_type} created successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create agent: {str(e)}"
            }
    
    def execute_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LlamaIndex Agent
        
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
            chat_history = input_data.get("chat_history", [])
            
            config = self.imported_agents[agent_id]
            
            if config.agent_type == "query_engine":
                response = agent.query(query)
                output = str(response)
            elif config.agent_type == "chat_engine":
                response = agent.chat(query)
                output = str(response)
            elif config.agent_type == "agent":
                # 支持多种输入格式
                if isinstance(input_data.get("messages"), list):
                    response = agent.chat(input_data["messages"])
                else:
                    response = agent.chat(query)
                output = str(response)
            else:
                return {
                    "success": False,
                    "error": f"Unknown agent type: {config.agent_type}"
                }
            
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
    
    async def execute_agent_async(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行Agent"""
        # 获取或创建实例
        if agent_id not in self._agent_instances:
            result = self.create_agent_instance(agent_id)
            if not result["success"]:
                return result
            agent = result["instance"]
        else:
            agent = self._agent_instances[agent_id]
        
        try:
            query = input_data.get("query", "")
            config = self.imported_agents[agent_id]
            
            if hasattr(agent, 'achat'):
                response = await agent.achat(query)
            else:
                response = agent.query(query)
            
            return {
                "success": True,
                "output": str(response),
                "agent_id": agent_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    def get_agent_config(self, agent_id: str) -> Optional[LlamaIndexAgentConfig]:
        """获取Agent配置"""
        return self.imported_agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有导入的Agent"""
        return [
            {
                "id": aid,
                "name": config.name,
                "description": config.description,
                "agent_type": config.agent_type,
                "llm_type": config.llm_type,
                "index_type": config.index_type,
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
            if agent_id in self._indices:
                del self._indices[agent_id]
            return True
        return False
    
    def add_documents_to_index(self, agent_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """向索引添加文档"""
        if agent_id not in self._indices:
            return {"success": False, "error": "Index not found"}
        
        index = self._indices[agent_id]
        
        try:
            # 转换文档格式
            from llama_index.core import Document
            docs = []
            for doc in documents:
                text = doc.get("text", doc.get("content", ""))
                if text:
                    docs.append(Document(text=text, metadata=doc.get("metadata", {})))
            
            if hasattr(index, 'insert'):
                index.insert(docs)
            elif hasattr(index, 'add'):
                index.add(docs)
            
            return {"success": True, "count": len(docs)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_llm(self, llm_type: str, config: Dict[str, Any]):
        """创建LLM实例"""
        llm_type = llm_type.lower()
        
        try:
            if llm_type == "openai":
                from llama_index.llms.openai import OpenAI
                return OpenAI(
                    model=config.get("model", "gpt-3.5-turbo"),
                    api_key=config.get("api_key"),
                    temperature=config.get("temperature", 0.1)
                )
            elif llm_type == "anthropic":
                from llama_index.llms.anthropic import Anthropic
                return Anthropic(
                    model=config.get("model", "claude-3-haiku-20240307"),
                    api_key=config.get("api_key")
                )
            elif llm_type == "huggingface":
                from llama_index.llms.huggingface import HuggingFaceLLM
                return HuggingFaceLLM(
                    model_name=config.get("model_name", "google/flan-t5-large"),
                    tokenizer_name=config.get("tokenizer_name", "google/flan-t5-large")
                )
            elif llm_type == "local":
                from llama_index.llms.ollama import Ollama
                return Ollama(
                    model=config.get("model", "llama2"),
                    base_url=config.get("base_url", "http://localhost:11434")
                )
        except ImportError:
            pass
        
        # 返回虚拟LLM
        return self._create_dummy_llm()
    
    def _create_dummy_llm(self):
        """创建虚拟LLM用于测试"""
        class DummyLLM:
            def __call__(self, prompt, **kwargs):
                return f"Dummy response to: {prompt[:50]}..."
        
        return DummyLLM()
    
    def _create_index(self, config: LlamaIndexAgentConfig):
        """创建索引"""
        index_type = config.index_type.lower() if config.index_type else "vector"
        index_config = config.index_config or {}
        
        try:
            if index_type == "vector":
                from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
                
                # 加载文档或创建空索引
                if index_config.get("documents"):
                    from llama_index.core import Document
                    docs = [Document(text=d) for d in index_config["documents"]]
                    return VectorStoreIndex.from_documents(docs)
                elif index_config.get("dir_path"):
                    documents = SimpleDirectoryReader(index_config["dir_path"]).load_data()
                    return VectorStoreIndex.from_documents(documents)
                else:
                    return VectorStoreIndex([])
            
            elif index_type == "list":
                from llama_index.core import ListIndex
                return ListIndex([])
            
            elif index_type == "tree":
                from llama_index.core import TreeIndex
                return TreeIndex([])
            
            elif index_type == "keyword":
                from llama_index.core import KeywordIndex
                return KeywordIndex([])
                
        except ImportError:
            pass
        
        return None
    
    def _create_engine(self, config: LlamaIndexAgentConfig, llm, index):
        """创建Engine/Agent"""
        agent_type = config.agent_type.lower()
        
        try:
            if agent_type == "query_engine":
                return index.as_query_engine(
                    llm=llm,
                    similarity_top_k=config.index_config.get("similarity_top_k", 2) if config.index_config else 2
                )
            
            elif agent_type == "chat_engine":
                return index.as_chat_engine(
                    llm=llm,
                    chat_mode=config.index_config.get("chat_mode", "context") if config.index_config else "context"
                )
            
            elif agent_type == "agent":
                from llama_index.core.agent import ReActAgent
                from llama_index.core.tools import QueryEngineTool, ToolMetadata
                
                tools = []
                
                # 添加查询引擎工具
                query_engine = index.as_query_engine(llm=llm)
                tools.append(QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="query_engine",
                        description="Query the document index"
                    )
                ))
                
                # 添加其他工具
                if config.tools:
                    for tool_name in config.tools:
                        tool = self._create_tool(tool_name)
                        if tool:
                            tools.append(tool)
                
                return ReActAgent.from_tools(
                    tools=tools,
                    llm=llm,
                    system_prompt=config.system_prompt or "You are a helpful assistant."
                )
                
        except ImportError:
            pass
        
        return None
    
    def _create_tool(self, tool_name: str):
        """创建工具"""
        tool_name = tool_name.lower()
        
        try:
            if tool_name == "arxiv":
                from llama_index.core.tools import FunctionTool
                import arxiv
                
                def search_arxiv(query: str) -> str:
                    """Search arXiv for papers"""
                    search = arxiv.Search(query=query, max_results=5)
                    results = []
                    for r in search.results():
                        results.append(f"- {r.title} ({r.id})")
                    return "\n".join(results) if results else "No results found"
                
                return FunctionTool.from_defaults(search_arxiv)
            
            elif tool_name == "wikipedia":
                from llama_index.core.tools import FunctionTool
                import wikipedia
                
                def search_wikipedia(query: str) -> str:
                    """Search Wikipedia"""
                    try:
                        return wikipedia.summary(query, sentences=3)
                    except:
                        return "No results found"
                
                return FunctionTool.from_defaults(search_wikipedia)
                
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
                "agent_type": config.agent_type,
                "llm_type": config.llm_type,
                "llm_config": config.llm_config,
                "index_type": config.index_type,
                "index_config": config.index_config,
                "prompt_template": config.prompt_template,
                "system_prompt": config.system_prompt,
                "tools": config.tools,
                "created_at": config.created_at,
                "updated_at": config.updated_at
            }
        return None


# 全局LlamaIndex适配器实例
llamaindex_adapter = LlamaIndexAdapter()
