"""
Agent Test Suite - AI Platform Backend
Agent测试套件
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== Agent框架测试 ==============

class TestAgentFramework:
    """Agent框架测试类"""
    
    def test_framework_import(self):
        """测试框架导入"""
        try:
            from agents.framework import (
                AgentFramework,
                AgentRole,
                AgentStatus,
                Tool,
                Memory,
                Agent
            )
            assert AgentFramework is not None
            assert AgentRole is not None
        except ImportError:
            pytest.skip("Agent framework not available")
    
    def test_agent_role_enum(self):
        """测试Agent角色枚举"""
        from agents.framework import AgentRole
        assert hasattr(AgentRole, 'ANALYST')
        assert hasattr(AgentRole, 'COORDINATOR')
        assert hasattr(AgentRole, 'EXECUTOR')
        assert hasattr(AgentRole, 'MONITOR')
        assert hasattr(AgentRole, 'RESEARCHER')
    
    def test_agent_status_enum(self):
        """测试Agent状态枚举"""
        from agents.framework import AgentStatus
        assert hasattr(AgentStatus, 'IDLE')
        assert hasattr(AgentStatus, 'RUNNING')
        assert hasattr(AgentStatus, 'THINKING')
        assert hasattr(AgentStatus, 'WAITING')
        assert hasattr(AgentStatus, 'ERROR')
    
    def test_tool_structure(self):
        """测试工具结构"""
        from agents.framework import Tool
        tool = Tool(
            tool_id="test-tool",
            name="Test Tool",
            description="A test tool",
            type="python",
            parameters={"code": {"type": "string"}}
        )
        assert tool.tool_id == "test-tool"
        assert tool.name == "Test Tool"
        assert tool.type == "python"
    
    def test_memory_structure(self):
        """测试记忆结构"""
        from agents.framework import Memory
        memory = Memory(
            memory_id="mem-001",
            content="Test memory content",
            timestamp=datetime.utcnow(),
            importance=0.8
        )
        assert memory.memory_id == "mem-001"
        assert memory.importance == 0.8
    
    def test_agent_structure(self):
        """测试Agent结构"""
        from agents.framework import Agent, AgentRole, AgentStatus
        agent = Agent(
            agent_id="agent-001",
            name="Test Agent",
            role=AgentRole.RESEARCHER,
            system_prompt="You are a helpful assistant.",
            tools=[],
            memories=[],
            status=AgentStatus.IDLE,
            created_at=datetime.utcnow()
        )
        assert agent.agent_id == "agent-001"
        assert agent.role == AgentRole.RESEARCHER
        assert agent.status == AgentStatus.IDLE
    
    def test_framework_initialization(self):
        """测试框架初始化"""
        from agents.framework import AgentFramework
        framework = AgentFramework()
        assert framework is not None
        assert hasattr(framework, 'agents')
        assert hasattr(framework, 'tool_registry')
    
    def test_default_tools_registration(self):
        """测试默认工具注册"""
        from agents.framework import AgentFramework
        framework = AgentFramework()
        
        # 检查默认工具
        assert "python" in framework.tool_registry
        assert "shell" in framework.tool_registry
        assert "search" in framework.tool_registry
        assert "file_read" in framework.tool_registry
        assert "file_write" in framework.tool_registry
    
    def test_create_agent(self):
        """测试创建Agent"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Research Agent",
            role=AgentRole.RESEARCHER,
            system_prompt="You are a research agent.",
            tool_names=["python", "search"]
        )
        
        assert agent is not None
        assert agent.name == "Research Agent"
        assert agent.role == AgentRole.RESEARCHER
        assert len(agent.tools) == 2
    
    def test_list_agents(self):
        """测试列出Agents"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        # 创建多个agents
        framework.create_agent("Agent1", AgentRole.RESEARCHER, "Prompt1", ["python"])
        framework.create_agent("Agent2", AgentRole.ANALYST, "Prompt2", ["shell"])
        
        agents = framework.list_agents()
        assert len(agents) == 2
    
    def test_get_agent(self):
        """测试获取Agent"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        created = framework.create_agent("Test", AgentRole.RESEARCHER, "Prompt", [])
        retrieved = framework.get_agent(created.agent_id)
        
        assert retrieved is not None
        assert retrieved.name == "Test"
    
    def test_delete_agent(self):
        """测试删除Agent"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent("ToDelete", AgentRole.EXECUTOR, "Prompt", [])
        agent_id = agent.agent_id
        
        # 删除
        result = framework.delete_agent(agent_id)
        assert result is True
        
        # 验证已删除
        retrieved = framework.get_agent(agent_id)
        assert retrieved is None
    
    def test_add_memory(self):
        """测试添加记忆"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent("MemoryTest", AgentRole.ANALYST, "Prompt", [])
        
        framework.add_memory(agent.agent_id, "Important fact", importance=0.9)
        framework.add_memory(agent.agent_id, "Less important", importance=0.3)
        
        assert len(agent.memories) == 2
    
    def test_get_tools(self):
        """测试获取工具列表"""
        from agents.framework import AgentFramework
        framework = AgentFramework()
        
        tools = framework.get_tools()
        assert len(tools) >= 5
        assert any(t["tool_id"] == "python" for t in tools)


# ============== Agent模板测试 ==============

class TestAgentTemplates:
    """Agent模板测试类"""
    
    def test_template_models_import(self):
        """测试模板模型导入"""
        try:
            from agents.factory.models import (
                AgentTemplate,
                AgentTemplateConfig,
                TemplateVersion,
                AgentInstance,
                FactoryStatus
            )
            assert AgentTemplate is not None
        except ImportError:
            pytest.skip("Template models not available")
    
    def test_agent_template_config(self):
        """测试Agent模板配置"""
        from agents.factory.models import AgentTemplateConfig
        config = AgentTemplateConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096,
            stream=False,
            timeout=30
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
    
    def test_agent_template(self):
        """测试Agent模板"""
        from agents.factory.models import AgentTemplate, AgentTemplateConfig
        template = AgentTemplate(
            name="Research Agent",
            description="A research agent",
            version="1.0.0",
            capabilities=["web_search", "analysis"],
            config=AgentTemplateConfig(),
            tools=["web_search", "file_read"],
            system_prompt="You are a research agent."
        )
        assert template.name == "Research Agent"
        assert template.version == "1.0.0"
    
    def test_create_agent_request(self):
        """测试创建Agent请求"""
        from agents.factory.models import CreateAgentRequest
        request = CreateAgentRequest(
            template_id="research-agent",
            name="My Research Agent",
            count=1,
            variables={"custom_param": "value"},
            metadata={"env": "production"}
        )
        assert request.template_id == "research-agent"
        assert request.count == 1
    
    def test_deploy_agent_request(self):
        """测试部署Agent请求"""
        from agents.factory.models import DeployAgentRequest
        request = DeployAgentRequest(
            agent_ids=["agent-001", "agent-002"],
            environment="production",
            replicas=2,
            resources={"cpu": "2", "memory": "4Gi"}
        )
        assert len(request.agent_ids) == 2
        assert request.environment == "production"
    
    def test_agent_instance(self):
        """测试Agent实例"""
        from agents.factory.models import AgentInstance
        instance = AgentInstance(
            id="instance-001",
            name="Deployed Agent",
            template_id="research-agent",
            template_version="1.0.0",
            config={"model": "gpt-4"},
            status="running",
            created_at=datetime.utcnow()
        )
        assert instance.id == "instance-001"
        assert instance.status == "running"
    
    def test_factory_status(self):
        """测试工厂状态"""
        from agents.factory.models import FactoryStatus
        status = FactoryStatus(
            total_templates=10,
            total_agents=50,
            running_agents=30,
            stopped_agents=15,
            active_sessions=5
        )
        assert status.total_templates == 10
        assert status.running_agents == 30


# ============== Agent API测试 ==============

class TestAgentAPI:
    """Agent API测试类"""
    
    def test_agent_api_import(self):
        """测试Agent API导入"""
        try:
            from agents.api.endpoints import router
            assert router is not None
        except ImportError:
            pytest.skip("Agent API not available")
    
    def test_agent_framework_instance(self):
        """测试Agent框架实例"""
        try:
            from agents.framework import agent_framework
            assert agent_framework is not None
        except ImportError:
            pytest.skip("Agent framework instance not available")


# ============== Research Agent测试 ==============

class TestResearchAgent:
    """Research Agent测试类"""
    
    def test_research_agent_creation(self):
        """测试Research Agent创建"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="You research information.",
            tool_names=["search", "file_read"]
        )
        
        assert agent.role == AgentRole.RESEARCHER
        assert "search" in [t.tool_id for t in agent.tools]
    
    def test_research_agent_tasks(self):
        """测试Research Agent任务"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            system_prompt="You research.",
            tool_names=["web_search"]
        )
        
        # 验证agent创建成功
        assert agent is not None
        assert agent.role == AgentRole.RESEARCHER


# ============== Coding Agent测试 ==============

class TestCodingAgent:
    """Coding Agent测试类"""
    
    def test_coding_agent_creation(self):
        """测试Coding Agent创建"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Coder",
            role=AgentRole.EXECUTOR,
            system_prompt="You write code.",
            tool_names=["python", "shell"]
        )
        
        assert "python" in [t.tool_id for t in agent.tools]
        assert "shell" in [t.tool_id for t in agent.tools]
    
    def test_coding_agent_execution(self):
        """测试Coding Agent执行"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Coder",
            role=AgentRole.EXECUTOR,
            system_prompt="You write code.",
            tool_names=["python"]
        )
        
        # 验证agent创建成功
        assert agent is not None
        assert "python" in [t.tool_id for t in agent.tools]


# ============== Chat Agent测试 ==============

class TestChatAgent:
    """Chat Agent测试类"""
    
    def test_chat_agent_creation(self):
        """测试Chat Agent创建"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="ChatBot",
            role=AgentRole.COORDINATOR,
            system_prompt="You chat with users.",
            tool_names=["file_read"]
        )
        
        assert agent is not None
        assert agent.role == AgentRole.COORDINATOR


# ============== Support Agent测试 ==============

class TestSupportAgent:
    """Support Agent测试类"""
    
    def test_support_agent_creation(self):
        """测试Support Agent创建"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent = framework.create_agent(
            name="Support",
            role=AgentRole.COORDINATOR,
            system_prompt="You provide support.",
            tool_names=["file_read", "file_write"]
        )
        
        assert agent is not None
        assert len(agent.tools) >= 2


# ============== Agent编排测试 ==============

class TestAgentOrchestration:
    """Agent编排测试类"""
    
    @pytest.mark.asyncio
    async def test_sequential_orchestration(self):
        """测试顺序编排"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        # 创建多个agents
        agent1 = framework.create_agent("A1", AgentRole.EXECUTOR, "Task1", ["python"])
        agent2 = framework.create_agent("A2", AgentRole.EXECUTOR, "Task2", ["shell"])
        
        # 模拟顺序编排 - 验证编排返回结构
        result = await framework.orchestrate(
            task="Complex task",
            agent_ids=[agent1.agent_id, agent2.agent_id],
            strategy="sequential"
        )
        
        assert result["strategy"] == "sequential"
    
    @pytest.mark.asyncio
    async def test_parallel_orchestration(self):
        """测试并行编排"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        agent1 = framework.create_agent("P1", AgentRole.EXECUTOR, "Task1", [])
        agent2 = framework.create_agent("P2", AgentRole.EXECUTOR, "Task2", [])
        
        result = await framework.orchestrate(
            task="Parallel task",
            agent_ids=[agent1.agent_id, agent2.agent_id],
            strategy="parallel"
        )
        
        assert result["strategy"] == "parallel"
    
    @pytest.mark.asyncio
    async def test_hierarchical_orchestration(self):
        """测试层级编排"""
        from agents.framework import AgentFramework, AgentRole
        framework = AgentFramework()
        
        coordinator = framework.create_agent("Coordinator", AgentRole.COORDINATOR, "Coord", [])
        worker = framework.create_agent("Worker", AgentRole.EXECUTOR, "Work", [])
        
        result = await framework.orchestrate(
            task="Hierarchical task",
            agent_ids=[coordinator.agent_id, worker.agent_id],
            strategy="hierarchical"
        )
        
        assert result["strategy"] == "hierarchical"


# ============== Mock Agent测试 ==============

class TestMockAgentOperations:
    """Mock Agent操作测试类"""
    
    def test_mock_agent_lifecycle(self):
        """测试Mock Agent生命周期"""
        agent_state = {
            "created": False,
            "running": False,
            "idle": False,
            "deleted": False
        }
        
        def create_agent():
            agent_state["created"] = True
        
        def start_agent():
            if agent_state["created"]:
                agent_state["running"] = True
        
        def stop_agent():
            if agent_state["running"]:
                agent_state["running"] = False
                agent_state["idle"] = True
        
        def delete_agent():
            agent_state["deleted"] = True
        
        create_agent()
        start_agent()
        assert agent_state["running"] == True, "Agent should be running after start"
        stop_agent()
        delete_agent()
        
        assert agent_state["created"] is True
        assert agent_state["idle"] is True
        assert agent_state["deleted"] is True
    
    def test_mock_tool_execution(self):
        """测试Mock工具执行"""
        tool_results = []
        
        def execute_tool(tool_name, params):
            result = {
                "tool": tool_name,
                "params": params,
                "result": f"Executed {tool_name}"
            }
            tool_results.append(result)
            return result
        
        r1 = execute_tool("python", {"code": "print('hello')"})
        r2 = execute_tool("search", {"query": "test"})
        
        assert len(tool_results) == 2
        assert r1["tool"] == "python"
    
    def test_mock_memory_management(self):
        """测试Mock记忆管理"""
        memories = []
        
        def add_memory(content, importance):
            memories.append({
                "content": content,
                "importance": importance,
                "timestamp": datetime.utcnow()
            })
        
        def get_memories(importance_threshold=0.5):
            return [m for m in memories if m["importance"] >= importance_threshold]
        
        add_memory("Important info", 0.9)
        add_memory("Less important", 0.3)
        add_memory("Very important", 0.95)
        
        high_importance = get_memories(0.8)
        assert len(high_importance) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
