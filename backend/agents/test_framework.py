"""
Agent框架测试脚本
验证Agent编排框架的基本功能
"""

import sys
import os

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import Agent, ReActReasoningEngine, ToolRegistry, MemoryManager
from backend.agents.tools import builtin_tools


def test_agent_creation():
    """测试Agent创建"""
    print("=" * 50)
    print("测试1: Agent创建")
    print("=" * 50)
    
    agent = Agent(
        name="TestAgent",
        description="一个测试Agent"
    )
    
    print(f"Agent ID: {agent.id}")
    print(f"Agent Name: {agent.name}")
    print(f"Tools count: {len(agent.get_tools())}")
    print(f"Agent created successfully!")
    print()


def test_tool_registry():
    """测试工具注册表"""
    print("=" * 50)
    print("测试2: 工具注册表")
    print("=" * 50)
    
    registry = ToolRegistry()
    
    # 注册自定义工具
    def my_tool(query: str, **kwargs):
        return {"result": f"Processed: {query}"}
    
    registry.register({
        "name": "my_custom_tool",
        "func": my_tool,
        "description": "My custom tool",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    })
    
    # 获取工具列表
    tools = registry.list_tools()
    print(f"Registered tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    print()


def test_memory_manager():
    """测试记忆管理器"""
    print("=" * 50)
    print("测试3: 记忆管理器")
    print("=" * 50)
    
    memory = MemoryManager()
    
    # 添加记忆
    memory.add_memory({"content": "Task started"}, "task")
    memory.add_memory({"content": "Thinking about the problem"}, "reasoning")
    memory.add_memory({"content": "Found solution"}, "reasoning")
    
    # 获取统计
    stats = memory.get_statistics()
    print(f"Memory statistics: {stats}")
    
    # 获取记忆
    memories = memory.get_memories()
    print(f"Total memories: {len(memories)}")
    
    # 搜索记忆
    results = memory.search_memories("solution")
    print(f"Search results for 'solution': {len(results)}")
    print()


def test_builtin_tools():
    """测试内置工具"""
    print("=" * 50)
    print("测试4: 内置工具")
    print("=" * 50)
    
    from backend.agents.tools.builtin_tools import get_all_tools
    tools = get_all_tools()
    print(f"Total builtin tools: {len(tools)}")
    
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    print()


def test_agent_execution():
    """测试Agent执行"""
    print("=" * 50)
    print("测试5: Agent执行")
    print("=" * 50)
    
    agent = Agent(
        name="ExecutionTestAgent",
        description="测试执行功能"
    )
    
    # 执行简单任务
    result = agent.execute("What is 2 + 2?")
    
    print(f"Task: {result['task']}")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Summary: {result['summary']}")
    print()


def test_all():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("AI Platform v3 - Agent编排框架测试")
    print("=" * 60 + "\n")
    
    try:
        test_agent_creation()
        test_tool_registry()
        test_memory_manager()
        test_builtin_tools()
        test_agent_execution()
        
        print("=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_all()
