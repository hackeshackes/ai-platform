"""
Agent协作平台测试脚本
"""

from backend.agents.collab.manager import collab_manager, TaskStatus, SessionStatus
from backend.agents.langchain.adapter import langchain_adapter
from backend.agents.llamaindex.adapter import llamaindex_adapter


def test_collab_manager():
    """测试协作管理器"""
    print("=== 测试协作管理器 ===")
    
    # 创建会话
    session = collab_manager.create_session(
        name="测试协作会话",
        description="测试多Agent协作"
    )
    print(f"✓ 创建会话: {session.id}")
    
    # 添加任务
    task1 = collab_manager.create_task(
        session_id=session.id,
        name="数据收集",
        description="收集用户数据",
        agent_type="langchain",
        input_data={"query": "收集数据"}
    )
    print(f"✓ 添加任务1: {task1.id}")
    
    task2 = collab_manager.create_task(
        session_id=session.id,
        name="数据分析",
        description="分析收集的数据",
        agent_type="llamaindex",
        input_data={"query": "分析数据"},
        dependencies=[task1.id]
    )
    print(f"✓ 添加任务2: {task2.id}")
    
    # 列出任务
    tasks = session.tasks
    print(f"✓ 会话中有 {len(tasks)} 个任务")
    
    # 导出会话
    export = collab_manager.export_session(session.id)
    print(f"✓ 会话导出成功，包含 {len(export['tasks'])} 个任务")
    
    return True


def test_langchain_adapter():
    """测试LangChain适配器"""
    print("\n=== 测试LangChain适配器 ===")
    
    # 导入Agent
    result = langchain_adapter.import_agent({
        "name": "测试LangChain Agent",
        "description": "用于测试",
        "agent_class": "zero_shot_react",
        "llm_type": "openai",
        "llm_config": {"model": "gpt-3.5-turbo"},
        "tools": ["serpapi"]
    })
    print(f"✓ 导入Agent: {result.get('agent_id')}")
    
    # 列出Agents
    agents = langchain_adapter.list_agents()
    print(f"✓ LangChain Agents数量: {len(agents)}")
    
    return True


def test_llamaindex_adapter():
    """测试LlamaIndex适配器"""
    print("\n=== 测试LlamaIndex适配器 ===")
    
    # 导入Agent
    result = llamaindex_adapter.import_agent({
        "name": "测试LlamaIndex Agent",
        "description": "用于测试",
        "agent_type": "query_engine",
        "llm_type": "openai",
        "llm_config": {"model": "gpt-3.5-turbo"},
        "index_type": "vector"
    })
    print(f"✓ 导入Agent: {result.get('agent_id')}")
    
    # 列出Agents
    agents = llamaindex_adapter.list_agents()
    print(f"✓ LlamaIndex Agents数量: {len(agents)}")
    
    return True


def test_delete():
    """测试清理"""
    print("\n=== 测试清理 ===")
    
    sessions = collab_manager.list_sessions()
    for s in sessions:
        collab_manager.delete_session(s.id)
    print(f"✓ 清理了 {len(sessions)} 个测试会话")
    
    return True


if __name__ == "__main__":
    print("开始Agent协作平台测试...\n")
    
    try:
        test_collab_manager()
        test_langchain_adapter()
        test_llamaindex_adapter()
        test_delete()
        
        print("\n✅ 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
