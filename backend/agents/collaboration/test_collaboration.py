"""
Agent协作网络测试脚本
"""

import asyncio
import sys
import os

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_collaboration():
    """测试协作功能"""
    print("=" * 50)
    print("Agent Collaboration Network Test")
    print("=" * 50)
    
    try:
        # 导入协作模块
        from agents.collaboration import (
            get_orchestrator,
            create_task_decomposer,
            create_consensus_manager
        )
        from agents.collaboration.models import (
            CollaborationMode,
            AgentRole
        )
        
        # 初始化编排器
        orchestrator = get_orchestrator()
        await orchestrator.initialize()
        print("[OK] Orchestrator initialized")
        
        # 1. 创建协作会话
        print("\n--- Test 1: Create Session ---")
        session_result = await orchestrator.create_collaboration_session(
            name="research_team",
            description="Research Team Collaboration",
            mode="hierarchical",
            agent_ids=["researcher", "analyst", "writer"]
        )
        print(f"Session created: {session_result}")
        
        session_id = session_result.get("session_id")
        if not session_id:
            print("[FAIL] Failed to create session")
            return False
        
        # 2. Agent加入会话
        print("\n--- Test 2: Agent Join ---")
        join_result = await orchestrator.join_session(
            session_id=session_id,
            agent_id="researcher",
            role="worker",
            metadata={"name": "Researcher Agent", "capabilities": ["web_search", "data_analysis"]}
        )
        print(f"Researcher joined: {join_result}")
        
        join_result = await orchestrator.join_session(
            session_id=session_id,
            agent_id="analyst",
            role="worker",
            metadata={"name": "Analyst Agent", "capabilities": ["pattern_recognition", "statistics"]}
        )
        print(f"Analyst joined: {join_result}")
        
        join_result = await orchestrator.join_session(
            session_id=session_id,
            agent_id="writer",
            role="worker",
            metadata={"name": "Writer Agent", "capabilities": ["content_generation", "summarization"]}
        )
        print(f"Writer joined: {join_result}")
        
        # 3. 分配任务
        print("\n--- Test 3: Assign Tasks ---")
        task1 = await orchestrator.assign_task(
            session_id=session_id,
            task_name="收集行业数据",
            description="从多个来源收集相关行业数据",
            assigned_agent="researcher",
            priority=1
        )
        print(f"Task 1 assigned: {task1}")
        
        task2 = await orchestrator.assign_task(
            session_id=session_id,
            task_name="数据分析",
            description="分析收集的数据并识别模式",
            assigned_agent="analyst",
            priority=2,
            dependencies=[task1.get("task_id")]
        )
        print(f"Task 2 assigned: {task2}")
        
        task3 = await orchestrator.assign_task(
            session_id=session_id,
            task_name="生成报告",
            description="基于分析结果生成报告",
            assigned_agent="writer",
            priority=3,
            dependencies=[task2.get("task_id")]
        )
        print(f"Task 3 assigned: {task3}")
        
        # 4. 获取会话详情
        print("\n--- Test 4: Get Session Details ---")
        session = await orchestrator.session_manager.get_session(session_id)
        if session:
            print(f"Session: {session.name}")
            print(f"Mode: {session.mode.value}")
            print(f"Status: {session.status.value}")
            print(f"Agents: {len(session.agents)}")
            print(f"Tasks: {len(session.tasks)}")
        
        # 5. 测试任务分解
        print("\n--- Test 5: Task Decomposition ---")
        decomposer = create_task_decomposer()
        from agents.collaboration.models import TaskInput
        
        task = TaskInput(
            name="研究AI发展趋势",
            description="深入研究AI技术的发展趋势和未来方向",
            payload={},
            priority=0,
            dependencies=[]
        )
        
        subtasks = await decomposer.decompose(task, strategy="hierarchical")
        print(f"Decomposed into {len(subtasks)} subtasks:")
        for i, subtask in enumerate(subtasks):
            print(f"  {i+1}. {subtask.name}")
        
        # 6. 测试共识机制
        print("\n--- Test 6: Consensus Mechanism ---")
        consensus_manager = create_consensus_manager()
        
        proposal = await consensus_manager.create_proposal(
            session_id=session_id,
            proposer_id="researcher",
            proposal_type="task_priority",
            content={"task_id": task1.get("task_id"), "new_priority": 5},
            participants=["researcher", "analyst", "writer"],
            threshold=0.5
        )
        print(f"Proposal created: {proposal.proposal_id}")
        
        # 提交投票
        await consensus_manager.submit_vote(proposal.proposal_id, "researcher", True, reason="任务紧急")
        await consensus_manager.submit_vote(proposal.proposal_id, "analyst", True, reason="同意")
        await consensus_manager.submit_vote(proposal.proposal_id, "writer", True, reason="合理")
        
        # 检查共识
        consensus_reached = await consensus_manager.check_consensus(proposal.proposal_id)
        print(f"Consensus reached: {consensus_reached}")
        
        # 7. 获取活跃会话列表
        print("\n--- Test 7: List Active Sessions ---")
        active_sessions = await orchestrator.list_active_sessions()
        print(f"Active sessions: {len(active_sessions)}")
        for s in active_sessions:
            print(f"  - {s['name']} ({s['mode']})")
        
        # 8. 获取进度
        print("\n--- Test 8: Get Progress ---")
        progress = await orchestrator.get_session_progress(session_id)
        if progress:
            print(f"Progress: {progress.completed_tasks}/{progress.total_tasks} tasks completed")
            print(f"Percentage: {progress.progress_percentage:.1f}%")
        
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_engine():
    """测试工作流引擎"""
    print("\n" + "=" * 50)
    print("Workflow Engine Test")
    print("=" * 50)
    
    try:
        from agents.collaboration import (
            get_workflow_engine,
            create_task_decomposer
        )
        from agents.collaboration.models import (
            WorkflowDefinition,
            TaskInput,
            AgentInfo,
            CollaborationMode,
            AgentRole
        )
        
        # 初始化引擎
        engine = get_workflow_engine()
        await engine.initialize()
        print("[OK] Workflow Engine initialized")
        
        # 创建工作流
        workflow = await engine.create_workflow(
            name="test_workflow",
            description="Test Workflow",
            mode=CollaborationMode.PARALLEL,
            agent_ids=["agent1", "agent2", "agent3"]
        )
        print(f"[OK] Workflow created: {workflow.workflow_id}")
        
        # 创建任务
        tasks = [
            TaskInput(
                name="Task A",
                description="First task",
                payload={"data": "test_a"},
                priority=1
            ),
            TaskInput(
                name="Task B",
                description="Second task", 
                payload={"data": "test_b"},
                priority=2
            ),
            TaskInput(
                name="Task C",
                description="Third task",
                payload={"data": "test_c"},
                priority=3
            )
        ]
        
        # 创建Agent映射
        agent_map = {
            "agent1": AgentInfo(agent_id="agent1", name="Agent 1", role=AgentRole.WORKER),
            "agent2": AgentInfo(agent_id="agent2", name="Agent 2", role=AgentRole.WORKER),
            "agent3": AgentInfo(agent_id="agent3", name="Agent 3", role=AgentRole.WORKER)
        }
        
        # 执行工作流
        print("\n[OK] Executing workflow...")
        executor = await engine.execute(workflow, tasks, agent_map)
        
        # 获取状态
        status = await engine.get_status(workflow.workflow_id)
        print(f"Workflow status: {status}")
        
        print("\n" + "=" * 50)
        print("Workflow Engine Test Passed!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("\nStarting Agent Collaboration Network Tests...\n")
    
    # 运行协作测试
    success1 = await test_collaboration()
    
    # 运行工作流引擎测试
    success2 = await test_workflow_engine()
    
    if success1 and success2:
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        return 0
    else:
        print("\n" + "=" * 50)
        print("SOME TESTS FAILED!")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
