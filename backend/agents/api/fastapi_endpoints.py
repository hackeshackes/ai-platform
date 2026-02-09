"""
Agent API端点 - FastAPI版本
提供RESTful API接口
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
import logging

logger = logging.getLogger(__name__)

# 全局Agent管理器
agent_manager = {}


def get_agent(agent_id: str):
    """依赖注入：获取Agent"""
    agent = agent_manager.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def create_agent_router():
    """创建Agent路由器"""
    router = APIRouter(tags=["Agents"])
    
    # ============ Agent管理端点 ============
    
    @router.post("/agents", status_code=201)
    async def create_agent(
        name: str,
        description: str = "",
        max_steps: int = 10
    ):
        """
        创建新Agent
        
        - **name**: Agent名称 (必需)
        - **description**: Agent描述
        - **max_steps**: 最大执行步数
        """
        try:
            from ..core.agent import Agent
            
            agent = Agent(
                name=name,
                description=description,
                max_steps=max_steps
            )
            
            # 保存到管理器
            agent_manager[agent.id] = agent
            
            logger.info(f"Agent created: {agent.id} - {agent.name}")
            
            return {
                "success": True,
                "agent_id": agent.id,
                "agent": agent.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/agents")
    async def list_agents():
        """列出所有Agent"""
        try:
            agents = [
                {
                    "agent_id": agent_id,
                    **agent.to_dict()
                }
                for agent_id, agent in agent_manager.items()
            ]
            
            return {
                "success": True,
                "count": len(agents),
                "agents": agents
            }
            
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/agents/{agent_id}")
    async def get_agent_info(agent_id: str):
        """获取Agent信息"""
        agent = get_agent(agent_id)
        
        return {
            "success": True,
            "agent": agent.to_dict(),
            "state": agent.get_state()
        }
    
    @router.delete("/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """删除Agent"""
        agent = get_agent(agent_id)
        del agent_manager[agent_id]
        
        return {
            "success": True,
            "message": f"Agent {agent_id} deleted"
        }
    
    # ============ Agent执行端点 ============
    
    @router.post("/{agent_id}/execute")
    async def execute_task(
        agent_id: str,
        task: str,
        context: Optional[Dict] = None
    ):
        """
        执行Agent任务
        
        - **agent_id**: Agent ID
        - **task**: 任务描述
        - **context**: 额外上下文
        """
        agent = get_agent(agent_id)
        
        if not task:
            raise HTTPException(status_code=400, detail="Task is required")
        
        result = agent.execute(task, context or {})
        
        return {
            "success": True,
            "result": result
        }
    
    # ============ 记忆管理端点 ============
    
    @router.get("/{agent_id}/memory")
    async def get_memory(
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50
    ):
        """获取Agent记忆"""
        agent = get_agent(agent_id)
        
        memories = agent.get_memory(memory_type)
        
        # 应用限制
        if limit > 0 and len(memories) > limit:
            memories = memories[-limit:]
        
        return {
            "success": True,
            "count": len(memories),
            "memories": memories,
            "statistics": agent.memory_manager.get_statistics()
        }
    
    @router.delete("/{agent_id}/memory")
    async def clear_memory(agent_id: str, memory_type: Optional[str] = None):
        """清空Agent记忆"""
        agent = get_agent(agent_id)
        agent.clear_memory(memory_type)
        
        return {
            "success": True,
            "message": f"Memory cleared: {memory_type or 'all'}"
        }
    
    @router.get("/{agent_id}/memory/search")
    async def search_memory(
        agent_id: str,
        query: str,
        limit: int = 10
    ):
        """搜索Agent记忆"""
        agent = get_agent(agent_id)
        
        results = agent.memory_manager.search_memories(query, limit)
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
    
    # ============ 工具管理端点 ============
    
    @router.post("/tools/register")
    async def register_tool(
        agent_id: str,
        name: str,
        description: str = "",
        parameters: Optional[Dict] = None
    ):
        """
        注册自定义工具
        
        - **agent_id**: Agent ID
        - **name**: 工具名称
        - **description**: 工具描述
        - **parameters**: 参数定义
        """
        agent = get_agent(agent_id)
        
        # 简化实现：注册占位函数
        def placeholder_func(**kwargs):
            return {"message": f"Tool {name} executed", "args": kwargs}
        
        agent.register_tool(name, placeholder_func, description, parameters or {})
        
        return {
            "success": True,
            "message": f"Tool '{name}' registered",
            "agent_id": agent_id
        }
    
    @router.get("/tools/list")
    async def list_tools(agent_id: Optional[str] = None, category: Optional[str] = None):
        """列出Agent可用工具"""
        if agent_id:
            agent = get_agent(agent_id)
            tools = agent.get_tools()
        else:
            # 返回所有Agent的工具（简化）
            tools = []
            for agent in agent_manager.values():
                tools.extend(agent.get_tools())
        
        if category:
            tools = [t for t in tools if t.get('category') == category]
        
        return {
            "success": True,
            "count": len(tools),
            "tools": tools
        }
    
    # ============ 调试器端点 ============
    
    @router.get("/{agent_id}/debug")
    async def get_debug_info(agent_id: str):
        """获取Agent调试信息"""
        agent = get_agent(agent_id)
        
        return {
            "success": True,
            "debug": {
                "state": agent.get_state(),
                "tools": agent.get_tools(),
                "memory_stats": agent.memory_manager.get_statistics(),
                "execution_history": agent.execution_history[-10:]
            }
        }
    
    @router.get("/{agent_id}/trace")
    async def get_execution_trace(agent_id: str):
        """获取Agent执行轨迹"""
        agent = get_agent(agent_id)
        
        traces = []
        for entry in agent.execution_history:
            traces.append({
                "task": entry["task"],
                "timestamp": entry["timestamp"],
                "result_summary": entry["result"].get("summary", "")
            })
        
        return {
            "success": True,
            "count": len(traces),
            "traces": traces
        }
    
    return router


# 多Agent协作端点
def create_collaboration_router():
    """创建多Agent协作路由器"""
    router = APIRouter(tags=["Collaboration"])
    
    @router.post("/create")
    async def create_team(
        name: str,
        agent_ids: List[str]
    ):
        """
        创建Agent团队
        
        - **name**: 团队名称
        - **agent_ids**: Agent ID列表
        """
        team_id = f"team_{len(agent_ids)}"
        
        return {
            "success": True,
            "team_id": team_id,
            "name": name,
            "agents": agent_ids
        }
    
    @router.post("/{team_id}/execute")
    async def team_execute(
        team_id: str,
        task: str,
        strategy: str = "sequential"
    ):
        """团队协作执行"""
        return {
            "success": True,
            "team_id": team_id,
            "task": task,
            "strategy": strategy,
            "message": "Collaboration execution placeholder"
        }
    
    return router


# 创建路由器实例
agent_router = create_agent_router()
collaboration_router = create_collaboration_router()
