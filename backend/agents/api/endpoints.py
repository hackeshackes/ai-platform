"""
Agent API端点
提供RESTful API接口
"""

from typing import Any, Dict, List
from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

# 全局Agent管理器
agent_manager = {}


def create_agent_blueprint():
    """创建Agent蓝图"""
    bp = Blueprint('agents', __name__, url_prefix='/api/v1/agents')
    
    # ============ Agent管理端点 ============
    
    @bp.route('/agents', methods=['POST'])
    def create_agent():
        """
        创建新Agent
        
        Request Body:
            {
                "name": "Agent Name",
                "description": "Agent Description",
                "max_steps": 10,
                "llm_provider": null
            }
        """
        try:
            data = request.get_json()
            
            if not data or 'name' not in data:
                return jsonify({"error": "Agent name is required"}), 400
            
            # 导入Agent类
            from ..core.agent import Agent
            
            agent = Agent(
                name=data['name'],
                description=data.get('description', ''),
                max_steps=data.get('max_steps', 10)
            )
            
            # 保存到管理器
            agent_manager[agent.id] = agent
            
            logger.info(f"Agent created: {agent.id} - {agent.name}")
            
            return jsonify({
                "success": True,
                "agent_id": agent.id,
                "agent": agent.to_dict()
            }), 201
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/agents', methods=['GET'])
    def list_agents():
        """列出所有Agent"""
        try:
            agents = [
                {
                    "id": agent_id,
                    **agent.to_dict()
                }
                for agent_id, agent in agent_manager.items()
            ]
            
            return jsonify({
                "success": True,
                "count": len(agents),
                "agents": agents
            })
            
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/agents/<agent_id>', methods=['GET'])
    def get_agent(agent_id):
        """获取Agent信息"""
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            return jsonify({
                "success": True,
                "agent": agent.to_dict(),
                "state": agent.get_state()
            })
            
        except Exception as e:
            logger.error(f"Error getting agent: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/agents/<agent_id>', methods=['DELETE'])
    def delete_agent(agent_id):
        """删除Agent"""
        try:
            if agent_id not in agent_manager:
                return jsonify({"error": "Agent not found"}), 404
            
            del agent_manager[agent_id]
            
            return jsonify({
                "success": True,
                "message": f"Agent {agent_id} deleted"
            })
            
        except Exception as e:
            logger.error(f"Error deleting agent: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ============ Agent执行端点 ============
    
    @bp.route('/<agent_id>/execute', methods=['POST'])
    def execute_task(agent_id):
        """
        执行Agent任务
        
        Path Parameters:
            agent_id: Agent ID
            
        Request Body:
            {
                "task": "Task description",
                "context": {}
            }
        """
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            data = request.get_json()
            if not data or 'task' not in data:
                return jsonify({"error": "Task is required"}), 400
            
            task = data['task']
            context = data.get('context', {})
            
            result = agent.execute(task, context)
            
            return jsonify({
                "success": True,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ============ 记忆管理端点 ============
    
    @bp.route('/<agent_id>/memory', methods=['GET'])
    def get_memory(agent_id):
        """
        获取Agent记忆
        
        Query Parameters:
            type: 记忆类型过滤
            limit: 返回数量限制
        """
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            memory_type = request.args.get('type')
            limit = request.args.get('limit', 50, type=int)
            
            memories = agent.get_memory(memory_type)
            
            # 应用限制
            if limit > 0 and len(memories) > limit:
                memories = memories[-limit:]
            
            return jsonify({
                "success": True,
                "count": len(memories),
                "memories": memories,
                "statistics": agent.memory_manager.get_statistics()
            })
            
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/<agent_id>/memory', methods=['DELETE'])
    def clear_memory(agent_id):
        """清空Agent记忆"""
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            memory_type = request.args.get('type')
            agent.clear_memory(memory_type)
            
            return jsonify({
                "success": True,
                "message": f"Memory cleared: {memory_type or 'all'}"
            })
            
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/<agent_id>/memory/search', methods=['GET'])
    def search_memory(agent_id):
        """
        搜索Agent记忆
        
        Query Parameters:
            query: 搜索关键词
            limit: 返回数量限制
        """
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            query = request.args.get('query')
            if not query:
                return jsonify({"error": "Search query is required"}), 400
            
            limit = request.args.get('limit', 10, type=int)
            results = agent.memory_manager.search_memories(query, limit)
            
            return jsonify({
                "success": True,
                "query": query,
                "count": len(results),
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ============ 工具管理端点 ============
    
    @bp.route('/tools/register', methods=['POST'])
    def register_tool():
        """
        注册自定义工具
        
        Request Body:
            {
                "agent_id": "agent-id",
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...}
                }
            }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "Request body is required"}), 400
            
            agent_id = data.get('agent_id')
            name = data.get('name')
            description = data.get('description', '')
            parameters = data.get('parameters', {})
            
            if not agent_id or not name:
                return jsonify({"error": "agent_id and name are required"}), 400
            
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            # 简化实现：注册占位函数
            def placeholder_func(**kwargs):
                return {"message": f"Tool {name} executed", "args": kwargs}
            
            agent.register_tool(name, placeholder_func, description, parameters)
            
            return jsonify({
                "success": True,
                "message": f"Tool '{name}' registered",
                "agent_id": agent_id
            })
            
        except Exception as e:
            logger.error(f"Error registering tool: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/tools/list', methods=['GET'])
    def list_tools():
        """
        列出Agent可用工具
        
        Query Parameters:
            agent_id: Agent ID
            category: 工具类别过滤
        """
        try:
            agent_id = request.args.get('agent_id')
            category = request.args.get('category')
            
            if agent_id:
                agent = agent_manager.get(agent_id)
                if not agent:
                    return jsonify({"error": "Agent not found"}), 404
                tools = agent.get_tools()
            else:
                # 返回所有Agent的工具（简化）
                tools = []
                for agent in agent_manager.values():
                    tools.extend(agent.get_tools())
            
            if category:
                tools = [t for t in tools if t.get('category') == category]
            
            return jsonify({
                "success": True,
                "count": len(tools),
                "tools": tools
            })
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ============ 调试器端点 ============
    
    @bp.route('/<agent_id>/debug', methods=['GET'])
    def get_debug_info(agent_id):
        """获取Agent调试信息"""
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            return jsonify({
                "success": True,
                "debug": {
                    "state": agent.get_state(),
                    "tools": agent.get_tools(),
                    "memory_stats": agent.memory_manager.get_statistics(),
                    "execution_history": agent.execution_history[-10:]  # 最近10次执行
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting debug info: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/<agent_id>/trace', methods=['GET'])
    def get_execution_trace(agent_id):
        """获取Agent执行轨迹"""
        try:
            agent = agent_manager.get(agent_id)
            if not agent:
                return jsonify({"error": "Agent not found"}), 404
            
            traces = []
            for entry in agent.execution_history:
                traces.append({
                    "task": entry["task"],
                    "timestamp": entry["timestamp"],
                    "result_summary": entry["result"].get("summary", "")
                })
            
            return jsonify({
                "success": True,
                "count": len(traces),
                "traces": traces
            })
            
        except Exception as e:
            logger.error(f"Error getting trace: {e}")
            return jsonify({"error": str(e)}), 500
    
    return bp


def get_agent_blueprint():
    """获取Agent蓝图（用于Flask注册）"""
    return create_agent_blueprint()


# 多Agent协作端点蓝图
def create_collaboration_blueprint():
    """创建多Agent协作蓝图"""
    bp = Blueprint('collaboration', __name__, url_prefix='/api/v1/agents/collaboration')
    
    @bp.route('/create', methods=['POST'])
    def create_team():
        """
        创建Agent团队
        
        Request Body:
            {
                "name": "Team Name",
                "agents": ["agent-id-1", "agent-id-2"]
            }
        """
        try:
            data = request.get_json()
            
            team_id = f"team_{len(agent_manager)}"
            
            # 简化实现
            return jsonify({
                "success": True,
                "team_id": team_id,
                "message": "Team creation placeholder"
            })
            
        except Exception as e:
            logger.error(f"Error creating team: {e}")
            return jsonify({"error": str(e)}), 500
    
    @bp.route('/<team_id>/execute', methods=['POST'])
    def team_execute(team_id):
        """团队协作执行"""
        try:
            return jsonify({
                "success": True,
                "team_id": team_id,
                "message": "Collaboration execution placeholder"
            })
        except Exception as e:
            logger.error(f"Error in team execution: {e}")
            return jsonify({"error": str(e)}), 500
    
    return bp
