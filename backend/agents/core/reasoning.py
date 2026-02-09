"""
ReAct推理引擎
实现ReAct (Reasoning + Acting) 范式
"""

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import logging

from ..tools.registry import ToolRegistry
from ..memory.memory import MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActReasoningEngine:
    """
    ReAct推理引擎
    
    实现了思考(Thought) -> 行动(Action) -> 观察(Observation)的循环
    """
    
    def __init__(
        self,
        llm_provider: Optional[Callable] = None,
        max_steps: int = 10,
        system_prompt: Optional[str] = None
    ):
        self.llm_provider = llm_provider
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        return """You are a ReAct agent that can think and act to solve problems.
You follow the pattern: Thought -> Action -> Observation -> ...
Think step by step and use tools when needed.
"""
    
    def execute(
        self,
        task: str,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager
    ) -> Dict[str, Any]:
        """
        执行ReAct推理循环
        
        Args:
            task: 用户任务
            tool_registry: 工具注册表
            memory_manager: 记忆管理器
            
        Returns:
            执行结果
        """
        # 初始化执行轨迹
        trajectory = []
        current_state = {
            "task": task,
            "step": 0,
            "thought": "",
            "action": None,
            "observation": None,
            "finished": False
        }
        
        logger.info(f"Starting ReAct execution for task: {task}")
        
        # ReAct循环
        for step in range(self.max_steps):
            current_state["step"] = step + 1
            
            # 1. 思考阶段 (Thought)
            thought = self._think(
                task=task,
                trajectory=trajectory,
                available_tools=tool_registry.list_tools(),
                memory=memory_manager.get_memories()
            )
            current_state["thought"] = thought
            
            # 检查是否应该停止
            if self._should_finish(thought):
                current_state["finished"] = True
                current_state["final_answer"] = thought
                break
            
            # 2. 决定行动 (Action)
            action = self._decide_action(
                thought=thought,
                available_tools=tool_registry.list_tools()
            )
            
            if action is None:
                # 没有可用行动，直接完成
                current_state["finished"] = True
                current_state["final_answer"] = thought
                break
            
            current_state["action"] = action
            
            # 3. 执行行动 (Execute Action)
            observation = self._execute_action(action, tool_registry)
            current_state["observation"] = observation
            
            # 记录到轨迹
            trajectory.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "observation": observation,
                "timestamp": datetime.now().isoformat()
            })
            
            # 更新记忆
            memory_manager.add_memory({
                "type": "reasoning_step",
                "step": step + 1,
                "thought": thought,
                "action": action.get("name") if action else None,
                "observation": str(observation)[:500],
                "timestamp": datetime.now().isoformat()
            })
        
        # 生成最终结果
        return self._generate_result(task, trajectory, current_state, memory_manager)
    
    def _think(
        self,
        task: str,
        trajectory: List[Dict],
        available_tools: List[Dict],
        memory: List[Dict]
    ) -> str:
        """
        思考阶段 - 分析任务并决定下一步
        如果有LLM提供商，使用LLM生成思考
        否则使用启发式方法
        """
        if self.llm_provider:
            prompt = self._build_thinking_prompt(
                task, trajectory, available_tools, memory
            )
            try:
                response = self.llm_provider(prompt)
                return response
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
        
        # 启发式思考
        return f"Task: {task}\nPrevious steps: {len(trajectory)}\nAvailable tools: {len(available_tools)}"
    
    def _build_thinking_prompt(
        self,
        task: str,
        trajectory: List[Dict],
        available_tools: List[Dict],
        memory: List[Dict]
    ) -> str:
        """构建思考提示"""
        # 简化版本：构建一个提示，让LLM决定下一步
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}" 
            for t in available_tools
        ])
        
        history = ""
        if trajectory:
            recent_steps = trajectory[-3:]  # 只取最近3步
            history = "Recent steps:\n"
            for step in recent_steps:
                history += f"- Thought: {step['thought']}\n"
                if step['action']:
                    history += f"  Action: {step['action']}\n"
                history += f"  Observation: {step['observation']}\n"
        
        prompt = f"""{self.system_prompt}

Task: {task}

Available tools:
{tools_desc}

{history}

What should I do next? Think step by step and decide on an action or provide the final answer.
"""
        return prompt
    
    def _decide_action(
        self,
        thought: str,
        available_tools: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        从思考中决定行动
        尝试从LLM响应中解析出行动参数
        """
        # 简单解析：检查thought中是否提到工具使用
        # 实际实现中应该让LLM输出结构化的JSON
        if "FINAL_ANSWER:" in thought or "FINAL ANSWER:" in thought.upper():
            return None
        
        # 查找被提及的工具
        for tool in available_tools:
            tool_name_lower = tool["name"].lower()
            if tool_name_lower in thought.lower():
                # 提取参数（简化版）
                params = self._extract_parameters(thought, tool)
                return {
                    "name": tool["name"],
                    "parameters": params
                }
        
        return None
    
    def _extract_parameters(self, thought: str, tool: Dict) -> Dict:
        """从思考中提取工具参数"""
        params = {}
        
        # 检查是否有参数定义
        tool_params = tool.get("parameters", {})
        
        # 简化实现：尝试从thought中查找常见的参数模式
        # 实际应该让LLM输出结构化参数
        if "query" in tool_params.get("properties", {}):
            # 尝试提取查询内容
            if '"' in thought or "'" in thought:
                import re
                quotes = re.findall(r'["\']([^"\']+)["\']', thought)
                if quotes:
                    params["query"] = quotes[0]
        
        return params
    
    def _execute_action(
        self,
        action: Dict[str, Any],
        tool_registry: ToolRegistry
    ) -> Any:
        """执行工具调用"""
        tool_name = action.get("name")
        params = action.get("parameters", {})
        
        logger.info(f"Executing action: {tool_name} with params: {params}")
        
        try:
            result = tool_registry.call_tool(tool_name, **params)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": str(e)}
    
    def _should_finish(self, thought: str) -> bool:
        """判断是否应该结束"""
        finish_patterns = [
            "final_answer:",
            "final answer:",
            "FINAL_ANSWER:",
            "FINAL ANSWER:",
            "答案是:",
            "最终答案是:"
        ]
        
        thought_lower = thought.lower()
        for pattern in finish_patterns:
            if pattern.lower() in thought_lower:
                return True
        
        return False
    
    def _generate_result(
        self,
        task: str,
        trajectory: List[Dict],
        final_state: Dict,
        memory_manager: MemoryManager
    ) -> Dict[str, Any]:
        """生成最终结果"""
        # 提取最终答案
        final_answer = final_state.get("final_answer", "")
        if not final_answer and trajectory:
            last_step = trajectory[-1]
            if last_step.get("observation"):
                final_answer = str(last_step.get("observation"))
        
        # 计算执行统计
        total_thoughts = len(trajectory)
        total_actions = sum(1 for t in trajectory if t.get("action"))
        tools_used = list(set(
            t.get("action", {}).get("name") 
            for t in trajectory 
            if t.get("action")
        ))
        
        return {
            "task": task,
            "success": final_state.get("finished", False) or len(trajectory) > 0,
            "steps": total_thoughts,
            "tools_used": tools_used,
            "trajectory": trajectory,
            "summary": final_answer,
            "memory_summary": {
                "total_memories": len(memory_manager.get_memories()),
                "reasoning_steps": len(memory_manager.get_memories("reasoning_step"))
            }
        }
