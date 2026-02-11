"""
Data Models - Adaptive Learning
自适应学习核心数据模型
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import uuid


class IntentType(Enum):
    """意图类型枚举"""
    QUERY = "query"
    ACTION = "action"
    CREATION = "creation"
    ANALYSIS = "analysis"
    LEARNING = "learning"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """动作类型枚举"""
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    RESPONSE = "response"
    ERROR = "error"


@dataclass
class Entity:
    """实体类"""
    name: str
    value: Any
    type: str
    confidence: float = 1.0


@dataclass
class ContextInfo:
    """上下文信息"""
    session_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStep:
    """执行步骤"""
    step_number: int
    action_type: ActionType
    action_name: str
    input_params: Dict[str, Any]
    output: Optional[Any] = None
    duration_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class Pattern:
    """
    交互模式类
    核心数据结构，用于存储从交互中提取的模式
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent: IntentType = IntentType.UNKNOWN
    intent_confidence: float = 0.0
    entities: List[Entity] = field(default_factory=list)
    context: Optional[ContextInfo] = None
    execution_path: List[ExecutionStep] = field(default_factory=list)
    success: bool = True
    reward: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "intent": self.intent.value,
            "intent_confidence": self.intent_confidence,
            "entities": [{"name": e.name, "value": e.value, "type": e.type} for e in self.entities],
            "context": {
                "session_id": self.context.session_id,
                "user_id": self.context.user_id,
                "timestamp": self.context.timestamp.isoformat() if self.context else None
            } if self.context else None,
            "execution_path": [
                {
                    "step_number": s.step_number,
                    "action_type": s.action_type.value,
                    "action_name": s.action_name,
                    "success": s.success
                } for s in self.execution_path
            ],
            "success": self.success,
            "reward": self.reward,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Interaction:
    """交互数据类"""
    text: str
    context: Dict[str, Any]
    actions: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """从字典创建"""
        return cls(
            text=data.get("text", ""),
            context=data.get("context", {}),
            actions=data.get("actions", []),
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )


@dataclass
class LearningResult:
    """学习结果类"""
    success: bool
    pattern_id: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizedStrategy:
    """优化策略结果类"""
    action: str
    q_value: float
    exploration: bool = False
    confidence: float = 0.0
    alternatives: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """评估结果类"""
    agent_id: str
    success_rate: float
    improvement_rate: float
    consistency: float
    total_interactions: int
    evaluated_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_id": self.agent_id,
            "success_rate": self.success_rate,
            "improvement_rate": self.improvement_rate,
            "consistency": self.consistency,
            "total_interactions": self.total_interactions,
            "evaluated_at": self.evaluated_at.isoformat(),
            "details": self.details
        }


@dataclass
class PatternRecord:
    """模式记录类"""
    id: str
    pattern_type: str
    frequency: int = 1
    success_count: int = 0
    avg_reward: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
