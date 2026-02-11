"""
Agent协作网络模块
提供多Agent协同工作、通信和编排能力
"""

from .models import (
    CollaborationMode,
    TaskStatus,
    SessionStatus,
    AgentRole,
    AgentInfo,
    TaskInput,
    TaskOutput,
    TaskDefinition,
    WorkflowDefinition,
    Message,
    ConsensusProposal,
    SessionProgress,
    CollaborationSession,
    WorkflowExecutionResult
)

from .communication import (
    CommunicationManager,
    MessageRouter,
    CommunicationChannel,
    MessageType,
    get_communication_manager,
    init_communication,
    shutdown_communication
)

from .task_decomposer import (
    TaskDecomposer,
    SmartTaskAnalyzer,
    DecompositionStrategy,
    SubTask,
    create_task_decomposer,
    create_smart_analyzer
)

from .consensus import (
    ConsensusManager,
    ConsensusAlgorithm,
    VotingConsensus,
    Vote,
    ConsensusRound,
    create_consensus_manager,
    create_voting_system
)

from .workflow import (
    WorkflowEngine,
    WorkflowExecutor,
    SequentialExecutor,
    ParallelExecutor,
    HierarchicalExecutor,
    WorkflowState,
    get_workflow_engine,
    init_workflow_engine
)

from .orchestrator import (
    CollaborationOrchestrator,
    OrchestrationSession,
    get_orchestrator,
    init_orchestrator,
    shutdown_orchestrator
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "CollaborationMode",
    "TaskStatus",
    "SessionStatus",
    "AgentRole",
    "AgentInfo",
    "TaskInput",
    "TaskOutput",
    "TaskDefinition",
    "WorkflowDefinition",
    "Message",
    "ConsensusProposal",
    "SessionProgress",
    "CollaborationSession",
    "WorkflowExecutionResult",
    
    # Communication
    "CommunicationManager",
    "MessageRouter",
    "CommunicationChannel",
    "MessageType",
    "get_communication_manager",
    "init_communication",
    "shutdown_communication",
    
    # Task Decomposition
    "TaskDecomposer",
    "SmartTaskAnalyzer",
    "DecompositionStrategy",
    "SubTask",
    "create_task_decomposer",
    "create_smart_analyzer",
    
    # Consensus
    "ConsensusManager",
    "ConsensusAlgorithm",
    "VotingConsensus",
    "Vote",
    "ConsensusRound",
    "create_consensus_manager",
    "create_voting_system",
    
    # Workflow
    "WorkflowEngine",
    "WorkflowExecutor",
    "SequentialExecutor",
    "ParallelExecutor",
    "HierarchicalExecutor",
    "WorkflowState",
    "get_workflow_engine",
    "init_workflow_engine",
    
    # Orchestration
    "CollaborationOrchestrator",
    "OrchestrationSession",
    "get_orchestrator",
    "init_orchestrator",
    "shutdown_orchestrator"
]
