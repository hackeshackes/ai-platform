"""
自动化运维平台 - 统一导出
=============================

提供完整的自动化运维解决方案:
- pipeline_engine.py: 流水线引擎
- cron_scheduler.py: 定时调度
- workflow_automation.py: 工作流自动化
- notification_center.py: 通知中心
- api.py: API接口
- config.py: 配置管理
- examples.py: 使用示例
- test_automation_ops.py: 测试用例

作者: AI Platform Team
版本: 1.0.0
"""

from .pipeline_engine import (
    PipelineEngine,
    Task,
    PipelineResult,
    TaskStatus,
    PipelineStatus
)

from .cron_scheduler import (
    CronScheduler,
    CronJob,
    SchedulerStatus
)

from .workflow_automation import (
    WorkflowAutomation,
    Workflow,
    WorkflowStep,
    WorkflowStatus,
    ConditionBranch,
    ManualIntervention,
    ApprovalProcess
)

from .notification_center import (
    NotificationCenter,
    NotificationChannel,
    AlertLevel,
    AlertRule,
    EscalationPolicy,
    SilencePeriod
)

from .api import (
    create_app,
    run_server
)

__version__ = "1.0.0"
__all__ = [
    # Pipeline Engine
    "PipelineEngine",
    "Task",
    "PipelineResult",
    "TaskStatus",
    "PipelineStatus",
    
    # Cron Scheduler
    "CronScheduler",
    "CronJob",
    "SchedulerStatus",
    
    # Workflow Automation
    "WorkflowAutomation",
    "Workflow",
    "WorkflowStep",
    "WorkflowStatus",
    "ConditionBranch",
    "ManualIntervention",
    "ApprovalProcess",
    
    # Notification Center
    "NotificationCenter",
    "NotificationChannel",
    "AlertLevel",
    "AlertRule",
    "EscalationPolicy",
    "SilencePeriod",
    
    # API
    "create_app",
    "run_server"
]
