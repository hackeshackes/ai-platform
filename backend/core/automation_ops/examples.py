"""
自动化运维平台 - 使用示例
=============================

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
from typing import Any, Dict, List
from datetime import datetime

from .pipeline_engine import PipelineEngine, Task, PipelineStatus
from .cron_scheduler import CronScheduler, CronExpressions
from .workflow_automation import WorkflowAutomation, StepType
from .notification_center import NotificationCenter, AlertLevel, NotificationChannel


# ================== Pipeline 示例 ==================

async def example_pipeline():
    """流水线执行示例"""
    print("\n=== Pipeline Engine Example ===")
    
    async def code_check():
        print("✓ Running code checks...")
        await asyncio.sleep(1)
        return {"issues": 0, "passed": True}
    
    async def run_tests():
        print("✓ Running unit tests...")
        await asyncio.sleep(2)
        return {"tests_run": 150, "passed": 149, "failed": 1}
    
    async def deploy_service():
        print("✓ Deploying service...")
        await asyncio.sleep(2)
        return {"deployed": True, "version": "1.2.0"}
    
    async def health_check():
        print("✓ Running health check...")
        await asyncio.sleep(1)
        return {"status": "healthy"}
    
    engine = PipelineEngine(max_concurrent_tasks=5)
    
    tasks = [
        Task("代码检查", code_check),
        Task("单元测试", run_tests, dependencies=["代码检查"]),
        Task("部署服务", deploy_service, dependencies=["单元测试"]),
        Task("健康检查", health_check, dependencies=["部署服务"])
    ]
    
    result = await engine.run(tasks, pipeline_name="CI/CD Pipeline")
    
    print(f"\nResult: {result.status.value}, Duration: {result.duration:.2f}s")
    return result


# ================== Cron Scheduler 示例 ==================

async def example_cron_scheduler():
    """定时任务调度示例"""
    print("\n=== Cron Scheduler Example ===")
    
    async def backup_database():
        print("  [Cron] Backing up database...")
        return {"backup_created": True}
    
    scheduler = CronScheduler(timezone="Asia/Shanghai")
    
    scheduler.add_job(
        func=backup_database,
        name="daily_backup",
        cron_expression=CronExpressions.DAILY_2AM,
        retry_count=3
    )
    
    print(f"Registered {len(scheduler.get_all_jobs())} cron jobs")
    
    # 启动并立即运行一个任务
    await scheduler.start()
    await asyncio.sleep(1)
    
    metrics = scheduler.get_metrics()
    print(f"Metrics: {metrics}")
    
    await scheduler.stop()
    return scheduler


# ================== Workflow 示例 ==================

async def example_workflow():
    """工作流自动化示例"""
    print("\n=== Workflow Automation Example ===")
    
    automation = WorkflowAutomation()
    
    workflow_id = automation.create_workflow(
        name="Server Provisioning",
        description="自动化服务器配置流程"
    )
    
    async def allocate_resources(context):
        print("  [Step] Allocating cloud resources...")
        return {"instance_id": "i-123456"}
    
    async def install_software(context):
        print("  [Step] Installing software...")
        return {"packages_installed": 25}
    
    async def configure_security(context):
        print("  [Step] Configuring security...")
        return {"security_configured": True}
    
    step1 = automation.add_step(workflow_id, "Allocate", StepType.TASK, allocate_resources)
    step2 = automation.add_step(workflow_id, "Install", StepType.TASK, install_software, dependencies=[step1])
    step3 = automation.add_step(workflow_id, "Security", StepType.TASK, configure_security, dependencies=[step2])
    automation.set_end_step(workflow_id, step3)
    
    print(f"Created workflow with {len(automation.get_workflow(workflow_id).steps)} steps")
    
    execution_id = automation.start_execution(workflow_id, context={"env": "prod"})
    
    await asyncio.sleep(2)
    
    status = automation.get_execution_status(execution_id)
    print(f"Execution Status: {status['status']}")
    
    return automation


# ================== Notification Center 示例 ==================

async def example_notification_center():
    """通知中心示例"""
    print("\n=== Notification Center Example ===")
    
    notifier = NotificationCenter(default_channels=[NotificationChannel.EMAIL])
    
    async def email_handler(notification):
        print(f"  [Email] {notification.title}")
        return True
    
    notifier.register_channel(NotificationChannel.EMAIL, email_handler)
    
    await notifier.send_alert(
        alert_name="High CPU",
        level=AlertLevel.WARNING,
        title="CPU使用率过高",
        message="Server cpu-01 CPU使用率达到95%"
    )
    
    notifications = notifier.get_notifications()
    print(f"Sent {len(notifications)} notifications")
    
    return notifier


# ================== 综合示例 ==================

async def example_full_pipeline():
    """完整的CI/CD流水线示例"""
    print("\n=== Full CI/CD Pipeline Example ===")
    
    pipeline = PipelineEngine(max_concurrent_tasks=10)
    notifier = NotificationCenter()
    
    async def lint_code():
        print("  [1/4] Linting code...")
        await asyncio.sleep(2)
        return {"errors": 0}
    
    async def run_tests():
        print("  [2/4] Running tests...")
        await asyncio.sleep(3)
        return {"passed": True}
    
    async def build():
        print("  [3/4] Building...")
        await asyncio.sleep(2)
        return {"image": "myapp:v1"}
    
    async def deploy():
        print("  [4/4] Deploying...")
        await asyncio.sleep(2)
        return {"deployed": True}
    
    tasks = [
        Task("Lint", lint_code),
        Task("Test", run_tests, dependencies=["Lint"]),
        Task("Build", build, dependencies=["Test"]),
        Task("Deploy", deploy, dependencies=["Build"])
    ]
    
    result = await pipeline.run(tasks, pipeline_name="CI/CD")
    
    print(f"\nPipeline: {result.status.value} ({result.duration:.2f}s)")
    return result


async def run_all_examples():
    """运行所有示例"""
    print("=" * 50)
    print("Automation Ops Platform - Examples")
    print("=" * 50)
    
    await example_pipeline()
    await example_cron_scheduler()
    await example_workflow()
    await example_notification_center()
    await example_full_pipeline()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
