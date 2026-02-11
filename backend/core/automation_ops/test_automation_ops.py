"""
自动化运维平台 - 测试用例
=============================

提供完整的单元测试和集成测试

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import pytest
import sys
from typing import Any, Dict, List
from datetime import datetime, timedelta

# 导入被测试模块
from .pipeline_engine import (
    PipelineEngine, Task, TaskStatus, PipelineStatus, PipelineResult
)
from .cron_scheduler import (
    CronScheduler, ScheduledJob, SchedulerStatus, JobStatus
)
from .workflow_automation import (
    WorkflowAutomation, Workflow, WorkflowStep, StepType, WorkflowStatus as WFStatus
)
from .notification_center import (
    NotificationCenter, AlertLevel, NotificationChannel, NotificationStatus
)


# ================== Pipeline Engine Tests ==================

class TestPipelineEngine:
    """流水线引擎测试"""
    
    @pytest.fixture
    def engine(self):
        """创建测试引擎"""
        return PipelineEngine(max_concurrent_tasks=5)
    
    @pytest.fixture
    def sample_tasks(self):
        """示例任务"""
        async def task1():
            await asyncio.sleep(0.1)
            return {"result": "task1"}
        
        async def task2():
            await asyncio.sleep(0.1)
            return {"result": "task2"}
        
        async def task3(data):
            return {"combined": data}
        
        return [
            Task("Task1", task1),
            Task("Task2", task2),
            Task("Task3", task3, dependencies=["Task1", "Task2"])
        ]
    
    @pytest.mark.asyncio
    async def test_single_task_execution(self, engine):
        """测试单个任务执行"""
        async def simple_task():
            return {"status": "completed"}
        
        tasks = [Task("Simple", simple_task)]
        result = await engine.run(tasks, pipeline_name="Test")
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.task_results) == 1
        assert result.task_results[0].status == TaskStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_task_with_dependencies(self, engine, sample_tasks):
        """测试任务依赖"""
        result = await engine.run(sample_tasks, pipeline_name="Dependency Test")
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.task_results) == 3
        
        # 验证执行顺序
        task_names = [r.task_name for r in result.task_results]
        assert task_names.index("Task3") > task_names.index("Task1")
        assert task_names.index("Task3") > task_names.index("Task2")
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, engine):
        """测试任务失败和重试"""
        call_count = 0
        
        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return {"success": True}
        
        task = Task("FailingTask", failing_task, retry_count=3)
        result = await engine.run([task], pipeline_name="Retry Test")
        
        assert result.status == PipelineStatus.COMPLETED
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_parallel_tasks(self, engine):
        """测试并行任务"""
        async def slow_task(name):
            await asyncio.sleep(0.5)
            return {"task": name}
        
        tasks = [
            Task("Parallel1", lambda: slow_task("1"), parallel=True),
            Task("Parallel2", lambda: slow_task("2"), parallel=True),
            Task("Sequential", lambda: slow_task("3"))
        ]
        
        result = await engine.run(tasks, pipeline_name="Parallel Test")
        
        assert result.status == PipelineStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_with_skip_condition(self, engine):
        """测试条件跳过任务"""
        async def conditional_task():
            return {"executed": True}
        
        async def skip_condition():
            return True
        
        task = Task("Conditional", conditional_task, skip_if=skip_condition)
        result = await engine.run([task], pipeline_name="Skip Test")
        
        assert result.status == PipelineStatus.COMPLETED
        assert result.task_results[0].status == TaskStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_pipeline_failure(self, engine):
        """测试流水线失败"""
        async def failing_task():
            raise RuntimeError("Task failed")
        
        task = Task("Failing", failing_task, retry_count=0)
        result = await engine.run([task], pipeline_name="Failure Test")
        
        assert result.status == PipelineStatus.FAILED
    
    def test_engine_status(self, engine):
        """测试引擎状态查询"""
        status = engine.get_status()
        
        assert "total_tasks" in status
        assert "running_tasks" in status
        assert "completed_tasks" in status


# ================== Cron Scheduler Tests ==================

class TestCronScheduler:
    """定时调度器测试"""
    
    @pytest.fixture
    def scheduler(self):
        """创建测试调度器"""
        return CronScheduler(check_interval=1)
    
    def test_add_job(self, scheduler):
        """测试添加任务"""
        def simple_job():
            return True
        
        job_id = scheduler.add_job(
            func=simple_job,
            name="test_job",
            cron_expression="* * * * *"
        )
        
        assert job_id is not None
        assert len(scheduler.get_all_jobs()) == 1
    
    def test_remove_job(self, scheduler):
        """测试删除任务"""
        def simple_job():
            pass
        
        job_id = scheduler.add_job(simple_job, "test", "* * * * *")
        success = scheduler.remove_job(job_id)
        
        assert success is True
        assert len(scheduler.get_all_jobs()) == 0
    
    def test_cron_expression_parsing(self, scheduler):
        """测试Cron表达式解析"""
        result = scheduler.parse_cron("0 2 * * *")
        
        assert result["minute"] == "0"
        assert result["hour"] == "2"
        assert result["day"] == "*"
        assert result["month"] == "*"
        assert result["weekday"] == "*"
    
    def test_invalid_cron_expression(self, scheduler):
        """测试无效Cron表达式"""
        with pytest.raises(ValueError):
            scheduler.parse_cron("invalid expression")
    
    @pytest.mark.asyncio
    async def test_run_job_now(self, scheduler):
        """测试立即运行任务"""
        executed = []
        
        def test_job():
            executed.append(True)
        
        job_id = scheduler.add_job(test_job, "immediate", "0 0 * * *")
        scheduler.run_now(job_id)
        
        await asyncio.sleep(0.5)
        
        assert len(executed) == 1
    
    def test_job_status(self, scheduler):
        """测试任务状态"""
        def test_job():
            pass
        
        scheduler.add_job(test_job, "status_test", "* * * * *", enabled=False)
        jobs = scheduler.get_all_jobs()
        
        assert jobs[0]["enabled"] is False
    
    def test_cron_expressions_constants(self):
        """测试常用Cron表达式"""
        from .cron_scheduler import CronExpressions
        
        assert CronExpressions.EVERY_MINUTE == "* * * * *"
        assert CronExpressions.DAILY_MIDNIGHT == "0 0 * * *"
        assert CronExpressions.WEEKLY_MONDAY == "0 0 * * 1"


# ================== Workflow Automation Tests ==================

class TestWorkflowAutomation:
    """工作流自动化测试"""
    
    @pytest.fixture
    def automation(self):
        """创建测试工作流引擎"""
        return WorkflowAutomation()
    
    def test_create_workflow(self, automation):
        """测试创建工作流"""
        workflow_id = automation.create_workflow(
            name="Test Workflow",
            description="A test workflow"
        )
        
        assert workflow_id is not None
        assert len(automation.get_all_workflows()) == 1
    
    def test_add_step(self, automation):
        """测试添加步骤"""
        workflow_id = automation.create_workflow("Test")
        
        step_id = automation.add_step(
            workflow_id=workflow_id,
            name="First Step",
            step_type=StepType.TASK
        )
        
        assert step_id is not None
        
        workflow = automation.get_workflow(workflow_id)
        assert len(workflow.steps) == 1
    
    def test_connect_steps(self, automation):
        """测试连接步骤"""
        workflow_id = automation.create_workflow("Connect Test")
        
        step1 = automation.add_step(workflow_id, "Step1", StepType.TASK)
        step2 = automation.add_step(workflow_id, "Step2", StepType.TASK)
        
        automation.connect_steps(workflow_id, step1, step2)
        
        workflow = automation.get_workflow(workflow_id)
        assert workflow.get_step(step1).next_step == step2
    
    def test_start_execution(self, automation):
        """测试启动执行"""
        workflow_id = automation.create_workflow("Execution Test")
        automation.add_step(workflow_id, "Task1", StepType.TASK, lambda ctx: {"done": True})
        
        execution_id = automation.start_execution(workflow_id)
        
        assert execution_id is not None
        
        status = automation.get_execution_status(execution_id)
        assert status is not None
        assert "status" in status
    
    def test_cancel_execution(self, automation):
        """测试取消执行"""
        workflow_id = automation.create_workflow("Cancel Test")
        automation.add_step(workflow_id, "Task1", StepType.TASK, lambda ctx: {"done": True})
        
        execution_id = automation.start_execution(workflow_id)
        success = automation.cancel_execution(execution_id)
        
        assert success is True


# ================== Notification Center Tests ==================

class TestNotificationCenter:
    """通知中心测试"""
    
    @pytest.fixture
    def notifier(self):
        """创建测试通知中心"""
        return NotificationCenter()
    
    def test_send_alert(self, notifier):
        """测试发送告警"""
        async def dummy_handler(notification):
            return True
        
        notifier.register_channel(NotificationChannel.EMAIL, dummy_handler)
        
        notification = asyncio.run(notifier.send_alert(
            alert_name="Test Alert",
            level=AlertLevel.INFO,
            title="Test Title",
            message="Test Message"
        ))
        
        assert notification is not None
        assert notification.alert_name == "Test Alert"
        assert notification.level == AlertLevel.INFO
    
    def test_alert_levels(self):
        """测试告警级别"""
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.INFO.value == "info"
    
    def test_notification_channels(self):
        """测试通知渠道"""
        channels = list(NotificationChannel)
        assert NotificationChannel.EMAIL in channels
        assert NotificationChannel.SLACK in channels
        assert NotificationChannel.SMS in channels
        assert NotificationChannel.PHONE in channels
    
    def test_add_alert_rule(self, notifier):
        """测试添加告警规则"""
        def condition(data):
            return data.get("value", 0) > 100
        
        rule_id = notifier.add_alert_rule(
            name="Test Rule",
            condition=condition,
            level=AlertLevel.WARNING,
            channels=[NotificationChannel.EMAIL]
        )
        
        assert rule_id is not None
        assert len(notifier.get_alert_rules()) == 1
    
    def test_evaluate_rules(self, notifier):
        """测试规则评估"""
        def condition_high_cpu(data):
            return data.get("cpu", 0) > 90
        
        notifier.add_alert_rule(
            name="High CPU",
            condition=condition_high_cpu,
            level=AlertLevel.WARNING,
            channels=[NotificationChannel.EMAIL]
        )
        
        # 应该触发
        triggered = notifier.evaluate_rules({"cpu": 95})
        assert len(triggered) == 1
        
        # 不应该触发
        triggered = notifier.evaluate_rules({"cpu": 50})
        assert len(triggered) == 0
    
    def test_silence_period(self, notifier):
        """测试静默期"""
        start = datetime.now()
        end = start + timedelta(hours=1)
        
        silence_id = notifier.add_silence_period(
            name="Maintenance",
            start_time=start,
            end_time=end,
            reason="Planned maintenance"
        )
        
        assert silence_id is not None
        assert notifier.is_silenced() is True
    
    def test_get_notifications(self, notifier):
        """测试获取通知列表"""
        async def handler(n):
            return True
        
        notifier.register_channel(NotificationChannel.EMAIL, handler)
        
        asyncio.run(notifier.send_alert(
            alert_name="Alert1",
            level=AlertLevel.INFO,
            title="Title1",
            message="Message1"
        ))
        
        notifications = notifier.get_notifications()
        assert len(notifications) >= 1
    
    def test_get_metrics(self, notifier):
        """测试获取指标"""
        metrics = notifier.get_metrics()
        
        assert "total_notifications" in metrics
        assert "sent_notifications" in metrics
        assert "failed_notifications" in metrics


# ================== 集成测试 ==================

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_notification(self):
        """测试流水线与通知集成"""
        pipeline = PipelineEngine()
        notifier = NotificationCenter()
        
        executed = []
        
        async def tracked_task():
            executed.append(True)
            return {"done": True}
        
        # 运行流水线
        task = Task("Tracked", tracked_task)
        result = await pipeline.run([task], pipeline_name="Integration Test")
        
        assert result.status == PipelineStatus.COMPLETED
        assert len(executed) == 1
    
    @pytest.mark.asyncio
    async def test_workflow_with_context(self):
        """测试工作流上下文传递"""
        automation = WorkflowAutomation()
        
        workflow_id = automation.create_workflow("Context Test")
        
        received_context = {}
        
        async def capture_context(ctx):
            received_context.update(ctx)
            return {"captured": True}
        
        automation.add_step(workflow_id, "Capture", StepType.TASK, capture_context)
        
        execution_id = automation.start_execution(
            workflow_id,
            context={"custom_key": "custom_value"}
        )
        
        await asyncio.sleep(0.5)
        
        assert "custom_key" in received_context
        assert received_context["custom_key"] == "custom_value"


# ================== 测试运行器 ==================

def run_tests():
    """运行所有测试"""
    import pytest
    
    # 测试文件路径
    test_paths = [
        str(__file__)
    ]
    
    # pytest 配置
    result = pytest.main([
        "-v",  # 详细输出
        "--tb=short",  # 简短的 traceback
        "-x",  # 遇到第一个失败即停止
    ] + test_paths)
    
    return result == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
