"""
自动化运维平台 - API接口
=============================

提供RESTful API用于管理:
- 流水线执行
- 定时任务
- 工作流
- 通知

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .pipeline_engine import PipelineEngine, Task
from .cron_scheduler import CronScheduler
from .workflow_automation import WorkflowAutomation, StepType
from .notification_center import NotificationCenter, AlertLevel, NotificationChannel

logger = logging.getLogger(__name__)


# ================== 数据模型 ==================

class TaskCreate(BaseModel):
    name: str
    func_name: str
    dependencies: List[str] = []
    retry_count: int = 3
    timeout: int = 300
    parallel: bool = False


class PipelineRunRequest(BaseModel):
    name: str
    tasks: List[TaskCreate]
    context: Optional[Dict[str, Any]] = None


class PipelineResponse(BaseModel):
    pipeline_id: str
    status: str
    message: str


class CronJobCreate(BaseModel):
    name: str
    func_name: str
    cron_expression: str
    timezone: str = "UTC"
    retry_count: int = 3
    enabled: bool = True


class WorkflowCreate(BaseModel):
    name: str
    description: str = ""


class WorkflowStepCreate(BaseModel):
    name: str
    step_type: str
    func_name: Optional[str] = None
    next_step: Optional[str] = None


class AlertSendRequest(BaseModel):
    alert_name: str
    level: str
    title: str
    message: str
    channels: List[str] = ["email"]
    labels: Dict[str, str] = {}


class StatusResponse(BaseModel):
    status: str
    timestamp: str
    data: Dict[str, Any]


# ================== 全局实例 ==================

_pipeline_engine: Optional[PipelineEngine] = None
_cron_scheduler: Optional[CronScheduler] = None
_workflow_automation: Optional[WorkflowAutomation] = None
_notification_center: Optional[NotificationCenter] = None

# 任务函数注册表
_task_functions: Dict[str, Any] = {}


def register_task(name: str, func: Any):
    """注册任务函数"""
    _task_functions[name] = func
    logger.info(f"Registered task function: {name}")


# ================== FastAPI 应用 ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _pipeline_engine, _cron_scheduler, _workflow_automation, _notification_center
    
    # 初始化引擎
    _pipeline_engine = PipelineEngine(max_concurrent_tasks=10)
    _cron_scheduler = CronScheduler()
    _workflow_automation = WorkflowAutomation()
    _notification_center = NotificationCenter()
    
    # 启动调度器
    await _cron_scheduler.start()
    
    logger.info("Automation Ops API started")
    
    yield
    
    # 清理
    await _cron_scheduler.stop()
    logger.info("Automation Ops API stopped")


app = FastAPI(
    title="Automation Ops API",
    description="自动化运维平台 API",
    version="1.0.0",
    lifespan=lifespan
)


# ================== 健康检查 ==================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status")
async def get_status():
    """获取所有服务状态"""
    return {
        "pipeline": _pipeline_engine.get_status() if _pipeline_engine else None,
        "scheduler": {
            "status": _cron_scheduler.status.value if _cron_scheduler else None,
            "metrics": _cron_scheduler.get_metrics() if _cron_scheduler else None
        },
        "workflow": {
            "workflows_count": len(_workflow_automation._workflows) if _workflow_automation else 0,
            "executions_count": len(_workflow_automation._executions) if _workflow_automation else 0
        },
        "notification": _notification_center.get_metrics() if _notification_center else None
    }


# ================== Pipeline API ==================

@app.post("/api/v1/pipeline/run")
async def run_pipeline(request: PipelineRunRequest) -> PipelineResponse:
    """运行流水线"""
    if not _pipeline_engine:
        raise HTTPException(status_code=500, detail="Pipeline engine not initialized")
    
    # 创建任务
    tasks = []
    for task_config in request.tasks:
        func = _task_functions.get(task_config.func_name)
        if not func:
            raise HTTPException(status_code=400, detail=f"Task function not found: {task_config.func_name}")
        
        task = Task(
            name=task_config.name,
            func=func,
            dependencies=task_config.dependencies,
            retry_count=task_config.retry_count,
            timeout=task_config.timeout,
            parallel=task_config.parallel
        )
        tasks.append(task)
    
    # 运行
    result = await _pipeline_engine.run(tasks, pipeline_name=request.name, context=request.context)
    
    return PipelineResponse(
        pipeline_id=result.pipeline_id,
        status=result.status.value,
        message="Pipeline executed successfully" if result.status.value == "completed" else f"Pipeline failed: {result.error}"
    )


@app.get("/api/v1/pipeline/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """获取流水线状态"""
    status = _pipeline_engine.get_status() if _pipeline_engine else {}
    return {
        "pipeline_id": pipeline_id,
        "status": status
    }


@app.get("/api/v1/pipeline/history")
async def get_pipeline_history(limit: int = 10):
    """获取流水线执行历史"""
    history = _pipeline_engine.get_history() if _pipeline_engine else []
    return {"history": history[-limit:]}


# ================== Cron Scheduler API ==================

@app.post("/api/v1/cron/jobs")
async def create_cron_job(job: CronJobCreate):
    """创建定时任务"""
    if not _cron_scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
    
    func = _task_functions.get(job.func_name)
    if not func:
        raise HTTPException(status_code=400, detail=f"Function not found: {job.func_name}")
    
    job_id = _cron_scheduler.add_job(
        func=func,
        name=job.name,
        cron_expression=job.cron_expression,
        timezone=job.timezone,
        retry_count=job.retry_count,
        enabled=job.enabled
    )
    
    return {"job_id": job_id, "message": "Cron job created"}


@app.get("/api/v1/cron/jobs")
async def list_cron_jobs():
    """列出所有定时任务"""
    if not _cron_scheduler:
        return {"jobs": []}
    
    return {"jobs": _cron_scheduler.get_all_jobs()}


@app.delete("/api/v1/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str):
    """删除定时任务"""
    if not _cron_scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
    
    success = _cron_scheduler.remove_job(job_id)
    if success:
        return {"message": "Job deleted"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/api/v1/cron/jobs/{job_id}/run")
async def run_cron_job_now(job_id: str):
    """立即运行定时任务"""
    if not _cron_scheduler:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
    
    success = _cron_scheduler.run_now(job_id)
    if success:
        return {"message": "Job triggered"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/v1/cron/metrics")
async def get_scheduler_metrics():
    """获取调度器指标"""
    if not _cron_scheduler:
        return {"metrics": None}
    return {"metrics": _cron_scheduler.get_metrics()}


# ================== Workflow API ==================

@app.post("/api/v1/workflows")
async def create_workflow(workflow: WorkflowCreate):
    """创建工作流"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    workflow_id = _workflow_automation.create_workflow(
        name=workflow.name,
        description=workflow.description
    )
    
    return {"workflow_id": workflow_id, "message": "Workflow created"}


@app.post("/api/v1/workflows/{workflow_id}/steps")
async def add_workflow_step(workflow_id: str, step: WorkflowStepCreate):
    """添加工作流步骤"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    func = None
    if step.func_name:
        func = _task_functions.get(step.func_name)
        if not func:
            raise HTTPException(status_code=400, detail=f"Function not found: {step.func_name}")
    
    step_type = StepType(step.step_type)
    
    step_id = _workflow_automation.add_step(
        workflow_id=workflow_id,
        name=step.name,
        step_type=step_type,
        func=func,
        next_step=step.next_step
    )
    
    return {"step_id": step_id, "message": "Step added"}


@app.post("/api/v1/workflows/{workflow_id}/connect")
async def connect_workflow_steps(workflow_id: str, from_step: str, to_step: str):
    """连接工作流步骤"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    try:
        _workflow_automation.connect_steps(workflow_id, from_step, to_step)
        return {"message": "Steps connected"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str, context: Optional[Dict[str, Any]] = None):
    """启动工作流"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    try:
        execution_id = _workflow_automation.start_execution(workflow_id, context=context)
        return {"execution_id": execution_id, "message": "Workflow started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/workflows")
async def list_workflows():
    """列出所有工作流"""
    if not _workflow_automation:
        return {"workflows": []}
    return {"workflows": _workflow_automation.get_all_workflows()}


@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """获取工作流详情"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    workflow = _workflow_automation.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "steps": [{"id": s.id, "name": s.name, "type": s.step_type.value} for s in workflow.steps]
    }


@app.get("/api/v1/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """获取执行状态"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    status = _workflow_automation.get_execution_status(execution_id)
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return status


@app.post("/api/v1/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """取消执行"""
    if not _workflow_automation:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    
    success = _workflow_automation.cancel_execution(execution_id)
    if success:
        return {"message": "Execution cancelled"}
    raise HTTPException(status_code=404, detail="Execution not found")


# ================== Notification API ==================

@app.post("/api/v1/notifications/alert")
async def send_alert(request: AlertSendRequest):
    """发送告警"""
    if not _notification_center:
        raise HTTPException(status_code=500, detail="Notification center not initialized")
    
    try:
        level = AlertLevel(request.level)
        channels = [NotificationChannel(c) for c in request.channels]
        
        notification = await _notification_center.send_alert(
            alert_name=request.alert_name,
            level=level,
            title=request.title,
            message=request.message,
            channels=channels,
            labels=request.labels
        )
        
        return {"notification_id": notification.id, "status": notification.status.value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/notifications")
async def list_notifications(
    status: Optional[str] = None,
    alert_name: Optional[str] = None,
    limit: int = 50
):
    """列出通知"""
    if not _notification_center:
        return {"notifications": []}
    
    status_enum = None
    if status:
        try:
            status_enum = NotificationStatus(status)
        except ValueError:
            pass
    
    notifications = _notification_center.get_notifications(
        status=status_enum,
        alert_name=alert_name,
        limit=limit
    )
    
    return {"notifications": notifications}


@app.get("/api/v1/notifications/metrics")
async def get_notification_metrics():
    """获取通知指标"""
    if not _notification_center:
        return {"metrics": None}
    return {"metrics": _notification_center.get_metrics()}


@app.post("/api/v1/notifications/acknowledge/{notification_id}")
async def acknowledge_notification(notification_id: str):
    """确认通知"""
    if not _notification_center:
        raise HTTPException(status_code=500, detail="Notification center not initialized")
    
    success = _notification_center.acknowledge_notification(notification_id)
    if success:
        return {"message": "Notification acknowledged"}
    raise HTTPException(status_code=404, detail="Notification not found")


# ================== 任务注册 API ==================

@app.post("/api/v1/tasks/register")
async def register_task_function(name: str, func_body: str):
    """注册任务函数"""
    try:
        # 动态创建函数
        exec_globals = {}
        exec(func_body, exec_globals)
        
        for key in exec_globals:
            if callable(exec_globals[key]):
                register_task(name, exec_globals[key])
                return {"message": f"Task '{name}' registered"}
        
        raise HTTPException(status_code=400, detail="No callable function found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ================== 便捷函数 ==================

def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """运行服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=debug)
