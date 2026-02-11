"""
Visualization API Endpoints - AI Platform v5

可视化API端点 - 提供训练过程可视化接口
"""
import asyncio
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json

from visualization.charts import (
    ChartGenerator,
    get_chart_generator,
    LossChartConfig,
    GPUChartConfig,
    MetricsChartConfig,
)
from visualization.realtime import (
    TrainingDataStore,
    RealtimeDataHandler,
    TrainingMetrics,
    TrainingStatus,
    get_training_store,
    get_realtime_handler,
)

router = APIRouter()


# ============ 依赖注入 ============

def get_chart_generator_instance() -> ChartGenerator:
    """获取图表生成器实例"""
    return get_chart_generator()


def get_training_store_instance() -> TrainingDataStore:
    """获取训练数据存储实例"""
    return get_training_store()


def get_realtime_handler_instance() -> RealtimeDataHandler:
    """获取实时数据处理器实例"""
    return get_realtime_handler()


# ============ 请求/响应模型 ============

# Training Job Models
class CreateJobRequest(BaseModel):
    """创建训练作业请求"""
    name: str = Field(..., description="作业名称")
    model_name: str = Field(..., description="模型名称")
    total_epochs: int = Field(default=10, ge=1, description="总训练轮次")
    total_steps: int = Field(default=0, ge=0, description="总步数")
    config: Optional[Dict[str, Any]] = Field(default=None, description="额外配置")


class JobResponse(BaseModel):
    """作业响应"""
    job_id: str
    name: str
    model_name: str
    status: str
    total_epochs: int
    total_steps: int
    current_epoch: int
    current_step: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics_count: int


class JobListResponse(BaseModel):
    """作业列表响应"""
    jobs: List[JobResponse]
    total: int


# Metrics Models
class MetricsRequest(BaseModel):
    """添加指标请求"""
    step: int
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory: Optional[float] = None
    gpu_temperature: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    progress_percent: Optional[float] = None


class MetricsResponse(BaseModel):
    """指标响应"""
    success: bool
    job_id: str
    step: int
    epoch: int
    timestamp: str


# Chart Models
class ChartDataResponse(BaseModel):
    """图表数据响应"""
    job_id: str
    chart_type: str
    data: Dict[str, Any]
    generated_at: str


class DashboardResponse(BaseModel):
    """仪表盘响应"""
    overview: Dict[str, Any]
    charts: Dict[str, Any]


# Loss Chart Query Params
class LossChartQuery(BaseModel):
    """Loss图表查询参数"""
    smooth: Optional[bool] = Field(default=True, description="是否平滑")
    smooth_factor: Optional[float] = Field(default=0.1, ge=0, le=1, description="平滑因子")


# ============ 训练作业管理 ============

@router.post("/training/jobs", response_model=JobResponse, tags=["Training Jobs"])
async def create_training_job(request: CreateJobRequest):
    """
    创建新的训练作业
    
    - **name**: 作业名称
    - **model_name**: 模型名称
    - **total_epochs**: 总训练轮次
    - **total_steps**: 总步数
    - **config**: 额外配置
    """
    store = get_training_store()
    
    job = await store.create_job(
        name=request.name,
        model_name=request.model_name,
        total_epochs=request.total_epochs,
        total_steps=request.total_steps,
        config=request.config,
    )
    
    return JobResponse(**job.to_dict())


@router.get("/training/jobs", response_model=JobListResponse, tags=["Training Jobs"])
async def list_training_jobs(
    status: Optional[str] = Query(default=None, description="按状态筛选"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    列出训练作业
    
    - **status**: 状态筛选 (pending, running, paused, completed, failed, cancelled)
    - **limit**: 返回数量限制
    """
    store = get_training_store()
    
    status_enum = TrainingStatus(status) if status else None
    jobs = await store.list_jobs(status=status_enum)
    
    return JobListResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs[:limit]],
        total=len(jobs),
    )


@router.get("/training/jobs/{job_id}", response_model=JobResponse, tags=["Training Jobs"])
async def get_training_job(job_id: str):
    """
    获取训练作业详情
    """
    store = get_training_store()
    
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(**job.to_dict())


@router.delete("/training/jobs/{job_id}", tags=["Training Jobs"])
async def delete_training_job(job_id: str):
    """
    删除训练作业
    """
    store = get_training_store()
    
    success = await store.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"success": True, "message": f"Job {job_id} deleted"}


@router.put("/training/jobs/{job_id}/status", tags=["Training Jobs"])
async def update_job_status(
    job_id: str,
    status: str = Query(..., description="新状态"),
):
    """
    更新作业状态
    """
    store = get_training_store()
    
    try:
        status_enum = TrainingStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    success = await store.update_status(job_id, status_enum)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"success": True, "job_id": job_id, "status": status}


# ============ 指标管理 ============

@router.post("/training/{job_id}/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def add_metrics(job_id: str, request: MetricsRequest):
    """
    添加训练指标
    
    用于训练过程中实时上报指标数据
    """
    store = get_training_store()
    
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    metrics = TrainingMetrics(
        step=request.step,
        epoch=request.epoch,
        train_loss=request.train_loss,
        val_loss=request.val_loss,
        learning_rate=request.learning_rate,
        gpu_utilization=request.gpu_utilization,
        gpu_memory=request.gpu_memory,
        gpu_temperature=request.gpu_temperature,
        accuracy=request.accuracy,
        f1=request.f1,
        precision=request.precision,
        recall=request.recall,
        progress_percent=request.progress_percent,
    )
    
    success = await store.add_metrics(job_id, metrics)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add metrics")
    
    # 广播给订阅者
    handler = get_realtime_handler()
    await handler.publish_metrics(job_id, metrics)
    
    return MetricsResponse(
        success=True,
        job_id=job_id,
        step=request.step,
        epoch=request.epoch,
        timestamp=metrics.timestamp,
    )


@router.get("/training/{job_id}/metrics/history", tags=["Metrics"])
async def get_metrics_history(job_id: str):
    """
    获取指标历史记录
    """
    store = get_training_store()
    
    data = await store.get_metrics_history(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return data


# ============ 可视化图表API ============

@router.get("/training/{job_id}/loss", response_model=ChartDataResponse, tags=["Visualization"])
async def get_loss_chart(
    job_id: str,
    smooth: bool = Query(default=True, description="是否平滑处理"),
    smooth_factor: float = Query(default=0.1, ge=0, le=1, description="平滑因子"),
):
    """
    获取Loss曲线图表数据
    
    返回可用于Chart.js/Recharts的图表配置
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 获取数据
    loss_data = await store.get_loss_history(job_id)
    if loss_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 生成图表配置
    config = LossChartConfig(
        title=f"Loss - {job_id}",
        smooth_factor=smooth_factor if smooth else 0,
    )
    
    train_loss = loss_data.get("train_loss", [])
    val_loss = loss_data.get("val_loss")
    
    chart_data = chart_gen.generate_loss_chart(
        train_loss=train_loss,
        val_loss=val_loss,
        steps=loss_data.get("steps"),
        config=config,
    )
    
    return ChartDataResponse(
        job_id=job_id,
        chart_type="loss",
        data=chart_data,
        generated_at=datetime.now().isoformat(),
    )


@router.get("/training/{job_id}/gpu", response_model=ChartDataResponse, tags=["Visualization"])
async def get_gpu_chart(job_id: str):
    """
    获取GPU监控图表数据
    
    返回GPU利用率、内存、温度等监控数据
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 获取数据
    gpu_data = await store.get_gpu_history(job_id)
    if gpu_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 生成图表配置
    config = GPUChartConfig(
        title=f"GPU Monitor - {job_id}",
    )
    
    chart_data = chart_gen.generate_gpu_chart(
        gpu_utilization=gpu_data.get("utilization", []),
        gpu_memory=gpu_data.get("memory"),
        gpu_temperature=gpu_data.get("temperature"),
        timestamps=gpu_data.get("timestamps"),
        config=config,
    )
    
    return ChartDataResponse(
        job_id=job_id,
        chart_type="gpu",
        data=chart_data,
        generated_at=datetime.now().isoformat(),
    )


@router.get("/training/{job_id}/metrics/chart", response_model=ChartDataResponse, tags=["Visualization"])
async def get_metrics_chart(job_id: str):
    """
    获取评估指标图表数据
    
    返回accuracy、f1、precision、recall等指标曲线
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 获取作业数据
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 生成图表配置
    config = MetricsChartConfig(
        title=f"Metrics - {job_id}",
    )
    
    metrics_data = job.get_metrics_data()
    epochs = list(range(len(job.metrics_history)))
    
    chart_data = chart_gen.generate_metrics_chart(
        metrics=metrics_data,
        epochs=epochs,
        config=config,
    )
    
    return ChartDataResponse(
        job_id=job_id,
        chart_type="metrics",
        data=chart_data,
        generated_at=datetime.now().isoformat(),
    )


@router.get("/training/{job_id}/learning-rate", response_model=ChartDataResponse, tags=["Visualization"])
async def get_learning_rate_chart(job_id: str):
    """
    获取学习率曲线图表数据
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 获取作业
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    lr_data = job.get_lr_data()
    steps = list(range(len(lr_data)))
    
    chart_data = chart_gen.generate_lr_chart(
        learning_rates=lr_data,
        steps=steps,
    )
    
    return ChartDataResponse(
        job_id=job_id,
        chart_type="learning_rate",
        data=chart_data,
        generated_at=datetime.now().isoformat(),
    )


# ============ 仪表盘API ============

@router.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
async def get_dashboard():
    """
    获取可视化仪表盘数据
    
    返回所有活跃训练作业的汇总信息
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 获取活跃作业
    running_jobs = await store.list_jobs(status=TrainingStatus.RUNNING)
    pending_jobs = await store.list_jobs(status=TrainingStatus.PENDING)
    
    dashboard = {
        "overview": {
            "total_jobs": len(running_jobs) + len(pending_jobs),
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
            "completed_jobs": len(await store.list_jobs(status=TrainingStatus.COMPLETED)),
            "failed_jobs": len(await store.list_jobs(status=TrainingStatus.FAILED)),
            "timestamp": datetime.now().isoformat(),
        },
        "jobs": [job.to_dict() for job in running_jobs[:10]],  # 最近10个运行中的作业
        "charts": {},
    }
    
    # 为运行中的作业生成图表
    for job in running_jobs[:5]:  # 限制为5个
        job_data = await store.get_job_data(job.job_id)
        if job_data:
            dashboard["charts"][job.job_id] = chart_gen.generate_dashboard(job_data)
    
    return dashboard


@router.get("/dashboard/job/{job_id}", response_model=DashboardResponse, tags=["Dashboard"])
async def get_job_dashboard(job_id: str):
    """
    获取指定作业的完整仪表盘数据
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    job_data = await store.get_job_data(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    dashboard = chart_gen.generate_dashboard(job_data)
    return dashboard


# ============ SSE 实时流 ============

from fastapi.responses import StreamingResponse

@router.get("/training/{job_id}/stream", tags=["Real-time"])
async def stream_training_updates(job_id: str):
    """
    SSE实时流 - 订阅训练更新
    
    通过Server-Sent Events实时推送训练指标更新
    """
    from visualization.realtime import SSEPublisher
    
    store = get_training_store()
    handler = get_realtime_handler()
    
    # 检查作业是否存在
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 生成会话ID
    import uuid
    session_id = str(uuid.uuid4())[:8]
    
    # 创建订阅队列
    queue = await handler.subscribe(job_id, session_id)
    
    # 返回SSE流
    async def generate():
        try:
            async for message in SSEPublisher.stream_generator(queue, job_id):
                yield message
        finally:
            await handler.unsubscribe(job_id, session_id)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============ 便捷工具 ============

@router.post("/demo/create", tags=["Demo"])
async def create_demo_job():
    """
    创建演示用训练作业
    
    自动生成模拟数据的演示作业
    """
    from visualization.realtime import create_demo_job
    
    import uuid
    job_id = f"demo-{uuid.uuid4().hex[:6]}"
    
    await create_demo_job(job_id)
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Demo job created with simulated data",
    }


@router.get("/demo/data/{job_id}", tags=["Demo"])
async def get_demo_data(job_id: str):
    """
    获取演示数据
    """
    store = get_training_store()
    chart_gen = get_chart_generator()
    
    # 确保演示作业存在
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Demo job not found. Create it first with POST /demo/create")
    
    job_data = await store.get_job_data(job_id)
    dashboard = chart_gen.generate_dashboard(job_data)
    
    return dashboard
