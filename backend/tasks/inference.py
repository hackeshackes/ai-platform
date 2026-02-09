"""
推理任务 v2.0
"""
from celery import shared_task
from core.celery_app import celery_app, TaskPriority

@shared_task(
    name="tasks.inference.run",
    queue="inference",
    priority=TaskPriority.CRITICAL
)
def run_inference(model_id: int, prompt: str, config: dict):
    """执行推理"""
    # TODO: 实现推理逻辑
    return {"status": "success", "model_id": model_id, "result": "inference result"}

@shared_task(name="tasks.inference.batch", queue="inference")
def batch_inference(model_id: int, prompts: list, config: dict):
    """批量推理"""
    results = []
    for prompt in prompts:
        result = run_inference(model_id, prompt, config)
        results.append(result)
    return {"status": "success", "count": len(results)}
