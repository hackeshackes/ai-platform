"""
异步任务使用示例
"""
import asyncio
from .async_queue import background_task, submit_task, get_task_status

@background_task
def long_running_task(name: str, duration: int = 3):
    """长时间运行的任务"""
    import time
    time.sleep(duration)
    return f"Task {name} completed"

async def example():
    """示例"""
    # 提交后台任务
    task = submit_task("training", long_running_task, "模型训练", 5)
    print(f"Task submitted: {task.id}")
    
    # 检查任务状态
    status = get_task_status(task.id)
    print(f"Status: {status.status}")
    
    # 等待完成
    while status.status.value != "completed":
        await asyncio.sleep(1)
        status = get_task_status(task.id)
    
    print(f"Result: {status.result}")
