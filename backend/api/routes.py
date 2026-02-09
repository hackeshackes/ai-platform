"""API Router"""
from fastapi import APIRouter

# 导入所有端点
from api.endpoints import auth, users, projects, experiments, tasks, datasets, models, health, gpu, metrics, training, inference, settings, versions, quality, permissions, pipeline

router = APIRouter()

# 认证
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# 用户
router.include_router(users.router, prefix="/users", tags=["Users"])

# 项目
router.include_router(projects.router, prefix="/projects", tags=["Projects"])

# 实验
router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])

# 任务
router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])

# 数据集
router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])

# 模型
router.include_router(models.router, prefix="/models", tags=["Models"])

# GPU监控
router.include_router(gpu.router, prefix="/gpu", tags=["GPU"])

# 训练指标
router.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

# 训练任务
router.include_router(training.router, prefix="/training", tags=["Training"])

# 推理服务
router.include_router(inference.router, prefix="/inference", tags=["Inference"])

# 系统设置
router.include_router(settings.router, prefix="/settings", tags=["Settings"])

# v1.1: 数据集版本
router.include_router(versions.router, prefix="/datasets", tags=["Versions"])

# v1.1: 数据质量检查
router.include_router(quality.router, prefix="/datasets", tags=["Quality"])

# v1.1: 权限管理
router.include_router(permissions.router, prefix="/permissions", tags=["Permissions"])

# v2.0 Phase 2: Pipeline编排
router.include_router(pipeline.router, prefix="/pipelines", tags=["Pipelines"])

# v2.0 Phase 2: CI/CD
try:
    from api.endpoints.cicd import deploy, distributed
    router.include_router(deploy.router, prefix="/cicd", tags=["CI/CD"])
    router.include_router(distributed.router, prefix="/cluster", tags=["Cluster"])
    CICD_ENABLED = True
except ImportError:
    CICD_ENABLED = False

# ML集成 - 使用条件导入，Docker环境自动启用
try:
    from api.endpoints import mlflow, ollama
    router.include_router(mlflow.router, prefix="/mlflow", tags=["MLflow"])
    router.include_router(ollama.router, prefix="/ollama", tags=["Ollama"])
    ML_INTEGRATION_ENABLED = True
except ImportError:
    ML_INTEGRATION_ENABLED = False

# 健康检查
router.include_router(health.router, prefix="/health", tags=["Health"])

# 系统信息端点
@router.get("/info")
async def system_info():
    """获取系统集成状态"""
    return {
        "ml_integration": ML_INTEGRATION_ENABLED,
        "services": {
            "backend": "running",
            "database": "postgresql",
            "cache": "redis",
            "pipeline": "v2.0"
        },
        "version": "2.0.0-beta"
    }
