"""
AI Platform Backend - Main Entry Point
大模型全生命周期管理平台后端
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.routes import router

# V9 API Routes
from api.endpoints.v9_adaptive import router as v9_adaptive_router
from api.endpoints.v9_federated import router as v9_federated_router
from api.endpoints.v9_decision import router as v9_decision_router

# V9 API Routes
from api.endpoints.v9_adaptive import router as v9_adaptive_router
from api.endpoints.v9_federated import router as v9_federated_router
from api.endpoints.v9_decision import router as v9_decision_router
from core.config import settings as config_settings
from core.logging import logger

# 创建FastAPI应用
app = FastAPI(
    title="AI Platform API",
    description="大模型全生命周期管理平台 API",
    version="2.3.0-beta",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")

# V9 Routes
app.include_router(v9_adaptive_router, prefix="/api/v1")
app.include_router(v9_federated_router, prefix="/api/v1")
app.include_router(v9_decision_router, prefix="/api/v1")

# API根路径 - 列出所有可用端点
@app.get("/api/v1")
async def api_root():
    return {
        "message": "AI Platform API v2.3",
        "version": "2.3.0-beta",
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users",
            "projects": "/api/v1/projects",
            "experiments": "/api/v1/experiments",
            "tasks": "/api/v1/tasks",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "health": "/api/v1/health",
            "gateway": "/api/v1/gateway",
            "assistant": "/api/v1/assistant",
            "judges": "/api/v1/judges",
            "ray": "/api/v1/ray",
            "optimization": "/api/v1/optimization"
        },
        "documentation": "/docs"
    }

# 根路径
@app.get("/")
async def root():
    return {
        "message": "AI Platform API",
        "version": "2.3.0-beta",
        "docs": "/docs"
    }

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
