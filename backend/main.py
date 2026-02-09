"""
AI Platform Backend - Main Entry Point
大模型全生命周期管理平台后端
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api.routes import router
from core.config import settings
from core.logging import logger

# 创建FastAPI应用
app = FastAPI(
    title="AI Platform API",
    description="大模型全生命周期管理平台 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")

# API根路径 - 列出所有可用端点
@app.get("/api/v1")
async def api_root():
    return {
        "message": "AI Platform API v1",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users",
            "projects": "/api/v1/projects",
            "experiments": "/api/v1/experiments",
            "tasks": "/api/v1/tasks",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "health": "/api/v1/health"
        },
        "documentation": "/docs"
    }

# 根路径
@app.get("/")
async def root():
    return {
        "message": "AI Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
