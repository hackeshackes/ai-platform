"""
兼容性路由配置 - 修复v6/v7/v8/v9模块导入问题
"""
import sys
import logging

logger = logging.getLogger(__name__)

def safe_import(module_name: str, router_attr: str = "router", prefix: str = None, tags: list = None):
    """安全导入模块"""
    try:
        # 尝试多种导入路径
        paths = [
            f"api.endpoints.{module_name}",
            f"backend.{module_name}",
            f"backend.api.{module_name}",
        ]
        
        for path in paths:
            try:
                module = __import__(path, fromlist=[router_attr])
                router = getattr(module, router_attr)
                logger.info(f"✅ {module_name} imported from {path}")
                return router
            except ImportError:
                continue
        
        # 模块不存在，创建空路由
        from fastapi import APIRouter
        router = APIRouter()
        logger.warning(f"⚠️ {module_name} not found, using empty router")
        return router
        
    except Exception as e:
        logger.error(f"❌ {module_name} import failed: {e}")
        from fastapi import APIRouter
        return APIRouter()

def create_compat_router():
    """创建兼容路由"""
    from fastapi import APIRouter
    router = APIRouter()
    
    # v6 模块 (标记为可选)
    optional_modules = [
        ("jwt_handler", "router", "/auth", ["Auth"]),
        ("auth", "router", "/auth", ["Auth"]),
        ("ss_handler", "router", "/auth/sso", ["SSO"]),
        ("tenant", "router", "/tenant", ["Tenant"]),
        ("gateway", "router", "/gateway", ["Gateway"]),
        ("lowcode", "router", "/lowcode", ["Lowcode"]),
        ("distributed", "router", "/distributed", ["Distributed"]),
        ("registry", "router", "/registry", ["Registry"]),
    ]
    
    for module_name, attr, prefix, tags in optional_modules:
        try:
            router_temp = safe_import(module_name, attr)
            if prefix:
                router.include_router(router_temp, prefix=prefix, tags=tags)
        except Exception:
            pass
    
    return router

# 创建兼容路由实例
compat_router = create_compat_router()
