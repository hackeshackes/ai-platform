"""
Providers Package - Multi-LLM Provider Support v3.0

LLM提供商包 - 使用延迟导入避免依赖问题
"""
from .base import BaseLLMProvider, ProviderCapability
from .registry import (
    ProviderRegistry,
    register_provider,
    get_registry,
    get_provider,
    list_providers,
    list_available_providers,
    get_all_models,
    router
)

# 延迟导入提供商（避免依赖缺失导致整个模块不可用）
def _register_providers():
    """延迟注册所有提供商"""
    try:
        from .google import GoogleProvider
        registry = ProviderRegistry()
        registry.register("google", GoogleProvider)
    except ImportError:
        pass
    
    try:
        from .deepseek import DeepSeekProvider
        registry = ProviderRegistry()
        registry.register("deepseek", DeepSeekProvider)
    except ImportError:
        pass
    
    try:
        from .meta import MetaProvider
        registry = ProviderRegistry()
        registry.register("meta", MetaProvider)
    except ImportError:
        pass

# 注册提供商
_register_providers()

__all__ = [
    # Base Classes
    "BaseLLMProvider",
    "ProviderCapability",
    # Registry
    "ProviderRegistry",
    "register_provider",
    "get_registry",
    "get_provider",
    "list_providers",
    "list_available_providers",
    "get_all_models",
    # Router
    "router",
]
