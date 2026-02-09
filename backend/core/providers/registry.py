"""
Provider Registry - Multi-LLM Provider Support v3.0

LLM提供商注册表，管理所有可用提供商
"""
from typing import Dict, List, Optional, Type
from core.providers.base import BaseLLMProvider


class ProviderRegistry:
    """
    LLM提供商注册表
    
    提供统一的接口来管理和访问所有注册的LLM提供商
    """
    
    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _instances: Dict[str, BaseLLMProvider] = {}
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化注册表"""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._register_default_providers()
    
    def _register_default_providers(self):
        """注册默认提供商"""
        # 延迟导入以避免循环依赖
        pass
    
    @classmethod
    def register(
        cls,
        provider_name: str,
        provider_class: Type[BaseLLMProvider]
    ):
        """
        注册LLM提供商
        
        Args:
            provider_name: 提供商标识符
            provider_class: 提供商类
        """
        cls._providers[provider_name] = provider_class
        print(f"Registered provider: {provider_name}")
    
    @classmethod
    def get_provider_class(
        cls,
        provider_name: str
    ) -> Optional[Type[BaseLLMProvider]]:
        """
        获取提供商类
        
        Args:
            provider_name: 提供商标识符
            
        Returns:
            提供商类，如果未找到则返回None
        """
        return cls._providers.get(provider_name)
    
    def get_provider(
        self,
        provider_name: str,
        **kwargs
    ) -> Optional[BaseLLMProvider]:
        """
        获取提供商实例
        
        Args:
            provider_name: 提供商标识符
            **kwargs: 初始化参数
            
        Returns:
            提供商实例
        """
        # 检查是否已有缓存实例
        cache_key = f"{provider_name}_{id(kwargs)}"
        if cache_key in self._instances:
            return self._instances[cache_key]
        
        # 获取提供商类
        provider_class = self.get_provider_class(provider_name)
        if provider_class is None:
            return None
        
        # 创建实例
        instance = provider_class(**kwargs)
        self._instances[cache_key] = instance
        
        return instance
    
    def get_provider_models(self, provider_name: str) -> Optional[List[Dict]]:
        """
        获取某提供商的模型列表
        
        Args:
            provider_name: 提供商标识符
            
        Returns:
            模型列表，如果提供商未找到则返回None
        """
        provider = self.get_provider(provider_name)
        if provider is None:
            return None
        return provider.get_models()
    
    def list_providers(self) -> List[str]:
        """
        列出所有注册的提供商
        
        Returns:
            提供商标识符列表
        """
        return list(self._providers.keys())
    
    def list_available_providers(self) -> List[Dict]:
        """
        列出所有可用的提供商及其模型
        
        Returns:
            提供商信息列表
        """
        available = []
        
        for name in self._providers.keys():
            provider = self.get_provider(name)
            if provider:
                models = provider.get_models()
                available.append({
                    "provider": name,
                    "name": provider.name,
                    "models_count": len(models),
                    "models": [
                        {
                            "id": m["id"],
                            "name": m["name"]
                        }
                        for m in models[:5]  # 只返回前5个
                    ]
                })
        
        return available
    
    def get_all_models(self) -> List[Dict]:
        """
        获取所有提供商的所有模型
        
        Returns:
            模型信息列表
        """
        all_models = []
        
        for provider_name in self._providers.keys():
            provider = self.get_provider(provider_name)
            if provider:
                models = provider.get_models()
                for model in models:
                    model["provider"] = provider_name
                    all_models.append(model)
        
        return all_models
    
    def get_models_by_capability(
        self,
        capability: str
    ) -> List[Dict]:
        """
        根据能力获取模型
        
        Args:
            capability: 能力标识符
            
        Returns:
            支持该能力的模型列表
        """
        all_models = self.get_all_models()
        
        return [
            model
            for model in all_models
            if model.get("capabilities", {}).get(capability, False)
        ]
    
    def search_models(
        self,
        query: str,
        provider: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索模型
        
        Args:
            query: 搜索查询
            provider: 可选，限定提供商
            
        Returns:
            匹配的模型列表
        """
        all_models = self.get_all_models()
        query_lower = query.lower()
        
        filtered = all_models
        if provider:
            filtered = [
                m for m in filtered
                if m.get("provider") == provider
            ]
        
        return [
            m for m in filtered
            if query_lower in m.get("name", "").lower() or
               query_lower in m.get("id", "").lower() or
               query_lower in m.get("description", "").lower()
        ]
    
    def remove_instance(self, provider_name: str, **kwargs):
        """
        移除缓存的实例
        
        Args:
            provider_name: 提供商标识符
            **kwargs: 初始化参数
        """
        cache_key = f"{provider_name}_{id(kwargs)}"
        if cache_key in self._instances:
            instance = self._instances.pop(cache_key)
            if hasattr(instance, 'close'):
                asyncio.get_event_loop().create_task(instance.close())
    
    def clear_cache(self):
        """清除所有缓存实例"""
        for instance in self._instances.values():
            if hasattr(instance, 'close'):
                try:
                    asyncio.get_event_loop().create_task(instance.close())
                except Exception:
                    pass
        self._instances.clear()
    
    @classmethod
    def reset(cls):
        """重置注册表（主要用于测试）"""
        cls._providers.clear()
        cls._instance = None


# 全局注册表实例
registry = ProviderRegistry()


def register_provider(provider_name: str):
    """
    装饰器：注册LLM提供商
    
    Usage:
        @register_provider("openai")
        class OpenAIProvider(BaseLLMProvider):
            ...
    """
    def decorator(provider_class: Type[BaseLLMProvider]):
        ProviderRegistry.register(provider_name, provider_class)
        return provider_class
    return decorator


def get_registry() -> ProviderRegistry:
    """获取全局注册表"""
    return registry


def get_provider(provider_name: str, **kwargs) -> Optional[BaseLLMProvider]:
    """便捷函数：获取提供商实例"""
    return registry.get_provider(provider_name, **kwargs)


def list_providers() -> List[str]:
    """便捷函数：列出所有注册的提供商"""
    return registry.list_providers()


def list_available_providers() -> List[Dict]:
    """便捷函数：列出所有可用的提供商"""
    return registry.list_available_providers()


def get_all_models() -> List[Dict]:
    """便捷函数：获取所有模型"""
    return registry.get_all_models()


# FastAPI Router
from fastapi import APIRouter

router = APIRouter()

@router.get("")
async def list_all_providers():
    """列出所有LLM提供商"""
    return {
        "providers": list_available_providers(),
        "total": len(list_providers())
    }

@router.get("/{provider_name}/models")
async def get_provider_models(provider_name: str):
    """获取某提供商的模型列表"""
    models = registry.get_provider_models(provider_name)
    if models is None:
        return {"error": f"Provider {provider_name} not found", "available_providers": list_providers()}
    return {
        "provider": provider_name,
        "models": models
    }

@router.get("/{provider_name}/status")
async def get_provider_status(provider_name: str):
    """获取某提供商的状态"""
    return {
        "provider": provider_name,
        "registered": provider_name in list_providers(),
        "available": provider_name in registry.get_provider_class(provider_name).__name__ if registry.get_provider_class(provider_name) else False
    }
