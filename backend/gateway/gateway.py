"""
AI Gateway模块 v2.3
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import asyncio
import httpx
from enum import Enum

class ProviderType(str, Enum):
    """提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"

@dataclass
class Provider:
    """LLM提供商"""
    provider_id: str
    name: str
    provider_type: ProviderType
    base_url: str
    api_key: str
    models: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小越高
    rate_limit: Dict = field(default_factory=dict)  # rpm, tpm

@dataclass
class Route:
    """路由规则"""
    route_id: str
    name: str
    patterns: List[str] = field(default_factory=list)  # 模型名匹配
    provider_id: Optional[str] = None
    weight: int = 100  # 流量权重

@dataclass
class RequestLog:
    """请求日志"""
    log_id: str
    provider_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str
    cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UsageStats:
    """使用统计"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0

class AIGateway:
    """AI Gateway - LLM API网关"""
    
    def __init__(self):
        self.providers: Dict[str, Provider] = {}
        self.routes: Dict[str, Route] = {}
        self.request_logs: List[RequestLog] = []
        self.usage_stats: Dict[str, UsageStats] = {}  # provider_id -> stats
        self.rate_limiter = RateLimiter()
        self.cost_calculator = CostCalculator()
        
        # 默认提供商
        self._init_default_providers()
    
    def _init_default_providers(self):
        """初始化默认提供商"""
        # OpenAI
        self.register_provider(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="",  # 需要配置
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            priority=0,
            rate_limit={"rpm": 500, "tpm": 1000000}
        )
        
        # Anthropic
        self.register_provider(
            name="Anthropic",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key="",  # 需要配置
            models=["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
            priority=1,
            rate_limit={"rpm": 1000, "tpm": 5000000}
        )
    
    # 提供商管理
    def register_provider(
        self,
        name: str,
        provider_type: ProviderType,
        base_url: str,
        api_key: str,
        models: Optional[List[str]] = None,
        priority: int = 0,
        rate_limit: Optional[Dict] = None
    ) -> Provider:
        """注册提供商"""
        provider = Provider(
            provider_id=str(uuid4()),
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            api_key=api_key,
            models=models or [],
            priority=priority
        )
        
        self.providers[provider.provider_id] = provider
        self.usage_stats[provider.provider_id] = UsageStats()
        
        return provider
    
    def get_provider(self, provider_id: str) -> Optional[Provider]:
        """获取提供商"""
        return self.providers.get(provider_id)
    
    def list_providers(self, enabled: Optional[bool] = None) -> List[Provider]:
        """列出提供商"""
        providers = list(self.providers.values())
        if enabled is not None:
            providers = [p for p in providers if p.enabled == enabled]
        return sorted(providers, key=lambda p: p.priority)
    
    def list_routes(self) -> List[Route]:
        """列出路由"""
        return list(self.routes.values())
    
    def update_provider(self, provider_id: str, **kwargs) -> bool:
        """更新提供商"""
        provider = self.providers.get(provider_id)
        if not provider:
            return False
        
        for key, value in kwargs.items():
            if hasattr(provider, key):
                setattr(provider, key, value)
        
        return True
    
    # 路由管理
    def create_route(
        self,
        name: str,
        patterns: List[str],
        provider_id: Optional[str] = None,
        weight: int = 100
    ) -> Route:
        """创建路由规则"""
        route = Route(
            route_id=str(uuid4()),
            name=name,
            patterns=patterns,
            provider_id=provider_id,
            weight=weight
        )
        
        self.routes[route.route_id] = route
        return route
    
    def get_route_for_model(self, model: str) -> Optional[Route]:
        """根据模型名获取路由"""
        for route in self.routes.values():
            for pattern in route.patterns:
                if pattern in model or model.endswith(pattern.replace("*", "")):
                    return route
        return None
    
    # 请求处理
    async def chat_completions(
        self,
        model: str,
        messages: List[Dict],
        provider_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """发送Chat Completions请求"""
        # 1. 选择提供商
        if not provider_id:
            route = self.get_route_for_model(model)
            provider_id = route.provider_id if route else None
        
        if not provider_id:
            # 使用最高优先级可用提供商
            available = [p for p in self.list_providers(enabled=True)]
            if not available:
                raise ValueError("No available providers")
            provider = available[0]
            provider_id = provider.provider_id
        else:
            provider = self.get_provider(provider_id)
            if not provider or not provider.enabled:
                raise ValueError(f"Provider {provider_id} not available")
        
        # 2. 检查速率限制
        await self.rate_limiter.check(provider_id, provider.rate_limit)
        
        # 3. 发送请求
        start_time = datetime.utcnow()
        try:
            if provider.provider_type == ProviderType.OPENAI:
                response = await self._call_openai(provider, model, messages, **kwargs)
            elif provider.provider_type == ProviderType.ANTHROPIC:
                response = await self._call_anthropic(provider, model, messages, **kwargs)
            else:
                response = await self._call_local(provider, model, messages, **kwargs)
            
            # 4. 记录日志
            self._log_request(provider_id, model, response, start_time, "success")
            
            return response
            
        except Exception as e:
            self._log_request(provider_id, model, {}, start_time, "error", str(e))
            raise
    
    async def _call_openai(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """调用OpenAI API"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def _call_anthropic(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """调用Anthropic API"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{provider.base_url}/messages",
                headers={
                    "x-api-key": provider.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def _call_local(
        self,
        provider: Provider,
        model: str,
        messages: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """调用本地API"""
        # 本地模型调用
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{provider.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()
    
    def _log_request(
        self,
        provider_id: str,
        model: str,
        response: Dict,
        start_time: datetime,
        status: str,
        error: str = ""
    ):
        """记录请求日志"""
        # 计算token使用
        prompt_tokens = 0
        completion_tokens = 0
        
        if status == "success" and "usage" in response:
            usage = response["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        cost = self.cost_calculator.calculate(provider_id, prompt_tokens, completion_tokens)
        
        log = RequestLog(
            log_id=str(uuid4()),
            provider_id=provider_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status=status,
            cost=cost
        )
        
        self.request_logs.append(log)
        
        # 更新统计
        stats = self.usage_stats.get(provider_id, UsageStats())
        stats.total_requests += 1
        stats.total_tokens += prompt_tokens + completion_tokens
        stats.total_cost += cost
        stats.avg_latency_ms = (stats.avg_latency_ms * (stats.total_requests - 1) + latency_ms) / stats.total_requests
        if status == "error":
            stats.error_count += 1
    
    # 使用统计
    def get_usage_stats(self, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """获取使用统计"""
        if provider_id:
            stats = self.usage_stats.get(provider_id)
            if stats:
                return {
                    "provider_id": provider_id,
                    "total_requests": stats.total_requests,
                    "total_tokens": stats.total_tokens,
                    "total_cost": round(stats.total_cost, 4),
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                    "error_count": stats.error_count
                }
            return {}
        else:
            return {
                pid: {
                    "total_requests": s.total_requests,
                    "total_tokens": s.total_tokens,
                    "total_cost": round(s.total_cost, 4)
                }
                for pid, s in self.usage_stats.items()
            }
    
    def get_cost_breakdown(self, days: int = 30) -> Dict[str, Any]:
        """获取成本分析"""
        # 简化实现
        return {
            "total_cost": sum(s.total_cost for s in self.usage_stats.values()),
            "by_provider": self.get_usage_stats(),
            "by_model": {}
        }

class RateLimiter:
    """速率限制器"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
    
    async def check(self, provider_id: str, limits: Dict[str, int]):
        """检查速率限制"""
        now = datetime.utcnow()
        minute_ago = now.timestamp() - 60
        
        if provider_id not in self.requests:
            self.requests[provider_id] = []
        
        # 清理过期记录
        self.requests[provider_id] = [
            t for t in self.requests[provider_id]
            if t.timestamp() > minute_ago
        ]
        
        # 检查RPM
        if "rpm" in limits:
            if len(self.requests[provider_id]) >= limits["rpm"]:
                raise ValueError(f"Rate limit exceeded: {limits['rpm']} RPM")
        
        self.requests[provider_id].append(now)

class CostCalculator:
    """成本计算器"""
    
    # 价格 (每1M tokens)
    PRICING = {
        "openai": {
            "gpt-4o": {"prompt": 5000, "completion": 15000},
            "gpt-4o-mini": {"prompt": 150, "completion": 600},
            "gpt-4-turbo": {"prompt": 10000, "completion": 30000},
            "gpt-3.5-turbo": {"prompt": 500, "completion": 1500},
        },
        "anthropic": {
            "claude-sonnet-4-20250514": {"prompt": 3000, "completion": 15000},
            "claude-opus-4-20250514": {"prompt": 15000, "completion": 75000},
        }
    }
    
    def calculate(
        self,
        provider_id: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """计算成本"""
        # 简化实现
        return 0.001  # 占位价格

# AIGateway实例
ai_gateway = AIGateway()
