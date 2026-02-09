"""
Meta Llama Provider - Multi-LLM Provider Support v3.0

提供Meta Llama模型的集成支持
"""
from typing import List, Dict, AsyncGenerator, Optional
from abc import ABC, abstractmethod
import httpx
from core.providers.base import BaseLLMProvider


class MetaProvider(BaseLLMProvider):
    """
    Meta Llama提供商实现
    
    支持的模型:
    - llama-3.1-8b-instruct
    - llama-3.1-70b-instruct
    - llama-3.1-405b-instruct
    - llama-3.2-1b-instruct
    - llama-3.2-3b-instruct
    - llama-3.3-70b-instruct
    - llama-4-scout
    - llama-4-marble
    """
    
    provider_name = "meta"
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "llama-3.1-8b-instruct": {
            "name": "Llama 3.1 8B Instruct",
            "description": "Efficient instruction-tuned model",
            "max_tokens": 4096,
            "context_length": 131072,
            "type": "open"
        },
        "llama-3.1-70b-instruct": {
            "name": "Llama 3.1 70B Instruct",
            "description": "Powerful instruction-tuned model",
            "max_tokens": 4096,
            "context_length": 131072,
            "type": "open"
        },
        "llama-3.1-405b-instruct": {
            "name": "Llama 3.1 405B Instruct",
            "description": "Frontier instruction-tuned model",
            "max_tokens": 4096,
            "context_length": 131072,
            "type": "open"
        },
        "llama-3.2-1b-instruct": {
            "name": "Llama 3.2 1B Instruct",
            "description": "Lightweight efficient model",
            "max_tokens": 2048,
            "context_length": 131072,
            "type": "open"
        },
        "llama-3.2-3b-instruct": {
            "name": "Llama 3.2 3B Instruct",
            "description": "Compact instruction-tuned model",
            "max_tokens": 2048,
            "context_length": 131072,
            "type": "open"
        },
        "llama-3.3-70b-instruct": {
            "name": "Llama 3.3 70B Instruct",
            "description": "Enhanced instruction-tuned model",
            "max_tokens": 8192,
            "context_length": 131072,
            "type": "open"
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化Meta Llama提供商
        
        Args:
            api_key: Meta API密钥 (Llama API)
            base_url: API基础URL，默认使用Llama API
        """
        self.api_key = api_key or self._get_api_key()
        self.base_url = base_url or self._get_base_url()
        self.http_client = httpx.AsyncClient(timeout=300.0)
        
        if not self.api_key:
            print("Warning: No API key provided for Meta provider")
    
    def _get_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        import os
        return os.getenv("META_API_KEY") or os.getenv("LLAMA_API_KEY")
    
    def _get_base_url(self) -> str:
        """获取API基础URL"""
        import os
        return os.getenv("META_API_BASE", "https://api.llama.com/compatible/v1")
    
    def get_models(self) -> List[Dict]:
        """
        获取支持的模型列表
        
        Returns:
            模型信息列表
        """
        return [
            {
                "id": model_id,
                "name": info["name"],
                "description": info["description"],
                "max_tokens": info["max_tokens"],
                "context_length": info["context_length"],
                "provider": self.provider_name,
                "model_type": info["type"],
                "capabilities": {
                    "streaming": True,
                    "tools": True,
                    "json_mode": True
                }
            }
            for model_id, info in self.SUPPORTED_MODELS.items()
        ]
    
    async def _call_api(
        self,
        endpoint: str,
        payload: Dict
    ) -> Dict:
        """
        调用Meta Llama API
        
        Args:
            endpoint: API端点
            payload: 请求负载
            
        Returns:
            API响应
        """
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = await self.http_client.post(
            url,
            json=payload,
            headers=headers
        )
        
        response.raise_for_status()
        return response.json()
    
    async def generate(
        self,
        prompt: str,
        model: str = "llama-3.1-8b-instruct",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        if not self.api_key:
            raise ValueError("Meta/Llama API key not configured")
        
        if model not in self.SUPPORTED_MODELS:
            # 尝试使用动态模型
            pass
        
        model_info = self.SUPPORTED_MODELS.get(model, {})
        max_tokens = max_tokens or model_info.get("max_tokens", 4096)
        
        # 构建消息格式
        messages = kwargs.get("messages", [])
        if messages:
            chat_messages = messages + [{"role": "user", "content": prompt}]
        else:
            chat_messages = [{"role": "user", "content": prompt}]
        
        # 添加系统消息
        if kwargs.get("system_prompt"):
            chat_messages.insert(0, {
                "role": "system",
                "content": kwargs["system_prompt"]
            })
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "stream": False,
            "response_format": kwargs.get("response_format", {"type": "text"})
        }
        
        response = await self._call_api("chat/completions", payload)
        
        return response["choices"][0]["message"]["content"]
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = "llama-3.1-8b-instruct",
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Yields:
            生成的文本块
        """
        if not self.api_key:
            raise ValueError("Meta/Llama API key not configured")
        
        model_info = self.SUPPORTED_MODELS.get(model, {})
        max_tokens = max_tokens or model_info.get("max_tokens", 4096)
        
        messages = kwargs.get("messages", [])
        if messages:
            chat_messages = messages + [{"role": "user", "content": prompt}]
        else:
            chat_messages = [{"role": "user", "content": prompt}]
        
        if kwargs.get("system_prompt"):
            chat_messages.insert(0, {
                "role": "system",
                "content": kwargs["system_prompt"]
            })
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.9),
            "stream": True,
            "response_format": kwargs.get("response_format", {"type": "text"})
        }
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with self.http_client.stream(
            "POST",
            url,
            json=payload,
            headers=headers
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() != "[DONE]":
                        chunk = __import__("json").loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
    
    async def get_model_info(self, model: str) -> Dict:
        """
        获取特定模型信息
        
        Args:
            model: 模型ID
            
        Returns:
            模型详细信息
        """
        if model in self.SUPPORTED_MODELS:
            return {
                "id": model,
                **self.SUPPORTED_MODELS[model],
                "provider": self.provider_name
            }
        
        # 尝试从API获取
        try:
            response = await self._call_api("models", {})
            for m in response.get("data", []):
                if m["id"] == model:
                    return {
                        "id": model,
                        "name": m.get("id", model),
                        "provider": self.provider_name,
                        "capabilities": m.get("capabilities", {})
                    }
        except Exception:
            pass
        
        return {
            "id": model,
            "name": model,
            "provider": self.provider_name,
            "capabilities": {}
        }
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.http_client.aclose()


# 便捷函数
def get_meta_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> MetaProvider:
    """获取Meta提供商实例"""
    return MetaProvider(api_key, base_url)
