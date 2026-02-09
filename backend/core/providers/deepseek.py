"""
DeepSeek Provider - Multi-LLM Provider Support v3.0

提供DeepSeek模型的集成支持
"""
from typing import List, Dict, AsyncGenerator, Optional
from abc import ABC, abstractmethod
import httpx
from core.providers.base import BaseLLMProvider


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek提供商实现
    
    支持的模型:
    - deepseek-chat
    - deepseek-coder
    - deepseek-math
    - deepseek-reasoner
    """
    
    provider_name = "deepseek"
    
    # API基础URL
    API_BASE = "https://api.deepseek.com"
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "deepseek-chat": {
            "name": "DeepSeek Chat",
            "description": "General purpose conversational model",
            "max_tokens": 16384,
            "context_length": 65536,
            "capabilities": ["chat", "reasoning"]
        },
        "deepseek-coder": {
            "name": "DeepSeek Coder",
            "description": "Specialized code generation model",
            "max_tokens": 16384,
            "context_length": 65536,
            "capabilities": ["code", "chat"]
        },
        "deepseek-math": {
            "name": "DeepSeek Math",
            "description": "Mathematical reasoning model",
            "max_tokens": 8192,
            "context_length": 32768,
            "capabilities": ["math", "reasoning"]
        },
        "deepseek-reasoner": {
            "name": "DeepSeek Reasoner",
            "description": "Advanced reasoning model",
            "max_tokens": 16384,
            "context_length": 131072,
            "capabilities": ["reasoning", "analysis"]
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化DeepSeek提供商
        
        Args:
            api_key: DeepSeek API密钥，默认从环境变量读取
        """
        self.api_key = api_key or self._get_api_key()
        self.http_client = httpx.AsyncClient(timeout=120.0)
        
        if not self.api_key:
            print("Warning: No API key provided for DeepSeek provider")
    
    def _get_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        import os
        return os.getenv("DEEPSEEK_API_KEY")
    
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
                "capabilities": {
                    "streaming": True,
                    "functions": True
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
        调用DeepSeek API
        
        Args:
            endpoint: API端点
            payload: 请求负载
            
        Returns:
            API响应
        """
        url = f"{self.API_BASE}/{endpoint}"
        
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
        model: str = "deepseek-chat",
        temperature: float = 0.7,
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
            raise ValueError("DeepSeek API key not configured")
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        model_info = self.SUPPORTED_MODELS[model]
        max_tokens = max_tokens or model_info["max_tokens"]
        
        # 构建消息格式
        messages = kwargs.get("messages", [])
        if messages:
            # 多轮对话格式
            chat_messages = messages + [{"role": "user", "content": prompt}]
        else:
            # 单轮对话
            chat_messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "stream": False
        }
        
        response = await self._call_api("chat/completions", payload)
        
        return response["choices"][0]["message"]["content"]
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
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
            raise ValueError("DeepSeek API key not configured")
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        model_info = self.SUPPORTED_MODELS[model]
        max_tokens = max_tokens or model_info["max_tokens"]
        
        messages = kwargs.get("messages", [])
        if messages:
            chat_messages = messages + [{"role": "user", "content": prompt}]
        else:
            chat_messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "stream": True
        }
        
        url = f"{self.API_BASE}/chat/completions"
        
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
    
    async def get_token_usage(self) -> Dict:
        """
        获取API使用量
        
        Returns:
            使用量统计
        """
        payload = {"action": "user/status"}
        response = await self._call_api("user/status", payload)
        
        return {
            "total_usage": response.get("quota_used", 0),
            "total_quota": response.get("quota_limit", 0),
            "remaining": response.get("quota_remaining", 0)
        }
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.http_client.aclose()


# 便捷函数
def get_deepseek_provider(api_key: Optional[str] = None) -> DeepSeekProvider:
    """获取DeepSeek提供商实例"""
    return DeepSeekProvider(api_key)
