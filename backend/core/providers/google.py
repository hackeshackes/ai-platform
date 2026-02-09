"""
Google Gemini Provider - Multi-LLM Provider Support v3.0

提供Google Gemini模型的集成支持
"""
from typing import List, Dict, AsyncGenerator, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai
from core.providers.base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini提供商实现
    
    支持的模型:
    - gemini-pro
    - gemini-pro-vision
    - gemini-1.0-pro
    - gemini-1.5-pro
    """
    
    provider_name = "google"
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "gemini-pro": {
            "name": "Gemini Pro",
            "description": "General purpose model for text generation",
            "max_tokens": 32768,
            "supports_vision": False
        },
        "gemini-pro-vision": {
            "name": "Gemini Pro Vision",
            "description": "Multimodal model supporting images",
            "max_tokens": 16384,
            "supports_vision": True
        },
        "gemini-1.0-pro": {
            "name": "Gemini 1.0 Pro",
            "description": "Stable Gemini Pro model",
            "max_tokens": 32768,
            "supports_vision": False
        },
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "description": "Latest Gemini Pro with extended context",
            "max_tokens": 1048576,
            "supports_vision": True
        },
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "description": "Fast and efficient Gemini model",
            "max_tokens": 1048576,
            "supports_vision": True
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化Google提供商
        
        Args:
            api_key: Google API密钥，默认从环境变量读取
        """
        self.api_key = api_key or self._get_api_key()
        if self.api_key:
            self._configure_client()
        else:
            print("Warning: No API key provided for Google provider")
    
    def _get_api_key(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        import os
        return os.getenv("GOOGLE_API_KEY")
    
    def _configure_client(self):
        """配置Google Generative AI客户端"""
        genai.configure(api_key=self.api_key)
        self.client_configured = True
        print("Google Generative AI client configured successfully")
    
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
                "provider": self.provider_name,
                "capabilities": {
                    "vision": info["supports_vision"],
                    "streaming": True
                }
            }
            for model_id, info in self.SUPPORTED_MODELS.items()
        ]
    
    async def generate(
        self,
        prompt: str,
        model: str = "gemini-pro",
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
            raise ValueError("Google API key not configured")
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        model_info = self.SUPPORTED_MODELS[model]
        max_tokens = max_tokens or model_info["max_tokens"]
        
        # 配置generation_config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 64)
        }
        
        # 创建模型实例
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=kwargs.get("safety_settings", None)
        )
        
        # 生成响应
        response = await gemini_model.generate_content_async(prompt)
        
        return response.text
    
    async def stream_generate(
        self,
        prompt: str,
        model: str = "gemini-pro",
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
            raise ValueError("Google API key not configured")
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        
        model_info = self.SUPPORTED_MODELS[model]
        max_tokens = max_tokens or model_info["max_tokens"]
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 64)
        }
        
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        response = await gemini_model.generate_content_stream_async(prompt)
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def embed(self, text: str, model: str = "embedding-001") -> List[float]:
        """
        生成文本嵌入
        
        Args:
            text: 输入文本
            model: 嵌入模型ID
            
        Returns:
            嵌入向量
        """
        if not self.api_key:
            raise ValueError("Google API key not configured")
        
        embedding_model = genai.GenerativeModel(model_name=model)
        embedding = await embedding_model.embed_content_async(text)
        
        return embedding


# 便捷函数
def get_google_provider(api_key: Optional[str] = None) -> GoogleProvider:
    """获取Google提供商实例"""
    return GoogleProvider(api_key)
