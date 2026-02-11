"""
图像理解模块
支持多种第三方API的图像理解和分析
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx

from .base import (
    BaseVisionModel,
    MultimodalProvider,
    MultimodalResult,
    MediaType
)


class OpenAIVisionModel(BaseVisionModel):
    """OpenAI GPT-4V 图像理解模型"""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.OPENAI, model, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        """使用GPT-4V分析图像"""
        try:
            image_data = self._prepare_image_data(image_path)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                return MultimodalResult(
                    success=True,
                    content=data["choices"][0]["message"]["content"],
                    provider=self.provider.value,
                    model=self.model,
                    usage=data.get("usage")
                )
        except Exception as e:
            return MultimodalResult(
                success=False,
                content=None,
                provider=self.provider.value,
                model=self.model,
                error=str(e)
            )
    
    async def describe_image(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        """描述图像内容"""
        return await self.analyze_image(
            image_path,
            "请详细描述这张图片的内容，包括场景、物体、颜色、氛围等。"
        )
    
    async def detect_objects(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        """检测图像中的对象"""
        return await self.analyze_image(
            image_path,
            "请识别并列出图片中的所有物体和人物，包括它们的位置关系。"
        )
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVisionModel 不支持音频处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVisionModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVisionModel 不支持内容生成")
    
    def _prepare_image_data(self, image_path: Union[str, Path, bytes]) -> str:
        """准备base64编码的图像数据"""
        if isinstance(image_path, bytes):
            return base64.b64encode(image_path).decode("utf-8")
        elif isinstance(image_path, str):
            image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class AnthropicVisionModel(BaseVisionModel):
    """Anthropic Claude 3 Vision 模型"""
    
    API_URL = "https://api.anthropic.com/v1/complete"
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.ANTHROPIC, model, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        """使用Claude Vision分析图像"""
        try:
            image_data = self._prepare_image_data(image_path)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                        "max_tokens_to_sample": kwargs.get("max_tokens", 4096),
                        "media": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                return MultimodalResult(
                    success=True,
                    content=data.get("completion", ""),
                    provider=self.provider.value,
                    model=self.model
                )
        except Exception as e:
            return MultimodalResult(
                success=False,
                content=None,
                provider=self.provider.value,
                model=self.model,
                error=str(e)
            )
    
    async def describe_image(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请详细描述这张图片的内容，包括场景、物体、颜色、氛围等。"
        )
    
    async def detect_objects(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请识别并列出图片中的所有物体和人物，包括它们的位置关系。"
        )
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVisionModel 不支持音频处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVisionModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVisionModel 不支持内容生成")
    
    def _prepare_image_data(self, image_path: Union[str, Path, bytes]) -> str:
        if isinstance(image_path, bytes):
            return base64.b64encode(image_path).decode("utf-8")
        elif isinstance(image_path, str):
            image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class GoogleVisionModel(BaseVisionModel):
    """Google Gemini Pro Vision 模型"""
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.GOOGLE, model, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        """使用Gemini Vision分析图像"""
        try:
            image_data = self._prepare_image_data(image_path)
            
            url = self.API_URL.format(model=self.model)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    params={"key": self.api_key},
                    json={
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_data
                                    }
                                }
                            ]
                        }]
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                
                return MultimodalResult(
                    success=True,
                    content=content,
                    provider=self.provider.value,
                    model=self.model
                )
        except Exception as e:
            return MultimodalResult(
                success=False,
                content=None,
                provider=self.provider.value,
                model=self.model,
                error=str(e)
            )
    
    async def describe_image(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请详细描述这张图片的内容，包括场景、物体、颜色、氛围等。"
        )
    
    async def detect_objects(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请识别并列出图片中的所有物体和人物，包括它们的位置关系。"
        )
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVisionModel 不支持音频处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVisionModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVisionModel 不支持内容生成")
    
    def _prepare_image_data(self, image_path: Union[str, Path, bytes]) -> str:
        if isinstance(image_path, bytes):
            return base64.b64encode(image_path).decode("utf-8")
        elif isinstance(image_path, str):
            image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class DeepSeekVLModel(BaseVisionModel):
    """DeepSeekVL 视觉语言模型"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self, model: str = "deepseek-vl", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.DEEPSEEK, model, api_key)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        """使用DeepSeekVL分析图像"""
        try:
            image_data = self._prepare_image_data(image_path)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                return MultimodalResult(
                    success=True,
                    content=data["choices"][0]["message"]["content"],
                    provider=self.provider.value,
                    model=self.model,
                    usage=data.get("usage")
                )
        except Exception as e:
            return MultimodalResult(
                success=False,
                content=None,
                provider=self.provider.value,
                model=self.model,
                error=str(e)
            )
    
    async def describe_image(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请详细描述这张图片的内容，包括场景、物体、颜色、氛围等。"
        )
    
    async def detect_objects(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        return await self.analyze_image(
            image_path,
            "请识别并列出图片中的所有物体和人物，包括它们的位置关系。"
        )
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVLModel 不支持音频处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVLModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVLModel 不支持内容生成")
    
    def _prepare_image_data(self, image_path: Union[str, Path, bytes]) -> str:
        if isinstance(image_path, bytes):
            return base64.b64encode(image_path).decode("utf-8")
        elif isinstance(image_path, str):
            image_path = Path(image_path)
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def create_vision_model(provider: str, model: str = None, api_key: str = None) -> BaseVisionModel:
    """工厂函数：创建视觉模型"""
    providers = {
        "openai": OpenAIVisionModel,
        "anthropic": AnthropicVisionModel,
        "google": GoogleVisionModel,
        "deepseek": DeepSeekVLModel
    }
    
    if provider not in providers:
        raise ValueError(f"不支持的提供商: {provider}")
    
    model_map = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-flash",
        "deepseek": "deepseek-vl"
    }
    
    return providers[provider](model=model or model_map[provider], api_key=api_key)
