"""
视频处理模块
支持多种第三方API的视频分析和理解
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx

from .base import (
    BaseVideoModel,
    MultimodalProvider,
    MultimodalResult,
    MediaType
)


class OpenAIVideoModel(BaseVideoModel):
    """OpenAI 视频理解模型"""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.OPENAI, model, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        """分析视频"""
        try:
            frames = await self.extract_frames(video_path, num_frames=kwargs.get("num_frames", 10))
            
            frame_contents = []
            for i, frame_data in enumerate(frames):
                frame_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_data}"
                    }
                })
            
            if prompt is None:
                prompt = "请分析这段视频的内容，包括场景变化、动作、对话等。"
            
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
                                    {"type": "text", "text": prompt}
                                ] + frame_contents
                            }
                        ],
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    },
                    timeout=120.0
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
    
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10
    ) -> List[bytes]:
        """从视频中提取关键帧 (需要本地处理)"""
        frames = []
        
        try:
            import cv2
            
            if isinstance(video_path, str):
                video_path = Path(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return frames
            
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(buffer.tobytes())
            
            cap.release()
            
        except ImportError:
            pass
        
        return frames
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVideoModel 不支持单张图像处理")
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVideoModel 不支持音频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIVideoModel 不支持内容生成")


class GoogleVideoModel(BaseVideoModel):
    """Google Gemini Pro Video 模型"""
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.GOOGLE, model, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        """使用Gemini分析视频"""
        try:
            video_data = self._prepare_video_data(video_path)
            
            if prompt is None:
                prompt = "请分析这段视频的内容，包括场景变化、动作、对话等。"
            
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
                                        "mime_type": "video/mp4",
                                        "data": video_data
                                    }
                                }
                            ]
                        }]
                    },
                    timeout=180.0
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
    
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10
    ) -> List[bytes]:
        """从视频中提取关键帧"""
        frames = []
        
        try:
            import cv2
            
            if isinstance(video_path, str):
                video_path = Path(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return frames
            
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(buffer.tobytes())
            
            cap.release()
            
        except ImportError:
            pass
        
        return frames
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVideoModel 不支持单张图像处理")
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVideoModel 不支持音频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleVideoModel 不支持内容生成")
    
    def _prepare_video_data(self, video_path: Union[str, Path]) -> str:
        """准备base64编码的视频数据"""
        if isinstance(video_path, str):
            video_path = Path(video_path)
        
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class AnthropicVideoModel(BaseVideoModel):
    """Anthropic Claude 3 Video 模型"""
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.ANTHROPIC, model, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        """使用Claude分析视频"""
        try:
            frames = await self.extract_frames(video_path, num_frames=kwargs.get("num_frames", 8))
            
            media_parts = []
            for frame_data in frames:
                media_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(frame_data).decode("utf-8")
                    }
                })
            
            if prompt is None:
                prompt = "请分析这段视频的内容，包括场景变化、动作、对话等。"
            
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
                        "max_tokens": 4096,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt}
                                ] + media_parts
                            }
                        ]
                    },
                    timeout=180.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                content = data["content"][0]["text"]
                
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
    
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 8
    ) -> List[bytes]:
        """从视频中提取关键帧"""
        frames = []
        
        try:
            import cv2
            
            if isinstance(video_path, str):
                video_path = Path(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return frames
            
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(buffer.tobytes())
            
            cap.release()
            
        except ImportError:
            pass
        
        return frames
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVideoModel 不支持单张图像处理")
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVideoModel 不支持音频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicVideoModel 不支持内容生成")


class DeepSeekVideoModel(BaseVideoModel):
    """DeepSeek 视频理解模型"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self, model: str = "deepseek-video", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.DEEPSEEK, model, api_key)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        """使用DeepSeek分析视频"""
        try:
            frames = await self.extract_frames(video_path, num_frames=kwargs.get("num_frames", 10))
            
            frame_contents = []
            for i, frame_data in enumerate(frames):
                frame_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(frame_data).decode('utf-8')}"
                    }
                })
            
            if prompt is None:
                prompt = "请分析这段视频的内容，包括场景变化、动作、对话等。"
            
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
                                    {"type": "text", "text": prompt}
                                ] + frame_contents
                            }
                        ],
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    },
                    timeout=120.0
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
    
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10
    ) -> List[bytes]:
        """从视频中提取关键帧"""
        frames = []
        
        try:
            import cv2
            
            if isinstance(video_path, str):
                video_path = Path(video_path)
            
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return frames
            
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(buffer.tobytes())
            
            cap.release()
            
        except ImportError:
            pass
        
        return frames
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVideoModel 不支持单张图像处理")
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVideoModel 不支持音频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekVideoModel 不支持内容生成")


def create_video_model(provider: str, model: str = None, api_key: str = None) -> BaseVideoModel:
    """工厂函数：创建视频模型"""
    providers = {
        "openai": OpenAIVideoModel,
        "google": GoogleVideoModel,
        "anthropic": AnthropicVideoModel,
        "deepseek": DeepSeekVideoModel
    }
    
    if provider not in providers:
        raise ValueError(f"不支持的提供商: {provider}")
    
    model_map = {
        "openai": "gpt-4o",
        "google": "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "deepseek": "deepseek-video"
    }
    
    return providers[provider](model=model or model_map[provider], api_key=api_key)
