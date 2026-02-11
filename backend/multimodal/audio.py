"""
音频处理模块
支持多种第三方API的语音转文字(ASR)
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import httpx

from .base import (
    BaseAudioModel,
    MultimodalProvider,
    MultimodalResult,
    MediaType
)


class OpenAIAudioModel(BaseAudioModel):
    """OpenAI Whisper 语音转文字模型"""
    
    API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    def __init__(self, model: str = "whisper-1", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.OPENAI, model, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    async def transcribe(
        self,
        audio_path: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> MultimodalResult:
        """使用Whisper转录音频"""
        try:
            audio_data, filename = self._prepare_audio_data(audio_path)
            
            async with httpx.AsyncClient() as client:
                files = {"file": (filename, audio_data, "audio/mpeg")}
                data = {"model": self.model}
                
                if language:
                    data["language"] = language
                
                response = await client.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                    timeout=120.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                return MultimodalResult(
                    success=True,
                    content=result["text"],
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
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        """转录音频"""
        return await self.transcribe(
            audio_path,
            language=kwargs.get("language")
        )
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIAudioModel 不支持图像处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIAudioModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("OpenAIAudioModel 不支持内容生成")
    
    def _prepare_audio_data(self, audio_path: Union[str, Path, bytes]) -> tuple:
        """准备音频数据"""
        if isinstance(audio_path, bytes):
            return audio_path, "audio.mp3"
        elif isinstance(audio_path, str):
            path = Path(audio_path)
            with open(path, "rb") as f:
                return f.read(), path.name
        return b"audio.mp3"


class AnthropicAudioModel(BaseAudioModel):
    """Anthropic 音频模型 (Claude 3.5支持音频)"""
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.ANTHROPIC, model, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    async def transcribe(
        self,
        audio_path: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> MultimodalResult:
        """使用Claude转录音频"""
        try:
            audio_data = self._prepare_audio_data(audio_path)
            
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
                                    {
                                        "type": "audio",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "audio/mp3",
                                            "data": audio_data
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": "请将这段音频转录成文字，保留标点符号和分段。"
                                    }
                                ]
                            }
                        ]
                    },
                    timeout=120.0
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
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        return await self.transcribe(
            audio_path,
            language=kwargs.get("language")
        )
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicAudioModel 不支持图像处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicAudioModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("AnthropicAudioModel 不支持内容生成")
    
    def _prepare_audio_data(self, audio_path: Union[str, Path, bytes]) -> str:
        if isinstance(audio_path, bytes):
            return base64.b64encode(audio_path).decode("utf-8")
        elif isinstance(audio_path, str):
            audio_path = Path(audio_path)
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""


class GoogleAudioModel(BaseAudioModel):
    """Google 语音转文字模型"""
    
    API_URL = "https://speech.googleapis.com/v1/speech:recognize"
    
    def __init__(self, model: str = "latest", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.GOOGLE, model, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
    
    async def transcribe(
        self,
        audio_path: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> MultimodalResult:
        """使用Google Speech-to-Text转录音频"""
        try:
            audio_data = self._prepare_audio_data(audio_path)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    params={"key": self.api_key},
                    json={
                        "config": {
                            "encoding": "LINEAR16",
                            "sampleRateHertz": 16000,
                            "languageCode": language or "zh-CN",
                            "enableAutomaticPunctuation": True
                        },
                        "audio": {
                            "content": audio_data
                        }
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                if data.get("results"):
                    content = "".join(
                        alt["transcript"]
                        for result in data["results"]
                        for alt in result["alternatives"]
                    )
                    
                    return MultimodalResult(
                        success=True,
                        content=content,
                        provider=self.provider.value,
                        model=self.model
                    )
                else:
                    return MultimodalResult(
                        success=False,
                        content=None,
                        provider=self.provider.value,
                        model=self.model,
                        error="未能识别到音频内容"
                    )
        except Exception as e:
            return MultimodalResult(
                success=False,
                content=None,
                provider=self.provider.value,
                model=self.model,
                error=str(e)
            )
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        return await self.transcribe(
            audio_path,
            language=kwargs.get("language")
        )
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleAudioModel 不支持图像处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleAudioModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("GoogleAudioModel 不支持内容生成")
    
    def _prepare_audio_data(self, audio_path: Union[str, Path, bytes]) -> str:
        if isinstance(audio_path, bytes):
            return base64.b64encode(audio_path).decode("utf-8")
        elif isinstance(audio_path, str):
            audio_path = Path(audio_path)
            with open(audio_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""


class DeepSeekAudioModel(BaseAudioModel):
    """DeepSeek 音频模型"""
    
    API_URL = "https://api.deepseek.com/v1/audio/transcriptions"
    
    def __init__(self, model: str = "deepseek-asr", api_key: Optional[str] = None):
        super().__init__(MultimodalProvider.DEEPSEEK, model, api_key)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    
    async def transcribe(
        self,
        audio_path: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> MultimodalResult:
        """使用DeepSeek ASR转录音频"""
        try:
            audio_data, filename = self._prepare_audio_data(audio_path)
            
            async with httpx.AsyncClient() as client:
                files = {"file": (filename, audio_data, "audio/mpeg")}
                data = {"model": self.model}
                
                if language:
                    data["language"] = language
                
                response = await client.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                    timeout=120.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                return MultimodalResult(
                    success=True,
                    content=result["text"],
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
    
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        return await self.transcribe(
            audio_path,
            language=kwargs.get("language")
        )
    
    async def analyze_image(
        self,
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekAudioModel 不支持图像处理")
    
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekAudioModel 不支持视频处理")
    
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        raise NotImplementedError("DeepSeekAudioModel 不支持内容生成")
    
    def _prepare_audio_data(self, audio_path: Union[str, Path, bytes]) -> tuple:
        if isinstance(audio_path, bytes):
            return audio_path, "audio.mp3"
        elif isinstance(audio_path, str):
            path = Path(audio_path)
            with open(path, "rb") as f:
                return f.read(), path.name
        return b"audio.mp3"


def create_audio_model(provider: str, model: str = None, api_key: str = None) -> BaseAudioModel:
    """工厂函数：创建音频模型"""
    providers = {
        "openai": OpenAIAudioModel,
        "anthropic": AnthropicAudioModel,
        "google": GoogleAudioModel,
        "deepseek": DeepSeekAudioModel
    }
    
    if provider not in providers:
        raise ValueError(f"不支持的提供商: {provider}")
    
    model_map = {
        "openai": "whisper-1",
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "latest",
        "deepseek": "deepseek-asr"
    }
    
    return providers[provider](model=model or model_map[provider], api_key=api_key)
