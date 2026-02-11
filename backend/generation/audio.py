"""
语音合成模块 (TTS)
支持多种TTS引擎的语音生成
"""

import os
import uuid
import asyncio
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .base import (
    BaseGenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse
)


class VoiceGender(str, Enum):
    """语音性别枚举"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAccent(str, Enum):
    """语音口音枚举"""
    ZH_CN = "zh-CN"
    ZH_TW = "zh-TW"
    EN_US = "en-US"
    EN_GB = "en-GB"
    JA_JP = "ja-JP"
    KO_KR = "ko-KR"


class AudioFormat(str, Enum):
    """音频格式枚举"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"


@dataclass
class TTSRequest(GenerationRequest):
    """语音合成请求"""
    text: str = ""
    voice: Optional[str] = None
    voice_gender: VoiceGender = VoiceGender.FEMALE
    accent: VoiceAccent = VoiceAccent.ZH_CN
    speed: float = 1.0  # 0.5 - 2.0
    pitch: float = 1.0  # 0.5 - 2.0
    volume: float = 1.0  # 0.0 - 1.0
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 44100
    duration_ms: Optional[int] = None


@dataclass
class TTSResponse(GenerationResponse):
    """语音合成响应"""
    audio_data: Optional[bytes] = None  # 音频数据
    audio_path: Optional[str] = None  # 音频文件路径
    duration_ms: Optional[float] = None  # 音频时长
    format: Optional[str] = None  # 音频格式
    sample_rate: Optional[int] = None  # 采样率
    characters: Optional[int] = None  # 处理的字符数


@dataclass
class VoiceConfig:
    """语音配置"""
    voice_id: str
    name: str
    provider: str
    gender: VoiceGender
    accent: VoiceAccent
    supported_languages: List[str]
    sample_url: Optional[str] = None


class BaseTTSEngine(BaseGenerationModel):
    """TTS引擎基类"""
    
    def __init__(self):
        super().__init__()
        self.voices: Dict[str, VoiceConfig] = {}
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """合成语音"""
        raise NotImplementedError
    
    async def get_voices(self) -> List[VoiceConfig]:
        """获取可用语音列表"""
        raise NotImplementedError
    
    async def get_voice_info(self, voice_id: str) -> Optional[VoiceConfig]:
        """获取语音信息"""
        return self.voices.get(voice_id)


class OpenAITTSEngine(BaseTTSEngine):
    """OpenAI TTS引擎"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "tts-1",
        voice_preset: str = "alloy"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice_preset = voice_preset
        self.provider = GenerationProvider.OPENAI
        
        # 初始化语音列表
        self._init_voices()
    
    def _init_voices(self):
        """初始化语音列表"""
        voices = [
            VoiceConfig(
                voice_id="alloy",
                name="Alloy",
                provider="openai",
                gender=VoiceGender.NEUTRAL,
                accent=VoiceAccent.EN_US,
                supported_languages=["en"]
            ),
            VoiceConfig(
                voice_id="echo",
                name="Echo",
                provider="openai",
                gender=VoiceGender.MALE,
                accent=VoiceAccent.EN_US,
                supported_languages=["en"]
            ),
            VoiceConfig(
                voice_id="fable",
                name="Fable",
                provider="openai",
                gender=VoiceGender.NEUTRAL,
                accent=VoiceAccent.EN_GB,
                supported_languages=["en", "zh"]
            ),
            VoiceConfig(
                voice_id="onyx",
                name="Onyx",
                provider="openai",
                gender=VoiceGender.MALE,
                accent=VoiceAccent.EN_US,
                supported_languages=["en"]
            ),
            VoiceConfig(
                voice_id="nova",
                name="Nova",
                provider="openai",
                gender=VoiceGender.FEMALE,
                accent=VoiceAccent.EN_US,
                supported_languages=["en"]
            ),
            VoiceConfig(
                voice_id="shimmer",
                name="Shimmer",
                provider="openai",
                gender=VoiceGender.FEMALE,
                accent=VoiceAccent.EN_US,
                supported_languages=["en"]
            )
        ]
        
        for voice in voices:
            self.voices[voice.voice_id] = voice
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """生成内容（通用接口，委托给synthesize）"""
        if isinstance(request, TTSRequest):
            return await self.synthesize(request)
        return GenerationResponse(
            success=False,
            error="OpenAITTSEngine只支持TTSRequest类型的请求"
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """使用OpenAI TTS合成语音"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # 选择语音
            voice = request.voice or self.voice_preset
            
            # 构建请求
            params = {
                "model": self.model,
                "input": request.text,
                "voice": voice,
                "speed": request.speed,
                "response_format": request.format.value if isinstance(request.format, AudioFormat) else "mp3"
            }
            
            # 调用API
            response = await client.audio.speech.create(**params)
            
            # 获取音频数据
            audio_data = response.content
            
            return TTSResponse(
                success=True,
                task_id=str(uuid.uuid4()),
                provider=self.provider.value,
                model=self.model,
                audio_data=audio_data,
                format=params["response_format"],
                characters=len(request.text)
            )
            
        except Exception as e:
            return TTSResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.model
            )
    
    async def get_voices(self) -> List[VoiceConfig]:
        """获取OpenAI可用语音列表"""
        return list(self.voices.values())
    
    async def stream_synthesize(self, request: TTSRequest):
        """流式合成语音"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            voice = request.voice or self.voice_preset
            
            response = await client.audio.speech.create(
                model=self.model,
                input=request.text,
                voice=voice,
                speed=request.speed,
                response_format=request.format.value if isinstance(request.format, AudioFormat) else "mp3"
            )
            
            async for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk
                
        except Exception as e:
            yield f"Error: {e}".encode()


class AzureTTSEngine(BaseTTSEngine):
    """Azure认知服务TTS引擎"""
    
    def __init__(
        self,
        subscription_key: str = None,
        region: str = "eastus",
        voice: str = "zh-CN-XiaoxiaoNeural"
    ):
        super().__init__()
        self.subscription_key = subscription_key or os.getenv("AZURE_TTS_KEY")
        self.region = region or os.getenv("AZURE_TTS_REGION", "eastus")
        self.voice = voice
        self.provider = GenerationProvider.AZURE
        
        self._init_voices()
    
    def _init_voices(self):
        """初始化Azure语音列表"""
        voices = [
            VoiceConfig(
                voice_id="zh-CN-XiaoxiaoNeural",
                name="晓晓",
                provider="azure",
                gender=VoiceGender.FEMALE,
                accent=VoiceAccent.ZH_CN,
                supported_languages=["zh-CN"],
                sample_url="https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/"
            ),
            VoiceConfig(
                voice_id="zh-CN-YunxiNeural",
                name="云希",
                provider="azure",
                gender=VoiceGender.MALE,
                accent=VoiceAccent.ZH_CN,
                supported_languages=["zh-CN"]
            ),
            VoiceConfig(
                voice_id="zh-CN-YunyangNeural",
                name="云扬",
                provider="azure",
                gender=VoiceGender.MALE,
                accent=VoiceAccent.ZH_CN,
                supported_languages=["zh-CN"]
            ),
            VoiceConfig(
                voice_id="en-US-JennyNeural",
                name="Jenny",
                provider="azure",
                gender=VoiceGender.FEMALE,
                accent=VoiceAccent.EN_US,
                supported_languages=["en-US"]
            )
        ]
        
        for voice in voices:
            self.voices[voice.voice_id] = voice
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """生成内容（通用接口，委托给synthesize）"""
        if isinstance(request, TTSRequest):
            return await self.synthesize(request)
        return GenerationResponse(
            success=False,
            error="AzureTTSEngine只支持TTSRequest类型的请求"
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """使用Azure TTS合成语音"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            
            # 设置音频格式
            audio_format = request.format.value.upper() if isinstance(request.format, AudioFormat) else "MP3"
            speech_config.speech_synthesis_output_format = getattr(
                speechsdk.SpeechSynthesisOutputFormat,
                audio_format
            )
            
            # 创建语音合成器
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            # 异步合成
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: synthesizer.speak_text_async(request.text).get()
            )
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return TTSResponse(
                    success=True,
                    task_id=str(uuid.uuid4()),
                    provider=self.provider.value,
                    model=self.voice,
                    audio_data=result.audio_data,
                    format=request.format.value,
                    characters=len(request.text)
                )
            else:
                return TTSResponse(
                    success=False,
                    error=f"语音合成失败: {result.cancellation_details.reason}",
                    provider=self.provider.value,
                    model=self.voice
                )
                
        except Exception as e:
            return TTSResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.voice
            )
    
    async def get_voices(self) -> List[VoiceConfig]:
        """获取Azure可用语音列表"""
        return list(self.voices.values())
    
    async def get_voices_by_language(self, language: str) -> List[VoiceConfig]:
        """按语言获取语音列表"""
        return [
            voice for voice in self.voices.values()
            if language in voice.supported_languages
        ]


class ElevenLabsTTSEngine(BaseTTSEngine):
    """ElevenLabs TTS引擎"""
    
    def __init__(
        self,
        api_key: str = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id
        self.provider = GenerationProvider.ELEVENLABS
        
        self.base_url = "https://api.elevenlabs.io/v1"
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """生成内容（通用接口，委托给synthesize）"""
        if isinstance(request, TTSRequest):
            return await self.synthesize(request)
        return GenerationResponse(
            success=False,
            error="ElevenLabsTTSEngine只支持TTSRequest类型的请求"
        )
    
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """使用ElevenLabs合成语音"""
        try:
            import aiohttp
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": request.text,
                "voice_id": request.voice or self.voice_id,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0,
                    "use_speaker_boost": True
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/text-to-speech/{data['voice_id']}",
                    json=data,
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        
                        return TTSResponse(
                            success=True,
                            task_id=str(uuid.uuid4()),
                            provider=self.provider.value,
                            model=self.voice_id,
                            audio_data=audio_data,
                            format="mp3",
                            characters=len(request.text)
                        )
                    else:
                        error_text = await resp.text()
                        return TTSResponse(
                            success=False,
                            error=f"API错误: {error_text}",
                            provider=self.provider.value,
                            model=self.voice_id
                        )
                        
        except Exception as e:
            return TTSResponse(
                success=False,
                error=str(e),
                provider=self.provider.value,
                model=self.voice_id
            )
    
    async def get_voices(self) -> List[VoiceConfig]:
        """获取ElevenLabs可用语音"""
        try:
            import aiohttp
            
            headers = {"xi-api-key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/voices",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        voices = []
                        for voice in data.get("voices", []):
                            voices.append(VoiceConfig(
                                voice_id=voice["voice_id"],
                                name=voice.get("name", "Unknown"),
                                provider="elevenlabs",
                                gender=VoiceGender.NEUTRAL,
                                accent=VoiceAccent.EN_US,
                                supported_languages=voice.get("languages", ["en"])
                            ))
                        return voices
                    
            return []
            
        except Exception as e:
            print(f"获取语音列表失败: {e}")
            return []


class TTSEngineManager:
    """TTS引擎管理器"""
    
    def __init__(self):
        self.engines: Dict[str, BaseTTSEngine] = {}
        self._register_default_engines()
    
    def _register_default_engines(self):
        """注册默认TTS引擎"""
        self.engines["openai"] = OpenAITTSEngine()
        self.engines["azure"] = AzureTTSEngine()
        self.engines["elevenlabs"] = ElevenLabsTTSEngine()
    
    def register_engine(self, name: str, engine: BaseTTSEngine):
        """注册自定义TTS引擎"""
        self.engines[name] = engine
    
    def get_engine(self, name: str) -> Optional[BaseTTSEngine]:
        """获取TTS引擎"""
        return self.engines.get(name)
    
    async def synthesize(
        self,
        provider: str,
        request: TTSRequest
    ) -> TTSResponse:
        """合成语音"""
        engine = self.get_engine(provider)
        if not engine:
            return TTSResponse(
                success=False,
                error=f"未找到TTS引擎: {provider}"
            )
        return await engine.synthesize(request)
    
    async def get_all_voices(self) -> Dict[str, List[VoiceConfig]]:
        """获取所有引擎的语音列表"""
        result = {}
        for name, engine in self.engines.items():
            voices = await engine.get_voices()
            result[name] = voices
        return result
    
    async def get_voices_by_provider(self, provider: str) -> List[VoiceConfig]:
        """获取指定提供商的语音列表"""
        engine = self.get_engine(provider)
        if engine:
            return await engine.get_voices()
        return []


# 全局引擎实例
_tts_engine: Optional[TTSEngineManager] = None


def get_tts_engine() -> TTSEngineManager:
    """获取TTS引擎"""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngineManager()
    return _tts_engine


async def create_tts_engine(
    provider: str,
    **kwargs
) -> BaseTTSEngine:
    """工厂函数：创建TTS引擎"""
    if provider == "openai":
        return OpenAITTSEngine(**kwargs)
    elif provider == "azure":
        return AzureTTSEngine(**kwargs)
    elif provider == "elevenlabs":
        return ElevenLabsTTSEngine(**kwargs)
    else:
        # 尝试从已注册引擎获取
        engine = get_tts_engine().get_engine(provider)
        if engine:
            return engine
        raise ValueError(f"不支持的TTS提供商: {provider}")
