"""
多模态能力抽象基类
定义图像、音频、视频理解和生成的基础接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class MultimodalProvider(Enum):
    """支持的第三方多模态提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    SELF_HOSTED = "self_hosted"


class MediaType(Enum):
    """媒体类型"""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalResult:
    """多模态处理结果"""
    success: bool
    content: Any
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class BaseMultimodalModel(ABC):
    """多模态模型抽象基类"""
    
    def __init__(self, provider: MultimodalProvider, model: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    async def analyze_image(
        self, 
        image_path: Union[str, Path, bytes],
        prompt: str,
        **kwargs
    ) -> MultimodalResult:
        """分析图像"""
        pass
    
    @abstractmethod
    async def transcribe_audio(
        self,
        audio_path: Union[str, Path, bytes],
        **kwargs
    ) -> MultimodalResult:
        """音频转文字"""
        pass
    
    @abstractmethod
    async def analyze_video(
        self,
        video_path: Union[str, Path],
        prompt: Optional[str] = None,
        **kwargs
    ) -> MultimodalResult:
        """分析视频"""
        pass
    
    @abstractmethod
    async def generate_content(
        self,
        prompt: str,
        media_type: MediaType,
        **kwargs
    ) -> MultimodalResult:
        """多模态内容生成"""
        pass


class BaseVisionModel(BaseMultimodalModel):
    """视觉模型基类"""
    
    @abstractmethod
    async def describe_image(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        """描述图像内容"""
        pass
    
    @abstractmethod
    async def detect_objects(self, image_path: Union[str, Path, bytes]) -> MultimodalResult:
        """检测图像中的对象"""
        pass


class BaseAudioModel(BaseMultimodalModel):
    """音频模型基类"""
    
    @abstractmethod
    async def transcribe(
        self,
        audio_path: Union[str, Path, bytes],
        language: Optional[str] = None
    ) -> MultimodalResult:
        """转录音频"""
        pass


class BaseVideoModel(BaseMultimodalModel):
    """视频模型基类"""
    
    @abstractmethod
    async def extract_frames(
        self,
        video_path: Union[str, Path],
        num_frames: int = 10
    ) -> List[bytes]:
        """从视频中提取关键帧"""
        pass


class BaseMultimodalEngine:
    """多模态引擎 - 管理所有模型"""
    
    def __init__(self):
        self.vision_models: Dict[str, BaseVisionModel] = {}
        self.audio_models: Dict[str, BaseAudioModel] = {}
        self.video_models: Dict[str, BaseVideoModel] = {}
        self._initialized = False
    
    def register_vision_model(self, name: str, model: BaseVisionModel):
        """注册视觉模型"""
        self.vision_models[name] = model
    
    def register_audio_model(self, name: str, model: BaseAudioModel):
        """注册音频模型"""
        self.audio_models[name] = model
    
    def register_video_model(self, name: str, model: BaseVideoModel):
        """注册视频模型"""
        self.video_models[name] = model
    
    async def initialize(self):
        """初始化所有注册的模型"""
        self._initialized = True
    
    async def shutdown(self):
        """关闭所有模型连接"""
        self._initialized = False
