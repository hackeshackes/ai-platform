"""
Base LLM Provider - Multi-LLM Provider Support v3.0

所有LLM提供商的抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, AsyncGenerator, Optional
import asyncio


class BaseLLMProvider(ABC):
    """
    LLM提供商抽象基类
    
    所有具体提供商必须实现以下方法:
    - generate(): 异步生成文本
    - stream_generate(): 异步流式生成文本
    - get_models(): 获取支持的模型列表
    
    属性:
    - provider_name: 提供商标识符
    """
    
    provider_name: str = "base"
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        同步生成文本响应
        
        Args:
            prompt: 输入提示
            model: 模型ID
            temperature: 温度参数 (0.0 - 2.0)
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Returns:
            生成的文本字符串
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成文本响应
        
        Args:
            prompt: 输入提示
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Yields:
            文本块
        """
        pass
    
    @abstractmethod
    def get_models(self) -> List[Dict]:
        """
        获取支持的模型列表
        
        Returns:
            模型信息字典列表，每个字典包含:
            - id: 模型标识符
            - name: 模型名称
            - description: 模型描述
            - max_tokens: 最大token数
            - provider: 提供商名称
            - capabilities: 模型能力
        """
        pass
    
    @property
    def name(self) -> str:
        """获取提供商名称"""
        return self.provider_name
    
    async def agenerate(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步生成方法 (简写)
        
        Args:
            prompt: 输入提示
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        return await self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        对话式生成
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型ID
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        last_message = messages[-1]
        if last_message.get("role") != "user":
            raise ValueError("Last message must be from user")
        
        return await self.generate(
            prompt=last_message["content"],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages[:-1],
            **kwargs
        )
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步对话式生成 (简写)
        """
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def embed(self, text: str, model: str = "default") -> List[float]:
        """
        生成文本嵌入
        
        Args:
            text: 输入文本
            model: 嵌入模型ID
            
        Returns:
            嵌入向量
        """
        raise NotImplementedError(
            f"Provider {self.provider_name} does not support embeddings"
        )
    
    async def close(self):
        """
        关闭提供商连接
        """
        pass
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
        return False


class ProviderCapability:
    """提供商能力枚举"""
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    JSON_MODE = "json_mode"
    SYSTEM_PROMPTS = "system_prompts"
    TOOL_USE = "tool_use"
