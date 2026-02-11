"""
边缘AI部署 - 离线推理引擎
支持离线模型推理、缓存管理、本地推理服务
"""

import os
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    model_id: str
    inputs: Dict[str, Any]
    mode: InferenceMode = InferenceMode.SYNC
    timeout: int = 60
    priority: int = 5  # 1-10, higher is more important
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    model_id: str
    outputs: Dict[str, Any]
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelCacheInfo:
    """模型缓存信息"""
    model_id: str
    model_path: str
    model_type: str  # onnx, tensorrt, torchscript
    size_bytes: int
    load_time_ms: float
    last_used: datetime
    access_count: int = 0
    hot: bool = False  # frequently accessed


class InferenceCache:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir: str = "/tmp/edge_inference_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ModelCacheInfo] = {}
        self._lock = threading.Lock()
        self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """加载缓存元数据"""
        metadata_path = self.cache_dir / "cache_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['last_used'] = datetime.fromisoformat(item['last_used'])
                        self._cache[item['model_id']] = ModelCacheInfo(**item)
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """保存缓存元数据"""
        metadata_path = self.cache_dir / "cache_metadata.json"
        data = [cache_info.__dict__ for cache_info in self._cache.values()]
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str
    ) -> ModelCacheInfo:
        """
        添加模型到缓存
        
        Args:
            model_id: 模型ID
            model_path: 模型路径
            model_type: 模型类型
            
        Returns:
            缓存信息
        """
        with self._lock:
            size_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            
            cache_info = ModelCacheInfo(
                model_id=model_id,
                model_path=model_path,
                model_type=model_type,
                size_bytes=size_bytes,
                load_time_ms=0,
                last_used=datetime.now(),
                access_count=0,
                hot=False
            )
            
            self._cache[model_id] = cache_info
            self._save_cache_metadata()
            
            return cache_info
    
    def get_model(self, model_id: str) -> Optional[ModelCacheInfo]:
        """
        获取模型缓存信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            缓存信息或None
        """
        if model_id in self._cache:
            cache_info = self._cache[model_id]
            cache_info.last_used = datetime.now()
            cache_info.access_count += 1
            # 标记热门模型
            if cache_info.access_count > 100:
                cache_info.hot = True
            return cache_info
        return None
    
    def remove_model(self, model_id: str) -> bool:
        """从缓存中移除模型"""
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                self._save_cache_metadata()
                return True
        return False
    
    def list_cached_models(self) -> List[ModelCacheInfo]:
        """列出所有缓存的模型"""
        return list(self._cache.values())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(c.size_bytes for c in self._cache.values())
        hot_count = sum(1 for c in self._cache.values() if c.hot)
        
        return {
            "total_models": len(self._cache),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "hot_models": hot_count,
            "total_accesses": sum(c.access_count for c in self._cache.values())
        }
    
    def cleanup(self, max_size_gb: float = 10.0, max_age_days: int = 30):
        """清理过期模型"""
        max_size_bytes = max_size_gb * 1024**3
        cutoff_date = datetime.now()
        
        models_to_remove = []
        
        total_size = sum(c.size_bytes for c in self._cache.values())
        
        for model_id, cache_info in self._cache.items():
            age = (cutoff_date - cache_info.last_used).days
            is_expired = age > max_age_days
            is_overflow = total_size > max_size_bytes and not cache_info.hot
            
            if is_expired or is_overflow:
                models_to_remove.append(model_id)
                total_size -= cache_info.size_bytes
        
        for model_id in models_to_remove:
            self.remove_model(model_id)
            logger.info(f"Removed expired model from cache: {model_id}")
        
        return {"removed_count": len(models_to_remove)}


class OfflineInferenceEngine:
    """离线推理引擎"""
    
    def __init__(self, cache_dir: str = "/tmp/edge_inference_cache"):
        self.cache = InferenceCache(cache_dir=cache_dir)
        self._models: Dict[str, Any] = {}  # loaded model instances
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._inference_lock = threading.Lock()
        self._request_queue: List[InferenceRequest] = []
        self._running = False
    
    def load_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str = "onnx",
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        加载模型到内存
        
        Args:
            model_id: 模型ID
            model_path: 模型文件路径
            model_type: 模型类型 (onnx, torchscript, tensorrt)
            device: 设备 (cpu, cuda)
            
        Returns:
            加载结果
        """
        start_time = time.time()
        
        try:
            if model_type == "onnx":
                import onnxruntime as ort
                providers = ['CPUExecutionProvider'] if device == "cpu" else ['CUDAExecutionProvider']
                session = ort.InferenceSession(model_path, providers=providers)
                self._models[model_id] = session
                
            elif model_type == "torchscript":
                import torch
                model = torch.jit.load(model_path)
                model.eval()
                if device == "cuda":
                    model.cuda()
                self._models[model_id] = model
                
            elif model_type == "tensorrt":
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.ERROR)
                with open(model_path, 'rb') as f:
                    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
                context = engine.create_execution_context()
                self._models[model_id] = {"engine": engine, "context": context}
            
            load_time_ms = (time.time() - start_time) * 1000
            
            # 更新缓存
            self.cache.add_model(model_id, model_path, model_type)
            
            cache_info = self.cache.get_model(model_id)
            if cache_info:
                cache_info.load_time_ms = load_time_ms
                cache_info.hot = True
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "device": device,
                "load_time_ms": load_time_ms
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unloaded model: {model_id}")
            return True
        return False
    
    def infer(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        mode: InferenceMode = InferenceMode.SYNC
    ) -> InferenceResponse:
        """
        执行推理
        
        Args:
            model_id: 模型ID
            inputs: 输入数据
            mode: 推理模式
            
        Returns:
            推理响应
        """
        start_time = time.time()
        
        if model_id not in self._models:
            return InferenceResponse(
                request_id="",
                model_id=model_id,
                outputs={},
                latency_ms=0,
                success=False,
                error_message=f"Model {model_id} not loaded"
            )
        
        try:
            model = self._models[model_id]
            
            # ONNX Runtime推理
            if isinstance(model, type(ort.InferenceSession)):
                import onnxruntime as ort
                input_names = [inp.name for inp in model.get_inputs()]
                feed_dict = {name: np.array(inputs.get(name, [])) for name in input_names}
                output_names = [out.name for out in model.get_outputs()]
                outputs = model.run(output_names, feed_dict)
                output_dict = {name: arr.tolist() for name, arr in zip(output_names, outputs)}
            
            # PyTorch推理
            elif hasattr(model, 'forward'):
                import torch
                input_tensors = {k: torch.tensor(v) for k, v in inputs.items()}
                with torch.no_grad():
                    result = model(**input_tensors)
                output_dict = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in result.items()}
            
            else:
                raise ValueError(f"Unsupported model type for inference")
            
            latency_ms = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id="",
                model_id=model_id,
                outputs=output_dict,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_id}: {e}")
            return InferenceResponse(
                request_id="",
                model_id=model_id,
                outputs={},
                latency_ms=0,
                success=False,
                error_message=str(e)
            )
    
    def batch_infer(
        self,
        model_id: str,
        batch_inputs: List[Dict[str, Any]]
    ) -> List[InferenceResponse]:
        """
        批量推理
        
        Args:
            model_id: 模型ID
            batch_inputs: 输入列表
            
        Returns:
            推理响应列表
        """
        results = []
        for inputs in batch_inputs:
            response = self.infer(model_id, inputs)
            results.append(response)
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "loaded_models": list(self._models.keys()),
            "cached_models": len(self.cache.list_cached_models()),
            "cache_stats": self.cache.get_cache_stats(),
            "active_workers": self._executor._max_workers
        }
    
    def shutdown(self):
        """关闭推理引擎"""
        self._running = False
        self._executor.shutdown(wait=True)
        for model_id in list(self._models.keys()):
            self.unload_model(model_id)
        logger.info("Inference engine shutdown complete")


class LocalInferenceServer:
    """本地推理服务器 - 为边缘设备提供本地推理API"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.engine = OfflineInferenceEngine()
        self._app = None
    
    def create_app(self):
        """创建FastAPI应用"""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            
            app = FastAPI(title="Edge Inference Server")
            
            class LoadModelRequest(BaseModel):
                model_id: str
                model_path: str
                model_type: str = "onnx"
                device: str = "cpu"
            
            class InferenceRequest(BaseModel):
                model_id: str
                inputs: Dict[str, Any]
            
            @app.post("/models/load")
            async def load_model(request: LoadModelRequest):
                result = self.engine.load_model(
                    request.model_id,
                    request.model_path,
                    request.model_type,
                    request.device
                )
                return result
            
            @app.post("/models/unload/{model_id}")
            async def unload_model(model_id: str):
                success = self.engine.unload_model(model_id)
                return {"success": success}
            
            @app.get("/models/list")
            async def list_models():
                return self.engine.get_engine_status()
            
            @app.post("/inference")
            async def inference(request: InferenceRequest):
                response = self.engine.infer(request.model_id, request.inputs)
                if response.success:
                    return response
                else:
                    raise HTTPException(status_code=400, detail=response.error_message)
            
            @app.get("/health")
            async def health():
                return {"status": "healthy", "engine": self.engine.get_engine_status()}
            
            self._app = app
            return app
            
        except ImportError:
            logger.warning("FastAPI not available, using placeholder server")
            return None
    
    def run(self, debug: bool = False):
        """运行服务器"""
        if self._app:
            import uvicorn
            uvicorn.run(self._app, host=self.host, port=self.port, debug=debug)
        else:
            logger.error("No app configured")


# 便捷函数
def create_inference_engine(
    cache_dir: str = "/tmp/edge_inference_cache"
) -> OfflineInferenceEngine:
    """创建离线推理引擎"""
    return OfflineInferenceEngine(cache_dir=cache_dir)


def create_inference_server(
    host: str = "0.0.0.0",
    port: int = 8080
) -> LocalInferenceServer:
    """创建本地推理服务器"""
    server = LocalInferenceServer(host=host, port=port)
    server.create_app()
    return server
