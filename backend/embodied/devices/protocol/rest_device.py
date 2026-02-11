"""
REST Device - REST协议设备集成

提供REST API设备的通信功能
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class RESTDevice:
    """
    REST设备适配器
    
    用于通过REST API与设备通信
    """
    
    def __init__(
        self,
        device_id: str,
        base_url: str,
        headers: Dict[str, str] = None,
        timeout: float = 10.0,
        simulation_mode: bool = True,
        auth_token: str = None
    ):
        """
        初始化REST设备适配器
        
        Args:
            device_id: 设备ID
            base_url: API基础URL
            headers: 请求头
            timeout: 超时时间
            simulation_mode: 模拟模式
            auth_token: 认证令牌
        """
        self.device_id = device_id
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.timeout = timeout
        self.simulation_mode = simulation_mode
        self.auth_token = auth_token
        
        # 添加默认请求头
        self.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        
        # 模拟数据存储
        self._simulated_state: Dict[str, Any] = {}
        self._simulated_history: List[Dict] = []
        
        logger.info(f"RESTDevice initialized: {device_id} @ {base_url}")
    
    async def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            query_params: 查询参数
            
        Returns:
            响应数据
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        if self.simulation_mode:
            return await self._simulated_request(method, url, data, params)
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                
                async with session.request(
                    method=method.value,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    
                    if response.content_type == 'application/json':
                        result = await response.json()
                    else:
                        result = {"text": await response.text()}
                    
                    return {
                        "status_code": response.status,
                        "data": result,
                        "headers": dict(response.headers)
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"REST request timeout: {url}")
            raise Exception(f"Request timeout: {url}")
        except Exception as e:
            logger.error(f"REST request failed: {e}")
            raise
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        GET请求
        
        Args:
            endpoint: API端点
            params: 查询参数
            
        Returns:
            响应数据
        """
        return await self.request(HTTPMethod.GET, endpoint, params=params)
    
    async def post(
        self,
        endpoint: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        POST请求
        
        Args:
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应数据
        """
        return await self.request(HTTPMethod.POST, endpoint, data=data)
    
    async def put(
        self,
        endpoint: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        PUT请求
        
        Args:
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应数据
        """
        return await self.request(HTTPMethod.PUT, endpoint, data=data)
    
    async def patch(
        self,
        endpoint: str,
        data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        PATCH请求
        
        Args:
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应数据
        """
        return await self.request(HTTPMethod.PATCH, endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """
        DELETE请求
        
        Args:
            endpoint: API端点
            
        Returns:
            响应数据
        """
        return await self.request(HTTPMethod.DELETE, endpoint)
    
    async def execute_action(
        self,
        action: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        执行设备动作
        
        Args:
            action: 动作名称
            params: 动作参数
            
        Returns:
            执行结果
        """
        if self.simulation_mode:
            return await self._simulated_action(action, params)
        
        # 实际API调用
        return await self.post(f"/actions/{action}", params)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        获取设备状态
        
        Returns:
            设备状态
        """
        if self.simulation_mode:
            return {
                "device_id": self.device_id,
                "status": "online",
                "state": self._simulated_state,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return await self.get("/status")
    
    async def turn_on(self) -> Dict[str, Any]:
        """
        开启设备
        
        Returns:
            操作结果
        """
        return await self.execute_action("turn_on")
    
    async def turn_off(self) -> Dict[str, Any]:
        """
        关闭设备
        
        Returns:
            操作结果
        """
        return await self.execute_action("turn_off")
    
    async def set_value(
        self,
        key: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        设置设备值
        
        Args:
            key: 值键
            value: 值
            
        Returns:
            操作结果
        """
        params = {"key": key, "value": value}
        return await self.execute_action("set_value", params)
    
    async def get_value(self, key: str) -> Any:
        """
        获取设备值
        
        Args:
            key: 值键
            
        Returns:
            值
        """
        if self.simulation_mode:
            return self._simulated_state.get(key)
        
        result = await self.get(f"/values/{key}")
        return result.get("data", {}).get("value")
    
    async def subscribe_telemetry(
        self,
        callback: callable,
        interval: float = 1.0
    ) -> asyncio.Task:
        """
        订阅遥测数据
        
        Args:
            callback: 回调函数
            interval: 采样间隔
            
        Returns:
            订阅任务
        """
        async def telemetry_loop():
            while True:
                try:
                    status = await self.get_status()
                    if callback:
                        callback(status)
                except Exception as e:
                    logger.error(f"Telemetry error: {e}")
                await asyncio.sleep(interval)
        
        task = asyncio.create_task(telemetry_loop())
        return task
    
    async def _simulated_request(
        self,
        method: HTTPMethod,
        url: str,
        data: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模拟HTTP请求"""
        self._simulated_history.append({
            "method": method.value,
            "url": url,
            "data": data,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"REST simulated: {method.value} {url}")
        
        return {
            "status_code": 200,
            "data": {"success": True, "simulated": True},
            "headers": {}
        }
    
    async def _simulated_action(
        self,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模拟动作执行"""
        result = {"action": action, "success": True}
        
        if action == "turn_on":
            self._simulated_state["power"] = "on"
            result["state"] = "on"
        elif action == "turn_off":
            self._simulated_state["power"] = "off"
            result["state"] = "off"
        elif action == "set_value":
            key = params.get("key")
            value = params.get("value")
            if key:
                self._simulated_state[key] = value
            result["key"] = key
            result["value"] = value
        else:
            result["acknowledged"] = True
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取模拟状态
        
        Returns:
            状态信息
        """
        return {
            "device_id": self.device_id,
            "state": self._simulated_state,
            "history_count": len(self._simulated_history),
            "simulation_mode": self.simulation_mode
        }
    
    def clear_history(self):
        """清空历史记录"""
        self._simulated_history.clear()
