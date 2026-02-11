"""
推荐系统API接口 - api.py

提供RESTful API接口
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: List[str] = None


class RecommenderAPI:
    """推荐系统API类"""
    
    def __init__(self, recommender, config: APIConfig = None):
        """
        初始化API
        
        Args:
            recommender: 推荐器实例
            config: API配置
        """
        self.recommender = recommender
        self.config = config or APIConfig()
        
        # 状态
        self._running = False
        self._request_count = 0
        self._error_count = 0
        
    def start(self):
        """启动API服务器"""
        self._running = True
        print(f"Starting API server on {self.config.host}:{self.config.port}")
        
    def stop(self):
        """停止API服务器"""
        self._running = False
        
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running
    
    def handle_request(self, method: str, path: str, 
                       body: Dict = None) -> Dict:
        """
        处理请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            body: 请求体
            
        Returns:
            响应
        """
        self._request_count += 1
        
        try:
            if method == "POST":
                return self._handle_post(path, body or {})
            elif method == "GET":
                return self._handle_get(path)
            else:
                return {"status": "error", "message": "Method not allowed", "code": 405}
        except Exception as e:
            self._error_count += 1
            return {"status": "error", "message": str(e), "code": 500}
    
    def _handle_post(self, path: str, body: Dict) -> Dict:
        """处理POST请求"""
        if path == "/recommend":
            return self._recommend(body)
        elif path == "/feedback":
            return self._feedback(body)
        elif path == "/interactions":
            return self._add_interactions(body)
        else:
            return {"status": "error", "message": "Path not found", "code": 404}
    
    def _handle_get(self, path: str) -> Dict:
        """处理GET请求"""
        if path == "/health":
            return self._health()
        elif path == "/stats":
            return self._stats()
        elif path == "/metrics":
            return self._metrics()
        elif path.startswith("/item/"):
            item_id = path.split("/")[-1]
            return self._get_item(item_id)
        elif path.startswith("/user/"):
            user_id = path.split("/")[-1]
            return self._get_user_profile(user_id)
        else:
            return {"status": "error", "message": "Path not found", "code": 404}
    
    def _recommend(self, body: Dict) -> Dict:
        """推荐接口"""
        start_time = time.time()
        
        user_id = body.get("user_id")
        if not user_id:
            return {"status": "error", "message": "user_id required", "code": 400}
        
        # 构建请求
        from .hybrid_recommender import RecommendationRequest
        
        request = RecommendationRequest(
            user_id=user_id,
            context=body.get("context", {}),
            item_type=body.get("item_type"),
            top_k=body.get("top_k", 10),
            exclude_items=body.get("exclude_items", []),
            diversity_weight=body.get("diversity_weight", 0.1),
            freshness_weight=body.get("freshness_weight", 0.1)
        )
        
        # 执行推荐
        result = self.recommender.recommend(request)
        
        response_time = time.time() - start_time
        
        return {
            "status": "success",
            "data": {
                "items": result.items,
                "total_count": result.total_count,
                "response_time": round(response_time * 1000, 2),  # ms
                "metadata": result.metadata
            }
        }
    
    def _feedback(self, body: Dict) -> Dict:
        """反馈接口"""
        user_id = body.get("user_id")
        item_id = body.get("item_id")
        action = body.get("action")
        
        if not all([user_id, item_id, action]):
            return {"status": "error", "message": "user_id, item_id, action required", "code": 400}
        
        # 记录反馈
        from .metrics import InteractionRecord
        
        record = InteractionRecord(
            user_id=user_id,
            item_id=item_id,
            recommended=body.get("recommended", True),
            timestamp=int(time.time()),
            action=action,
            reward=body.get("reward", 0.0)
        )
        
        self.recommender.metrics.log_interaction(record)
        
        return {"status": "success", "message": "Feedback recorded"}
    
    def _add_interactions(self, body: Dict) -> Dict:
        """添加交互数据"""
        interactions = body.get("interactions", [])
        
        from .collaborative_filtering import UserItemInteraction
        
        for interaction in interactions:
            self.recommender.cf.add_interaction(UserItemInteraction(**interaction))
        
        return {"status": "success", "count": len(interactions)}
    
    def _health(self) -> Dict:
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": self._request_count,
            "error_rate": self._error_count / max(1, self._request_count)
        }
    
    def _stats(self) -> Dict:
        """统计信息"""
        return {
            "status": "success",
            "data": {
                "total_requests": self._request_count,
                "total_errors": self._error_count,
                "recommender_stats": self.recommender.get_stats() if hasattr(self.recommender, 'get_stats') else {}
            }
        }
    
    def _metrics(self) -> Dict:
        """指标数据"""
        if hasattr(self.recommender, 'metrics'):
            return {
                "status": "success",
                "data": self.recommender.metrics.export_metrics()
            }
        return {"status": "error", "message": "Metrics not available"}
    
    def _get_item(self, item_id: str) -> Dict:
        """获取物品信息"""
        if hasattr(self.recommender, 'item_features') and self.recommender.item_features:
            if item_id in self.recommender.item_features.items:
                item = self.recommender.item_features.items[item_id]
                return {
                    "status": "success",
                    "data": {
                        "item_id": item.item_id,
                        "name": item.name,
                        "type": item.item_type,
                        "tags": list(item.tags)
                    }
                }
        return {"status": "error", "message": "Item not found", "code": 404}
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        if hasattr(self.recommender, 'user_profiles') and user_id in self.recommender.user_profiles:
            profile = self.recommender.user_profiles[user_id]
            return {
                "status": "success",
                "data": profile.to_dict()
            }
        return {"status": "error", "message": "User not found", "code": 404}


def create_app(recommender) -> RecommenderAPI:
    """创建API应用"""
    config = APIConfig()
    return RecommenderAPI(recommender, config)


# Flask风格的路由适配器
class FlaskAdapter:
    """Flask适配器"""
    
    def __init__(self, api: RecommenderAPI):
        self.api = api
    
    def wsgi_app(self, environ, start_response):
        """WSGI应用"""
        method = environ.get('REQUEST_METHOD', 'GET')
        path = environ.get('PATH_INFO', '/')
        body_size = int(environ.get('CONTENT_LENGTH', 0))
        
        body = {}
        if body_size > 0:
            import json
            body_bytes = environ['wsgi.input'].read(body_size)
            body = json.loads(body_bytes.decode())
        
        response = self.api.handle_request(method, path, body)
        
        status = f"{response.get('code', 200)} OK"
        headers = [('Content-Type', 'application/json')]
        
        start_response(status, headers)
        return [json.dumps(response).encode()]
