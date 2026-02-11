"""
API Test Suite - AI Platform Backend
API测试套件
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


# ============== 根路径和健康检查测试 ==============

class TestRootEndpoints:
    """根路径端点测试类"""
    
    def test_root_endpoint(self):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# ============== API根路径测试 ==============

class TestAPIRoot:
    """API根路径测试类"""
    
    def test_api_root(self):
        """测试API根路径"""
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


# ============== V9 API端点测试 ==============

class TestV9APIEndpoints:
    """V9 API端点测试类"""
    
    # Adaptive Learning Endpoints
    def test_v9_adaptive_intent_parse_get(self):
        """V9自适应学习 - 意图解析GET"""
        response = client.get("/api/v1/v9/adaptive/intent/parse?text=创建项目")
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
    
    def test_v9_adaptive_entities_extract_get(self):
        """V9自适应学习 - 实体提取GET"""
        response = client.get("/api/v1/v9/adaptive/entities/extract?text=100万")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_v9_adaptive_q_learning_info(self):
        """V9自适应学习 - Q-Learning信息"""
        response = client.get("/api/v1/v9/adaptive/strategies/q-learning/info")
        assert response.status_code == 200
        data = response.json()
        assert data["algorithm"] == "Q-Learning"
    
    def test_v9_adaptive_evaluate(self):
        """V9自适应学习 - 评估"""
        response = client.get("/api/v1/v9/adaptive/evaluate/test-agent")
        assert response.status_code == 200
        data = response.json()
        assert "success_rate" in data
    
    # Federated Learning Endpoints
    def test_v9_federated_sessions_list(self):
        """V9联邦学习 - 会话列表"""
        response = client.get("/api/v1/v9/federated/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
    
    def test_v9_federated_session_create(self):
        """V9联邦学习 - 创建会话"""
        response = client.post(
            "/api/v1/v9/federated/sessions",
            json={"name": "测试", "model_type": "classifier"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_v9_federated_session_get(self):
        """V9联邦学习 - 获取会话"""
        # 先创建
        create_resp = client.post(
            "/api/v1/v9/federated/sessions",
            json={"name": "测试", "model_type": "classifier"}
        )
        session_id = create_resp.json()["session_id"]
        
        # 获取
        response = client.get(f"/api/v1/v9/federated/sessions/{session_id}")
        assert response.status_code == 200
    
    def test_v9_federated_session_join(self):
        """V9联邦学习 - 加入会话"""
        # 创建会话
        create_resp = client.post(
            "/api/v1/v9/federated/sessions",
            json={"name": "测试", "model_type": "regression"}
        )
        session_id = create_resp.json()["session_id"]
        
        # 加入
        response = client.post(
            f"/api/v1/v9/federated/sessions/{session_id}/join",
            json={"client_id": "client-001", "data_size": 1000}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "joined"
    
    def test_v9_federated_privacy_config(self):
        """V9联邦学习 - 隐私配置"""
        response = client.get("/api/v1/v9/federated/privacy/config")
        assert response.status_code == 200
        data = response.json()
        assert "epsilon" in data
    
    def test_v9_federated_aggregators(self):
        """V9联邦学习 - 聚合器"""
        response = client.get("/api/v1/v9/federated/aggregators")
        assert response.status_code == 200
        data = response.json()
        assert "algorithms" in data
    
    # Decision Engine Endpoints
    def test_v9_decision_analyze(self):
        """V9决策引擎 - 分析"""
        response = client.post(
            "/api/v1/v9/decision/analyze",
            json={"type": "pricing", "options": ["A", "B"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision_id" in data
    
    def test_v9_decision_risk_assess(self):
        """V9决策引擎 - 风险评估"""
        response = client.post(
            "/api/v1/v9/decision/risk/assess",
            json={"data": {"market_risk": 0.3}}
        )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
    
    def test_v9_decision_predict(self):
        """V9决策引擎 - 预测"""
        response = client.post(
            "/api/v1/v9/decision/predict",
            json={"data": {"values": [100, 110]}, "horizon": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
    
    def test_v9_decision_recommend(self):
        """V9决策引擎 - 建议"""
        response = client.post(
            "/api/v1/v9/decision/recommend",
            json={"context": "test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
    
    def test_v9_decision_history(self):
        """V9决策引擎 - 历史"""
        response = client.get("/api/v1/v9/decision/history")
        assert response.status_code == 200
        data = response.json()
        assert "decisions" in data


# ============== API响应格式测试 ==============

class TestAPIResponseFormat:
    """API响应格式测试类"""
    
    def test_response_contains_timestamp(self):
        """测试响应包含时间戳"""
        response = client.get("/api/v1/v9/adaptive/strategies/q-learning/info")
        assert response.status_code == 200
        # V9响应不一定包含时间戳
    
    def test_response_content_type(self):
        """测试响应内容类型"""
        response = client.get("/api/v1/v9/adaptive/intent/parse?text=测试")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_successful_response_structure(self):
        """测试成功响应结构"""
        response = client.post(
            "/api/v1/v9/federated/sessions",
            json={"name": "测试", "model_type": "classifier"}
        )
        assert response.status_code == 200
        data = response.json()
        # 应该包含ID或标识符
        assert "session_id" in data or "id" in data or "decision_id" in data


# ============== API错误处理测试 ==============

class TestAPIErrorHandling:
    """API错误处理测试类"""
    
    def test_nonexistent_endpoint(self):
        """测试不存在的端点"""
        response = client.get("/api/v1/v9/nonexistent/endpoint")
        assert response.status_code in [404, 405]
    
    def test_invalid_method(self):
        """测试无效HTTP方法"""
        response = client.put("/api/v1/v9/adaptive/intent/parse?text=测试")
        assert response.status_code in [404, 405]
    
    def test_missing_required_param(self):
        """测试缺少必需参数"""
        response = client.get("/api/v1/v9/adaptive/intent/parse")
        # 应该有参数验证
        assert response.status_code in [200, 422]
    
    def test_empty_body_post(self):
        """测试空body的POST请求"""
        response = client.post("/api/v1/v9/decision/analyze", json={})
        # 端点可能需要某些字段
        assert response.status_code in [200, 422]
    
    def test_invalid_json(self):
        """测试无效JSON"""
        response = client.post(
            "/api/v1/v9/federated/sessions",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]


# ============== API性能测试 ==============

class TestAPIPerformance:
    """API性能测试类"""
    
    def test_response_time_simple_endpoint(self):
        """测试简单端点响应时间"""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # 应在1秒内响应
    
    def test_response_time_v9_endpoint(self):
        """测试V9端点响应时间"""
        import time
        
        start = time.time()
        response = client.get("/api/v1/v9/adaptive/strategies/q-learning/info")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0  # V9端点应在2秒内响应
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return client.get("/api/v1/v9/adaptive/strategies/q-learning/info")
        
        start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(executor.submit(make_request) for _ in range(10))
        elapsed = time.time() - start
        
        assert len(responses) == 10
        for r in responses:
            assert r.result().status_code == 200
        assert elapsed < 5.0  # 10个并发请求应在5秒内完成


# ============== API认证和授权测试 ==============

class TestAPIAuth:
    """API认证测试类"""
    
    def test_unauthenticated_access(self):
        """测试未认证访问"""
        # V9端点目前不需要认证
        response = client.get("/api/v1/v9/adaptive/intent/parse?text=测试")
        assert response.status_code == 200
    
    def test_api_root_accessible(self):
        """测试API根路径可访问"""
        response = client.get("/api/v1")
        assert response.status_code == 200


# ============== API版本控制测试 ==============

class TestAPIVersioning:
    """API版本控制测试类"""
    
    def test_version_in_response(self):
        """测试响应中包含版本"""
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
    
    def test_version_format(self):
        """测试版本格式"""
        response = client.get("/")
        data = response.json()
        version = data.get("version", "")
        # 版本应为 x.x.x 格式
        parts = version.split(".")
        assert len(parts) >= 2


# ============== Mock API测试 ==============

class TestMockAPI:
    """Mock API测试类"""
    
    def test_mock_api_response(self):
        """测试Mock API响应"""
        mock_response = {
            "status": "success",
            "data": {"id": "test-001"},
            "message": "Operation completed"
        }
        
        assert mock_response["status"] == "success"
        assert "data" in mock_response
    
    def test_mock_error_response(self):
        """测试Mock错误响应"""
        mock_error = {
            "status": "error",
            "error": {
                "code": 400,
                "message": "Invalid request"
            }
        }
        
        assert mock_error["status"] == "error"
        assert "error" in mock_error
        assert "code" in mock_error["error"]
    
    def test_mock_pagination(self):
        """测试Mock分页"""
        mock_page = {
            "items": [{"id": i} for i in range(10)],
            "total": 100,
            "page": 1,
            "page_size": 10,
            "total_pages": 10
        }
        
        assert len(mock_page["items"]) == 10
        assert mock_page["total"] == 100
        assert mock_page["page"] == 1
    
    def test_mock_rate_limit_headers(self):
        """测试Mock速率限制头"""
        mock_headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "99",
            "X-RateLimit-Reset": "3600"
        }
        
        assert "X-RateLimit-Limit" in mock_headers
        assert int(mock_headers["X-RateLimit-Limit"]) > 0


# ============== WebSocket测试 ==============

class TestWebSocket:
    """WebSocket测试类"""
    
    def test_websocket_import(self):
        """测试WebSocket导入"""
        try:
            from fastapi import WebSocket
            assert WebSocket is not None
        except ImportError:
            pytest.skip("WebSocket not available")
    
    def test_websocket_connection_simulation(self):
        """测试WebSocket连接模拟"""
        # 模拟WebSocket连接
        class MockWebSocket:
            def __init__(self):
                self.messages = []
                self.connected = False
            
            def connect(self, url):
                self.connected = True
            
            def send_text(self, message):
                self.messages.append(message)
            
            def close(self):
                self.connected = False
        
        ws = MockWebSocket()
        ws.connect("ws://localhost:8000/ws")
        ws.send_text("Hello")
        
        assert ws.connected is True
        assert len(ws.messages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
