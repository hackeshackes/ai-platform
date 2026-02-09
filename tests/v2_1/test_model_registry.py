"""
Model Registry 端到端测试
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestModelRegistry:
    """Model Registry API测试"""
    
    def test_list_registered_models(self):
        """测试列出注册模型"""
        response = client.get("/model-registry")
        # 可能需要认证
        assert response.status_code in [200, 401, 403]
    
    def test_search_models(self):
        """测试搜索模型"""
        response = client.get("/model-registry/search")
        assert response.status_code in [200, 401, 403]
        data = response.json()
        assert "results" in data
    
    def test_create_registered_model(self):
        """测试创建注册模型"""
        response = client.post("/model-registry/register", json={
            "name": "test-model-v2-1",
            "description": "Test model for v2.1"
        })
        assert response.status_code in [200, 400, 401, 403]
        if response.status_code == 200:
            data = response.json()
            assert "model_id" in data
            assert data["name"] == "test-model-v2-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
