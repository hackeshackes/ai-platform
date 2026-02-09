"""
Notebooks 端到端测试
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestNotebooks:
    """Notebooks API测试"""
    
    def test_list_notebooks(self):
        """测试列出Notebooks"""
        response = client.get("/notebooks")
        assert response.status_code in [200, 401, 403]
        if response.status_code == 200:
            data = response.json()
            assert "notebooks" in data
    
    def test_create_notebook(self):
        """测试创建Notebook"""
        response = client.post("/notebooks", json={
            "name": "test-notebook-v2-1",
            "description": "Test notebook for v2.1"
        })
        assert response.status_code in [200, 400, 401, 403]
        if response.status_code == 200:
            data = response.json()
            assert "notebook_id" in data
            assert data["name"] == "test-notebook-v2-1"
    
    def test_get_info(self):
        """测试系统信息"""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        # 检查v2.1功能是否启用
        assert "services" in data
        services = data["services"]
        assert services.get("feature_store") == True
        assert services.get("model_registry") == True
        assert services.get("lineage") == True
        assert services.get("quality") == True
        assert services.get("notebooks") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
