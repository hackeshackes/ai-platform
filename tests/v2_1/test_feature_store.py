"""
Feature Store 端到端测试
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestFeatureStore:
    """Feature Store API测试"""
    
    def test_list_feature_groups(self):
        """测试列出特征组"""
        response = client.get("/feature-store/groups")
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
    
    def test_create_feature_group(self):
        """测试创建特征组"""
        response = client.post("/feature-store/groups", json={
            "name": "test-feature-group",
            "description": "Test feature group",
            "source_type": "batch",
            "features": [
                {"name": "user_id", "dtype": "int32", "description": "User ID"},
                {"name": "user_age", "dtype": "int32", "description": "User age"}
            ]
        })
        assert response.status_code in [200, 400]  # 200=成功, 400=已存在
    
    def test_ingest_features(self):
        """测试特征摄入"""
        response = client.post("/feature-store/ingest", json={
            "group_name": "test-feature-group",
            "features": [
                {"user_id": 1, "user_age": 25},
                {"user_id": 2, "user_age": 30}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "ingested" in data
        assert data["ingested"] == 2
    
    def test_online_serving(self):
        """测试在线特征服务"""
        response = client.post("/feature-store/online", json={
            "group_name": "test-feature-group",
            "entity_ids": [1, 2]
        })
        assert response.status_code == 200
        data = response.json()
        assert "features" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
