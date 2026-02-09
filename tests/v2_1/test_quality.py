"""
数据质量 端到端测试
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestDataQuality:
    """数据质量 API测试"""
    
    def test_list_rules(self):
        """测试列出质量规则"""
        response = client.get("/quality/rules")
        assert response.status_code == 200
        data = response.json()
        assert "rules" in data
    
    def test_create_rule(self):
        """测试创建质量规则"""
        response = client.post("/quality/rules", json={
            "name": "test-no-nulls",
            "description": "Column cannot be null",
            "rule_type": "completeness",
            "column": "email",
            "condition": "is not null",
            "severity": "error"
        })
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "rule_id" in data
    
    def test_validate_dataset(self):
        """测试验证数据集"""
        response = client.post("/quality/validate", json={
            "dataset_id": "test-dataset",
            "data": [
                {"id": 1, "name": "Alice", "email": "alice@test.com"},
                {"id": 2, "name": "Bob", "email": "bob@test.com"},
                {"id": 3, "name": "Charlie", "email": None}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "check_id" in data
        assert data["failed_count"] >= 1  # Charlie没有email
    
    def test_quality_summary(self):
        """测试获取质量总览"""
        response = client.get("/quality/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_rules" in data
        assert "total_checks" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
