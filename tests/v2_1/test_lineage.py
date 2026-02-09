"""
Model Lineage 端到端测试
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestModelLineage:
    """Model Lineage API测试"""
    
    def test_create_node(self):
        """测试创建血缘节点"""
        response = client.post("/lineage/nodes", json={
            "node_type": "dataset",
            "external_id": "test-dataset-001",
            "name": "Test Dataset",
            "metadata": {"size": "100MB"}
        })
        assert response.status_code == 200
        data = response.json()
        assert "node_id" in data
        assert data["node_type"] == "dataset"
    
    def test_connect_nodes(self):
        """测试连接节点"""
        # 先创建两个节点
        response = client.post("/lineage/nodes", json={
            "node_type": "dataset",
            "external_id": "test-dataset-002",
            "name": "Test Dataset 2"
        })
        dataset_node = response.json()
        
        response = client.post("/lineage/nodes", json={
            "node_type": "run",
            "external_id": "test-run-001",
            "name": "Test Run"
        })
        run_node = response.json()
        
        # 连接
        response = client.post("/lineage/connect", json={
            "source_type": "dataset",
            "source_id": "test-dataset-002",
            "target_type": "run",
            "target_id": "test-run-001",
            "edge_type": "execution_flow"
        })
        assert response.status_code == 200
        data = response.json()
        assert "edge_id" in data
    
    def test_get_graph(self):
        """测试获取完整血缘图"""
        response = client.get("/lineage/graph")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
