"""
V9 自适应学习 测试用例
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_adaptive_intent_parse():
    """测试意图解析"""
    response = client.get(
        "/api/v1/v9/adaptive/intent/parse",
        params={"text": "帮我创建新项目"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "type" in data["intent"]
    print("✅ 意图解析测试通过")

def test_adaptive_entities_extract():
    """测试实体提取"""
    response = client.get(
        "/api/v1/v9/adaptive/entities/extract",
        params={"text": "下载report-2026.xlsx"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("✅ 实体提取测试通过")

def test_adaptive_qlearning_info():
    """测试Q-Learning策略"""
    response = client.get("/api/v1/v9/adaptive/strategies/q-learning/info")
    assert response.status_code == 200
    data = response.json()
    assert data.get("algorithm") == "Q-Learning"
    assert "state_dim" in data
    assert "action_dim" in data
    print("✅ Q-Learning策略测试通过")

def test_adaptive_evaluate():
    """测试效果评估"""
    response = client.get("/api/v1/v9/adaptive/evaluate/agent-test")
    assert response.status_code == 200
    data = response.json()
    assert "success_rate" in data
    assert "total_interactions" in data
    print("✅ 效果评估测试通过")

if __name__ == "__main__":
    test_adaptive_intent_parse()
    test_adaptive_entities_extract()
    test_adaptive_qlearning_info()
    test_adaptive_evaluate()
    print("\n🎉 V9自适应学习: 全部测试通过!")
