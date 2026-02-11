"""
Pytest Configuration - AI Platform Backend Tests
"""
import pytest
import sys
import os
from typing import Generator
from fastapi.testclient import TestClient

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入应用
from main import app


@pytest.fixture(scope="session")
def app_client() -> Generator[TestClient, None, None]:
    """提供FastAPI测试客户端"""
    client = TestClient(app)
    yield client


@pytest.fixture(scope="function")
def test_client(app_client: TestClient) -> TestClient:
    """提供测试客户端，每个测试用例独立"""
    return app_client


@pytest.fixture
def sample_intent_texts():
    """提供意图解析测试文本"""
    return {
        "creation": "创建新项目",
        "analysis": "分析数据报告",
        "query": "搜索文件内容",
        "learning": "训练模型",
        "action": "执行任务"
    }


@pytest.fixture
def sample_entity_texts():
    """提供实体提取测试文本"""
    return {
        "numbers": "项目预算100万，执行周期6个月",
        "dates": "截止日期2026-03-15，报告日期2026-02-10",
        "emails": "联系邮箱: test@example.com",
        "mixed": "2026年预算500万，联系test@company.com"
    }


@pytest.fixture
def sample_decision_contexts():
    """提供决策分析测试数据"""
    return [
        {"type": "pricing", "options": ["A", "B", "C"]},
        {"type": "resource", "options": ["allocate", "defer"]},
        {"type": "strategy", "amount": 100000, "options": ["expand", "maintain"]}
    ]


@pytest.fixture
def sample_risk_data():
    """提供风险评估测试数据"""
    return [
        {"market_risk": 0.3, "financial_risk": 0.2, "operational_risk": 0.4},
        {"market_risk": 0.1, "financial_risk": 0.1},
        {"market_risk": 0.7, "financial_risk": 0.6, "operational_risk": 0.8}
    ]


@pytest.fixture
def sample_predict_data():
    """提供预测分析测试数据"""
    return [
        {"data": {"values": [100, 110, 105, 108, 112]}, "horizon": 7},
        {"data": {"values": [50, 55, 52, 58]}, "horizon": 5},
        {"data": {"values": [1000, 1005, 1010]}, "horizon": 10}
    ]


@pytest.fixture
def sample_session_requests():
    """提供联邦学习会话创建测试数据"""
    return [
        {"name": "测试训练", "model_type": "regression", "num_rounds": 5},
        {"name": "分类模型", "model_type": "classifier", "num_rounds": 10},
        {"name": "聚类分析", "model_type": "clustering", "num_rounds": 8}
    ]


@pytest.fixture
def sample_join_requests():
    """提供联邦学习加入会话测试数据"""
    return [
        {"client_id": "client-001", "data_size": 1000},
        {"client_id": "client-002", "data_size": 2500},
        {"client_id": "client-003", "data_size": 500}
    ]


def pytest_configure(config):
    """配置pytest marker"""
    config.addinivalue_line(
        "markers", "v9: mark test as v9 module test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "database: mark test as database test"
    )
    config.addinivalue_line(
        "markers", "agent: mark test as agent test"
    )
    config.addinivalue_line(
        "markers", "monitoring: mark test as monitoring test"
    )


def pytest_collection_modifyitems(config, items):
    """根据测试类别分组"""
    for item in items:
        fspath = str(item.fspath)
        if "test_v9" in fspath:
            item.add_marker(pytest.mark.v9)
        elif "test_api" in fspath:
            item.add_marker(pytest.mark.api)
        elif "test_database" in fspath:
            item.add_marker(pytest.mark.database)
        elif "test_agents" in fspath:
            item.add_marker(pytest.mark.agent)
        elif "test_monitoring" in fspath:
            item.add_marker(pytest.mark.monitoring)
