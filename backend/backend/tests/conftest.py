"""
Test configuration for Federated Learning Platform
"""
import pytest
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def platform():
    """Create a fresh platform for each test"""
    from federated.fl_platform import FederatedLearningPlatform
    from federated.storage import SessionStore
    return FederatedLearningPlatform(storage=SessionStore())


@pytest.fixture
def sample_config():
    """Sample FL configuration"""
    from federated.models import FLConfig
    return FLConfig(
        model_name="test-model",
        local_epochs=5,
        learning_rate=0.01,
        min_clients=2,
        max_clients=5,
        rounds=3
    )


@pytest.fixture
def sample_client_info():
    """Sample client information"""
    from federated.models import FLClientInfo
    return FLClientInfo(
        client_id="test-client-001",
        data_size=1000,
        model_version="1.0"
    )


@pytest.fixture
def sample_session(platform, sample_config):
    """Create a sample session"""
    async def _create():
        return await platform.create_session(sample_config)
    return asyncio.run(_create())
