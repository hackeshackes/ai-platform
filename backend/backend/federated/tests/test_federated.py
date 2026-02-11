"""
Unit tests for Federated Learning Platform
"""
import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from federated.models import (
    FLSession,
    FLConfig,
    LocalModel,
    GlobalModel,
    FLClientInfo,
    SessionStatus
)
from federated.privacy import PrivacyManager
from federated.aggregator import Aggregator
from federated.storage import SessionStore
from federated.client import FLClient, ModelTrainer
from federated.fl_platform import FederatedLearningPlatform


class TestModels:
    """Test Pydantic models"""
    
    def test_fl_config_defaults(self):
        """Test FLConfig default values"""
        config = FLConfig()
        assert config.model_name == "default_model"
        assert config.local_epochs == 5
        assert config.learning_rate == 0.01
        assert config.min_clients == 2
        assert config.differential_privacy is False
    
    def test_fl_config_custom(self):
        """Test FLConfig with custom values"""
        config = FLConfig(
            model_name="resnet18",
            local_epochs=10,
            learning_rate=0.001,
            differential_privacy=True,
            dp_epsilon=0.5
        )
        assert config.model_name == "resnet18"
        assert config.local_epochs == 10
        assert config.differential_privacy is True
        assert config.dp_epsilon == 0.5
    
    def test_fl_session_creation(self):
        """Test FLSession creation"""
        config = FLConfig()
        session = FLSession(
            id="test-session-123",
            config=config,
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        assert session.id == "test-session-123"
        assert session.status == SessionStatus.PENDING
        assert len(session.participants) == 0
    
    def test_fl_client_info(self):
        """Test FLClientInfo"""
        client = FLClientInfo(
            client_id="client-001",
            data_size=1000,
            model_version="1.0"
        )
        assert client.client_id == "client-001"
        assert client.data_size == 1000
    
    def test_local_model(self):
        """Test LocalModel"""
        model = LocalModel(
            client_id="client-001",
            weights={"layer_0": [1.0, 2.0, 3.0]},
            data_size=500,
            accuracy=0.85,
            loss=0.15
        )
        assert model.client_id == "client-001"
        assert model.accuracy == 0.85


class TestPrivacyManager:
    """Test Privacy Manager with differential privacy"""
    
    def test_initialization(self):
        """Test PrivacyManager initialization"""
        pm = PrivacyManager(epsilon=1.0, delta=1e-5, max_norm=1.0)
        assert pm.epsilon == 1.0
        assert pm.delta == 1e-5
        assert pm.max_norm == 1.0
    
    def test_clip_gradients(self):
        """Test gradient clipping"""
        pm = PrivacyManager(max_norm=1.0)
        gradients = np.array([0.5, 0.5, 0.5, 0.5])
        clipped = pm.clip_gradients(gradients)
        norm = np.linalg.norm(clipped)
        assert norm <= 1.0
    
    def test_clip_gradients_no_clip(self):
        """Test gradient clipping when not needed"""
        pm = PrivacyManager(max_norm=5.0)
        gradients = np.array([1.0, 1.0, 1.0])
        clipped = pm.clip_gradients(gradients)
        assert np.allclose(clipped, gradients)
    
    def test_add_noise(self):
        """Test noise addition"""
        pm = PrivacyManager(epsilon=1.0, max_norm=1.0)
        gradients = np.array([0.5, 0.5, 0.5])
        noisy = pm.add_noise(gradients, sample_size=100)
        assert noisy.shape == gradients.shape
        assert not np.allclose(noisy, gradients)
    
    def test_add_noise_to_dict(self):
        """Test noise addition to dictionary"""
        pm = PrivacyManager(epsilon=1.0, max_norm=1.0)
        gradients = {
            "layer_0": np.array([1.0, 2.0, 3.0]),
            "layer_1": [4.0, 5.0, 6.0]
        }
        noisy = pm.add_noise_to_dict(gradients, sample_size=100)
        assert "layer_0" in noisy
        assert "layer_1" in noisy
        assert noisy["layer_0"].shape == (3,)
    
    def test_clip_and_noise_dict(self):
        """Test clip and noise for dictionary"""
        pm = PrivacyManager(epsilon=1.0, max_norm=1.0)
        gradients = {
            "layer_0": np.array([10.0, 10.0, 10.0]),
            "scalar": 5.0
        }
        processed = pm.clip_and_noise_dict(gradients, sample_size=100)
        assert processed is not None
    
    def test_privacy_accounting(self):
        """Test privacy accounting"""
        pm = PrivacyManager(epsilon=1.0)
        accounting = pm.privacy_accounting()
        assert len(accounting) == 1
        assert accounting[0]["epsilon"] == 1.0
    
    def test_update_privacy_budget(self):
        """Test privacy budget update"""
        pm = PrivacyManager(epsilon=1.0)
        pm.update_privacy_budget(rounds=1)
        assert pm.privacy_budget_used > 0
    
    def test_get_privacy_spent(self):
        """Test getting privacy spent"""
        pm = PrivacyManager(epsilon=1.0)
        pm.update_privacy_budget(rounds=5)
        spent = pm.get_privacy_spent()
        assert "epsilon" in spent
        assert "budget_used" in spent
        assert "budget_remaining" in spent
    
    def test_compose_privacy_budget(self):
        """Test privacy budget composition"""
        pm = PrivacyManager(epsilon=1.0)
        composed = pm.compose_privacy_budget(num_clients=10)
        assert composed > 0


class TestAggregator:
    """Test Model Aggregator"""
    
    def test_fedavg_single_client(self):
        """Test FedAvg with single client"""
        aggregator = Aggregator()
        weights = [{"layer_0": np.array([1.0, 2.0, 3.0])}]
        result = aggregator.fedavg(weights)
        assert np.allclose(result["layer_0"], np.array([1.0, 2.0, 3.0]))
    
    def test_fedavg_two_clients_equal_data(self):
        """Test FedAvg with two clients equal data"""
        aggregator = Aggregator()
        client1 = {"layer_0": np.array([1.0, 1.0])}
        client2 = {"layer_0": np.array([3.0, 3.0])}
        weights = [client1, client2]
        result = aggregator.fedavg(weights)
        assert np.allclose(result["layer_0"], np.array([2.0, 2.0]))
    
    def test_fedavg_weighted_by_data_size(self):
        """Test FedAvg weighted by data size"""
        aggregator = Aggregator()
        client1 = {"layer_0": np.array([1.0])}
        client2 = {"layer_0": np.array([5.0])}
        weights = [client1, client2]
        data_sizes = [10, 90]  # client2 has 9x more data
        result = aggregator.fedavg(weights, data_sizes)
        expected = np.array([4.6])  # (1*10 + 5*90) / 100
        assert np.allclose(result["layer_0"], expected)
    
    def test_fedavg_empty_raises(self):
        """Test FedAvg with empty weights raises error"""
        aggregator = Aggregator()
        with pytest.raises(ValueError):
            aggregator.fedavg([])
    
    def test_fedmedian(self):
        """Test FedMedian"""
        aggregator = Aggregator()
        client1 = {"layer_0": np.array([1.0, 1.0])}
        client2 = {"layer_0": np.array([2.0, 2.0])}
        client3 = {"layer_0": np.array([3.0, 3.0])}
        weights = [client1, client2, client3]
        result = aggregator.fedmedian(weights)
        assert np.allclose(result["layer_0"], np.array([2.0, 2.0]))
    
    def test_fedtrimmedavg(self):
        """Test FedTrimmedAvg"""
        aggregator = Aggregator()
        client1 = {"layer_0": np.array([1.0, 1.0])}
        client2 = {"layer_0": np.array([2.0, 2.0])}
        client3 = {"layer_0": np.array([10.0, 10.0])}
        weights = [client1, client2, client3]
        result = aggregator.fedtrimmedavg(weights, trim_ratio=0.33)
        assert np.allclose(result["layer_0"], np.array([2.0, 2.0]))
    
    def test_weighted_fedavg(self):
        """Test weighted FedAvg"""
        aggregator = Aggregator()
        client1 = {"layer_0": np.array([1.0])}
        client2 = {"layer_0": np.array([5.0])}
        weights = [client1, client2]
        scores = [0.8, 1.0]  # client2 has higher score
        result = aggregator.weighted_fedavg(weights, scores)
        # (1*0.8 + 5*1.0) / (0.8 + 1.0) = 5.8 / 1.8 = 3.222...
        expected = np.array([3.22222222])
        assert np.allclose(result["layer_0"], expected)
    
    def test_aggregate_method(self):
        """Test unified aggregate method"""
        aggregator = Aggregator(aggregation_method="fedavg")
        weights = [{"layer_0": np.array([1.0, 2.0])}]
        result = aggregator.aggregate(weights)
        assert "layer_0" in result
    
    def test_aggregation_history(self):
        """Test aggregation history tracking"""
        aggregator = Aggregator()
        weights = [
            {"layer_0": np.array([1.0])},
            {"layer_0": np.array([2.0])}
        ]
        aggregator.fedavg(weights)
        history = aggregator.get_aggregation_history()
        assert len(history) >= 1


class TestSessionStore:
    """Test Session Storage"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.store = SessionStore()
    
    def test_save_and_get(self):
        """Test save and retrieve session"""
        session = FLSession(
            id="test-123",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        retrieved = asyncio.run(self.store.get("test-123"))
        assert retrieved is not None
        assert retrieved.id == "test-123"
    
    def test_get_nonexistent(self):
        """Test getting nonexistent session"""
        result = asyncio.run(self.store.get("nonexistent"))
        assert result is None
    
    def test_update_session(self):
        """Test updating session"""
        session = FLSession(
            id="test-456",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        session.status = SessionStatus.TRAINING
        asyncio.run(self.store.update(session))
        
        retrieved = asyncio.run(self.store.get("test-456"))
        assert retrieved.status == SessionStatus.TRAINING
    
    def test_delete_session(self):
        """Test deleting session"""
        session = FLSession(
            id="test-789",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        deleted = asyncio.run(self.store.delete("test-789"))
        assert deleted is True
        
        retrieved = asyncio.run(self.store.get("test-789"))
        assert retrieved is None
    
    def test_list_sessions(self):
        """Test listing sessions"""
        for i in range(5):
            session = FLSession(
                id=f"test-{i}",
                config=FLConfig(),
                status=SessionStatus.PENDING,
                created_at=datetime.now()
            )
            asyncio.run(self.store.save(session))
        
        sessions = asyncio.run(self.store.list_sessions())
        assert len(sessions) == 5
        
        sessions = asyncio.run(self.store.list_sessions(status=SessionStatus.PENDING))
        assert len(sessions) == 5
    
    def test_register_client(self):
        """Test registering client to session"""
        session = FLSession(
            id="test-client",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        client = FLClientInfo(
            client_id="client-001",
            data_size=500,
            model_version="1.0"
        )
        
        registered = asyncio.run(self.store.register_client("test-client", client))
        assert registered is True
        
        retrieved = asyncio.run(self.store.get("test-client"))
        assert len(retrieved.participants) == 1
        assert retrieved.participants[0].client_id == "client-001"
    
    def test_get_client_session(self):
        """Test getting client's session"""
        session = FLSession(
            id="client-session",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        client = FLClientInfo(
            client_id="client-xyz",
            data_size=100,
            model_version="1.0"
        )
        asyncio.run(self.store.register_client("client-session", client))
        
        session_id = asyncio.run(self.store.get_client_session("client-xyz"))
        assert session_id == "client-session"
    
    def test_session_count(self):
        """Test session count property"""
        assert self.store.session_count == 0
        
        session = FLSession(
            id="count-test",
            config=FLConfig(),
            status=SessionStatus.PENDING,
            created_at=datetime.now()
        )
        asyncio.run(self.store.save(session))
        
        assert self.store.session_count == 1
    
    def test_clear(self):
        """Test clearing all sessions"""
        for i in range(3):
            session = FLSession(
                id=f"clear-test-{i}",
                config=FLConfig(),
                status=SessionStatus.PENDING,
                created_at=datetime.now()
            )
            asyncio.run(self.store.save(session))
        
        assert self.store.session_count == 3
        
        self.store.clear()
        
        assert self.store.session_count == 0


class TestModelTrainer:
    """Test Model Trainer"""
    
    def test_train_with_dummy_data(self):
        """Test training with generated data"""
        data = {
            "features": [[1.0, 2.0, 3.0]] * 100,
            "labels": [0, 1] * 50
        }
        weights = {"layer_0": [0.1, 0.2, 0.3]}
        result = ModelTrainer.train(weights, data, epochs=1, lr=0.01)
        assert "layer_0" in result
    
    def test_train_empty_data(self):
        """Test training with empty data"""
        weights = {"layer_0": [0.1, 0.2]}
        result = ModelTrainer.train(weights, {}, epochs=1)
        assert result == weights
    
    def test_compute_gradients(self):
        """Test gradient computation"""
        weights = {"layer_0": [1.0, 2.0, 3.0]}
        data = {"features": [[1.0, 2.0, 3.0]], "labels": [0]}
        gradients = ModelTrainer.compute_gradients(weights, data)
        assert "layer_0" in gradients


class TestFLClient:
    """Test Federated Learning Client"""
    
    def test_initialization(self):
        """Test client initialization"""
        client = FLClient(client_id="test-client")
        assert client.client_id == "test-client"
    
    def test_initialization_auto_id(self):
        """Test automatic client ID generation"""
        client = FLClient()
        assert client.client_id is not None
        assert len(client.client_id) > 0
    
    def test_load_data_dummy(self):
        """Test loading dummy data"""
        client = FLClient(client_id="test-client")
        data = client.load_data()
        assert "features" in data
        assert "labels" in data
        assert len(data["features"]) > 0
    
    def test_load_data_nonexistent(self):
        """Test loading nonexistent data path"""
        client = FLClient(client_id="test-client", data_path="/nonexistent/path.json")
        data = client.load_data()
        assert "features" in data
    
    def test_get_client_info(self):
        """Test getting client info"""
        client = FLClient(client_id="test-client")
        client.data_size = 1000
        info = client.get_client_info()
        assert info.client_id == "test-client"
        assert info.data_size == 1000
    
    def test_set_model_weights(self):
        """Test setting model weights"""
        client = FLClient(client_id="test-client")
        weights = {"layer_0": [1.0, 2.0, 3.0]}
        client.set_model_weights(weights)
        assert client.local_weights is not None
        assert client.model_version == "2.0"
    
    def test_local_train(self):
        """Test local training"""
        client = FLClient(client_id="test-client")
        config = FLConfig(local_epochs=1, learning_rate=0.01)
        
        async def train():
            return await client.local_train(config)
        
        result = asyncio.run(train())
        assert result.client_id == "test-client"
        assert result.weights is not None
    
    def test_local_train_with_dp(self):
        """Test local training with differential privacy"""
        client = FLClient(client_id="test-client")
        config = FLConfig(
            local_epochs=1,
            learning_rate=0.01,
            differential_privacy=True,
            dp_epsilon=1.0
        )
        
        async def train():
            return await client.local_train(config)
        
        result = asyncio.run(train())
        assert result.gradients is not None
    
    def test_get_model_update(self):
        """Test getting model update"""
        client = FLClient(client_id="test-client")
        client.data_size = 500
        client.accuracy = 0.85
        
        async def get_update():
            return await client.get_model_update("session-123")
        
        update = asyncio.run(get_update())
        assert update["session_id"] == "session-123"
        assert update["client_id"] == "test-client"
        assert update["data_size"] == 500


class TestFederatedLearningPlatform:
    """Test Federated Learning Platform"""
    
    def test_initialization(self):
        """Test platform initialization"""
        platform = FederatedLearningPlatform()
        assert platform.tls_enabled is False
    
    def test_create_session(self):
        """Test creating a session"""
        platform = FederatedLearningPlatform()
        config = FLConfig(model_name="test-model")
        
        async def create():
            return await platform.create_session(config)
        
        session = asyncio.run(create())
        assert session.id is not None
        assert session.status == SessionStatus.PENDING
        assert session.config.model_name == "test-model"
    
    def test_join_session(self):
        """Test joining a session"""
        platform = FederatedLearningPlatform()
        config = FLConfig(min_clients=1, max_clients=5)
        
        async def setup():
            session = await platform.create_session(config)
            return session.id
        
        session_id = asyncio.run(setup())
        
        client = FLClientInfo(
            client_id="client-001",
            data_size=1000,
            model_version="1.0"
        )
        
        async def join():
            return await platform.join_session(session_id, client)
        
        result = asyncio.run(join())
        assert result is True
        
        session = asyncio.run(platform.get_session_status(session_id))
        assert len(session.participants) == 1
    
    def test_join_nonexistent_session(self):
        """Test joining nonexistent session"""
        platform = FederatedLearningPlatform()
        client = FLClientInfo(client_id="client-001", data_size=1000)
        
        async def join():
            return await platform.join_session("nonexistent", client)
        
        result = asyncio.run(join())
        assert result is False
    
    def test_start_training(self):
        """Test starting training"""
        platform = FederatedLearningPlatform()
        config = FLConfig(
            min_clients=1,
            max_clients=5,
            rounds=2
        )
        
        async def setup():
            session = await platform.create_session(config)
            
            client = FLClientInfo(
                client_id="client-001",
                data_size=1000
            )
            await platform.join_session(session.id, client)
            
            return session.id
        
        session_id = asyncio.run(setup())
        
        async def start():
            return await platform.start_training(session_id)
        
        session = asyncio.run(start())
        assert session.status == SessionStatus.TRAINING
        assert session.current_round == 1
    
    def test_start_training_insufficient_clients(self):
        """Test starting training with insufficient clients"""
        platform = FederatedLearningPlatform()
        config = FLConfig(min_clients=2, max_clients=5)
        
        async def setup():
            session = await platform.create_session(config)
            return session.id
        
        session_id = asyncio.run(setup())
        
        async def start():
            return await platform.start_training(session_id)
        
        with pytest.raises(ValueError):
            asyncio.run(start())
    
    def test_aggregate_models(self):
        """Test aggregating models"""
        platform = FederatedLearningPlatform()
        config = FLConfig(
            min_clients=1,
            max_clients=5,
            rounds=1
        )
        
        async def setup():
            session = await platform.create_session(config)
            
            client = FLClientInfo(
                client_id="client-001",
                data_size=1000
            )
            await platform.join_session(session.id, client)
            
            await platform.start_training(session.id)
            
            return session.id
        
        session_id = asyncio.run(setup())
        
        async def aggregate():
            return await platform.aggregate_models(session_id)
        
        global_model = asyncio.run(aggregate())
        assert global_model.session_id == session_id
        assert global_model.version == "2.0"
        assert global_model.round_number == 1
    
    def test_aggregate_models_completed_session(self):
        """Test that session completes after all rounds"""
        platform = FederatedLearningPlatform()
        config = FLConfig(
            min_clients=1,
            max_clients=5,
            rounds=1
        )
        
        async def setup():
            session = await platform.create_session(config)
            
            client = FLClientInfo(
                client_id="client-001",
                data_size=1000
            )
            await platform.join_session(session.id, client)
            
            await platform.start_training(session.id)
            
            return session.id
        
        session_id = asyncio.run(setup())
        
        async def aggregate():
            return await platform.aggregate_models(session_id)
        
        asyncio.run(aggregate())
        
        session = asyncio.run(platform.get_session_status(session_id))
        assert session.status == SessionStatus.COMPLETED
    
    def test_get_session_status(self):
        """Test getting session status"""
        platform = FederatedLearningPlatform()
        config = FLConfig()
        
        async def create():
            return await platform.create_session(config)
        
        session = asyncio.run(create())
        
        async def get_status():
            return await platform.get_session_status(session.id)
        
        status = asyncio.run(get_status())
        assert status.id == session.id
    
    def test_list_sessions(self):
        """Test listing sessions"""
        platform = FederatedLearningPlatform()
        
        async def create_multiple():
            for i in range(3):
                await platform.create_session(FLConfig(model_name=f"model-{i}"))
        
        asyncio.run(create_multiple())
        
        async def list_all():
            return await platform.list_sessions()
        
        sessions = asyncio.run(list_all())
        assert len(sessions) == 3
    
    def test_get_global_model_completed(self):
        """Test getting global model for completed session"""
        platform = FederatedLearningPlatform()
        config = FLConfig(
            min_clients=1,
            max_clients=5,
            rounds=1
        )
        
        async def setup():
            session = await platform.create_session(config)
            
            client = FLClientInfo(
                client_id="client-001",
                data_size=1000
            )
            await platform.join_session(session.id, client)
            
            await platform.start_training(session.id)
            await platform.aggregate_models(session.id)
            
            return session.id
        
        session_id = asyncio.run(setup())
        
        async def get_model():
            return await platform.get_global_model(session_id)
        
        model = asyncio.run(get_model())
        assert model is not None
        assert model.session_id == session_id
    
    def test_get_privacy_report(self):
        """Test getting privacy report"""
        platform = FederatedLearningPlatform()
        
        async def get_report():
            return await platform.get_privacy_report("nonexistent")
        
        report = asyncio.run(get_report())
        assert "epsilon" in report
        assert "budget_used" in report
    
    def test_close_session(self):
        """Test closing session"""
        platform = FederatedLearningPlatform()
        config = FLConfig()
        
        async def create():
            return await platform.create_session(config)
        
        session = asyncio.run(create())
        
        async def close():
            return await platform.close_session(session.id, reason="error")
        
        result = asyncio.run(close())
        assert result is True
        
        status = asyncio.run(platform.get_session_status(session.id))
        assert status.status == SessionStatus.FAILED
    
    def test_active_session_count(self):
        """Test active session count"""
        platform = FederatedLearningPlatform()
        
        async def create_session():
            await platform.create_session(FLConfig())
        
        assert platform.active_session_count == 0
        
        asyncio.run(create_session())
        
        assert platform.active_session_count == 1


class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_full_fl_workflow(self):
        """Test complete federated learning workflow"""
        platform = FederatedLearningPlatform()
        
        async def run_workflow():
            config = FLConfig(
                model_name="test-model",
                min_clients=2,
                max_clients=5,
                rounds=3,
                differential_privacy=True,
                dp_epsilon=2.0
            )
            
            session = await platform.create_session(config)
            
            for i in range(2):
                client = FLClientInfo(
                    client_id=f"client-{i}",
                    data_size=1000 + i * 500
                )
                await platform.join_session(session.id, client)
            
            session = await platform.start_training(session.id)
            
            for round_num in range(1, config.rounds + 1):
                session = await platform.get_session_status(session.id)
                for participant in session.participants:
                    local_model = LocalModel(
                        client_id=participant.client_id,
                        weights={"layer_0": [1.0] * 10},
                        data_size=participant.data_size,
                        accuracy=0.8 + round_num * 0.05,
                        loss=0.2 - round_num * 0.02
                    )
                    await platform.submit_local_model(session.id, local_model)
                
                global_model = await platform.aggregate_models(session.id)
                assert global_model.round_number == round_num
                
                session = await platform.get_session_status(session.id)
            
            assert session.status == SessionStatus.COMPLETED
            
            # Privacy configuration was used during aggregation
            assert session.config.differential_privacy is True
            assert session.config.dp_epsilon == 2.0
            
            return session
        
        final_session = asyncio.run(run_workflow())
        assert final_session.status == SessionStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
