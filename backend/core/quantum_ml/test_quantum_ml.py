"""
Quantum Machine Learning Tests

Comprehensive test suite for quantum ML algorithms.
"""

import pytest
import numpy as np
from typing import Tuple


# Fixtures
@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
    return X, y


@pytest.fixture
def clustering_data():
    """Generate clustering data for testing."""
    np.random.seed(42)
    n_clusters = 3
    n_samples = 150
    n_features = 4
    
    X = []
    labels = []
    
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 3
        n_points = n_samples // n_clusters
        points = np.random.randn(n_points, n_features) * 0.5 + center
        X.append(points)
        labels.extend([i] * n_points)
    
    X = np.vstack(X)
    labels = np.array(labels)
    
    return X, labels


class TestDataEncoding:
    """Tests for quantum data encoding module."""
    
    def test_amplitude_encoder_basic(self):
        """Test basic amplitude encoding."""
        from quantum_ml.data_encoding import AmplitudeEncoder
        
        encoder = AmplitudeEncoder()
        data = np.array([1.0, 2.0, 3.0, 4.0])
        
        state = encoder.encode(data)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-6
        
        # Check shape
        assert len(state) == 4  # No padding needed for power of 2
    
    def test_amplitude_encoder_padding(self):
        """Test amplitude encoding with padding."""
        from quantum_ml.data_encoding import AmplitudeEncoder
        
        encoder = AmplitudeEncoder(pad_to_power_of_two=True)
        data = np.array([1.0, 2.0, 3.0])  # Not power of 2
        
        state = encoder.encode(data)
        
        # Should be padded to 4 elements
        assert len(state) == 4
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6
    
    def test_angle_encoder(self):
        """Test angle encoding."""
        from quantum_ml.data_encoding import AngleEncoder
        
        encoder = AngleEncoder(scale=np.pi)
        data = np.array([0.5, 0.25, 0.75, 1.0])
        
        encoded = encoder.encode(data)
        
        # Check scaling
        expected = data * np.pi
        np.testing.assert_array_almost_equal(encoded, expected)
    
    def test_dictionary_encoder(self):
        """Test dictionary/sparse encoding."""
        from quantum_ml.data_encoding import DictionaryEncoder
        
        encoder = DictionaryEncoder(n_qubits=3, normalize=True)
        data = np.array([1.0, 2.0, 3.0])
        
        state = encoder.encode(data)
        
        # Should be 8 elements (2^3)
        assert len(state) == 8
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6
    
    def test_basis_encoder(self):
        """Test basis state encoding."""
        from quantum_ml.data_encoding import BasisEncoder
        
        encoder = BasisEncoder(n_qubits=4)
        data = np.array([0, 1, 2, 3])
        
        states = encoder.encode(data)
        
        # Should have 16 elements (2^4)
        assert states.shape == (16,)
        assert np.sum(np.abs(states) ** 2) > 0.9  # Probability sum
    
    def test_data_encoder_factory(self):
        """Test DataEncoder factory method."""
        from quantum_ml.data_encoding import DataEncoder
        
        encoder = DataEncoder(encoding="amplitude")
        data = np.array([1.0, 2.0])
        
        state = encoder.encode(data)
        
        assert state is not None
        assert len(state) > 0
    
    def test_auto_select_encoding(self):
        """Test automatic encoding selection."""
        from quantum_ml.data_encoding import DataEncoder
        
        # Low dimensions -> angle encoding
        small_data = np.random.randn(4)
        encoding = DataEncoder.auto_select_encoding(small_data, max_qubits=10)
        assert encoding == "angle"
        
        # High dimensions -> amplitude encoding
        large_data = np.random.randn(20)
        encoding = DataEncoder.auto_select_encoding(large_data, max_qubits=10)
        assert encoding == "amplitude"


class TestQuantumNeuralNetwork:
    """Tests for quantum neural network module."""
    
    def test_qnn_initialization(self):
        """Test QNN initialization."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        assert qnn.n_qubits == 4
        assert qnn.n_layers == 2
        assert qnn.learning_rate > 0
    
    def test_qnn_parameters(self):
        """Test QNN parameter count."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        # Expected: n_qubits * n_layers * 3 (ry, rz, entanglement pattern)
        expected_params = 4 * 2 * 3
        assert qnn.n_parameters == expected_params
    
    def test_qnn_encode_input(self):
        """Test input encoding."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(n_qubits=4)
        x = np.array([0.1, 0.5, 0.8, 0.3])
        
        encoded = qnn.encode_input(x)
        
        assert len(encoded) == 4
        np.testing.assert_array_less(encoded, np.pi + 0.1)
    
    def test_qnn_forward(self):
        """Test forward pass."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(n_qubits=4)
        x = np.array([0.1, 0.5, 0.8, 0.3])
        
        pred = qnn.forward(x)
        
        assert pred.shape == (1,)
        assert not np.isnan(pred).any()
    
    def test_qnn_train(self, sample_data):
        """Test QNN training."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        X, y = sample_data
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, learning_rate=0.1)
        
        # Train for a few epochs
        history = qnn.fit(X, y, epochs=10, verbose=False)
        
        assert 'loss' in history
        assert len(history['loss']) == 10
        assert history['loss'][-1] >= 0  # Loss should be non-negative
    
    def test_qnn_predict(self, sample_data):
        """Test QNN prediction."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        X, y = sample_data
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        qnn.fit(X, y, epochs=5, verbose=False)
        
        predictions = qnn.predict(X)
        
        assert len(predictions) == len(X)
    
    def test_qnn_classify(self, sample_data):
        """Test QNN classification."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        X, y = sample_data
        
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        qnn.fit(X, y, epochs=5, verbose=False)
        
        labels = qnn.classify(X, threshold=0.5)
        
        assert len(labels) == len(X)
        assert set(labels).issubset({0, 1})
    
    def test_qnn_save_load(self, sample_data):
        """Test QNN save and load."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        import tempfile
        import os
        
        X, y = sample_data
        
        qnn1 = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        qnn1.fit(X, y, epochs=5, verbose=False)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            qnn1.save(filepath)
            
            # Load
            qnn2 = QuantumNeuralNetwork.load(filepath)
            
            assert qnn2.n_qubits == qnn1.n_qubits
            assert qnn2.n_layers == qnn1.n_layers
            assert qnn2.is_trained == qnn1.is_trained
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_qnn_regressor(self, regression_data):
        """Test QNN regressor."""
        from quantum_ml.qnn import QuantumRegressor
        
        X, y = regression_data
        
        regressor = QuantumRegressor(n_qubits=4, n_layers=2)
        regressor.fit(X, y, epochs=10, verbose=False)
        
        predictions = regressor.predict(X[:10])
        
        assert len(predictions) == 10


class TestQuantumSVM:
    """Tests for quantum SVM module."""
    
    def test_qsvm_classification_init(self):
        """Test QSVM classifier initialization."""
        from quantum_ml.qsvm import QuantumSVM
        
        qsvm = QuantumSVM(mode="classification", n_qubits=4)
        
        assert qsvm.mode == "classification"
    
    def test_qsvm_regression_init(self):
        """Test QSVM regressor initialization."""
        from quantum_ml.qsvm import QuantumSVM
        
        qsvm = QuantumSVM(mode="regression", n_qubits=4)
        
        assert qsvm.mode == "regression"
    
    def test_qsvm_train_classification(self, sample_data):
        """Test QSVM classification training."""
        from quantum_ml.qsvm import QuantumSVM
        
        X, y = sample_data
        
        qsvm = QuantumSVM(mode="classification", n_qubits=4, kernel="quantum")
        qsvm.train(X, y)
        
        predictions = qsvm.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_qsvm_train_regression(self, regression_data):
        """Test QSVM regression training."""
        from quantum_ml.qsvm import QuantumSVM
        
        X, y = regression_data
        
        qsvm = QuantumSVM(mode="regression", n_qubits=4, kernel="quantum")
        qsvm.train(X, y)
        
        predictions = qsvm.predict(X[:10])
        
        assert len(predictions) == 10
    
    def test_qsvm_score(self, sample_data):
        """Test QSVM scoring."""
        from quantum_ml.qsvm import QuantumSVM
        
        X, y = sample_data
        
        qsvm = QuantumSVM(mode="classification", n_qubits=4, kernel="quantum")
        qsvm.train(X, y)
        
        score = qsvm.score(X, y)
        
        assert 0 <= score <= 1


class TestQuantumClustering:
    """Tests for quantum clustering module."""
    
    def test_quantum_kmeans_init(self):
        """Test Quantum K-Means initialization."""
        from quantum_ml.qclustering import QuantumKMeans
        
        qkmeans = QuantumKMeans(n_clusters=3, n_qubits=4)
        
        assert qkmeans.n_clusters == 3
        assert qkmeans.n_qubits == 4
    
    def test_quantum_kmeans_fit(self, clustering_data):
        """Test Quantum K-Means fitting."""
        from quantum_ml.qclustering import QuantumKMeans
        
        X, true_labels = clustering_data
        
        qkmeans = QuantumKMeans(n_clusters=3, n_qubits=4, max_iter=10)
        labels = qkmeans.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(set(labels)) <= 3
    
    def test_quantum_kmeans_predict(self, clustering_data):
        """Test Quantum K-Means prediction."""
        from quantum_ml.qclustering import QuantumKMeans
        
        X, true_labels = clustering_data
        
        qkmeans = QuantumKMeans(n_clusters=3, n_qubits=4)
        qkmeans.fit(X)
        
        # Should be able to predict on same data
        labels = qkmeans.predict(X)
        
        assert len(labels) == len(X)
    
    def test_quantum_dbscan_init(self):
        """Test Quantum DBSCAN initialization."""
        from quantum_ml.qclustering import QuantumDBSCAN
        
        qdbscan = QuantumDBSCAN(eps=0.5, min_samples=5, n_qubits=4)
        
        assert qdbscan.eps == 0.5
        assert qdbscan.min_samples == 5
    
    def test_quantum_dbscan_fit(self, clustering_data):
        """Test Quantum DBSCAN fitting."""
        from quantum_ml.qclustering import QuantumDBSCAN
        
        X, true_labels = clustering_data
        
        qdbscan = QuantumDBSCAN(eps=2.0, min_samples=3, n_qubits=4)
        labels = qdbscan.fit_predict(X)
        
        assert len(labels) == len(X)
    
    def test_quantum_spectral_init(self):
        """Test Quantum Spectral initialization."""
        from quantum_ml.qclustering import QuantumSpectralClustering
        
        qspectral = QuantumSpectralClustering(
            n_clusters=3, n_qubits=4
        )
        
        assert qspectral.n_clusters == 3
    
    def test_quantum_spectral_fit(self, clustering_data):
        """Test Quantum Spectral fitting."""
        from quantum_ml.qclustering import QuantumSpectralClustering
        
        X, true_labels = clustering_data
        
        qspectral = QuantumSpectralClustering(n_clusters=3, n_qubits=4)
        labels = qspectral.fit_predict(X)
        
        assert len(labels) == len(X)
    
    def test_quantum_centroid_init(self):
        """Test Quantum Centroid initialization."""
        from quantum_ml.qclustering import QuantumCentroidClustering
        
        qcentroid = QuantumCentroidClustering(
            n_clusters=3, n_qubits=4
        )
        
        assert qcentroid.n_clusters == 3
    
    def test_quantum_clustering_wrapper(self):
        """Test QuantumClustering wrapper class."""
        from quantum_ml.qclustering import QuantumClustering
        
        # Test with different algorithms
        for algo in ["kmeans", "dbscan", "spectral"]:
            qc = QuantumClustering(
                algorithm=algo,
                n_clusters=3,
                n_qubits=4
            )
            assert qc is not None


class TestQuantumDistance:
    """Tests for quantum distance metrics."""
    
    def test_fidelity_distance(self):
        """Test fidelity distance computation."""
        from quantum_ml.qclustering import QuantumDistance
        
        state1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        state2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        
        # Same state should have distance 0
        dist = QuantumDistance.fidelity_distance(state1, state2)
        assert abs(dist) < 1e-6
    
    def test_hellinger_distance(self):
        """Test Hellinger distance."""
        from quantum_ml.qclustering import QuantumDistance
        
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        
        dist = QuantumDistance.hellinger_distance(p, q)
        assert abs(dist) < 1e-6
    
    def test_compute_distance_metric(self):
        """Test compute_distance with different metrics."""
        from quantum_ml.qclustering import QuantumDistance
        
        state1 = np.array([1.0, 0.0])
        state2 = np.array([0.0, 1.0])
        
        for metric in ["fidelity", "trace", "hellinger"]:
            dist = QuantumDistance.compute_distance(state1, state2, metric)
            assert dist >= 0


class TestQuantumGradients:
    """Tests for quantum gradient computation."""
    
    def test_finite_difference(self):
        """Test finite difference gradient."""
        from quantum_ml.qnn import QuantumGradients
        
        # Create a mock circuit
        if hasattr(QuantumGradients, 'finite_difference'):
            # Finite difference should return an array
            gradients = QuantumGradients.finite_difference(
                None, np.array([0.1, 0.2, 0.3])
            )
            assert len(gradients) == 3


class TestQuantumLossFunction:
    """Tests for quantum loss functions."""
    
    def test_mse(self):
        """Test MSE loss."""
        from quantum_ml.qnn import QuantumLossFunction
        
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 1.9, 3.1])
        
        loss = QuantumLossFunction.mse(pred, target)
        
        expected = np.mean((pred - target) ** 2)
        assert abs(loss - expected) < 1e-6
    
    def test_mae(self):
        """Test MAE loss."""
        from quantum_ml.qnn import QuantumLossFunction
        
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        
        loss = QuantumLossFunction.mae(pred, target)
        
        assert abs(loss) < 1e-6
    
    def test_cross_entropy(self):
        """Test cross-entropy loss."""
        from quantum_ml.qnn import QuantumLossFunction
        
        probs = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        
        loss = QuantumLossFunction.cross_entropy(probs, labels)
        
        assert loss >= 0
    
    def test_hinge_loss(self):
        """Test hinge loss."""
        from quantum_ml.qnn import QuantumLossFunction
        
        pred = np.array([0.5, -0.5])
        labels = np.array([1, -1])
        
        loss = QuantumLossFunction.hinge(pred, labels)
        
        assert loss >= 0
    
    def test_fidelity_loss(self):
        """Test fidelity loss."""
        from quantum_ml.qnn import QuantumLossFunction
        
        state1 = np.array([1.0, 0.0])
        state2 = np.array([1.0, 0.0])
        
        loss = QuantumLossFunction.fidelity_loss(state1, state2)
        
        assert abs(loss) < 1e-6  # Same states should have 0 loss


class TestConfig:
    """Tests for configuration module."""
    
    def test_default_config(self):
        """Test default quantum config."""
        from quantum_ml.config import QuantumConfig
        
        config = QuantumConfig()
        
        assert config.backend == "qasm_simulator"
        assert config.shots == 1024
        assert config.learning_rate > 0
    
    def test_config_from_dict(self):
        """Test config creation from dictionary."""
        from quantum_ml.config import QuantumConfig
        
        config_dict = {
            "n_qubits": 8,
            "learning_rate": 0.05,
            "epochs": 200
        }
        
        config = QuantumConfig.from_dict(config_dict)
        
        assert config.n_qubits == 8
        assert config.learning_rate == 0.05
        assert config.epochs == 200
    
    def test_config_save_load(self):
        """Test config save and load."""
        from quantum_ml.config import QuantumConfig
        import tempfile
        import os
        
        config = QuantumConfig(n_qubits=8, learning_rate=0.05)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            config.save(filepath)
            loaded = QuantumConfig.load(filepath)
            
            assert loaded.n_qubits == config.n_qubits
            assert loaded.learning_rate == config.learning_rate
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestAPI:
    """Tests for API module."""
    
    def test_model_registry(self):
        """Test model registry."""
        from quantum_ml.api import ModelRegistry
        
        registry = ModelRegistry()
        
        assert len(registry.list_models()) == 0
        
        # Add a mock model
        class MockModel:
            def __init__(self, model_id):
                self.model_id = model_id
                self.is_trained = False
        
        mock = MockModel("test_id")
        model_id = registry.register(mock)
        
        assert model_id == "test_id"
        assert len(registry.list_models()) == 1
        
        # Get model
        retrieved = registry.get("test_id")
        assert retrieved is not None
        
        # Remove model
        assert registry.remove("test_id") is True
        assert len(registry.list_models()) == 0
    
    def test_pipeline(self):
        """Test quantum ML pipeline."""
        from quantum_ml.api import QuantumMLPipeline
        
        pipeline = QuantumMLPipeline()
        
        # Add steps
        pipeline.add_step("step1", "qclustering", n_clusters=3, n_qubits=4)
        
        assert len(pipeline.steps) == 1


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow."""
        from quantum_ml.qnn import QuantumNeuralNetwork
        from quantum_ml.qsvm import QuantumSVM
        
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.array([0] * 50 + [1] * 50)
        
        # Train QNN
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        qnn.fit(X, y, epochs=10, verbose=False)
        qnn_pred = qnn.predict(X[:10])
        
        # Train QSVM
        qsvm = QuantumSVM(mode="classification", n_qubits=4)
        qsvm.train(X, y)
        qsvm_pred = qsvm.predict(X[:10])
        
        assert len(qnn_pred) == 10
        assert len(qsvm_pred) == 10
    
    def test_full_clustering_workflow(self):
        """Test complete clustering workflow."""
        from quantum_ml.qclustering import QuantumKMeans
        
        np.random.seed(42)
        n_samples = 100
        n_features = 4
        
        X = np.random.randn(n_samples, n_features)
        
        qkmeans = QuantumKMeans(n_clusters=3, n_qubits=4)
        labels = qkmeans.fit_predict(X)
        
        assert len(labels) == n_samples


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
