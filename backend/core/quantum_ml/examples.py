"""
Quantum Machine Learning Examples

Demonstrates usage of quantum ML algorithms with practical examples.
"""

import numpy as np
from typing import Tuple, Dict


def generate_classification_data(n_samples: int = 100,
                                 n_features: int = 4,
                                 noise: float = 0.1,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        X: Feature matrix
        y: Labels
    """
    np.random.seed(random_state)
    
    # Create two classes
    n_class1 = n_samples // 2
    n_class2 = n_samples - n_class1
    
    # Class 1: centered at (-1, -1, ...)
    X1 = np.random.randn(n_class1, n_features) - 1
    # Class 2: centered at (1, 1, ...)
    X2 = np.random.randn(n_class2, n_features) + 1
    
    X = np.vstack([X1, X2])
    
    # Add noise
    X += np.random.randn(*X.shape) * noise
    
    # Create labels
    y = np.array([0] * n_class1 + [1] * n_class2)
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def generate_regression_data(n_samples: int = 100,
                             n_features: int = 4,
                             noise: float = 0.1,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        X: Feature matrix
        y: Target values
    """
    np.random.seed(random_state)
    
    # Create sinusoidal data
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + noise * np.random.randn(n_samples)
    
    return X, y


def generate_clustering_data(n_samples: int = 150,
                             n_clusters: int = 3,
                             n_features: int = 4,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustering data.
    
    Args:
        n_samples: Number of samples
        n_clusters: Number of clusters
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        X: Feature matrix
        true_labels: True cluster labels
    """
    np.random.seed(random_state)
    
    # Create spherical clusters
    X = []
    labels = []
    
    for i in range(n_clusters):
        # Random center
        center = np.random.randn(n_features) * 3
        
        # Generate points around center
        n_points = n_samples // n_clusters
        points = np.random.randn(n_points, n_features) * 0.5 + center
        
        X.append(points)
        labels.extend([i] * n_points)
    
    X = np.vstack(X)
    labels = np.array(labels)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return X, labels


def example_qnn_classification():
    """Example: Quantum Neural Network for classification."""
    print("=" * 60)
    print("Quantum Neural Network - Classification Example")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=100, n_features=4)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    # Create QNN
    try:
        from quantum_ml.qnn import QuantumNeuralNetwork
        
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=2,
            learning_rate=0.1
        )
        
        # Train
        print("\nTraining QNN...")
        history = qnn.fit(X_train, y_train, epochs=50, verbose=False)
        
        # Evaluate
        train_pred = qnn.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        test_pred = qnn.predict(X_test)
        test_acc = np.mean(test_pred == y_test)
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Final Loss: {history['loss'][-1]:.6f}")
        
    except ImportError as e:
        print(f"Could not import QNN: {e}")
    
    print()


def example_qnn_regression():
    """Example: Quantum Neural Network for regression."""
    print("=" * 60)
    print("Quantum Neural Network - Regression Example")
    print("=" * 60)
    
    # Generate data
    X, y = generate_regression_data(n_samples=100, n_features=4)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X.shape[1]}")
    
    try:
        from quantum_ml.qnn import QuantumRegressor
        
        qnn = QuantumRegressor(
            n_qubits=4,
            n_layers=2,
            learning_rate=0.1
        )
        
        # Train
        print("\nTraining QNN Regressor...")
        history = qnn.fit(X_train, y_train, epochs=50, verbose=False)
        
        # Evaluate
        train_pred = qnn.predict(X_train)
        train_mse = np.mean((train_pred - y_train) ** 2)
        
        test_pred = qnn.predict(X_test)
        test_mse = np.mean((test_pred - y_test) ** 2)
        
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Final Loss: {history['loss'][-1]:.6f}")
        
    except ImportError as e:
        print(f"Could not import QNN: {e}")
    
    print()


def example_qsvm_classification():
    """Example: Quantum SVM for classification."""
    print("=" * 60)
    print("Quantum SVM - Classification Example")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=100, n_features=4)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    try:
        from quantum_ml.qsvm import QuantumSVM
        
        qsvm = QuantumSVM(
            mode="classification",
            n_qubits=4,
            kernel="quantum"
        )
        
        # Train
        print("\nTraining Quantum SVM...")
        qsvm.train(X_train, y_train)
        
        # Evaluate
        train_pred = qsvm.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        test_pred = qsvm.predict(X_test)
        test_acc = np.mean(test_pred == y_test)
        
        score = qsvm.score(X_test, y_test)
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Model Score: {score:.4f}")
        
    except ImportError as e:
        print(f"Could not import QSVM: {e}")
    
    print()


def example_qsvm_regression():
    """Example: Quantum SVM for regression."""
    print("=" * 60)
    print("Quantum SVM - Regression Example")
    print("=" * 60)
    
    # Generate data
    X, y = generate_regression_data(n_samples=100, n_features=4)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    try:
        from quantum_ml.qsvm import QuantumSVM
        
        qsvm = QuantumSVM(
            mode="regression",
            n_qubits=4,
            kernel="quantum"
        )
        
        # Train
        print("\nTraining Quantum SVR...")
        qsvm.train(X_train, y_train)
        
        # Evaluate
        train_pred = qsvm.predict(X_train)
        train_mse = np.mean((train_pred - y_train) ** 2)
        
        test_pred = qsvm.predict(X_test)
        test_mse = np.mean((test_pred - y_test) ** 2)
        
        score = qsvm.score(X_test, y_test)
        
        print(f"Train MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"R² Score: {score:.4f}")
        
    except ImportError as e:
        print(f"Could not import QSVM: {e}")
    
    print()


def example_qclustering():
    """Example: Quantum Clustering."""
    print("=" * 60)
    print("Quantum Clustering Example")
    print("=" * 60)
    
    # Generate data
    X, true_labels = generate_clustering_data(
        n_samples=150, n_clusters=3, n_features=4
    )
    
    print(f"Samples: {len(X)}")
    print(f"Clusters: {len(np.unique(true_labels))}")
    print(f"Features: {X.shape[1]}")
    
    try:
        from quantum_ml.qclustering import QuantumKMeans, QuantumClustering
        
        # Quantum K-Means
        print("\nQuantum K-Means Clustering...")
        qkmeans = QuantumKMeans(
            n_clusters=3,
            n_qubits=4,
            max_iter=50,
            random_state=42
        )
        
        q_labels = qkmeans.fit_predict(X)
        q_acc = np.mean(q_labels == true_labels)
        
        print(f"Quantum K-Means Accuracy: {q_acc:.4f}")
        
        # Compare with classical
        print("\nComparison with Classical K-Means...")
        comparison = qkmeans.compare_with_classical(X)
        
        print(f"Quantum Silhouette: {comparison['quantum_silhouette']:.4f}")
        print(f"Classical Silhouette: {comparison['classical_silhouette']:.4f}")
        print(f"Labels Match: {comparison['labels_match']:.4f}")
        
    except ImportError as e:
        print(f"Could not import clustering: {e}")
    
    print()


def example_data_encoding():
    """Example: Data encoding methods."""
    print("=" * 60)
    print("Quantum Data Encoding Example")
    print("=" * 60)
    
    try:
        from quantum_ml.data_encoding import (
            DataEncoder, AmplitudeEncoder, AngleEncoder, 
            DictionaryEncoder, BasisEncoder
        )
        
        # Sample data
        data = np.array([0.1, 0.5, 0.8, 0.3, 0.7])
        print(f"Original data: {data}")
        print(f"Shape: {data.shape}")
        
        # Amplitude encoding
        print("\n1. Amplitude Encoding:")
        amp_encoder = DataEncoder(encoding="amplitude")
        amp_state = amp_encoder.encode(data)
        print(f"   State shape: {amp_state.shape}")
        print(f"   State (first 5): {amp_state[:5]}")
        print(f"   Norm: {np.linalg.norm(amp_state):.6f}")
        
        # Angle encoding
        print("\n2. Angle Encoding:")
        angle_encoder = DataEncoder(encoding="angle", scale=np.pi)
        angle_encoded = angle_encoder.encode(data)
        print(f"   Encoded: {angle_encoded}")
        print(f"   Required qubits: {angle_encoder.get_n_qubits(data.shape)}")
        
        # Auto-selection
        print("\n3. Auto-selection:")
        recommended = DataEncoder.auto_select_encoding(data, max_qubits=8)
        print(f"   Recommended encoding: {recommended}")
        
    except ImportError as e:
        print(f"Could not import encoders: {e}")
    
    print()


def example_pipeline():
    """Example: Quantum ML Pipeline."""
    print("=" * 60)
    print("Quantum ML Pipeline Example")
    print("=" * 60)
    
    try:
        from quantum_ml.api import QuantumMLPipeline
        
        # Generate data
        X, y = generate_classification_data(n_samples=100, n_features=4)
        
        # Create pipeline
        pipeline = QuantumMLPipeline()
        
        # Add steps
        print("Creating pipeline with 2 steps...")
        
        # Fit pipeline
        print("Fitting pipeline...")
        pipeline.fit(X)
        
        print("Pipeline fitted successfully!")
        
    except ImportError as e:
        print(f"Could not import pipeline: {e}")
    
    print()


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" QUANTUM MACHINE LEARNING - EXAMPLES")
    print("=" * 60 + "\n")
    
    examples = [
        ("Data Encoding", example_data_encoding),
        ("QNN Classification", example_qnn_classification),
        ("QNN Regression", example_qnn_regression),
        ("QSVM Classification", example_qsvm_classification),
        ("QSVM Regression", example_qsvm_regression),
        ("Quantum Clustering", example_qclustering),
        ("Pipeline", example_pipeline),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
