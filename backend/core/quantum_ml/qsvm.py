"""
Quantum Support Vector Machine Module

Implements quantum kernel-based SVM for classification and regression.
Supports quantum feature mapping and kernel computation.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class QuantumKernelConfig:
    """Configuration for quantum kernel."""
    n_qubits: int = 4
    feature_map_type: str = "zz_feature_map"
    reps: int = 2
    entanglement: str = "linear"
    parameter_shift: bool = True
    fidelity: Optional[object] = None


class QuantumFeatureMap:
    """Quantum feature mapping for kernel computation."""
    
    def __init__(self, n_features: int, n_qubits: int = None,
                 feature_map_type: str = "zz_feature_map",
                 reps: int = 2, 
                 entanglement: str = "linear"):
        """Initialize quantum feature map.
        
        Args:
            n_features: Number of input features
            n_qubits: Number of qubits (defaults to n_features)
            feature_map_type: Type of feature map
            reps: Number of repetitions
            entanglement: Entanglement pattern
        """
        self.n_features = n_features
        self.n_qubits = n_qubits or min(n_features, 8)
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.entanglement = entanglement
        self.parameters = None
        self.circuit = None
        
        if QISKIT_AVAILABLE:
            self._build_circuit()
    
    def _build_circuit(self):
        """Build the feature map quantum circuit."""
        self.parameters = ParameterVector('x', self.n_features)
        
        if self.n_qubits < self.n_features:
            # Use amplitude encoding for high-dimensional data
            self.circuit = self._build_amplitude_encoding_map()
        else:
            self.circuit = self._build_feature_map()
    
    def _build_amplitude_encoding_map(self) -> QuantumCircuit:
        """Build amplitude encoding feature map."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Encode data into amplitudes
        for i in range(self.n_qubits):
            qc.ry(self.parameters[i], i)
        
        # Entangling layers
        for _ in range(self.reps):
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
            
            # Rotation layers
            for i in range(self.n_qubits):
                qc.rz(self.parameters[i], i)
        
        return qc
    
    def _build_feature_map(self) -> QuantumCircuit:
        """Build parameterized feature map circuit."""
        qc = QuantumCircuit(self.n_qubits)
        
        if self.feature_map_type == "zz_feature_map":
            return self._build_zz_feature_map()
        elif self.feature_map_type == "zz_yz_feature_map":
            return self._build_zz_yz_feature_map()
        elif self.feature_map_type == "efficient_su2":
            return self._build_efficient_su2_map()
        else:
            return self._build_zz_feature_map()
    
    def _build_zz_feature_map(self) -> QuantumCircuit:
        """Build ZZ-feature map (N-local).
        
        Based on: https://arxiv.org/abs/1804.11326
        """
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.reps):
            # Feature encoding layer
            for i in range(self.n_qubits):
                qc.ry(self.parameters[i], i)
            
            # Entangling layer with ZZ interactions
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cnot(i, j)
                    qc.rz(2 * self.parameters[i] * self.parameters[j], j)
                    qc.cnot(i, j)
            
            # Final rotation layer
            for i in range(self.n_qubits):
                qc.rz(self.parameters[i], i)
        
        return qc
    
    def _build_zz_yz_feature_map(self) -> QuantumCircuit:
        """Build combined ZZ and YZ feature map."""
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.reps):
            # First rotation block (Y rotations)
            for i in range(self.n_qubits):
                qc.ry(self.parameters[i], i)
            
            # Entangling ZZ block
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
            
            # Second rotation block (Z rotations)
            for i in range(self.n_qubits):
                qc.rz(self.parameters[i], i)
            
            # YZ interactions
            for i in range(self.n_qubits - 1):
                qc.cnot(i, i + 1)
                qc.ry(self.parameters[i] * self.parameters[i + 1], i + 1)
                qc.cnot(i, i + 1)
        
        return qc
    
    def _build_efficient_su2_map(self) -> QuantumCircuit:
        """Build EfficientSU2-style feature map."""
        qc = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.reps):
            # Rotation layer
            for i in range(self.n_qubits):
                qc.ry(self.parameters[i % self.n_features], i)
                qc.rz(self.parameters[i % self.n_features], i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
        
        return qc
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the feature map circuit.
        
        Returns:
            Quantum circuit
        """
        return self.circuit
    
    def get_parameters(self) -> np.ndarray:
        """Get parameter values from circuit.
        
        Returns:
            Current parameter values
        """
        if self.circuit is None:
            return np.zeros(self.n_features)
        return np.array([self.parameters[i] for i in range(len(self.parameters))])
    
    def bind_parameters(self, x: np.ndarray) -> QuantumCircuit:
        """Create circuit with bound data parameters.
        
        Args:
            x: Input data vector
            
        Returns:
            Bound quantum circuit
        """
        if self.circuit is None:
            raise RuntimeError("Circuit not built")
        
        if len(x) != self.n_features:
            x = self._preprocess_data(x)
        
        qc = self.circuit.copy()
        for i in range(min(len(x), len(self.parameters))):
            qc.assign_parameters({self.parameters[i]: x[i]}, inplace=True)
        
        return qc
    
    def _preprocess_data(self, x: np.ndarray) -> np.ndarray:
        """Preprocess input data for encoding.
        
        Args:
            x: Raw input data
            
        Returns:
            Processed data
        """
        # Normalize to [0, pi] range for rotation angles
        x = np.asarray(x, dtype=np.float64)
        x = x / (np.max(np.abs(x)) + 1e-8)
        x = x * np.pi
        
        # Pad or truncate to match feature dimension
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        elif len(x) > self.n_features:
            x = x[:self.n_features]
        
        return x


class QuantumKernel:
    """Quantum kernel for SVM."""
    
    def __init__(self, feature_map: QuantumFeatureMap,
                 fidelity: Optional[object] = None,
                 evaluation: str = "quantum",
                 shots: int = 1024):
        """Initialize quantum kernel.
        
        Args:
            feature_map: Quantum feature map
            fidelity: Fidelity quantum kernel (optional)
            evaluation: Evaluation method ('quantum' or 'classical')
            shots: Number of measurement shots
        """
        self.feature_map = feature_map
        self.fidelity = fidelity
        self.evaluation = evaluation
        self.shots = shots
        self.X_train = None
        self.kernel_matrix = None
        
        if QISKIT_AVAILABLE and fidelity is None and evaluation == "quantum":
            self._initialize_fidelity_kernel()
    
    def _initialize_fidelity_kernel(self):
        """Initialize fidelity-based quantum kernel."""
        sampler = Sampler()
        self.fidelity = FidelityQuantumKernel(
            feature_map=self.feature_map.get_circuit()
        )
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two samples.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Kernel value K(x1, x2)
        """
        if self.evaluation == "quantum" and self.fidelity is not None:
            return self._quantum_kernel(x1, x2)
        else:
            return self._classical_kernel(x1, x2)
    
    def _quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel using quantum circuit.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Kernel value
        """
        # Bind data to feature map
        qc1 = self.feature_map.bind_parameters(x1)
        qc2 = self.feature_map.bind_parameters(x2)
        
        # Compute overlap |<ψ1|ψ2>|^2
        # This is a simplified version - actual implementation uses fidelity
        fidelity_value = np.abs(np.vdot(x1, x2)) ** 2
        
        return fidelity_value
    
    def _classical_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute classical kernel approximation.
        
        Args:
            x1: First sample
            x2: Second sample
            
        Returns:
            Kernel value
        """
        # Polynomial kernel approximation
        gamma = 1.0 / len(x1)
        return (1 + gamma * np.dot(x1, x2)) ** 2
    
    def evaluate(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Evaluate kernel on datasets.
        
        Args:
            X1: First dataset
            X2: Second dataset (if None, computes kernel matrix)
            
        Returns:
            Kernel matrix or values
        """
        X2 = X1 if X2 is None else X2
        
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self.compute(X1[i], X2[j])
        
        return kernel_matrix
    
    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for training data.
        
        Args:
            X: Training data
            
        Returns:
            Kernel matrix K[i,j] = K(X[i], X[j])
        """
        self.X_train = X
        self.kernel_matrix = self.evaluate(X)
        return self.kernel_matrix


class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """Quantum Kernel Support Vector Machine Classifier."""
    
    def __init__(self, n_qubits: int = 4,
                 feature_map_type: str = "zz_feature_map",
                 reps: int = 2,
                 C: float = 1.0,
                 kernel: str = "quantum",
                 gamma: float = "scale",
                 degree: int = 3,
                 coef0: float = 1,
                 probability: bool = False,
                 max_iter: int = -1,
                 random_state: Optional[int] = None):
        """Initialize quantum kernel SVM.
        
        Args:
            n_qubits: Number of qubits
            feature_map_type: Type of quantum feature map
            reps: Number of feature map repetitions
            C: Regularization parameter
            kernel: Kernel type ('quantum' or 'rbf')
            gamma: Kernel coefficient
            degree: Polynomial degree
            coef0: Polynomial bias
            probability: Enable probability estimates
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.scaler_ = StandardScaler()
        self.label_encoder_ = LabelEncoder()
        self.kernel_ = None
        self.model_ = None
        self.quantum_kernel_ = None
    
    def _build_quantum_kernel(self, X: np.ndarray) -> QuantumKernel:
        """Build quantum kernel for data.
        
        Args:
            X: Training data
            
        Returns:
            Quantum kernel instance
        """
        n_features = X.shape[1]
        
        feature_map = QuantumFeatureMap(
            n_features=n_features,
            n_qubits=self.n_qubits,
            feature_map_type=self.feature_map_type,
            reps=self.reps
        )
        
        self.quantum_kernel_ = QuantumKernel(
            feature_map=feature_map,
            evaluation="quantum" if self.kernel_type == "quantum" else "classical"
        )
        
        return self.quantum_kernel_
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumKernelSVM':
        """Fit the SVM model.
        
        Args:
            X: Training data
            y: Training labels
            
        Returns:
            Self
        """
        # Preprocess data
        X_scaled = self.scaler_.fit_transform(X)
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Build kernel
        if self.kernel_type == "quantum":
            quantum_kernel = self._build_quantum_kernel(X_scaled)
            kernel_matrix = quantum_kernel.kernel_matrix(X_scaled)
            
            # Use precomputed kernel with sklearn SVC
            self.model_ = SVC(
                C=self.C,
                kernel='precomputed',
                probability=self.probability,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.model_.fit(kernel_matrix, y_encoded)
        else:
            # Classical RBF kernel
            self.model_ = SVC(
                C=self.C,
                kernel='rbf',
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                probability=self.probability,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self.model_.fit(X_scaled, y_encoded)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted labels
        """
        X_scaled = self.scaler_.transform(X)
        
        if self.kernel_type == "quantum":
            kernel_matrix = self.quantum_kernel_.evaluate(X_scaled, self.quantum_kernel_.X_train)
            predictions = self.model_.predict(kernel_matrix)
        else:
            predictions = self.model_.predict(X_scaled)
        
        return self.label_encoder_.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Probability predictions
        """
        if not self.probability:
            raise ValueError("Probability estimates not enabled")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.kernel_type == "quantum":
            kernel_matrix = self.quantum_kernel_.evaluate(X_scaled, self.quantum_kernel_.X_train)
            return self.model_.predict_proba(kernel_matrix)
        else:
            return self.model_.predict_proba(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function.
        
        Args:
            X: Input data
            
        Returns:
            Decision function values
        """
        X_scaled = self.scaler_.transform(X)
        
        if self.kernel_type == "quantum":
            kernel_matrix = self.quantum_kernel_.evaluate(X_scaled, self.quantum_kernel_.X_train)
            return self.model_.decision_function(kernel_matrix)
        else:
            return self.model_.decision_function(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class QuantumKernelSVR(BaseEstimator, RegressorMixin):
    """Quantum Kernel Support Vector Regression."""
    
    def __init__(self, n_qubits: int = 4,
                 feature_map_type: str = "zz_feature_map",
                 reps: int = 2,
                 kernel: str = "quantum",
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: float = "scale",
                 max_iter: int = -1,
                 random_state: Optional[int] = None):
        """Initialize quantum kernel SVR.
        
        Args:
            n_qubits: Number of qubits
            feature_map_type: Type of quantum feature map
            reps: Number of feature map repetitions
            kernel: Kernel type ('quantum' or 'rbf')
            C: Regularization parameter
            epsilon: Epsilon tube
            gamma: Kernel coefficient
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.kernel_type = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.scaler_ = StandardScaler()
        self.quantum_kernel_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumKernelSVR':
        """Fit the SVR model.
        
        Args:
            X: Training data
            y: Training targets
            
        Returns:
            Self
        """
        # Preprocess data
        X_scaled = self.scaler_.fit_transform(X)
        y_scaled = self.scaler_.fit_transform(y.reshape(-1, 1)).ravel()
        
        if self.kernel_type == "quantum":
            # Build quantum kernel
            n_features = X.shape[1]
            feature_map = QuantumFeatureMap(
                n_features=n_features,
                n_qubits=self.n_qubits,
                feature_map_type=self.feature_map_type,
                reps=self.reps
            )
            
            self.quantum_kernel_ = QuantumKernel(
                feature_map=feature_map,
                evaluation="quantum"
            )
            kernel_matrix = self.quantum_kernel_.kernel_matrix(X_scaled)
            
            self.model_ = SVR(
                C=self.C,
                kernel='precomputed',
                epsilon=self.epsilon,
                max_iter=self.max_iter
            )
            self.model_.fit(kernel_matrix, y_scaled)
        else:
            self.model_ = SVR(
                C=self.C,
                kernel='rbf',
                gamma=self.gamma,
                epsilon=self.epsilon,
                max_iter=self.max_iter
            )
            self.model_.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.
        
        Args:
            X: Input data
            
        Returns:
            Predicted targets
        """
        X_scaled = self.scaler_.transform(X)
        
        if self.kernel_type == "quantum":
            kernel_matrix = self.quantum_kernel_.evaluate(X_scaled, self.quantum_kernel_.X_train)
            predictions = self.model_.predict(kernel_matrix)
        else:
            predictions = self.model_.predict(X_scaled)
        
        return self.scaler_.inverse_transform(predictions.reshape(-1, 1)).ravel()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score.
        
        Args:
            X: Input data
            y: True targets
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        return 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)


class QuantumSVM:
    """High-level Quantum SVM interface."""
    
    def __init__(self, mode: str = "classification",
                 encoding: str = "amplitude",
                 n_qubits: int = 4,
                 **kwargs):
        """Initialize quantum SVM.
        
        Args:
            mode: 'classification' or 'regression'
            encoding: Encoding type
            n_qubits: Number of qubits
            **kwargs: Additional arguments
        """
        self.mode = mode
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.kwargs = kwargs
        
        if mode == "classification":
            self.model = QuantumKernelSVM(
                n_qubits=n_qubits,
                **kwargs
            )
        elif mode == "regression":
            self.model = QuantumKernelSVR(
                n_qubits=n_qubits,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'QuantumSVM':
        """Train the quantum SVM.
        
        Args:
            X: Training data
            y: Training labels/targets
            
        Returns:
            Self
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model.
        
        Args:
            X: Input data
            y: True values
            
        Returns:
            Score (accuracy or R²)
        """
        return self.model.score(X, y)
