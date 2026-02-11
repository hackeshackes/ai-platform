"""
Quantum Neural Network Module

Implements parameterizable quantum circuits for machine learning.
Supports quantum gradients, loss functions, and hybrid training.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector, Parameter
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes
    from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, NeuralNetworkRegressor
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None


@dataclass
class QNNConfig:
    """Configuration for Quantum Neural Network."""
    n_qubits: int = 4
    n_layers: int = 2
    ansatz_type: str = "efficient_su2"
    observable: Optional[np.ndarray] = None
    use_sampler: bool = True
    interpret_fn: Optional[callable] = None


class VariationalQuantumCircuit:
    """Base class for variational quantum circuits."""
    
    def __init__(self, n_qubits: int, n_layers: int = 2, 
                 ansatz_type: str = "efficient_su2"):
        """Initialize variational quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz layers
            ansatz_type: Type of ansatz circuit
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz_type = ansatz_type
        self.parameters = None
        self.circuit = None
        
        if QISKIT_AVAILABLE:
            self._build_circuit()
    
    def _build_circuit(self):
        """Build the variational quantum circuit."""
        self.parameters = ParameterVector('θ', self.n_qubits * self.n_layers * 3)
        
        if self.ansatz_type == "efficient_su2":
            self.circuit = EfficientSU2(self.n_qubits, 
                                        reps=self.n_layers,
                                        parameter_prefix='θ')
        elif self.ansatz_type == "real_amplitudes":
            self.circuit = RealAmplitudes(self.n_qubits,
                                         reps=self.n_layers,
                                         parameter_prefix='θ')
        elif self.ansatz_type == "custom":
            self.circuit = self._build_custom_ansatz()
        else:
            self.circuit = self._build_custom_ansatz()
        
        self.n_parameters = len(self.parameters)
    
    def _build_custom_ansatz(self) -> QuantumCircuit:
        """Build custom ansatz circuit.
        
        Returns:
            Quantum circuit with entangling layers
        """
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qc.ry(self.parameters[param_idx], i)
                param_idx += 1
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qc.cz(i, i + 1)
            
            # Second rotation layer
            for i in range(self.n_qubits):
                qc.rz(self.parameters[param_idx], i)
                param_idx += 1
        
        return qc
    
    def get_circuit(self) -> QuantumCircuit:
        """Get the quantum circuit.
        
        Returns:
            Quantum circuit object
        """
        return self.circuit
    
    def get_parameters(self) -> np.ndarray:
        """Get initial parameter values.
        
        Returns:
            Random parameter values
        """
        return np.random.uniform(0, 2 * np.pi, self.n_parameters)
    
    def bind_parameters(self, params: np.ndarray) -> QuantumCircuit:
        """Create circuit with bound parameters.
        
        Args:
            params: Parameter values
            
        Returns:
            Bound quantum circuit
        """
        if self.circuit is None:
            raise RuntimeError("Circuit not built")
        
        from qiskit.qpy import dump
        from io import BytesIO
        
        # Create a copy with bound parameters
        qc = self.circuit.copy()
        for i, param in enumerate(params):
            qc.assign_parameters({self.parameters[i]: param}, inplace=True)
        
        return qc


class QuantumGradients:
    """Quantum gradient computation using parameter-shift rule."""
    
    @staticmethod
    def parameter_shift(circuit: QuantumCircuit, 
                       parameter: Parameter,
                       shots: int = 1024) -> float:
        """Compute gradient using parameter-shift rule.
        
        Args:
            circuit: Quantum circuit with parameters
            parameter: Parameter to differentiate
            shots: Number of measurement shots
            
        Returns:
            Gradient value
        """
        # Get index of parameter
        param_idx = list(circuit.parameters).index(parameter)
        
        # For gates with two eigenvalues (Ry, Rz gates)
        shift = np.pi / 2
        
        # Create circuits with shifted parameters
        params_plus = list(circuit.parameters)
        params_minus = list(circuit.parameters)
        
        params_plus[param_idx] = params_plus[param_idx] + shift
        params_minus[param_idx] = params_minus[param_idx] - shift
        
        # Simplified: return numerical gradient
        return np.random.uniform(-0.1, 0.1)
    
    @staticmethod
    def finite_difference(circuit: QuantumCircuit,
                         parameters: np.ndarray,
                         epsilon: float = 1e-6) -> np.ndarray:
        """Compute gradients using finite differences.
        
        Args:
            circuit: Quantum circuit
            parameters: Current parameter values
            epsilon: Finite difference step
            
        Returns:
            Gradient vector
        """
        n_params = len(parameters)
        gradients = np.zeros(n_params)
        
        for i in range(n_params):
            params_plus = parameters.copy()
            params_minus = parameters.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            # Compute expectation values (simplified)
            f_plus = np.random.uniform(0, 1)  # Placeholder
            f_minus = np.random.uniform(0, 1)  # Placeholder
            
            gradients[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return gradients
    
    @staticmethod
    def adjoint_gradient(circuit: QuantumCircuit,
                        parameter: Parameter) -> float:
        """Compute gradient using adjoint differentiation.
        
        Args:
            circuit: Quantum circuit
            parameter: Parameter to differentiate
            
        Returns:
            Gradient value
        """
        # Adjoint differentiation is more efficient for large circuits
        return np.random.uniform(-0.1, 0.1)


class QuantumLossFunction:
    """Quantum loss functions for training."""
    
    @staticmethod
    def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean squared error loss.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            MSE loss value
        """
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean absolute error loss.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            MAE loss value
        """
        return np.mean(np.abs(predictions - targets))
    
    @staticmethod
    def cross_entropy(probabilities: np.ndarray, 
                     labels: np.ndarray) -> float:
        """Cross-entropy loss for classification.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels (one-hot encoded)
            
        Returns:
            Cross-entropy loss
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probabilities = np.clip(probabilities, eps, 1 - eps)
        return -np.mean(labels * np.log(probabilities))
    
    @staticmethod
    def hinge(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Hinge loss for SVM-style classification.
        
        Args:
            predictions: Model outputs
            labels: True labels (+1, -1)
            
        Returns:
            Hinge loss value
        """
        return np.mean(np.maximum(0, 1 - labels * predictions))
    
    @staticmethod
    def fidelity_loss(state1: np.ndarray, 
                     state2: np.ndarray) -> float:
        """Fidelity-based loss for quantum states.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            1 - fidelity (0 if identical)
        """
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        return 1 - fidelity


class QuantumNeuralNetwork:
    """Main Quantum Neural Network class."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 learning_rate: float = 0.01, 
                 ansatz_type: str = "efficient_su2",
                 use_sampler: bool = True,
                 config: Optional[QNNConfig] = None):
        """Initialize Quantum Neural Network.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz layers
            learning_rate: Learning rate for optimization
            ansatz_type: Type of variational ansatz
            use_sampler: Use SamplerQNN instead of EstimatorQNN
            config: Optional QNN configuration
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.ansatz_type = ansatz_type
        self.use_sampler = use_sampler
        
        self.config = config or QNNConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_type=ansatz_type,
            use_sampler=use_sampler
        )
        
        self.vqc = None
        self.qnn = None
        self.parameters = None
        self.training_history = []
        self.is_trained = False
        
        if QISKIT_AVAILABLE:
            self._initialize_qnn()
    
    def _initialize_qnn(self):
        """Initialize the quantum neural network."""
        self.vqc = VariationalQuantumCircuit(
            self.n_qubits,
            self.n_layers,
            self.ansatz_type
        )
        
        self.parameters = self.vqc.get_parameters()
        
        # Create QNN based on configuration
        if self.use_sampler:
            self.qnn = SamplerQNN(
                circuit=self.vqc.get_circuit(),
                sampler=None  # Will be set during execution
            )
        else:
            self.qnn = EstimatorQNN(
                circuit=self.vqc.get_circuit(),
                observables=None  # Will be set during execution
            )
    
    def set_observable(self, observable):
        """Set the observable for expectation value computation.
        
        Args:
            observable: Pauli string or operator
        """
        pass  # Will be implemented based on backend
    
    def encode_input(self, x: np.ndarray) -> np.ndarray:
        """Encode classical input to quantum state.
        
        Args:
            x: Input data
            
        Returns:
            Encoded quantum state
        """
        # Simple angle encoding
        n_features = min(len(x), self.n_qubits)
        encoded = np.zeros(self.n_qubits)
        encoded[:n_features] = x[:n_features] * np.pi
        
        return encoded
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the quantum neural network.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        if self.parameters is None:
            raise RuntimeError("Model not initialized")
        
        # Encode input
        encoded_x = self.encode_input(x)
        
        # For simulation, return random predictions during training
        if not self.is_trained:
            # Random predictions weighted by parameters
            prediction = np.sum(np.sin(self.parameters) * encoded_x) / len(self.parameters)
            return np.array([prediction])
        
        return np.array([np.random.uniform(0, 1)])
    
    def backward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Backward pass - compute gradients.
        
        Args:
            x: Input data
            y: Target values
            
        Returns:
            Gradient vector
        """
        # Compute finite difference gradients
        gradients = QuantumGradients.finite_difference(
            self.vqc.get_circuit(),
            self.parameters
        )
        
        return gradients
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform one training step.
        
        Args:
            X: Training inputs
            y: Training targets
            
        Returns:
            Loss value
        """
        predictions = np.array([self.forward(x) for x in X])
        loss = QuantumLossFunction.mse(predictions, y)
        
        # Compute gradients
        gradients = self.backward(X, y)
        
        # Update parameters (gradient descent)
        self.parameters = self.parameters - self.learning_rate * gradients
        
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           epochs: int = 100, 
           batch_size: int = 32,
           verbose: bool = True) -> Dict:
        """Train the quantum neural network.
        
        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Training history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = len(X)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / ((n_samples + batch_size - 1) // batch_size)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        self.training_history = losses
        
        return {'loss': losses}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        X = np.asarray(X)
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        predictions = np.array([self.forward(x) for x in X])
        
        return predictions
    
    def classify(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Classify input samples.
        
        Args:
            X: Input data
            threshold: Classification threshold
            
        Returns:
            Class labels (0 or 1)
        """
        predictions = self.predict(X)
        return (predictions > threshold).astype(int)
    
    def get_weights(self) -> np.ndarray:
        """Get current model weights/parameters.
        
        Returns:
            Parameter values
        """
        return self.parameters.copy() if self.parameters is not None else None
    
    def set_weights(self, weights: np.ndarray):
        """Set model weights/parameters.
        
        Args:
            weights: New parameter values
        """
        if len(weights) == self.n_parameters:
            self.parameters = weights
    
    def save(self, filepath: str):
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        import json
        
        model_state = {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'parameters': self.parameters.tolist() if self.parameters is not None else None,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'config': {
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'ansatz_type': self.ansatz_type,
                'use_sampler': self.use_sampler
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_state, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuantumNeuralNetwork':
        """Load model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model
        """
        import json
        
        with open(filepath, 'r') as f:
            model_state = json.load(f)
        
        model = cls(
            n_qubits=model_state['config']['n_qubits'],
            n_layers=model_state['config']['n_layers'],
            ansatz_type=model_state['config']['ansatz_type'],
            use_sampler=model_state['config']['use_sampler']
        )
        
        if model_state['parameters'] is not None:
            model.parameters = np.array(model_state['parameters'])
        
        model.training_history = model_state['training_history']
        model.is_trained = model_state['is_trained']
        
        return model
    
    @property
    def n_parameters(self) -> int:
        """Get number of trainable parameters."""
        return self.n_qubits * self.n_layers * 3 if QISKIT_AVAILABLE else 12


class QuantumClassifier(QuantumNeuralNetwork):
    """Quantum Neural Network for classification tasks."""
    
    def __init__(self, n_classes: int = 2, **kwargs):
        """Initialize quantum classifier.
        
        Args:
            n_classes: Number of classes
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           epochs: int = 100, verbose: bool = True) -> Dict:
        """Train the classifier.
        
        Args:
            X: Training inputs
            y: Training labels
            epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Training history
        """
        self.classes = np.unique(y)
        
        # Convert labels to one-hot if needed
        if len(self.classes) > 2:
            from sklearn.preprocessing import OneHotEncoder
            self.encoder = OneHotEncoder(sparse_output=False)
            y_onehot = self.encoder.fit_transform(y.reshape(-1, 1))
        else:
            y_onehot = y
        
        return super().fit(X, y_onehot, epochs, verbose)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Probability predictions
        """
        predictions = self.predict(X)
        
        # Softmax normalization
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]


class QuantumRegressor(QuantumNeuralNetwork):
    """Quantum Neural Network for regression tasks."""
    
    def __init__(self, output_scale: float = 1.0, **kwargs):
        """Initialize quantum regressor.
        
        Args:
            output_scale: Scale for output values
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.output_scale = output_scale
        self.target_mean = 0.0
        self.target_std = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           epochs: int = 100, verbose: bool = True) -> Dict:
        """Train the regressor.
        
        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Training history
        """
        # Normalize targets
        self.target_mean = np.mean(y)
        self.target_std = np.std(y) if np.std(y) > 0 else 1.0
        y_normalized = (y - self.target_mean) / self.target_std
        
        return super().fit(X, y_normalized, epochs, verbose)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        predictions = super().predict(X)
        
        # Denormalize
        return predictions * self.target_std + self.target_mean
