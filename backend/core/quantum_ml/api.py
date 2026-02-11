"""
Quantum Machine Learning API Module

REST API interfaces for quantum ML algorithms.
Provides endpoints for training, prediction, and model management.
"""

import json
import uuid
from typing import Union, Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@dataclass
class APIResponse:
    """Standard API response format."""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class QuantumMLModel:
    """Base class for quantum ML models."""
    
    def __init__(self, model_id: str = None):
        """Initialize model.
        
        Args:
            model_id: Unique model identifier
        """
        self.model_id = model_id or str(uuid.uuid4())
        self.model = None
        self.config = None
        self.is_trained = False
        self.training_history = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train the model.
        
        Args:
            X: Training data
            y: Training targets
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model from file."""
        pass


class ModelRegistry:
    """Registry for managing quantum ML models."""
    
    def __init__(self):
        """Initialize model registry."""
        self._models: Dict[str, QuantumMLModel] = {}
        self._lock = threading.Lock()
    
    def register(self, model: QuantumMLModel) -> str:
        """Register a model.
        
        Args:
            model: Model to register
            
        Returns:
            Model ID
        """
        with self._lock:
            self._models[model.model_id] = model
            return model.model_id
    
    def get(self, model_id: str) -> Optional[QuantumMLModel]:
        """Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model or None if not found
        """
        with self._lock:
            return self._models.get(model_id)
    
    def remove(self, model_id: str) -> bool:
        """Remove model from registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if removed, False otherwise
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False
    
    def list_models(self) -> List[Dict]:
        """List all registered models.
        
        Returns:
            List of model info dictionaries
        """
        with self._lock:
            return [
                {
                    "model_id": mid,
                    "is_trained": model.is_trained,
                    "model_type": type(model).__name__
                }
                for mid, model in self._models.items()
            ]
    
    def clear(self):
        """Clear all models."""
        with self._lock:
            self._models.clear()


class QuantumMLAPI:
    """Quantum Machine Learning API wrapper."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 use_fastapi: bool = True):
        """Initialize API.
        
        Args:
            host: Host address
            port: Port number
            use_fastapi: Use FastAPI instead of Flask
        """
        self.host = host
        self.port = port
        self.registry = ModelRegistry()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if use_fastapi and FASTAPI_AVAILABLE:
            self._setup_fastapi()
        elif FLASK_AVAILABLE:
            self._setup_flask()
        else:
            raise RuntimeError("Neither FastAPI nor Flask is available")
    
    def _setup_fastapi(self):
        """Setup FastAPI application."""
        if not FASTAPI_AVAILABLE:
            return
        
        self.app = FastAPI(title="Quantum ML API")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_flask(self):
        """Setup Flask application."""
        if not FLASK_AVAILABLE:
            return
        
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        def root():
            return {"message": "Quantum ML API", "version": "1.0.0"}
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy"}
        
        @self.app.get("/models")
        def list_models():
            return {"models": self.registry.list_models()}
        
        @self.app.get("/models/{model_id}")
        def get_model(model_id: str):
            model = self.registry.get(model_id)
            if model is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Model not found"}
                )
            return {
                "model_id": model.model_id,
                "is_trained": model.is_trained,
                "model_type": type(model).__name__
            }
        
        @self.app.delete("/models/{model_id}")
        def delete_model(model_id: str):
            if self.registry.remove(model_id):
                return {"message": "Model deleted"}
            return JSONResponse(
                status_code=404,
                content={"error": "Model not found"}
            )
    
    def run(self, **kwargs):
        """Run the API server.
        
        Args:
            **kwargs: Additional arguments for server
        """
        if hasattr(self, 'app'):
            if hasattr(self.app, 'run'):
                self.app.run(host=self.host, port=self.port, **kwargs)
            else:
                import uvicorn
                uvicorn.run(self.app, host=self.host, port=self.port, **kwargs)
    
    def create_qnn(self, model_id: str = None, **kwargs) -> str:
        """Create a new quantum neural network model.
        
        Args:
            model_id: Optional model ID
            **kwargs: Model configuration
            
        Returns:
            Model ID
        """
        from .qnn import QuantumNeuralNetwork
        
        model = QuantumMLModelWrapper(model_id)
        model.model = QuantumNeuralNetwork(**kwargs)
        self.registry.register(model)
        return model.model_id
    
    def create_qsvm(self, model_id: str = None, 
                   mode: str = "classification", **kwargs) -> str:
        """Create a new quantum SVM model.
        
        Args:
            model_id: Optional model ID
            mode: Classification or regression
            **kwargs: Model configuration
            
        Returns:
            Model ID
        """
        from .qsvm import QuantumSVM
        
        model = QuantumMLModelWrapper(model_id)
        model.model = QuantumSVM(mode=mode, **kwargs)
        self.registry.register(model)
        return model.model_id
    
    def create_qclustering(self, model_id: str = None,
                          algorithm: str = "kmeans", **kwargs) -> str:
        """Create a new quantum clustering model.
        
        Args:
            model_id: Optional model ID
            algorithm: Clustering algorithm
            **kwargs: Model configuration
            
        Returns:
            Model ID
        """
        from .qclustering import QuantumClustering
        
        model = QuantumMLModelWrapper(model_id)
        model.model = QuantumClustering(algorithm=algorithm, **kwargs)
        self.registry.register(model)
        return model.model_id


class QuantumMLModelWrapper(QuantumMLModel):
    """Wrapper for quantum ML models with API support."""
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train the wrapped model.
        
        Args:
            X: Training data
            y: Training targets
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        if self.model is None:
            raise ValueError("No model wrapped")
        
        history = self.model.fit(X, y, **kwargs)
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with wrapped model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("No model wrapped")
        
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save wrapped model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is not None and hasattr(self.model, 'save'):
            self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load wrapped model.
        
        Args:
            filepath: Path to load model from
        """
        # Load logic depends on model type
        pass


class QuantumMLPipeline:
    """Pipeline for chaining quantum ML operations."""
    
    def __init__(self, steps: List[Dict] = None):
        """Initialize pipeline.
        
        Args:
            steps: List of pipeline steps
        """
        self.steps = steps or []
        self.models = []
    
    def add_step(self, name: str, model_type: str, **kwargs):
        """Add a step to the pipeline.
        
        Args:
            name: Step name
            model_type: Type of model
            **kwargs: Model configuration
        """
        self.steps.append({
            "name": name,
            "model_type": model_type,
            "config": kwargs
        })
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'QuantumMLPipeline':
        """Fit the entire pipeline.
        
        Args:
            X: Training data
            y: Training targets
            
        Returns:
            Self
        """
        current_data = X
        
        for step in self.steps:
            if step["model_type"] == "qnn":
                from .qnn import QuantumNeuralNetwork
                model = QuantumNeuralNetwork(**step.get("config", {}))
            elif step["model_type"] == "qsvm":
                from .qsvm import QuantumSVM
                model = QuantumSVM(**step.get("config", {}))
            elif step["model_type"] == "qclustering":
                from .qclustering import QuantumClustering
                model = QuantumClustering(**step.get("config", {}))
            else:
                raise ValueError(f"Unknown model type: {step['model_type']}")
            
            model.fit(current_data, y if y is not None else current_data)
            self.models.append(model)
            
            # Transform data for next step
            if hasattr(model, 'predict'):
                current_data = model.predict(current_data)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run data through pipeline.
        
        Args:
            X: Input data
            
        Returns:
            Final predictions
        """
        current_data = X
        
        for model in self.models:
            if hasattr(model, 'predict'):
                current_data = model.predict(current_data)
        
        return current_data
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through pipeline.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        return self.predict(X)


# Syntactic sugar for creating models
def create_qnn(n_qubits: int = 4, n_layers: int = 2,
              learning_rate: float = 0.01, **kwargs):
    """Create a quantum neural network.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        learning_rate: Learning rate
        **kwargs: Additional arguments
        
    Returns:
        QuantumNeuralNetwork instance
    """
    from .qnn import QuantumNeuralNetwork
    return QuantumNeuralNetwork(
        n_qubits=n_qubits,
        n_layers=n_layers,
        learning_rate=learning_rate,
        **kwargs
    )


def create_qsvm(mode: str = "classification", n_qubits: int = 4,
                encoding: str = "amplitude", **kwargs):
    """Create a quantum SVM.
    
    Args:
        mode: Classification or regression
        n_qubits: Number of qubits
        encoding: Encoding type
        **kwargs: Additional arguments
        
    Returns:
        QuantumSVM instance
    """
    from .qsvm import QuantumSVM
    return QuantumSVM(
        mode=mode,
        encoding=encoding,
        n_qubits=n_qubits,
        **kwargs
    )


def create_qclustering(algorithm: str = "kmeans", n_clusters: int = 3,
                       n_qubits: int = 4, **kwargs):
    """Create a quantum clustering model.
    
    Args:
        algorithm: Clustering algorithm
        n_clusters: Number of clusters
        n_qubits: Number of qubits
        **kwargs: Additional arguments
        
    Returns:
        QuantumClustering instance
    """
    from .qclustering import QuantumClustering
    return QuantumClustering(
        algorithm=algorithm,
        n_clusters=n_clusters,
        n_qubits=n_qubits,
        **kwargs
    )


# Import numpy for type hints
import numpy as np
