"""
Quantum Machine Learning Configuration Module

Provides configuration settings and utilities for quantum ML algorithms.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantumConfig:
    """Quantum computing configuration settings."""
    
    # Backend settings
    backend: str = "qasm_simulator"
    noise_model: Optional[str] = None
    coupling_map: Optional[list] = None
    basis_gates: Optional[list] = None
    
    # Simulation settings
    shots: int = 1024
    seed_simulator: int = 42
    optimization_level: int = 1
    
    # Circuit settings
    max_depth: int = 100
    qubit_threshold: float = 1e-6
    
    # Training settings
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    verbose: bool = True
    
    # Encoding settings
    default_encoding: str = "amplitude"
    n_qubits_auto: bool = True
    
    # Cluster settings
    n_clusters_auto: bool = True
    max_clusters: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'QuantumConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    def save(self, filepath: str):
        """Save config to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuantumConfig':
        """Load config from file."""
        import json
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


def get_default_config() -> QuantumConfig:
    """Get default quantum ML configuration."""
    return QuantumConfig()


def set_qiskit_aer_backend(backend_name: str = "qasm_simulator"):
    """Set the default Qiskit Aer backend."""
    os.environ.get('QISKIT_AER_BACKEND', backend_name)


# Environment variables for quantum backend
QISKIT_TOKEN_ENV = "QISKIT_TOKEN"
IBMQ_PROJECT_ENV = "IBMQ_PROJECT"

def get_ibm_token() -> Optional[str]:
    """Get IBM Quantum token from environment."""
    return os.environ.get(QISKIT_TOKEN_ENV)


def set_ibm_token(token: str):
    """Set IBM Quantum token."""
    os.environ[QISKIT_TOKEN_ENV] = token


# Supported encodings
SUPPORTED_ENCODINGS = [
    "amplitude",
    "angle", 
    "dictionary",
    "mixed",
    "basis"
]

# Supported algorithms
SUPPORTED_ALGORITHMS = [
    "qnn",
    "qsvm",
    "qclustering",
    "qPCA",
    "qAutoencoder"
]

# Version info
__version__ = "1.0.0"
