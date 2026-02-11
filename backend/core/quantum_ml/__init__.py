"""
Quantum Machine Learning Library

A comprehensive quantum machine learning library providing
implementations of quantum neural networks, quantum SVM,
quantum clustering, and data encoding schemes.

Modules:
- qnn: Quantum Neural Networks
- qsvm: Quantum Support Vector Machines
- qclustering: Quantum Clustering Algorithms
- data_encoding: Quantum Data Encoding
- api: REST API Interfaces
- config: Configuration Management
- examples: Usage Examples
- tests: Test Suite

Author: AI Platform Team
Version: 1.0.0
"""

__version__ = "1.0.0"

# Core modules
from .qnn import (
    QuantumNeuralNetwork,
    QuantumClassifier,
    QuantumRegressor,
    VariationalQuantumCircuit,
    QuantumGradients,
    QuantumLossFunction
)

from .qsvm import (
    QuantumSVM,
    QuantumKernelSVM,
    QuantumKernelSVR,
    QuantumFeatureMap,
    QuantumKernel
)

from .qclustering import (
    QuantumClustering,
    QuantumKMeans,
    QuantumDBSCAN,
    QuantumSpectralClustering,
    QuantumCentroidClustering,
    QuantumDistance
)

from .data_encoding import (
    DataEncoder,
    QuantumEncoder,
    AmplitudeEncoder,
    AngleEncoder,
    DictionaryEncoder,
    MixedEncoder,
    BasisEncoder
)

from .api import (
    QuantumMLAPI,
    QuantumMLPipeline,
    ModelRegistry,
    create_qnn,
    create_qsvm,
    create_qclustering
)

from .config import (
    QuantumConfig,
    get_default_config,
    SUPPORTED_ENCODINGS,
    SUPPORTED_ALGORITHMS
)

__all__ = [
    # Version
    "__version__",
    
    # QNN
    "QuantumNeuralNetwork",
    "QuantumClassifier", 
    "QuantumRegressor",
    "VariationalQuantumCircuit",
    "QuantumGradients",
    "QuantumLossFunction",
    
    # QSVM
    "QuantumSVM",
    "QuantumKernelSVM",
    "QuantumKernelSVR",
    "QuantumFeatureMap",
    "QuantumKernel",
    
    # Clustering
    "QuantumClustering",
    "QuantumKMeans",
    "QuantumDBSCAN",
    "QuantumSpectralClustering",
    "QuantumCentroidClustering",
    "QuantumDistance",
    
    # Encoding
    "DataEncoder",
    "QuantumEncoder",
    "AmplitudeEncoder",
    "AngleEncoder",
    "DictionaryEncoder",
    "MixedEncoder",
    "BasisEncoder",
    
    # API
    "QuantumMLAPI",
    "QuantumMLPipeline",
    "ModelRegistry",
    "create_qnn",
    "create_qsvm",
    "create_qclustering",
    
    # Config
    "QuantumConfig",
    "get_default_config",
    "SUPPORTED_ENCODINGS",
    "SUPPORTED_ALGORITHMS"
]


def test_installation():
    """Quick test to verify installation."""
    print("Testing Quantum ML Installation...")
    print(f"Version: {__version__}")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        from qiskit import __version__ as qiskit_version
        print(f"Qiskit: {qiskit_version}")
    except ImportError:
        print("Qiskit: Not installed (some features may be limited)")
    
    # Test basic functionality
    try:
        from quantum_ml.data_encoding import DataEncoder
        encoder = DataEncoder(encoding="amplitude")
        data = np.array([1.0, 2.0, 3.0, 4.0])
        state = encoder.encode(data)
        print(f"Data encoding: OK (state shape: {state.shape})")
    except Exception as e:
        print(f"Data encoding: FAILED ({e})")
    
    print("Installation test complete!")


# Import numpy for internal use
import numpy as np
