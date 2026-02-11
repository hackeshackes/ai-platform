"""
Quantum Data Encoding Module

Provides various methods for encoding classical data into quantum states.
Supports amplitude encoding, angle encoding, dictionary encoding, and mixed encoding.
"""

import numpy as np
from typing import Union, Optional, Tuple
from abc import ABC, abstractmethod


class QuantumEncoder(ABC):
    """Abstract base class for quantum data encoders."""
    
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum state.
        
        Args:
            data: Input classical data (1D or 2D array)
            
        Returns:
            Quantum state vector (complex array)
        """
        pass
    
    @abstractmethod
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode quantum state back to classical data.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Classical data
        """
        pass
    
    @abstractmethod
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Calculate required number of qubits for encoding.
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Number of qubits needed
        """
        pass


class AmplitudeEncoder(QuantumEncoder):
    """Amplitude encoding for quantum states.
    
    Encodes data into the amplitudes of a quantum state.
    Can represent up to 2^n dimensional vectors with n qubits.
    """
    
    def __init__(self, normalize: bool = True, pad_to_power_of_two: bool = True):
        """Initialize amplitude encoder.
        
        Args:
            normalize: Whether to normalize input data
            pad_to_power_of_two: Pad data to nearest power of 2
        """
        self.normalize = normalize
        self.pad_to_power_of_two = pad_to_power_of_two
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using amplitude encoding.
        
        Args:
            data: Input data (1D array of length N)
            
        Returns:
            Quantum state vector of length 2^n
            
        Raises:
            ValueError: If data cannot be encoded
        """
        data = np.asarray(data, dtype=np.float64)
        
        if data.ndim == 2:
            # Handle batch encoding - return states for each sample
            return np.array([self.encode(sample) for sample in data])
        
        if data.ndim != 1:
            raise ValueError("Input data must be 1D or 2D array")
        
        original_length = len(data)
        
        # Pad to power of 2 if needed
        if self.pad_to_power_of_two:
            target_length = 2 ** int(np.ceil(np.log2(original_length)))
            if target_length > original_length:
                data = np.pad(data, (0, target_length - original_length))
        
        # Normalize if required
        if self.normalize:
            norm = np.linalg.norm(data)
            if norm > 0:
                data = data / norm
        
        return data.astype(np.complex128)
    
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode quantum state amplitudes back to classical data.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Classical data (amplitudes)
        """
        # Return squared amplitudes (probabilities)
        return np.abs(state) ** 2
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Calculate qubits needed for amplitude encoding.
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Number of qubits needed
        """
        if len(data_shape) == 1:
            dim = data_shape[0]
        else:
            dim = np.prod(data_shape)
        
        target_length = dim
        if self.pad_to_power_of_two:
            target_length = 2 ** int(np.ceil(np.log2(dim)))
        
        return int(np.log2(target_length))


class AngleEncoder(QuantumEncoder):
    """Angle (basis state) encoding for quantum states.
    
    Encodes data into rotation angles of quantum gates.
    Each data dimension requires one qubit.
    """
    
    def __init__(self, scale: float = np.pi, offset: float = 0.0):
        """Initialize angle encoder.
        
        Args:
            scale: Scaling factor for angles
            offset: Offset for angles
        """
        self.scale = scale
        self.offset = offset
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using angle encoding.
        
        Args:
            data: Input data (1D array)
            
        Returns:
            Encoded angles
        """
        data = np.asarray(data, dtype=np.float64)
        
        if data.ndim == 2:
            return np.array([self.encode(sample) for sample in data])
        
        return (data * self.scale + self.offset).astype(np.float64)
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode angles back to classical data.
        
        Args:
            encoded: Encoded angles
            
        Returns:
            Original data
        """
        return (encoded - self.offset) / self.scale
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Calculate qubits needed for angle encoding.
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Number of qubits needed (one per feature)
        """
        if len(data_shape) == 1:
            return data_shape[0]
        return data_shape[1] if len(data_shape) >= 2 else 1


class DictionaryEncoder(QuantumEncoder):
    """Dictionary/sparse encoding for quantum states.
    
    Encodes data using sparse representation for high-dimensional spaces.
    Useful for very high-dimensional sparse data.
    """
    
    def __init__(self, n_qubits: int, normalize: bool = True):
        """Initialize dictionary encoder.
        
        Args:
            n_qubits: Fixed number of qubits
            normalize: Whether to normalize input
        """
        self.n_qubits = n_qubits
        self.normalize = normalize
        self.state_dim = 2 ** n_qubits
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using sparse/dictionary encoding.
        
        Args:
            data: Input data (can be sparse representation)
            
        Returns:
            Quantum state vector
        """
        data = np.asarray(data, dtype=np.float64)
        
        if data.ndim == 2:
            return np.array([self.encode(sample) for sample in data])
        
        # Initialize zero state
        state = np.zeros(self.state_dim, dtype=np.complex128)
        
        # Handle different input formats
        if len(data) == self.state_dim:
            # Full vector
            if self.normalize:
                norm = np.linalg.norm(data)
                if norm > 0:
                    data = data / norm
            state = data.astype(np.complex128)
        elif len(data) <= self.state_dim:
            # Sparse format: data contains (index, value) pairs
            # or just values to place in first len(data) positions
            if self.normalize:
                norm = np.linalg.norm(data)
                if norm > 0:
                    data = data / norm
            state[:len(data)] = data
        else:
            # Truncate if too large
            if self.normalize:
                norm = np.linalg.norm(data[:self.state_dim])
                if norm > 0:
                    data = data[:self.state_dim] / norm
            state = data[:self.state_dim]
        
        return state
    
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode quantum state back to sparse representation.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Sparse data representation
        """
        return np.abs(state) ** 2
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Return configured number of qubits.
        
        Args:
            data_shape: Ignored (uses configured n_qubits)
            
        Returns:
            Number of qubits
        """
        return self.n_qubits


class MixedEncoder(QuantumEncoder):
    """Mixed encoding combining multiple encoding schemes.
    
    Uses different encoding strategies for different data features.
    """
    
    def __init__(self, encoding_config: dict = None):
        """Initialize mixed encoder.
        
        Args:
            encoding_config: Dict mapping feature indices to encoders
        """
        self.encoding_config = encoding_config or {}
        self._encoders = {}
    
    def add_encoder(self, feature_range: slice, encoder: QuantumEncoder):
        """Add encoder for a range of features.
        
        Args:
            feature_range: Slice or range of feature indices
            encoder: Encoder to use for these features
        """
        self._encoders[feature_range] = encoder
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode using mixed encoding scheme.
        
        Args:
            data: Input data
            
        Returns:
            Encoded quantum state
        """
        if data.ndim == 2:
            return np.array([self.encode(sample) for sample in data])
        
        encoded_parts = []
        
        # Sort ranges and encode each part
        ranges = sorted(self._encoders.keys(), 
                       key=lambda r: r.start if hasattr(r, 'start') else 0)
        
        current_idx = 0
        for r in ranges:
            # Encode features in this range
            if hasattr(r, 'start'):
                start, stop = r.start, r.stop
            else:
                # Handle integer index
                start, stop = r, r + 1
            
            # Pad with zeros if gap exists
            if start > current_idx:
                n_pad = start - current_idx
                encoded_parts.append(np.zeros(n_pad))
            
            # Encode the features
            features = data[start:stop]
            encoded = self._encoders[r].encode(features)
            encoded_parts.append(encoded)
            current_idx = stop
        
        # Pad remaining features
        if current_idx < len(data):
            encoded_parts.append(np.zeros(len(data) - current_idx))
        
        # Concatenate and normalize
        result = np.concatenate(encoded_parts)
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.astype(np.complex128)
    
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode mixed encoded state.
        
        Args:
            state: Quantum state
            
        Returns:
            Original data
        """
        # Simplified decoding - returns probabilities
        return np.abs(state) ** 2
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Calculate total qubits needed for mixed encoding.
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Total number of qubits needed
        """
        total_dims = []
        
        for r in self._encoders.keys():
            if hasattr(r, 'start'):
                n_features = r.stop - r.start
            else:
                n_features = 1
            n_qubits = self._encoders[r].get_n_qubits((n_features,))
            total_dims.append(2 ** n_qubits)
        
        if total_dims:
            return int(np.ceil(np.log2(sum(total_dims))))
        
        return int(np.ceil(np.log2(data_shape[0])))


class BasisEncoder(QuantumEncoder):
    """Basis state encoding.
    
    Encodes data as computational basis states.
    Each data value is converted to a basis state index.
    """
    
    def __init__(self, n_qubits: int):
        """Initialize basis encoder.
        
        Args:
            n_qubits: Number of qubits for encoding
        """
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data using basis state encoding.
        
        Args:
            data: Input data (integer indices)
            
        Returns:
            Quantum basis state
        """
        data = np.asarray(data, dtype=np.int64)
        
        if data.ndim == 2:
            return np.array([self.encode(sample) for sample in data])
        
        state = np.zeros(self.state_dim, dtype=np.complex128)
        
        for val in data:
            if 0 <= val < self.state_dim:
                state[val] = 1.0 / np.sqrt(len(data))
        
        return state
    
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode basis state to indices.
        
        Args:
            state: Quantum basis state
            
        Returns:
            Data indices
        """
        return np.argmax(np.abs(state))
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Return configured qubits.
        
        Args:
            data_shape: Ignored
            
        Returns:
            Number of qubits
        """
        return self.n_qubits


class DataEncoder:
    """Main data encoding class with factory methods."""
    
    ENCODING_TYPES = {
        'amplitude': AmplitudeEncoder,
        'angle': AngleEncoder,
        'dictionary': DictionaryEncoder,
        'mixed': MixedEncoder,
        'basis': BasisEncoder
    }
    
    def __init__(self, encoding: str = "amplitude", **kwargs):
        """Initialize data encoder.
        
        Args:
            encoding: Type of encoding to use
            **kwargs: Additional arguments for encoder
        """
        if encoding not in self.ENCODING_TYPES:
            raise ValueError(f"Unknown encoding type: {encoding}. "
                            f"Available: {list(self.ENCODING_TYPES.keys())}")
        
        self.encoder = self.ENCODING_TYPES[encoding](**kwargs)
        self.encoding_type = encoding
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data into quantum state.
        
        Args:
            data: Input classical data
            
        Returns:
            Quantum state vector
        """
        return self.encoder.encode(data)
    
    def decode(self, state: np.ndarray) -> np.ndarray:
        """Decode quantum state to classical data.
        
        Args:
            state: Quantum state vector
            
        Returns:
            Classical data
        """
        return self.encoder.decode(state)
    
    def get_n_qubits(self, data_shape: Tuple[int, ...]) -> int:
        """Calculate required qubits for encoding.
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Number of qubits needed
        """
        return self.encoder.get_n_qubits(data_shape)
    
    @classmethod
    def create_encoder(cls, encoding: str, **kwargs) -> QuantumEncoder:
        """Factory method to create encoder instance.
        
        Args:
            encoding: Type of encoding
            **kwargs: Encoder arguments
            
        Returns:
            Encoder instance
        """
        if encoding not in cls.ENCODING_TYPES:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        return cls.ENCODING_TYPES[encoding](**kwargs)
    
    @staticmethod
    def auto_select_encoding(data: np.ndarray, 
                            max_qubits: int = 10) -> str:
        """Automatically select best encoding based on data.
        
        Args:
            data: Input data
            max_qubits: Maximum qubits available
            
        Returns:
            Recommended encoding type
        """
        data = np.asarray(data)
        
        if data.ndim == 1:
            n_features = len(data)
        else:
            n_features = data.shape[1]
        
        # For high-dimensional data, use amplitude encoding
        if n_features > max_qubits:
            return 'amplitude'
        
        # For moderate dimensions, angle encoding is efficient
        if n_features <= max_qubits:
            return 'angle'
        
        return 'amplitude'
