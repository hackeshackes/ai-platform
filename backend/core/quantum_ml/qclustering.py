"""
Quantum Clustering Module

Implements quantum versions of clustering algorithms:
- Quantum K-Means
- Quantum DBSCAN
- Quantum Spectral Clustering
- Quantum Centroid-based clustering
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings


@dataclass
class QuantumClusteringConfig:
    """Configuration for quantum clustering algorithms."""
    n_clusters: int = 3
    n_qubits: int = 4
    encoding: str = "amplitude"
    max_iter: int = 100
    n_init: int = 10
    init_method: str = "k-means++"
    tolerance: float = 1e-4
    quantum_enhancement: float = 0.5  # Weight for quantum vs classical
    distance_metric: str = "fidelity"


class QuantumDistance:
    """Quantum distance metrics for clustering."""
    
    @staticmethod
    def fidelity_distance(state1: np.ndarray, 
                         state2: np.ndarray) -> float:
        """Compute fidelity-based distance.
        
        Fidelity: F(ρ, σ) = [tr(√(√ρ σ √ρ))]²
        For pure states: F = |<ψ|φ>|²
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            Distance = 1 - fidelity
        """
        fidelity = np.abs(np.vdot(state1, state2)) ** 2
        return np.sqrt(1 - fidelity)
    
    @staticmethod
    def trace_distance(state1: np.ndarray, 
                      state2: np.ndarray) -> float:
        """Compute trace distance.
        
        D(ρ, σ) = 0.5 * ||ρ - σ||_1
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Trace distance
        """
        diff = state1 - state2
        return 0.5 * np.sum(np.abs(diff))
    
    @staticmethod
    def bray_curtis_distance(state1: np.ndarray,
                             state2: np.ndarray) -> float:
        """Compute Bray-Curtis distance.
        
        Args:
            state1: First vector
            state2: Second vector
            
        Returns:
            Bray-Curtis distance
        """
        numerator = np.sum(np.abs(state1 - state2))
        denominator = np.sum(state1 + state2)
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    @staticmethod
    def hellinger_distance(state1: np.ndarray,
                          state2: np.ndarray) -> float:
        """Compute Hellinger distance.
        
        Args:
            state1: First probability distribution
            state2: Second probability distribution
            
        Returns:
            Hellinger distance
        """
        return np.sqrt(0.5 * np.sum((np.sqrt(state1) - np.sqrt(state2)) ** 2))
    
    @classmethod
    def compute_distance(cls, state1: np.ndarray,
                       state2: np.ndarray,
                       metric: str = "fidelity") -> float:
        """Compute distance using specified metric.
        
        Args:
            state1: First state
            state2: Second state
            metric: Distance metric
            
        Returns:
            Distance value
        """
        metrics = {
            "fidelity": cls.fidelity_distance,
            "trace": cls.trace_distance,
            "bray_curtis": cls.bray_curtis_distance,
            "hellinger": cls.hellinger_distance
        }
        
        if metric not in metrics:
            warnings.warn(f"Unknown metric {metric}, using fidelity")
            metric = "fidelity"
        
        return metrics[metric](state1, state2)


class QuantumKMeans:
    """Quantum-enhanced K-Means clustering."""
    
    def __init__(self, n_clusters: int = 3,
                 n_qubits: int = 4,
                 max_iter: int = 100,
                 n_init: int = 10,
                 random_state: Optional[int] = None,
                 distance_metric: str = "fidelity",
                 quantum_weight: float = 0.5):
        """Initialize quantum K-means.
        
        Args:
            n_clusters: Number of clusters
            n_qubits: Number of qubits for encoding
            max_iter: Maximum iterations
            n_init: Number of initializations
            random_state: Random seed
            distance_metric: Distance metric to use
            quantum_weight: Weight for quantum vs classical distance
        """
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.quantum_weight = quantum_weight
        
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.classical_kmeans_ = None
        
        # Classical fallback
        self.classical_kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
    
    def _encode_data(self, X: np.ndarray) -> np.ndarray:
        """Encode classical data to quantum states.
        
        Args:
            X: Input data
            
        Returns:
            Quantum states
        """
        # Simple amplitude encoding
        n_samples = len(X)
        dim = 2 ** self.n_qubits
        
        # Reshape data to fit quantum dimension
        states = np.zeros((n_samples, dim), dtype=np.complex128)
        
        for i, x in enumerate(X):
            x = np.asarray(x, dtype=np.float64)
            # Normalize
            x = x / (np.linalg.norm(x) + 1e-8)
            # Pad or truncate
            if len(x) > dim:
                states[i, :dim] = x[:dim]
            else:
                states[i, :len(x)] = x
        
        return states
    
    def _compute_distances(self, X_encoded: np.ndarray,
                          centers: np.ndarray) -> np.ndarray:
        """Compute distances between data and centers.
        
        Args:
            X_encoded: Encoded data
            centers: Current cluster centers
            
        Returns:
            Distance matrix
        """
        n_samples, n_features = X_encoded.shape
        n_clusters = len(centers)
        
        distances = np.zeros((n_samples, n_clusters))
        
        for i in range(n_samples):
            for j in range(n_clusters):
                # Classical Euclidean distance
                classical_dist = np.linalg.norm(X_encoded[i] - centers[j])
                
                # Quantum distance
                quantum_dist = QuantumDistance.compute_distance(
                    X_encoded[i], centers[j], self.distance_metric
                )
                
                # Combine
                distances[i, j] = (1 - self.quantum_weight) * classical_dist + \
                                  self.quantum_weight * quantum_dist
        
        return distances
    
    def _update_centers(self, X_encoded: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
        """Update cluster centers.
        
        Args:
            X_encoded: Encoded data
            labels: Cluster assignments
            
        Returns:
            New cluster centers
        """
        n_clusters = len(np.unique(labels))
        centers = np.zeros((n_clusters, X_encoded.shape[1]))
        
        for k in range(n_clusters):
            cluster_points = X_encoded[labels == k]
            if len(cluster_points) > 0:
                centers[k] = np.mean(cluster_points, axis=0)
                # Renormalize
                norm = np.linalg.norm(centers[k])
                if norm > 0:
                    centers[k] = centers[k] / norm
        
        return centers
    
    def fit(self, X: np.ndarray) -> 'QuantumKMeans':
        """Fit K-means clustering.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        # Encode data to quantum states
        X_encoded = self._encode_data(X)
        
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        
        for init in range(self.n_init):
            # Initialize centers randomly from data points
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state + init)
            else:
                rng = np.random.RandomState()
            
            center_indices = rng.choice(n_samples, self.n_clusters, replace=False)
            centers = X_encoded[center_indices].copy()
            
            for iteration in range(self.max_iter):
                # Assign points to nearest center
                distances = self._compute_distances(X_encoded, centers)
                labels = np.argmin(distances, axis=1)
                
                # Update centers
                new_centers = self._update_centers(X_encoded, labels)
                
                # Check convergence
                center_shift = np.linalg.norm(new_centers - centers)
                if center_shift < 1e-6:
                    break
                
                centers = new_centers
            
            # Compute inertia
            inertia = 0.0
            for k in range(self.n_clusters):
                cluster_points = X_encoded[labels == k]
                if len(cluster_points) > 0:
                    inertia += np.sum((X_encoded[labels == k] - centers[k]) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
        
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = self.max_iter
        
        # Also run classical K-means for comparison
        self.classical_kmeans_.fit(X)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        X_encoded = self._encode_data(X)
        distances = self._compute_distances(X_encoded, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def score(self, X: np.ndarray) -> float:
        """Compute clustering score (negative inertia).
        
        Args:
            X: Input data
            
        Returns:
            Score
        """
        return -self.inertia_
    
    def compare_with_classical(self, X: np.ndarray) -> Dict:
        """Compare quantum K-means with classical K-means.
        
        Args:
            X: Input data
            
        Returns:
            Comparison metrics
        """
        quantum_labels = self.fit_predict(X)
        classical_labels = self.classical_kmeans_.fit_predict(X)
        
        # Compute scores
        quantum_score = silhouette_score(X, quantum_labels) if len(np.unique(quantum_labels)) > 1 else 0
        classical_score = silhouette_score(X, classical_labels) if len(np.unique(classical_labels)) > 1 else 0
        
        # Compute CH score
        quantum_ch = calinski_harabasz_score(X, quantum_labels)
        classical_ch = calinski_harabasz_score(X, classical_labels)
        
        return {
            "quantum_silhouette": quantum_score,
            "classical_silhouette": classical_score,
            "quantum_calinski_harabasz": quantum_ch,
            "classical_calinski_harabasz": classical_ch,
            "labels_match": np.mean(quantum_labels == classical_labels)
        }


class QuantumDBSCAN:
    """Quantum-enhanced DBSCAN clustering."""
    
    def __init__(self, n_qubits: int = 4,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 distance_metric: str = "fidelity",
                 quantum_weight: float = 0.5):
        """Initialize quantum DBSCAN.
        
        Args:
            n_qubits: Number of qubits for encoding
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            distance_metric: Distance metric
            quantum_weight: Weight for quantum distance
        """
        self.n_qubits = n_qubits
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.quantum_weight = quantum_weight
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
        self.classical_dbscan_ = None
    
    def _encode_data(self, X: np.ndarray) -> np.ndarray:
        """Encode classical data to quantum states."""
        n_samples = len(X)
        dim = 2 ** self.n_qubits
        
        states = np.zeros((n_samples, dim), dtype=np.complex128)
        
        for i, x in enumerate(X):
            x = np.asarray(x, dtype=np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            if len(x) > dim:
                states[i, :dim] = x[:dim]
            else:
                states[i, :len(x)] = x
        
        return states
    
    def _compute_distance_matrix(self, X_encoded: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n_samples = len(X_encoded)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                classical_dist = np.linalg.norm(X_encoded[i] - X_encoded[j])
                quantum_dist = QuantumDistance.compute_distance(
                    X_encoded[i], X_encoded[j], self.distance_metric
                )
                d = (1 - self.quantum_weight) * classical_dist + \
                    self.quantum_weight * quantum_dist
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def _region_query(self, distances: np.ndarray, 
                     point_idx: int) -> np.ndarray:
        """Find all points within eps of point_idx."""
        return np.where(distances[point_idx] <= self.eps)[0]
    
    def fit(self, X: np.ndarray) -> 'QuantumDBSCAN':
        """Fit DBSCAN clustering.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        # Encode data
        X_encoded = self._encode_data(X)
        
        # Compute distance matrix
        distances = self._compute_distance_matrix(X_encoded)
        
        # Initialize labels
        labels = np.full(n_samples, -1, dtype=int)
        core_samples = []
        cluster_id = 0
        
        # Find core samples
        for i in range(n_samples):
            neighbors = self._region_query(distances, i)
            if len(neighbors) >= self.min_samples:
                core_samples.append(i)
        
        # Expand clusters
        for core in core_samples:
            if labels[core] != -1:
                continue
            
            # Start new cluster
            labels[core] = cluster_id
            queue = [core]
            
            while queue:
                current = queue.pop(0)
                neighbors = self._region_query(distances, current)
                
                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        if neighbor in core_samples:
                            queue.append(neighbor)
            
            cluster_id += 1
        
        self.labels_ = labels
        self.core_sample_indices_ = np.array(core_samples)
        self.components_ = X_encoded[core_samples]
        self.n_clusters_ = cluster_id
        self.n_noise_ = np.sum(labels == -1)
        
        # Also run classical DBSCAN
        self.classical_dbscan_ = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.classical_dbscan_.fit(X)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_


class QuantumSpectralClustering:
    """Quantum-enhanced Spectral Clustering."""
    
    def __init__(self, n_clusters: int = 3,
                 n_qubits: int = 4,
                 affinity: str = "quantum",
                 n_components: int = None,
                 random_state: Optional[int] = None,
                 quantum_weight: float = 0.5):
        """Initialize quantum spectral clustering.
        
        Args:
            n_clusters: Number of clusters
            n_qubits: Number of qubits
            affinity: Affinity type ('quantum', 'rbf', 'nearest_neighbors')
            n_components: Number of eigenvectors
            random_state: Random seed
            quantum_weight: Weight for quantum affinity
        """
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.affinity = affinity
        self.n_components = n_components or n_clusters
        self.random_state = random_state
        self.quantum_weight = quantum_weight
        
        self.labels_ = None
        self.affinity_matrix_ = None
        self.classical_model_ = None
    
    def _encode_data(self, X: np.ndarray) -> np.ndarray:
        """Encode data to quantum states."""
        n_samples = len(X)
        dim = 2 ** self.n_qubits
        
        states = np.zeros((n_samples, dim), dtype=np.complex128)
        
        for i, x in enumerate(X):
            x = np.asarray(x, dtype=np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            if len(x) > dim:
                states[i, :dim] = x[:dim]
            else:
                states[i, :len(x)] = x
        
        return states
    
    def _compute_affinity_matrix(self, X_encoded: np.ndarray) -> np.ndarray:
        """Compute affinity matrix using quantum distances."""
        n_samples = len(X_encoded)
        affinity = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                classical_aff = np.exp(-np.linalg.norm(X_encoded[i] - X_encoded[j]) ** 2)
                quantum_dist = QuantumDistance.compute_distance(
                    X_encoded[i], X_encoded[j], "fidelity"
                )
                quantum_aff = np.exp(-quantum_dist ** 2)
                
                # Combine affinities
                aff = (1 - self.quantum_weight) * classical_aff + \
                      self.quantum_weight * quantum_aff
                
                affinity[i, j] = aff
                affinity[j, i] = aff
        
        return affinity
    
    def _compute_laplacian(self, affinity: np.ndarray) -> np.ndarray:
        """Compute normalized Laplacian."""
        degrees = np.sum(affinity, axis=1)
        degrees = np.where(degrees == 0, 1, degrees)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        Laplacian = np.eye(len(affinity)) - D_inv_sqrt @ affinity @ D_inv_sqrt
        return Laplacian
    
    def fit(self, X: np.ndarray) -> 'QuantumSpectralClustering':
        """Fit spectral clustering.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        # Encode data
        X_encoded = self._encode_data(X)
        
        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity_matrix(X_encoded)
        
        # Compute Laplacian
        laplacian = self._compute_laplacian(self.affinity_matrix_)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Select eigenvectors
        embedding = eigenvectors[:, :self.n_components]
        
        # Normalize rows
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        
        # K-means on embedding
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(embedding)
        
        # Also run classical spectral clustering
        self.classical_model_ = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            random_state=self.random_state
        )
        self.classical_model_.fit(X)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        self.fit(X)
        return self.labels_


class QuantumCentroidClustering:
    """Quantum Centroid-based clustering (medoid-based)."""
    
    def __init__(self, n_clusters: int = 3,
                 n_qubits: int = 4,
                 method: str = "k-medoids",
                 max_iter: int = 100,
                 random_state: Optional[int] = None,
                 distance_metric: str = "fidelity"):
        """Initialize quantum centroid clustering.
        
        Args:
            n_clusters: Number of clusters
            n_qubits: Number of qubits
            method: Clustering method ('k-medoids', 'k-means', 'pam')
            max_iter: Maximum iterations
            random_state: Random seed
            distance_metric: Distance metric
        """
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        self.distance_metric = distance_metric
        
        self.labels_ = None
        self.centroids_ = None
        self.medoids_ = None
    
    def _encode_data(self, X: np.ndarray) -> np.ndarray:
        """Encode data to quantum states."""
        n_samples = len(X)
        dim = 2 ** self.n_qubits
        
        states = np.zeros((n_samples, dim), dtype=np.complex128)
        
        for i, x in enumerate(X):
            x = np.asarray(x, dtype=np.float64)
            x = x / (np.linalg.norm(x) + 1e-8)
            if len(x) > dim:
                states[i, :dim] = x[:dim]
            else:
                states[i, :len(x)] = x
        
        return states
    
    def _compute_distance_matrix(self, X_encoded: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n_samples = len(X_encoded)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = QuantumDistance.compute_distance(
                    X_encoded[i], X_encoded[j], self.distance_metric
                )
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def fit(self, X: np.ndarray) -> 'QuantumCentroidClustering':
        """Fit centroid clustering.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        # Encode data
        X_encoded = self._encode_data(X)
        
        # Compute distance matrix
        distances = self._compute_distance_matrix(X_encoded)
        
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = np.random.RandomState()
        
        # Initialize medoids randomly
        medoid_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        medoids = X_encoded[medoid_indices].copy()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest medoid
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                min_dist = np.inf
                for k in range(self.n_clusters):
                    d = QuantumDistance.compute_distance(
                        X_encoded[i], medoids[k], self.distance_metric
                    )
                    if d < min_dist:
                        min_dist = d
                        labels[i] = k
            
            # Update medoids
            new_medoids = []
            for k in range(self.n_clusters):
                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) == 0:
                    # Empty cluster, keep random medoid
                    new_medoids.append(medoids[k])
                    continue
                
                # Find point that minimizes total distance
                cluster_points = X_encoded[cluster_indices]
                cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
                
                total_dists = np.sum(cluster_distances, axis=1)
                best_idx = cluster_indices[np.argmin(total_dists)]
                new_medoids.append(X_encoded[best_idx])
            
            new_medoids = np.array(new_medoids)
            
            # Check convergence
            shift = np.mean([
                QuantumDistance.compute_distance(medoids[k], new_medoids[k], self.distance_metric)
                for k in range(self.n_clusters)
            ])
            
            if shift < 1e-6:
                break
            
            medoids = new_medoids
        
        self.labels_ = labels
        self.medoids_ = medoids
        self.centroids_ = medoids  # Same as medoids
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels."""
        self.fit(X)
        return self.labels_


class QuantumClustering:
    """High-level interface for quantum clustering algorithms."""
    
    ALGORITHMS = {
        "kmeans": QuantumKMeans,
        "dbscan": QuantumDBSCAN,
        "spectral": QuantumSpectralClustering,
        "centroid": QuantumCentroidClustering
    }
    
    def __init__(self, algorithm: str = "kmeans",
                 n_clusters: int = 3,
                 n_qubits: int = 4,
                 **kwargs):
        """Initialize quantum clustering.
        
        Args:
            algorithm: Clustering algorithm name
            n_clusters: Number of clusters
            n_qubits: Number of qubits
            **kwargs: Algorithm-specific arguments
        """
        self.algorithm_name = algorithm
        
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Available: {list(self.ALGORITHMS.keys())}")
        
        self.model = self.ALGORITHMS[algorithm](
            n_clusters=n_clusters,
            n_qubits=n_qubits,
            **kwargs
        )
    
    def fit(self, X: np.ndarray) -> 'QuantumClustering':
        """Fit clustering model.
        
        Args:
            X: Input data
            
        Returns:
            Self
        """
        self.model.fit(X)
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        return self.model.fit_predict(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        return self.model.fit_predict(X)
    
    @property
    def labels_(self):
        """Get cluster labels."""
        return self.model.labels_
    
    @property
    def cluster_centers_(self):
        """Get cluster centers."""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None
