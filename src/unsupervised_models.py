"""
Unsupervised Learning Module
K-Means clustering and Isolation Forest for anomaly detection.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import joblib
from typing import Tuple, Optional
from pathlib import Path


class UnsupervisedDetector:
    """
    Unsupervised learning models for VPN traffic pattern detection.
    Uses K-Means for clustering and Isolation Forest for anomaly detection.
    """
    
    def __init__(self, n_clusters: int = 5, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize unsupervised models.
        
        Args:
            n_clusters: Number of clusters for K-Means
            contamination: Expected proportion of anomalies for Isolation Forest
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        
        self.kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=10
        )
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        self.cluster_vpn_likelihood = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit both clustering and anomaly detection models.
        
        Args:
            X: Feature matrix
            y: Optional labels (used to compute cluster VPN likelihood)
        """
        # Fit K-Means
        self.kmeans.fit(X)
        
        # Fit Isolation Forest
        self.isolation_forest.fit(X)
        
        # If labels provided, compute VPN likelihood per cluster
        if y is not None:
            self._compute_cluster_vpn_likelihood(X, y)
        
        self.is_fitted = True
        return self
    
    def _compute_cluster_vpn_likelihood(self, X: np.ndarray, y: np.ndarray):
        """Compute the proportion of VPN samples in each cluster."""
        clusters = self.kmeans.predict(X)
        self.cluster_vpn_likelihood = {}
        
        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            if mask.sum() > 0:
                vpn_ratio = y[mask].mean()
                self.cluster_vpn_likelihood[cluster_id] = vpn_ratio
            else:
                self.cluster_vpn_likelihood[cluster_id] = 0.5  # Default to uncertain
    
    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        return self.kmeans.predict(X)
    
    def get_cluster_vpn_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Get VPN likelihood based on cluster membership."""
        if self.cluster_vpn_likelihood is None:
            # If no likelihood computed, return 0.5 (uncertain)
            return np.full(len(X), 0.5)
        
        clusters = self.predict_cluster(X)
        return np.array([self.cluster_vpn_likelihood.get(c, 0.5) for c in clusters])
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from Isolation Forest.
        Higher scores indicate more anomalous (potentially VPN) traffic.
        
        Returns:
            Normalized anomaly scores in [0, 1] range
        """
        # Isolation Forest returns negative scores; more negative = more anomalous
        raw_scores = self.isolation_forest.decision_function(X)
        
        # Normalize to [0, 1] where 1 = more anomalous
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score - min_score > 0:
            normalized = 1 - (raw_scores - min_score) / (max_score - min_score)
        else:
            normalized = np.full(len(X), 0.5)
        
        return normalized
    
    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are anomalies (1) or normal (-1)."""
        return self.isolation_forest.predict(X)
    
    def get_combined_scores(self, X: np.ndarray, 
                            cluster_weight: float = 0.4, 
                            anomaly_weight: float = 0.6) -> np.ndarray:
        """
        Get combined unsupervised VPN probability scores.
        
        Args:
            X: Feature matrix
            cluster_weight: Weight for cluster-based VPN likelihood
            anomaly_weight: Weight for anomaly scores
            
        Returns:
            Combined scores in [0, 1] range
        """
        cluster_scores = self.get_cluster_vpn_likelihood(X)
        anomaly_scores = self.get_anomaly_scores(X)
        
        combined = cluster_weight * cluster_scores + anomaly_weight * anomaly_scores
        return np.clip(combined, 0, 1)
    
    def find_optimal_clusters(self, X: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to try
            
        Returns:
            Optimal number of clusters
        """
        best_score = -1
        best_k = k_range[0]
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def save(self, path: str):
        """Save models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.kmeans, path / "kmeans.joblib")
        joblib.dump(self.isolation_forest, path / "isolation_forest.joblib")
        joblib.dump(self.cluster_vpn_likelihood, path / "cluster_vpn_likelihood.joblib")
    
    def load(self, path: str):
        """Load models from disk."""
        path = Path(path)
        
        self.kmeans = joblib.load(path / "kmeans.joblib")
        self.isolation_forest = joblib.load(path / "isolation_forest.joblib")
        self.cluster_vpn_likelihood = joblib.load(path / "cluster_vpn_likelihood.joblib")
        self.is_fitted = True
        
        return self


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    detector = UnsupervisedDetector(n_clusters=3)
    detector.fit(X, y)
    
    scores = detector.get_combined_scores(X)
    print(f"Sample scores: {scores[:10]}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
