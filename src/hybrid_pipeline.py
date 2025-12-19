"""
Hybrid Pipeline Module
Combines unsupervised and supervised models for robust VPN detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib

from .feature_extractor import FeatureExtractor
from .unsupervised_models import UnsupervisedDetector
from .supervised_models import SupervisedClassifier


class HybridVPNDetector:
    """
    Hybrid ML pipeline combining unsupervised and supervised approaches.
    
    VPN Score = α × RF_Probability + β × Anomaly_Score + γ × Cluster_VPN_Likelihood
    """
    
    def __init__(self, 
                 supervised_weight: float = 0.6,
                 unsupervised_weight: float = 0.4,
                 vpn_threshold: float = 0.5):
        """
        Initialize hybrid detector.
        
        Args:
            supervised_weight: Weight for Random Forest predictions (α)
            unsupervised_weight: Weight for unsupervised scores (β + γ)
            vpn_threshold: Threshold for VPN classification
        """
        self.supervised_weight = supervised_weight
        self.unsupervised_weight = unsupervised_weight
        self.vpn_threshold = vpn_threshold
        
        self.feature_extractor = FeatureExtractor()
        self.unsupervised = UnsupervisedDetector()
        self.supervised = SupervisedClassifier()
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> Dict:
        """
        Train both unsupervised and supervised models.
        
        Args:
            X: Feature matrix
            y: Labels (0 = Non-VPN, 1 = VPN)
            feature_names: Names of features
            
        Returns:
            Training metrics from both models
        """
        # Train unsupervised models
        self.unsupervised.fit(X, y)
        
        # Train supervised model
        supervised_metrics = self.supervised.fit(X, y, feature_names)
        
        self.is_fitted = True
        
        return {
            'supervised_metrics': supervised_metrics,
            'feature_importance': self.supervised.get_feature_importance()
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict VPN probability using hybrid approach.
        
        Returns:
            VPN probability scores in [0, 1] range
        """
        # Get supervised predictions
        rf_proba = self.supervised.predict_proba(X)
        
        # Get unsupervised scores
        unsup_scores = self.unsupervised.get_combined_scores(X)
        
        # Combine using weighted average
        hybrid_scores = (
            self.supervised_weight * rf_proba + 
            self.unsupervised_weight * unsup_scores
        )
        
        return hybrid_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict VPN labels using threshold.
        
        Returns:
            Binary labels (0 = Non-VPN, 1 = VPN)
        """
        proba = self.predict_proba(X)
        return (proba >= self.vpn_threshold).astype(int)
    
    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file and make predictions.
        
        Returns:
            DataFrame with original data plus predictions
        """
        df = self.feature_extractor.load_data(file_path)
        X, _ = self.feature_extractor.preprocess(df, fit=False)
        
        proba = self.predict_proba(X)
        labels = self.predict(X)
        
        df['vpn_probability'] = proba
        df['vpn_predicted'] = labels
        df['vpn_label'] = df['vpn_predicted'].map({0: 'Non-VPN', 1: 'VPN'})
        
        return df
    
    def analyze_device(self, df: pd.DataFrame, 
                       device_column: str = 'source_ip') -> pd.DataFrame:
        """
        Aggregate predictions by device/IP.
        
        Args:
            df: DataFrame with predictions
            device_column: Column containing device identifiers
            
        Returns:
            DataFrame with per-device VPN metrics
        """
        if device_column not in df.columns:
            # If no device column, treat all as single device
            return pd.DataFrame({
                'device': ['all_flows'],
                'total_flows': [len(df)],
                'vpn_flows': [df['vpn_predicted'].sum()],
                'vpn_ratio': [df['vpn_predicted'].mean()],
                'avg_vpn_probability': [df['vpn_probability'].mean()],
                'max_vpn_probability': [df['vpn_probability'].max()],
                'is_suspicious': [df['vpn_probability'].mean() > self.vpn_threshold]
            })
        
        device_stats = df.groupby(device_column).agg({
            'vpn_predicted': ['count', 'sum', 'mean'],
            'vpn_probability': ['mean', 'max']
        }).reset_index()
        
        device_stats.columns = [
            'device', 'total_flows', 'vpn_flows', 'vpn_ratio',
            'avg_vpn_probability', 'max_vpn_probability'
        ]
        
        device_stats['is_suspicious'] = device_stats['vpn_ratio'] > self.vpn_threshold
        
        return device_stats.sort_values('vpn_ratio', ascending=False)
    
    def get_model_insights(self) -> Dict:
        """Get insights about model behavior."""
        return {
            'supervised_weight': self.supervised_weight,
            'unsupervised_weight': self.unsupervised_weight,
            'vpn_threshold': self.vpn_threshold,
            'feature_importance': self.supervised.get_feature_importance(10),
            'training_metrics': self.supervised.training_metrics,
            'cluster_vpn_likelihood': self.unsupervised.cluster_vpn_likelihood
        }
    
    def save(self, path: str):
        """Save all models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save component models
        self.unsupervised.save(path / "unsupervised")
        self.supervised.save(path / "supervised")
        
        # Save feature extractor stats
        joblib.dump(self.feature_extractor.feature_stats, path / "feature_stats.joblib")
        
        # Save pipeline config
        config = {
            'supervised_weight': self.supervised_weight,
            'unsupervised_weight': self.unsupervised_weight,
            'vpn_threshold': self.vpn_threshold
        }
        joblib.dump(config, path / "config.joblib")
    
    def load(self, path: str):
        """Load all models from disk."""
        path = Path(path)
        
        # Load component models
        self.unsupervised.load(path / "unsupervised")
        self.supervised.load(path / "supervised")
        
        # Load feature extractor stats
        self.feature_extractor.feature_stats = joblib.load(path / "feature_stats.joblib")
        
        # Load pipeline config
        config = joblib.load(path / "config.joblib")
        self.supervised_weight = config['supervised_weight']
        self.unsupervised_weight = config['unsupervised_weight']
        self.vpn_threshold = config['vpn_threshold']
        
        self.is_fitted = True
        return self


def train_vpn_detector(data_path: str, model_save_path: str = "models") -> HybridVPNDetector:
    """
    Train a complete VPN detection pipeline.
    
    Args:
        data_path: Path to training data (ARFF or CSV)
        model_save_path: Path to save trained models
        
    Returns:
        Trained HybridVPNDetector
    """
    from .feature_extractor import load_cic_dataset
    
    # Load and preprocess data
    X, y, feature_names = load_cic_dataset(data_path)
    
    print(f"Loaded {len(y)} samples ({sum(y)} VPN, {len(y) - sum(y)} Non-VPN)")
    
    # Train hybrid detector
    detector = HybridVPNDetector()
    metrics = detector.fit(X, y, feature_names)
    
    # Print training results
    print("\nTraining Results:")
    print(f"  Accuracy: {metrics['supervised_metrics']['accuracy']:.3f}")
    print(f"  Precision: {metrics['supervised_metrics']['precision']:.3f}")
    print(f"  Recall: {metrics['supervised_metrics']['recall']:.3f}")
    print(f"  F1 Score: {metrics['supervised_metrics']['f1']:.3f}")
    
    print("\nTop 5 Important Features:")
    for name, imp in metrics['feature_importance'][:5]:
        print(f"  {name}: {imp:.4f}")
    
    # Save models
    detector.save(model_save_path)
    print(f"\nModels saved to: {model_save_path}")
    
    return detector


if __name__ == "__main__":
    import os
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(
        base_path, "Scenario A1-ARFF", "Scenario A1-ARFF",
        "TimeBasedFeatures-Dataset-15s-VPN.arff"
    )
    model_path = os.path.join(base_path, "models")
    
    if os.path.exists(data_file):
        detector = train_vpn_detector(data_file, model_path)
    else:
        print(f"Data file not found: {data_file}")
