"""
Supervised Learning Module
Random Forest classifier for VPN detection with feature importance analysis.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class SupervisedClassifier:
    """
    Random Forest classifier for VPN vs Non-VPN traffic classification.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.feature_names = None
        self.is_fitted = False
        self.training_metrics = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            validation_split: float = 0.2) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels (0 = Non-VPN, 1 = VPN)
            feature_names: Names of features for importance analysis
            validation_split: Proportion of data for validation
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = feature_names
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]
        
        self.training_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_val, y_pred),
            'validation_size': len(y_val)
        }
        
        return self.training_metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        
        return {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 = Non-VPN, 1 = VPN)."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of VPN class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.
        
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        importance = self.model.feature_importances_
        
        if self.feature_names:
            feature_imp = list(zip(self.feature_names, importance))
        else:
            feature_imp = [(f"feature_{i}", imp) for i, imp in enumerate(importance)]
        
        # Sort by importance (descending)
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        return feature_imp[:top_n]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(
                y, y_pred, 
                target_names=['Non-VPN', 'VPN'],
                output_dict=True
            )
        }
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, path / "random_forest.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        joblib.dump(self.training_metrics, path / "training_metrics.joblib")
    
    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)
        
        self.model = joblib.load(path / "random_forest.joblib")
        self.feature_names = joblib.load(path / "feature_names.joblib")
        self.training_metrics = joblib.load(path / "training_metrics.joblib")
        self.is_fitted = True
        
        return self


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    feature_names = [f"feature_{i}" for i in range(10)]
    
    classifier = SupervisedClassifier(n_estimators=50)
    metrics = classifier.fit(X, y, feature_names)
    
    print("Training Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    
    print("\nTop 5 Features:")
    for name, imp in classifier.get_feature_importance(5):
        print(f"  {name}: {imp:.4f}")
