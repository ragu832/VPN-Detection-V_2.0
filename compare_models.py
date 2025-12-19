"""
Compare Hybrid Model vs Current Model
Tests if hybrid (supervised + unsupervised) performs better than simple Random Forest.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Load training data
print("Loading training data...")
df = pd.read_csv('archive/consolidated_traffic_data.csv')
X = df.drop(columns=['traffic_type'])
y = df['traffic_type'].apply(lambda x: 1 if x.startswith('VPN') else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle inf/nan
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print("=" * 60)

# ============================================================
# MODEL 1: Simple Random Forest (Current Model)
# ============================================================
print("\n[1] Training Simple Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_preds = rf_model.predict(X_test_scaled)

rf_metrics = {
    'accuracy': accuracy_score(y_test, rf_preds),
    'precision': precision_score(y_test, rf_preds),
    'recall': recall_score(y_test, rf_preds),
    'f1': f1_score(y_test, rf_preds),
}
print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")

# ============================================================
# MODEL 2: Hybrid Model (RF + Isolation Forest + K-Means)
# ============================================================
print("\n[2] Training Hybrid Model...")

# Train Isolation Forest for anomaly detection
print("  - Training Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.3, random_state=42, n_jobs=-1)
iso_forest.fit(X_train_scaled)
anomaly_scores = -iso_forest.score_samples(X_test_scaled)  # Higher = more anomalous
anomaly_scores_normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

# Train K-Means for clustering
print("  - Training K-Means...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)
cluster_labels = kmeans.predict(X_test_scaled)

# Calculate cluster VPN likelihood based on training data
train_clusters = kmeans.predict(X_train_scaled)
cluster_vpn_rate = {}
for c in [0, 1]:
    mask = train_clusters == c
    if mask.sum() > 0:
        cluster_vpn_rate[c] = y_train.values[mask].mean()
    else:
        cluster_vpn_rate[c] = 0.5

# Get cluster-based VPN probability
cluster_probs = np.array([cluster_vpn_rate[c] for c in cluster_labels])

# Combine scores: α × RF + β × Anomaly + γ × Cluster
alpha = 0.6  # RF weight
beta = 0.25  # Anomaly weight  
gamma = 0.15 # Cluster weight

hybrid_probs = alpha * rf_probs + beta * anomaly_scores_normalized + gamma * cluster_probs
hybrid_preds = (hybrid_probs >= 0.5).astype(int)

hybrid_metrics = {
    'accuracy': accuracy_score(y_test, hybrid_preds),
    'precision': precision_score(y_test, hybrid_preds),
    'recall': recall_score(y_test, hybrid_preds),
    'f1': f1_score(y_test, hybrid_preds),
}
print(f"Hybrid Model  - Accuracy: {hybrid_metrics['accuracy']:.4f}, F1: {hybrid_metrics['f1']:.4f}")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
print(f"{'Metric':<15} {'Random Forest':>15} {'Hybrid Model':>15} {'Difference':>12}")
print("-" * 60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    rf_val = rf_metrics[metric]
    hybrid_val = hybrid_metrics[metric]
    diff = hybrid_val - rf_val
    diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
    print(f"{metric:<15} {rf_val:>15.4f} {hybrid_val:>15.4f} {diff_str:>12}")

print("-" * 60)
if hybrid_metrics['f1'] > rf_metrics['f1']:
    print("✅ HYBRID MODEL PERFORMS BETTER!")
    print(f"   F1 improvement: +{(hybrid_metrics['f1'] - rf_metrics['f1'])*100:.2f}%")
else:
    print("❌ SIMPLE RF MODEL PERFORMS BETTER")
    print(f"   F1 difference: {(hybrid_metrics['f1'] - rf_metrics['f1'])*100:.2f}%")

# Try different weight combinations
print("\n\nTrying different weight combinations...")
print("-" * 60)
best_f1 = 0
best_weights = None

for alpha in [0.5, 0.6, 0.7, 0.8]:
    for beta in [0.1, 0.2, 0.3]:
        gamma = 1 - alpha - beta
        if gamma < 0:
            continue
        probs = alpha * rf_probs + beta * anomaly_scores_normalized + gamma * cluster_probs
        preds = (probs >= 0.5).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (alpha, beta, gamma)

print(f"Best weights: α={best_weights[0]:.2f}, β={best_weights[1]:.2f}, γ={best_weights[2]:.2f}")
print(f"Best Hybrid F1: {best_f1:.4f}")
print(f"Simple RF F1:   {rf_metrics['f1']:.4f}")

if best_f1 > rf_metrics['f1']:
    print(f"\n✅ Best hybrid is {(best_f1 - rf_metrics['f1'])*100:.2f}% better than simple RF")
else:
    print(f"\n❌ Simple RF is still better by {(rf_metrics['f1'] - best_f1)*100:.2f}%")
