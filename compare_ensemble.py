"""
Compare: Random Forest vs XGBoost vs Ensemble
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed, using GradientBoosting instead")

# Load data
print("Loading training data...")
df = pd.read_csv('archive/consolidated_traffic_data.csv')
X = df.drop(columns=['traffic_type'])
y = df['traffic_type'].apply(lambda x: 1 if x.startswith('VPN') else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clean data
X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
print("=" * 70)

# ============================================================
# MODEL 1: Random Forest (baseline)
# ============================================================
print("\n[1] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
rf_f1 = f1_score(y_test, rf_preds)
print(f"    Random Forest F1: {rf_f1:.4f}")

# ============================================================
# MODEL 2: XGBoost / Gradient Boosting
# ============================================================
print("\n[2] Training XGBoost/GradientBoosting...")
if HAS_XGBOOST:
    xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
else:
    xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)
xgb_f1 = f1_score(y_test, xgb_preds)
print(f"    XGBoost/GB F1: {xgb_f1:.4f}")

# ============================================================
# MODEL 3: Ensemble (Hard Voting)
# ============================================================
print("\n[3] Training Ensemble (Hard Voting)...")
ensemble_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', xgb.__class__(n_estimators=100, random_state=42) if not HAS_XGBOOST else XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)),
    ],
    voting='hard'
)
ensemble_hard.fit(X_train_scaled, y_train)
ens_hard_preds = ensemble_hard.predict(X_test_scaled)
ens_hard_f1 = f1_score(y_test, ens_hard_preds)
print(f"    Ensemble (Hard) F1: {ens_hard_f1:.4f}")

# ============================================================
# MODEL 4: Ensemble (Soft Voting)
# ============================================================
print("\n[4] Training Ensemble (Soft Voting)...")
ensemble_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0) if HAS_XGBOOST else GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    voting='soft'
)
ensemble_soft.fit(X_train_scaled, y_train)
ens_soft_preds = ensemble_soft.predict(X_test_scaled)
ens_soft_f1 = f1_score(y_test, ens_soft_preds)
print(f"    Ensemble (Soft) F1: {ens_soft_f1:.4f}")

# ============================================================
# RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

results = [
    ("Random Forest", rf_f1, rf_preds),
    ("XGBoost/GB", xgb_f1, xgb_preds),
    ("Ensemble (Hard)", ens_hard_f1, ens_hard_preds),
    ("Ensemble (Soft)", ens_soft_f1, ens_soft_preds),
]

print(f"{'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}")
print("-" * 70)

for name, f1, preds in results:
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(f"{name:<20} {acc:>12.4f} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f}")

print("-" * 70)

# Find best
best = max(results, key=lambda x: x[1])
baseline_f1 = rf_f1

print(f"\n🏆 WINNER: {best[0]} (F1: {best[1]:.4f})")
if best[1] > baseline_f1:
    print(f"   Improvement over RF: +{(best[1] - baseline_f1)*100:.2f}%")
else:
    print(f"   Same as or worse than baseline RF")
