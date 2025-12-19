"""
Retrain VPN Detection Model from Scratch
Includes user's real-world non-VPN data to reduce false positives
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RETRAINING VPN DETECTION MODEL")
print("=" * 60)

# ============================================================
# STEP 1: Load CIC Archive Data
# ============================================================
print("\n[1] Loading CIC Archive Data...")
archive = pd.read_csv('archive/consolidated_traffic_data.csv')
archive_vpn = archive[archive['traffic_type'].str.startswith('VPN')].copy()
archive_nonvpn = archive[~archive['traffic_type'].str.startswith('VPN')].copy()

print(f"    CIC VPN samples:     {len(archive_vpn):,}")
print(f"    CIC Non-VPN samples: {len(archive_nonvpn):,}")

# ============================================================
# STEP 2: Load User's Real-World Non-VPN Data
# ============================================================
print("\n[2] Loading User's Real-World Non-VPN Data...")
user_nonvpn_files = [
    'data/processed/cic_features/summa1_cic.csv',
    'data/processed/cic_features/summa2_cic.csv',
]

user_nonvpn_dfs = []
for f in user_nonvpn_files:
    if Path(f).exists():
        df = pd.read_csv(f)
        df['traffic_type'] = 'Non-VPN_RealWorld'
        user_nonvpn_dfs.append(df)
        print(f"    {Path(f).name}: {len(df):,} flows")

user_nonvpn = pd.concat(user_nonvpn_dfs, ignore_index=True) if user_nonvpn_dfs else pd.DataFrame()
print(f"    Total user Non-VPN:  {len(user_nonvpn):,}")

# ============================================================
# STEP 3: Combine All Data
# ============================================================
print("\n[3] Combining Training Data...")

# Get feature columns (exclude traffic_type)
feature_cols = [c for c in archive.columns if c != 'traffic_type']

# Prepare combined dataset
all_data = []

# Add CIC VPN
for _, row in archive_vpn.iterrows():
    all_data.append({'features': row[feature_cols].values, 'label': 1, 'source': 'CIC_VPN'})

# Add CIC Non-VPN
for _, row in archive_nonvpn.iterrows():
    all_data.append({'features': row[feature_cols].values, 'label': 0, 'source': 'CIC_NonVPN'})

# Add User Non-VPN (ensure same features)
for _, row in user_nonvpn.iterrows():
    features = []
    for col in feature_cols:
        if col in user_nonvpn.columns:
            features.append(row[col])
        else:
            features.append(0)
    all_data.append({'features': np.array(features), 'label': 0, 'source': 'User_NonVPN'})

# Create arrays
X = np.array([d['features'] for d in all_data])
y = np.array([d['label'] for d in all_data])
sources = np.array([d['source'] for d in all_data])

print(f"    Total samples: {len(X):,}")
print(f"    VPN samples:   {sum(y == 1):,}")
print(f"    Non-VPN samples: {sum(y == 0):,}")

# ============================================================
# STEP 4: Prepare Training Data
# ============================================================
print("\n[4] Preparing Training Data...")

# Handle inf and nan
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"    Training samples: {len(X_train):,}")
print(f"    Test samples:     {len(X_test):,}")

# ============================================================
# STEP 5: Train Model with Regularization
# ============================================================
print("\n[5] Training Random Forest (with regularization)...")

# Add regularization to reduce overfitting
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,           # Limit tree depth
    min_samples_split=10,   # Require more samples to split
    min_samples_leaf=5,     # Require more samples in leaves
    max_features='sqrt',    # Use subset of features
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ============================================================
# STEP 6: Evaluate Model
# ============================================================
print("\n[6] Evaluating Model...")

# Training performance
train_preds = model.predict(X_train_scaled)
train_f1 = f1_score(y_train, train_preds)

# Test performance
test_preds = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)

print(f"    Training F1: {train_f1:.4f}")
print(f"    Test F1:     {test_f1:.4f}")
print(f"    Gap:         {(train_f1 - test_f1)*100:.2f}% (lower is better)")

print(f"\n    Test Metrics:")
print(f"    - Accuracy:  {test_acc:.4f}")
print(f"    - Precision: {test_prec:.4f}")
print(f"    - Recall:    {test_rec:.4f}")
print(f"    - F1 Score:  {test_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, test_preds)
print(f"\n    Confusion Matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# ============================================================
# STEP 7: Cross-Validation
# ============================================================
print("\n[7] Cross-Validation (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"    CV F1 scores: {cv_scores}")
print(f"    Mean CV F1:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================
# STEP 8: Save Model
# ============================================================
print("\n[8] Saving Model...")
Path('models').mkdir(exist_ok=True)

joblib.dump(model, 'models/vpn_detector.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_cols, 'models/feature_names.joblib')
joblib.dump({
    'accuracy': test_acc,
    'precision': test_prec,
    'recall': test_rec,
    'f1_score': test_f1,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
}, 'models/training_metrics.joblib')

print("    Saved to models/")

# ============================================================
# STEP 9: Test on User's Data
# ============================================================
print("\n[9] Testing on User's Non-VPN Data...")

for f in user_nonvpn_files:
    if Path(f).exists():
        df = pd.read_csv(f)
        X_user = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
        X_user_scaled = scaler.transform(X_user)
        probs = model.predict_proba(X_user_scaled)[:, 1]
        preds = (probs >= 0.55).astype(int)
        vpn_pct = 100 * sum(preds) / len(preds)
        
        print(f"    {Path(f).name}:")
        print(f"      Flows: {len(df):,}")
        print(f"      VPN detected: {sum(preds):,} ({vpn_pct:.1f}%)")
        print(f"      Expected: 0% (since this is non-VPN data)")
        print()

print("=" * 60)
print("RETRAINING COMPLETE!")
print("=" * 60)
