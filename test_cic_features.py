"""
Test VPN detection on CIC-extracted features.
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Configuration
THRESHOLD = 0.55  # VPN probability threshold

# Load model
model = joblib.load('models/vpn_detector.joblib')
scaler = joblib.load('models/scaler.joblib')
features = joblib.load('models/feature_names.joblib')

# Test on CIC-extracted files
cic_dir = Path('data/processed/cic_features')

print(f'VPN DETECTION RESULTS (threshold={THRESHOLD})')
print('=' * 65)
print(f"File                                     Flows  VPN %  Verdict")
print('-' * 65)

for f in sorted(cic_dir.glob('*_cic.csv')):
    df = pd.read_csv(f)
    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    
    vpn_pct = 100 * sum(preds) / len(preds)
    if vpn_pct > 50:
        verdict = 'VPN DETECTED'
    elif vpn_pct > 25:
        verdict = 'LIKELY VPN'
    else:
        verdict = 'Non-VPN'
    
    name = f.stem.replace('_cic', '')[:38]
    print(f'{name:<40} {len(preds):>5}  {vpn_pct:>5.1f}%  {verdict}')

print('-' * 65)
print('\nValidation:')
print('  log1 (mobile+VPN), log3 (wifi+VPN), log4 (stealth): Should be VPN')
print('  log2 (wifi without VPN): Should be Non-VPN')
