"""
Analyze summa1_cic.csv and compare with training data.
"""

import pandas as pd
import numpy as np
import joblib

# Load the file
df = pd.read_csv('data/processed/cic_features/summa1_cic.csv')
print('summa1_cic.csv Analysis')
print('=' * 60)
print(f'Total flows: {len(df):,}')
print()

# Show feature statistics
print('Feature Statistics (first 10):')
print('-' * 60)
for col in df.columns[:10]:
    print(f"{col:20} min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

# Compare with training data
print()
print('=' * 60)
print('Comparing with Training Data:')
archive = pd.read_csv('archive/consolidated_traffic_data.csv')
archive_vpn = archive[archive['traffic_type'].str.startswith('VPN')]
archive_nonvpn = archive[~archive['traffic_type'].str.startswith('VPN')]

print(f'Training VPN samples: {len(archive_vpn):,}')
print(f'Training Non-VPN samples: {len(archive_nonvpn):,}')

# Compare key features
print()
print('Key Feature Comparison:')
print('-' * 60)

features_to_compare = ['duration', 'flowBytesPerSecond', 'flowPktsPerSecond', 'mean_flowiat']
for feat in features_to_compare:
    s_mean = df[feat].mean()
    vpn_mean = archive_vpn[feat].mean()
    nonvpn_mean = archive_nonvpn[feat].mean()
    
    # Which is closer?
    dist_to_vpn = abs(s_mean - vpn_mean)
    dist_to_nonvpn = abs(s_mean - nonvpn_mean)
    closer = "VPN" if dist_to_vpn < dist_to_nonvpn else "Non-VPN"
    
    print(f'{feat}:')
    print(f'  summa1:  {s_mean:,.2f}')
    print(f'  VPN:     {vpn_mean:,.2f}')
    print(f'  Non-VPN: {nonvpn_mean:,.2f}')
    print(f'  Closer to: {closer}')
    print()

# Run model predictions with different thresholds
print('=' * 60)
print('Predictions at Different Thresholds:')
print('-' * 60)

model = joblib.load('models/vpn_detector.joblib')
scaler = joblib.load('models/scaler.joblib')
feature_names = joblib.load('models/feature_names.joblib')

X = df[feature_names].replace([np.inf, -np.inf], 0).fillna(0)
X_scaled = scaler.transform(X)
probs = model.predict_proba(X_scaled)[:, 1]

for threshold in [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
    preds = (probs >= threshold).astype(int)
    vpn_pct = 100 * sum(preds) / len(preds)
    print(f'Threshold {threshold:.2f}: {sum(preds):,} VPN flows ({vpn_pct:.1f}%)')

print()
print('Probability distribution:')
print(f'  Min:    {probs.min():.3f}')
print(f'  Max:    {probs.max():.3f}')
print(f'  Mean:   {probs.mean():.3f}')
print(f'  Median: {np.median(probs):.3f}')
print(f'  >0.9:   {sum(probs > 0.9)} flows')
print(f'  >0.8:   {sum(probs > 0.8)} flows')
print(f'  <0.2:   {sum(probs < 0.2)} flows')
