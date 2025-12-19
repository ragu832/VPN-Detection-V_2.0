"""
Save model with all required files for the Streamlit app.
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load archive data
df = pd.read_csv('archive/consolidated_traffic_data.csv')
X = df.drop(columns=['traffic_type'])
y = df['traffic_type'].apply(lambda x: 1 if x.startswith('VPN') else 0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.replace([np.inf, -np.inf], 0).fillna(0))
X_test_scaled = scaler.transform(X_test.replace([np.inf, -np.inf], 0).fillna(0))

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

# Save all
joblib.dump(model, 'models/vpn_detector.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(list(X.columns), 'models/feature_names.joblib')
joblib.dump(metrics, 'models/training_metrics.joblib')

print('Model saved with metrics!')
print(f"Accuracy: {metrics['accuracy']:.4f}")
