"""
Feature Extractor Module
Handles loading and preprocessing of network traffic data from various formats.
Supports both CIC flow data and raw Wireshark packet captures.
"""

import pandas as pd
import numpy as np
from scipy.io import arff
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


def is_wireshark_data(df: pd.DataFrame) -> bool:
    """Check if DataFrame contains raw Wireshark packet data."""
    wireshark_columns = {'No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length'}
    return wireshark_columns.issubset(set(df.columns))


def packet_to_flow_features(df: pd.DataFrame, time_window: float = 15.0) -> pd.DataFrame:
    """
    Convert raw Wireshark packet data to flow-level features.
    
    Groups packets by (Source, Destination) pairs within time windows
    and computes statistical features similar to CIC dataset.
    
    Args:
        df: DataFrame with Wireshark columns (No., Time, Source, Destination, Protocol, Length)
        time_window: Time window in seconds for flow aggregation (default 15s)
        
    Returns:
        DataFrame with flow-level features matching CIC dataset format
    """
    # Ensure we have the required columns
    required_cols = ['Time', 'Source', 'Destination', 'Length']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    # Clean and prepare data
    df = df.copy()
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df = df.dropna(subset=['Time', 'Length', 'Source', 'Destination'])
    
    if len(df) == 0:
        raise ValueError("No valid packets after cleaning")
    
    # Create flow identifier (bidirectional)
    df['flow_id'] = df.apply(
        lambda row: tuple(sorted([str(row['Source']), str(row['Destination'])])),
        axis=1
    )
    
    # Assign time windows
    df['time_window'] = (df['Time'] // time_window).astype(int)
    
    # Group by flow and time window
    flows = []
    
    for (flow_id, time_win), group in df.groupby(['flow_id', 'time_window']):
        if len(group) < 2:
            continue
            
        source, dest = flow_id
        
        # Separate forward and backward packets
        fwd_mask = group['Source'] == source
        bwd_mask = ~fwd_mask
        
        fwd_packets = group[fwd_mask]
        bwd_packets = group[bwd_mask]
        
        # Time calculations
        times = group['Time'].values
        sorted_times = np.sort(times)
        duration = sorted_times[-1] - sorted_times[0] if len(sorted_times) > 1 else 0.001
        
        # Inter-arrival times
        if len(sorted_times) > 1:
            iats = np.diff(sorted_times)
            flow_iat_mean = np.mean(iats)
            flow_iat_std = np.std(iats) if len(iats) > 1 else 0
            flow_iat_min = np.min(iats)
            flow_iat_max = np.max(iats)
        else:
            flow_iat_mean = flow_iat_std = flow_iat_min = flow_iat_max = 0
        
        # Forward inter-arrival times
        if len(fwd_packets) > 1:
            fwd_times = np.sort(fwd_packets['Time'].values)
            fwd_iats = np.diff(fwd_times)
            fwd_iat_mean = np.mean(fwd_iats)
            fwd_iat_min = np.min(fwd_iats)
            fwd_iat_max = np.max(fwd_iats)
            fwd_iat_total = np.sum(fwd_iats)
        else:
            fwd_iat_mean = fwd_iat_min = fwd_iat_max = fwd_iat_total = 0
        
        # Backward inter-arrival times
        if len(bwd_packets) > 1:
            bwd_times = np.sort(bwd_packets['Time'].values)
            bwd_iats = np.diff(bwd_times)
            bwd_iat_mean = np.mean(bwd_iats)
            bwd_iat_min = np.min(bwd_iats)
            bwd_iat_max = np.max(bwd_iats)
            bwd_iat_total = np.sum(bwd_iats)
        else:
            bwd_iat_mean = bwd_iat_min = bwd_iat_max = bwd_iat_total = 0
        
        # Byte and packet statistics
        total_bytes = group['Length'].sum()
        total_packets = len(group)
        
        # Active/idle time estimation (simplified)
        # Consider packets within 1 second as "active", gaps > 1s as "idle"
        if len(sorted_times) > 1:
            iats = np.diff(sorted_times)
            active_times = iats[iats < 1.0]
            idle_times = iats[iats >= 1.0]
            
            active_mean = np.mean(active_times) if len(active_times) > 0 else 0
            active_min = np.min(active_times) if len(active_times) > 0 else 0
            active_max = np.max(active_times) if len(active_times) > 0 else 0
            active_std = np.std(active_times) if len(active_times) > 1 else 0
            
            idle_mean = np.mean(idle_times) if len(idle_times) > 0 else 0
            idle_min = np.min(idle_times) if len(idle_times) > 0 else 0
            idle_max = np.max(idle_times) if len(idle_times) > 0 else 0
            idle_std = np.std(idle_times) if len(idle_times) > 1 else 0
        else:
            active_mean = active_min = active_max = active_std = 0
            idle_mean = idle_min = idle_max = idle_std = 0
        
        # Build flow record matching CIC features
        flow_record = {
            'source_ip': source,
            'destination_ip': dest,
            'duration': duration,
            'total_fiat': fwd_iat_total,
            'total_biat': bwd_iat_total,
            'min_fiat': fwd_iat_min,
            'min_biat': bwd_iat_min,
            'max_fiat': fwd_iat_max,
            'max_biat': bwd_iat_max,
            'mean_fiat': fwd_iat_mean,
            'mean_biat': bwd_iat_mean,
            'flowPktsPerSecond': total_packets / max(duration, 0.001),
            'flowBytesPerSecond': total_bytes / max(duration, 0.001),
            'min_flowiat': flow_iat_min,
            'max_flowiat': flow_iat_max,
            'mean_flowiat': flow_iat_mean,
            'std_flowiat': flow_iat_std,
            'min_active': active_min,
            'mean_active': active_mean,
            'max_active': active_max,
            'std_active': active_std,
            'min_idle': idle_min,
            'mean_idle': idle_mean,
            'max_idle': idle_max,
            'std_idle': idle_std,
            'total_packets': total_packets,
            'total_bytes': total_bytes,
            'protocol': group['Protocol'].mode().iloc[0] if 'Protocol' in group.columns else 'Unknown'
        }
        
        flows.append(flow_record)
    
    if len(flows) == 0:
        raise ValueError("No flows could be extracted from packet data. Need at least 2 packets per flow.")
    
    return pd.DataFrame(flows)


class FeatureExtractor:
    """Extract and preprocess network flow features from various data sources."""
    
    # Feature columns from CIC dataset
    FEATURE_COLUMNS = [
        'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
        'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
        'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
        'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
        'min_idle', 'mean_idle', 'max_idle', 'std_idle'
    ]
    
    LABEL_COLUMN = 'class1'
    
    def __init__(self):
        self.scaler = None
        self.feature_stats = {}
    
    def load_arff(self, file_path: str) -> pd.DataFrame:
        """Load data from ARFF format (CIC dataset format)."""
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Decode byte strings to regular strings for label column
        if self.LABEL_COLUMN in df.columns:
            df[self.LABEL_COLUMN] = df[self.LABEL_COLUMN].str.decode('utf-8')
        
        return df
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV format. Auto-detects Wireshark data and converts to flows."""
        # Try multiple encodings to handle various CSV formats
        encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, engine='python', on_bad_lines='skip')
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                continue
        
        if df is None:
            # Last resort: read with latin-1 which accepts any byte
            df = pd.read_csv(file_path, encoding='latin-1', engine='python', on_bad_lines='skip')
        
        # Check if this is Wireshark packet data
        if is_wireshark_data(df):
            print(f"Detected Wireshark packet data ({len(df)} packets). Converting to flow features...")
            df = packet_to_flow_features(df)
            print(f"Extracted {len(df)} flows from packet data.")
        
        return df

    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Auto-detect file format and load data."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.arff':
            return self.load_arff(str(file_path))
        elif file_path.suffix.lower() == '.csv':
            return self.load_csv(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for ML models.
        
        Args:
            df: Input DataFrame
            fit: If True, fit the scaler on this data
            
        Returns:
            Tuple of (features, labels) where labels may be None
        """
        # Select feature columns (handle missing columns gracefully)
        available_features = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values (-1 in CIC dataset means missing)
        X = X.replace(-1, np.nan)
        
        # Fill NaN with column median
        if fit:
            self.feature_stats['medians'] = X.median()
        X = X.fillna(self.feature_stats.get('medians', X.median()))
        
        # Normalize features (Min-Max scaling to [0, 1])
        if fit:
            self.feature_stats['min'] = X.min()
            self.feature_stats['max'] = X.max()
        
        X_min = self.feature_stats.get('min', X.min())
        X_max = self.feature_stats.get('max', X.max())
        
        # Avoid division by zero
        range_vals = X_max - X_min
        range_vals = range_vals.replace(0, 1)
        
        X_normalized = (X - X_min) / range_vals
        
        # Handle any remaining NaN or inf values
        X_normalized = X_normalized.replace([np.inf, -np.inf], 0)
        X_normalized = X_normalized.fillna(0)
        
        # Extract labels if present
        y = None
        if self.LABEL_COLUMN in df.columns:
            # Convert to binary: VPN = 1, Non-VPN = 0
            y = (df[self.LABEL_COLUMN] == 'VPN').astype(int).values
        
        return X_normalized.values, y
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Get list of available feature names."""
        return [col for col in self.FEATURE_COLUMNS if col in df.columns]


def load_cic_dataset(scenario_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load and preprocess CIC VPN dataset.
    
    Args:
        scenario_path: Path to the ARFF file
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    extractor = FeatureExtractor()
    df = extractor.load_data(scenario_path)
    X, y = extractor.preprocess(df, fit=True)
    feature_names = extractor.get_feature_names(df)
    
    return X, y, feature_names


if __name__ == "__main__":
    # Test loading
    import os
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_path, "Scenario A1-ARFF", "Scenario A1-ARFF", 
                             "TimeBasedFeatures-Dataset-15s-VPN.arff")
    
    if os.path.exists(test_file):
        X, y, features = load_cic_dataset(test_file)
        print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"VPN samples: {sum(y)}, Non-VPN samples: {len(y) - sum(y)}")
        print(f"Features: {features}")
    else:
        print(f"Test file not found: {test_file}")
