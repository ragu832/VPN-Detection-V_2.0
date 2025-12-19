"""
Process a single large pcapng file and extract CIC features + run VPN detection.
"""

from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict
import pandas as pd
import numpy as np
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

def extract_cic_features(pcap_path, output_path=None):
    """Extract CIC-compatible features from a pcapng file."""
    
    print(f"Loading: {pcap_path}")
    print("This may take several minutes for large files...")
    packets = rdpcap(pcap_path)
    print(f"Loaded {len(packets):,} packets")
    
    # Group packets into flows
    print("Grouping packets into flows...")
    flows = defaultdict(list)
    
    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            
            src_port = 0
            dst_port = 0
            
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            
            flow_key = tuple(sorted([
                (src_ip, src_port),
                (dst_ip, dst_port)
            ])) + (proto,)
            
            flows[flow_key].append({
                'time': float(pkt.time),
                'size': len(pkt),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
            })
    
    print(f"Found {len(flows):,} flows")
    
    # Extract features
    print("Extracting CIC features...")
    flow_features = []
    
    for flow_key, pkts in flows.items():
        if len(pkts) < 2:
            continue
        
        pkts = sorted(pkts, key=lambda x: x['time'])
        times = [p['time'] for p in pkts]
        sizes = [p['size'] for p in pkts]
        
        duration = (times[-1] - times[0]) * 1000000
        if duration == 0:
            duration = 1
        
        iats = np.diff(times) * 1000000
        
        first_src = pkts[0]['src_ip']
        fwd_times = [p['time'] for p in pkts if p['src_ip'] == first_src]
        bwd_times = [p['time'] for p in pkts if p['src_ip'] != first_src]
        
        fiat = np.diff(fwd_times) * 1000000 if len(fwd_times) > 1 else [0]
        biat = np.diff(bwd_times) * 1000000 if len(bwd_times) > 1 else [0]
        
        active_threshold = 1000000
        active_times = iats[iats < active_threshold] if len(iats) > 0 else [0]
        idle_times = iats[iats >= active_threshold] if len(iats) > 0 else [0]
        
        if len(active_times) == 0:
            active_times = [0]
        if len(idle_times) == 0:
            idle_times = [0]
        
        features = {
            'duration': duration,
            'total_fiat': np.sum(fiat),
            'total_biat': np.sum(biat),
            'min_fiat': np.min(fiat) if len(fiat) > 0 else 0,
            'min_biat': np.min(biat) if len(biat) > 0 else 0,
            'max_fiat': np.max(fiat) if len(fiat) > 0 else 0,
            'max_biat': np.max(biat) if len(biat) > 0 else 0,
            'mean_fiat': np.mean(fiat) if len(fiat) > 0 else 0,
            'mean_biat': np.mean(biat) if len(biat) > 0 else 0,
            'flowPktsPerSecond': len(pkts) / (duration / 1000000) if duration > 0 else 0,
            'flowBytesPerSecond': sum(sizes) / (duration / 1000000) if duration > 0 else 0,
            'min_flowiat': np.min(iats) if len(iats) > 0 else 0,
            'max_flowiat': np.max(iats) if len(iats) > 0 else 0,
            'mean_flowiat': np.mean(iats) if len(iats) > 0 else 0,
            'std_flowiat': np.std(iats) if len(iats) > 0 else 0,
            'min_active': np.min(active_times),
            'mean_active': np.mean(active_times),
            'max_active': np.max(active_times),
            'std_active': np.std(active_times),
            'min_idle': np.min(idle_times),
            'mean_idle': np.mean(idle_times),
            'max_idle': np.max(idle_times),
            'std_idle': np.std(idle_times),
        }
        flow_features.append(features)
    
    print(f"Extracted features for {len(flow_features):,} flows")
    
    df = pd.DataFrame(flow_features)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    
    return df


def run_vpn_detection(df):
    """Run VPN detection on extracted features."""
    print("\nLoading model...")
    model = joblib.load('models/vpn_detector.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    
    # Ensure all features exist
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
    
    X = df[feature_names].replace([np.inf, -np.inf], 0).fillna(0)
    X_scaled = scaler.transform(X)
    
    probs = model.predict_proba(X_scaled)[:, 1]
    threshold = 0.55
    preds = (probs >= threshold).astype(int)
    
    vpn_count = sum(preds)
    total = len(preds)
    vpn_pct = 100 * vpn_count / total
    
    print("\n" + "=" * 50)
    print("VPN DETECTION RESULTS")
    print("=" * 50)
    print(f"Total flows analyzed: {total:,}")
    print(f"VPN flows detected:   {vpn_count:,} ({vpn_pct:.1f}%)")
    print(f"Non-VPN flows:        {total - vpn_count:,}")
    print(f"Average VPN prob:     {probs.mean():.1%}")
    print("=" * 50)
    
    if vpn_pct > 50:
        print("\n🔴 VERDICT: VPN DETECTED")
    elif vpn_pct > 25:
        print("\n🟡 VERDICT: LIKELY VPN")
    else:
        print("\n🟢 VERDICT: NO VPN DETECTED")
    
    return preds, probs


if __name__ == "__main__":
    pcap_file = r"C:\Users\ragur\Downloads\VPN Detection\data\processed\Wireshark files\summa1.pcapng"
    output_file = "data/processed/cic_features/summa1_cic.csv"
    
    df = extract_cic_features(pcap_file, output_file)
    run_vpn_detection(df)
