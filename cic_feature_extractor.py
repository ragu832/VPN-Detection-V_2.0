"""
CIC-style Feature Extractor for PCAP/PCAPNG files.
Extracts flow-level features compatible with CIC training data.
"""

from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def extract_cic_features(pcap_path: str, output_path: str = None):
    """
    Extract CIC-compatible flow features from a pcap/pcapng file.
    
    Args:
        pcap_path: Path to the pcap/pcapng file
        output_path: Path to save the output CSV (optional)
    
    Returns:
        DataFrame with flow features
    """
    print(f"Loading: {pcap_path}")
    packets = rdpcap(pcap_path)
    print(f"Loaded {len(packets)} packets")
    
    # Group packets into flows (by 5-tuple: src_ip, dst_ip, src_port, dst_port, protocol)
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
            
            # Create bidirectional flow key
            flow_key = tuple(sorted([
                (src_ip, src_port),
                (dst_ip, dst_port)
            ])) + (proto,)
            
            flows[flow_key].append({
                'time': float(pkt.time),
                'size': len(pkt),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'proto': proto
            })
    
    print(f"Found {len(flows)} flows")
    
    # Extract features for each flow
    flow_features = []
    
    for flow_key, pkts in flows.items():
        if len(pkts) < 2:
            continue
        
        # Sort by time
        pkts = sorted(pkts, key=lambda x: x['time'])
        
        # Basic stats
        times = [p['time'] for p in pkts]
        sizes = [p['size'] for p in pkts]
        
        duration = (times[-1] - times[0]) * 1000000  # microseconds
        if duration == 0:
            duration = 1
        
        # Inter-arrival times (in microseconds)
        iats = np.diff(times) * 1000000
        
        # Forward and backward packets
        first_src = pkts[0]['src_ip']
        fwd_pkts = [p for p in pkts if p['src_ip'] == first_src]
        bwd_pkts = [p for p in pkts if p['src_ip'] != first_src]
        
        # Forward IAT
        fwd_times = [p['time'] for p in fwd_pkts]
        fiat = np.diff(fwd_times) * 1000000 if len(fwd_times) > 1 else [0]
        
        # Backward IAT
        bwd_times = [p['time'] for p in bwd_pkts]
        biat = np.diff(bwd_times) * 1000000 if len(bwd_times) > 1 else [0]
        
        # Active/Idle times (threshold: 1 second = 1000000 microseconds)
        active_threshold = 1000000
        active_times = iats[iats < active_threshold] if len(iats) > 0 else [0]
        idle_times = iats[iats >= active_threshold] if len(iats) > 0 else [0]
        
        if len(active_times) == 0:
            active_times = [0]
        if len(idle_times) == 0:
            idle_times = [0]
        
        # Calculate CIC-compatible features
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
    
    print(f"Extracted features for {len(flow_features)} flows")
    
    # Create DataFrame
    df = pd.DataFrame(flow_features)
    
    # Handle inf and nan
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    
    return df


def main():
    import sys
    
    # Default paths
    pcap_dir = Path("data/processed/Wireshark files")
    output_dir = Path("data/processed/cic_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all pcapng files
    pcap_files = list(pcap_dir.glob("*.pcapng"))
    
    if not pcap_files:
        print(f"No pcapng files found in {pcap_dir}")
        return
    
    print(f"Found {len(pcap_files)} pcapng files")
    print("=" * 60)
    
    for pcap_file in pcap_files:
        output_file = output_dir / f"{pcap_file.stem}_cic.csv"
        try:
            df = extract_cic_features(str(pcap_file), str(output_file))
            print(f"  Features: {len(df.columns)}, Flows: {len(df)}")
        except Exception as e:
            print(f"  Error: {e}")
        print("-" * 60)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
