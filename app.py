"""
VPN Detection System - Streamlit Dashboard
Real-time VPN traffic detection using trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import tempfile
import os

# Import scapy for pcapng processing
try:
    from scapy.all import rdpcap, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="VPN Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for threshold
if 'vpn_threshold' not in st.session_state:
    st.session_state.vpn_threshold = 0.55


@st.cache_resource
def load_model():
    """Load trained VPN detector model."""
    model_path = Path(__file__).parent / "models"
    try:
        model = joblib.load(model_path / "vpn_detector.joblib")
        scaler = joblib.load(model_path / "scaler.joblib")
        feature_names = joblib.load(model_path / "feature_names.joblib")
        metrics = joblib.load(model_path / "training_metrics.joblib")
        return model, scaler, feature_names, metrics, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, False


def process_pcapng(file_bytes, feature_names):
    """
    Process pcapng file and extract CIC-compatible features.
    """
    if not SCAPY_AVAILABLE:
        st.error("Scapy not installed. Run: pip install scapy")
        return None
    
    from collections import defaultdict
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcapng') as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Read packets
        st.info("📦 Loading packets from pcapng...")
        packets = rdpcap(tmp_path)
        st.success(f"✅ Loaded {len(packets)} packets")
        
        # Group packets into flows
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
        
        st.info(f"📊 Found {len(flows)} network flows")
        
        # Extract features for each flow
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
            fwd_pkts = [p for p in pkts if p['src_ip'] == first_src]
            bwd_pkts = [p for p in pkts if p['src_ip'] != first_src]
            
            fwd_times = [p['time'] for p in fwd_pkts]
            fiat = np.diff(fwd_times) * 1000000 if len(fwd_times) > 1 else [0]
            
            bwd_times = [p['time'] for p in bwd_pkts]
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
        
        st.success(f"✅ Extracted features for {len(flow_features)} flows")
        
        if not flow_features:
            return None
        
        df = pd.DataFrame(flow_features)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Ensure all required features exist
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        
        return df[feature_names]
        
    finally:
        os.unlink(tmp_path)


def create_gauge_chart(value, title):
    """Create a gauge chart for VPN probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def create_pie_chart(vpn_count, non_vpn_count):
    """Create pie chart for VPN vs Non-VPN distribution."""
    fig = px.pie(
        values=[vpn_count, non_vpn_count],
        names=['VPN Traffic', 'Normal Traffic'],
        title="Traffic Classification",
        color_discrete_sequence=['#FF6B6B', '#90EE90'],
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def extract_features_from_csv(df, feature_names):
    """Extract flow features from raw packet CSV."""
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Debug: Show columns
    st.write(f"Columns in file: {list(df.columns)[:10]}...")  # Show first 10
    
    # Check if this is already a feature file (has all required features)
    missing_features = [f for f in feature_names if f not in df.columns]
    if len(missing_features) == 0:
        st.success(f"✅ Found all {len(feature_names)} features in the file")
        return df[feature_names]
    
    # Check if this is a raw packet file (has ip.src, ip.dst)
    has_ip_cols = 'ip.src' in df.columns and 'ip.dst' in df.columns
    
    if not has_ip_cols:
        st.error(f"Missing features: {missing_features[:5]}...")
        st.error("CSV must have either all flow features OR 'ip.src'/'ip.dst' columns for packet data")
        return None
    
    # Otherwise, try to aggregate packets into flows
    st.info("Extracting flow features from raw packet data...")
    
    # Ensure numeric columns
    for col in ['frame.time_relative', 'frame.time_delta', 'frame.len']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create flow key function
    def create_flow_key(row):
        src = str(row.get('ip.src', ''))
        dst = str(row.get('ip.dst', ''))
        return tuple(sorted([src, dst]))
    
    df['flow_key'] = df.apply(create_flow_key, axis=1)
    
    flows = []
    for flow_key, group in df.groupby('flow_key'):
        if len(group) < 2:
            continue
        
        group = group.sort_values('frame.time_relative')
        times = group['frame.time_relative'].values
        iats = np.diff(times) * 1000
        
        first_src = group['ip.src'].iloc[0]
        forward_mask = group['ip.src'] == first_src
        backward_mask = ~forward_mask
        
        forward_times = group.loc[forward_mask, 'frame.time_relative'].values
        fiat = np.diff(forward_times) * 1000 if len(forward_times) > 1 else np.array([0])
        
        backward_times = group.loc[backward_mask, 'frame.time_relative'].values
        biat = np.diff(backward_times) * 1000 if len(backward_times) > 1 else np.array([0])
        
        duration = (times[-1] - times[0]) * 1000000
        total_bytes = group['frame.len'].sum() if 'frame.len' in group.columns else 0
        total_packets = len(group)
        duration_sec = max(duration / 1000000, 0.001)
        
        active_times = iats[iats < 1000] if len(iats) > 0 else np.array([0])
        idle_times = iats[iats >= 1000] if len(iats) > 0 else np.array([0])
        
        features = {
            'duration': duration,
            'total_biat': biat.sum() if len(biat) > 0 else 0,
            'mean_flowiat': iats.mean() if len(iats) > 0 else 0,
            'max_biat': biat.max() if len(biat) > 0 else 0,
            'min_idle': idle_times.min() if len(idle_times) > 0 else 0,
            'max_fiat': fiat.max() if len(fiat) > 0 else 0,
            'mean_fiat': fiat.mean() if len(fiat) > 0 else 0,
            'total_fiat': fiat.sum() if len(fiat) > 0 else 0,
            'flowPktsPerSecond': total_packets / duration_sec,
            'max_active': active_times.max() if len(active_times) > 0 else 0,
            'std_active': active_times.std() if len(active_times) > 0 else 0,
            'std_flowiat': iats.std() if len(iats) > 0 else 0,
            'mean_biat': biat.mean() if len(biat) > 0 else 0,
            'flowBytesPerSecond': total_bytes / duration_sec,
            'min_flowiat': iats.min() if len(iats) > 0 else 0,
            'max_flowiat': iats.max() if len(iats) > 0 else 0,
            'min_active': active_times.min() if len(active_times) > 0 else 0,
            'mean_active': active_times.mean() if len(active_times) > 0 else 0,
            'max_idle': idle_times.max() if len(idle_times) > 0 else 0,
            'mean_idle': idle_times.mean() if len(idle_times) > 0 else 0,
            'min_fiat': fiat.min() if len(fiat) > 0 else 0,
            'std_idle': idle_times.std() if len(idle_times) > 0 else 0,
            'min_biat': biat.min() if len(biat) > 0 else 0,
        }
        flows.append(features)
    
    if not flows:
        return None
    
    return pd.DataFrame(flows)[feature_names]


def main():
    st.markdown('<h1 class="main-header">🔒 VPN Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Real-time VPN traffic detection using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names, metrics, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Model not loaded. Run `python save_model.py` first.")
        return
    
    # Main content
    st.markdown("### 📁 Upload Network Traffic Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV or PCAPNG file",
            type=['csv', 'pcapng', 'pcap'],
            help="Upload CIC features CSV or raw Wireshark capture (pcapng/pcap)"
        )
    
    with col2:
        use_sample = st.checkbox("Use sample data", value=False)
        if use_sample:
            feature_dir = Path("data/processed/cic_features")
            if feature_dir.exists():
                sample_files = list(feature_dir.glob("*_cic.csv"))
                if sample_files:
                    selected_sample = st.selectbox(
                        "Select sample",
                        options=sample_files,
                        format_func=lambda x: x.name
                    )
    
    # Process data
    if uploaded_file is not None or (use_sample and 'selected_sample' in dir() and selected_sample):
        try:
            with st.spinner("🔄 Analyzing traffic..."):
                if uploaded_file is not None:
                    file_name = uploaded_file.name.lower()
                    
                    if file_name.endswith('.pcapng') or file_name.endswith('.pcap'):
                        # Process PCAPNG file
                        st.info(f"📁 Processing {uploaded_file.name}...")
                        X = process_pcapng(uploaded_file.read(), feature_names)
                    else:
                        # Process CSV file
                        df = pd.read_csv(uploaded_file)
                        X = extract_features_from_csv(df, feature_names)
                else:
                    df = pd.read_csv(selected_sample)
                    X = extract_features_from_csv(df, feature_names)
                
                if X is None or len(X) == 0:
                    st.error("Could not extract features from the data")
                    return
                
                # Handle missing/inf values
                X = X.replace([np.inf, -np.inf], 0).fillna(0)
                
                # Scale and predict using threshold from session state
                X_scaled = scaler.transform(X)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                predictions = (probabilities >= st.session_state.vpn_threshold).astype(int)
            
            # Results
            st.markdown("---")
            st.markdown("### 📊 Detection Results")
            
            total_flows = len(predictions)
            vpn_flows = sum(predictions == 1)
            non_vpn_flows = total_flows - vpn_flows
            avg_prob = probabilities.mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Flows", f"{total_flows:,}")
            col2.metric("VPN Detected", f"{vpn_flows:,}", f"{vpn_flows/total_flows:.1%}")
            col3.metric("Normal Traffic", f"{non_vpn_flows:,}")
            col4.metric("Avg VPN Prob", f"{avg_prob:.1%}")
            
            # Charts
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.plotly_chart(create_pie_chart(vpn_flows, non_vpn_flows), use_container_width=True)
            
            with chart_col2:
                fig = px.histogram(x=probabilities, nbins=30, title="VPN Probability Distribution")
                fig.add_vline(x=st.session_state.vpn_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Gauge charts
            st.markdown("### 🎯 Risk Assessment")
            g1, g2, g3 = st.columns(3)
            with g1:
                st.plotly_chart(create_gauge_chart(avg_prob, "Avg VPN Probability"), use_container_width=True)
            with g2:
                st.plotly_chart(create_gauge_chart(vpn_flows/total_flows, "VPN Traffic Ratio"), use_container_width=True)
            with g3:
                high_risk = sum(probabilities > 0.8) / total_flows
                st.plotly_chart(create_gauge_chart(high_risk, "High-Risk Flows"), use_container_width=True)
            
            # Results table
            st.markdown("---")
            st.markdown("### 📋 Flow Details")
            
            results_df = X.copy()
            results_df['VPN_Probability'] = probabilities
            results_df['Prediction'] = ['VPN' if p == 1 else 'Non-VPN' for p in predictions]
            results_df = results_df.sort_values('VPN_Probability', ascending=False)
            
            st.dataframe(results_df.head(100), use_container_width=True)
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "vpn_results.csv", "text/csv")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👆 Upload a CSV file or select sample data to begin")
    
    # Footer with threshold and model info
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.vpn_threshold = st.slider(
            "Threshold", 0.1, 0.9, st.session_state.vpn_threshold, 0.05
        )
    with col2:
        st.markdown(
            f"<p style='text-align: right; color: #888; font-size: 0.85rem; margin-top: 1rem;'>"
            f"📈 Accuracy {metrics['accuracy']:.1%} | F1 {metrics['f1_score']:.1%} | "
            f"Precision {metrics['precision']:.1%} | Recall {metrics['recall']:.1%}</p>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
