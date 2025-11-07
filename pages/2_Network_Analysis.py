import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.styling import inject_custom_css

# Page config
st.set_page_config(
    page_title="Network Analysis - Supply Chain Fraud Detection",
    page_icon="🕸️",
    layout="wide"
)

# Apply custom styling
inject_custom_css()

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_sna_data():
    """Load Social Network Analysis results"""
    data_path = Path(__file__).parent.parent / "data" / "sna_results.json"
    
    if not data_path.exists():
        return None
    
    with open(data_path, 'r') as f:
        return json.load(f)

def create_network_graph(sna_data):
    """Create an interactive network visualization"""
    
    # Get top 5 products and create sample network
    top_products = sna_data['top_products'][:5]
    fraud_rings = sna_data['fraud_rings']
    
    # Create nodes
    nodes = []
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    # Add product nodes in center
    for i, product in enumerate(top_products):
        angle = 2 * np.pi * i / len(top_products)
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        node_x.append(x)
        node_y.append(y)
        node_colors.append('#4A90E2')  # Blue for products - more visible
        node_sizes.append(35)
        node_text.append(f"Product {product['product_id']}<br>{product['customers']} customers")
    
    # Add customer nodes around products (fraud rings)
    np.random.seed(42)
    for ring in fraud_rings:
        # Normal customers in ring
        normal_count = ring['size'] - ring['fraud_count']
        for j in range(normal_count):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.6, 0.9)
            node_x.append(0.5 + radius * np.cos(angle))
            node_y.append(0.5 + radius * np.sin(angle))
            node_colors.append('#7C7C7C')  # Medium gray for normal - clearer
            node_sizes.append(10)
            node_text.append(f"Normal Customer<br>Ring #{ring['ring_id']}")
        
        # Fraud customers in ring
        for j in range(ring['fraud_count']):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0.6, 0.9)
            node_x.append(0.5 + radius * np.cos(angle))
            node_y.append(0.5 + radius * np.sin(angle))
            node_colors.append('#E74C3C')  # Red for fraud - highly visible
            node_sizes.append(13)
            node_text.append(f"Fraud Customer<br>Ring #{ring['ring_id']}")
    
    # Create edges (connect ALL customers to products)
    edge_x = []
    edge_y = []
    
    # Connect all customers to products
    num_products = len(top_products)
    num_customers = len(node_x) - num_products
    
    # Connect EVERY customer to at least one product
    for i in range(num_products, len(node_x)):
        # Connect to 1-3 random products (realistic: customers buy multiple products)
        num_connections = np.random.randint(1, 4)
        for _ in range(num_connections):
            product_idx = np.random.randint(0, num_products)
            edge_x.extend([node_x[i], node_x[product_idx], None])
            edge_y.extend([node_y[i], node_y[product_idx], None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#9E9E9E'),  # Medium gray for edges - more visible
        hoverinfo='none',
        mode='lines',
        opacity=0.5
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white',
                       paper_bgcolor='white',
                       height=500
                   ))
    
    return fig

# Load data
sna_data = load_sna_data()

# ============================================
# HEADER
# ============================================
st.title("Network Analysis")
st.markdown("Social Network Analysis (SNA) results and fraud ring detection")
st.divider()

if sna_data:
    # ============================================
    # SECTION 1: NETWORK METRICS
    # ============================================
    st.subheader("Network Overview")
    
    metrics = sna_data['network_metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="NODES",
            value=f"{metrics['total_nodes']:,}",
            help="Total customers + products"
        )

    with col2:
        st.metric(
            label="EDGES", 
            value=f"{metrics['total_edges']:,}",
            help="Purchase connections"
        )

    with col3:
        st.metric(
            label="COMMUNITIES",
            value=f"{sna_data['communities']['total']}",
            help="Detected groups"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # SECTION 2: FRAUD RINGS DETECTION
    # ============================================
    st.subheader("Fraud Rings Detection")
    
    fraud_rings = sna_data['fraud_rings']
    
    # Convert to DataFrame
    df_rings = pd.DataFrame([
        {
            'Community': f"Ring #{ring['ring_id']}",
            'Members': ring['size'],
            'Fraud Count': ring['fraud_count'],
            'Fraud Rate': f"{ring['fraud_rate']*100:.1f}%"
        }
        for ring in fraud_rings
    ])
    
    # Display table
    st.dataframe(
        df_rings,
        use_container_width=True,
        hide_index=True
    )
    
    # Calculate insights
    total_frauds = sum(ring['fraud_count'] for ring in fraud_rings)
    total_members = sum(ring['size'] for ring in fraud_rings)
    avg_fraud_rate = (total_frauds / total_members * 100) if total_members > 0 else 0
    
    # Key insights
    st.info(f"""
    **Key Insights:**
    
    - These **{len(fraud_rings)} communities** show fraud rates between **75-87%**
    - Average fraud rate is **{avg_fraud_rate:.1f}%** — indicating highly coordinated fraud networks
    - Suggests organized fraud rings with systematic patterns targeting specific products
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # SECTION 3: CENTRALITY COMPARISON
    # ============================================
    st.subheader("Centrality Measures Comparison")
    st.caption("Comparing network centrality metrics between fraudulent and normal customers")
    
    centrality_data = sna_data['centrality_comparison']
    
    col1, col2, col3 = st.columns(3)
    
    # Betweenness Centrality
    with col1:
        st.markdown("##### Betweenness Centrality")
        st.caption("Bridge role in network")
        
        betweenness_fraud = centrality_data['betweenness']['fraud_avg']
        betweenness_normal = centrality_data['betweenness']['normal_avg']
        betweenness_increase = centrality_data['betweenness']['fraud_increase_pct']
        
        st.metric("Fraud", f"{betweenness_fraud:.8f}", f"+{betweenness_increase:.1f}%")
        st.metric("Normal", f"{betweenness_normal:.8f}")
        
        st.caption(f"→ {centrality_data['betweenness']['interpretation']}")
    
    # Degree Centrality
    with col2:
        st.markdown("##### Degree Centrality")
        st.caption("Number of products purchased")
        
        degree_fraud = centrality_data['degree']['fraud_avg']
        degree_normal = centrality_data['degree']['normal_avg']
        degree_increase = centrality_data['degree']['fraud_increase_pct']
        
        st.metric("Fraud", f"{degree_fraud:.2f}", f"+{degree_increase:.1f}%")
        st.metric("Normal", f"{degree_normal:.2f}")
        
        st.caption(f"→ {centrality_data['degree']['interpretation']}")
    
    # Closeness Centrality
    with col3:
        st.markdown("##### Closeness Centrality")
        st.caption("Distance to all nodes")
        
        closeness_fraud = centrality_data['closeness']['fraud_avg']
        closeness_normal = centrality_data['closeness']['normal_avg']
        closeness_increase = centrality_data['closeness']['fraud_increase_pct']
        
        st.metric("Fraud", f"{closeness_fraud:.3f}", f"+{closeness_increase:.1f}%")
        st.metric("Normal", f"{closeness_normal:.3f}")
        
        st.caption(f"→ {centrality_data['closeness']['interpretation']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # SECTION 4: NETWORK VISUALIZATION
    # ============================================
    st.subheader("Network Visualization")
    st.caption("Interactive network graph showing customer-product relationships (sample)")
    
    # Legend
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #7C7C7C; border-radius: 50%;"></span>
            <span style="font-size: 0.875rem; color: #5A5A5A;">Normal Customers</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="display: inline-block; width: 12px; height: 12px; background: #E74C3C; border-radius: 50%;"></span>
            <span style="font-size: 0.875rem; color: #0B0B0B;">Fraud Customers</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #4A90E2; border-radius: 50%;"></span>
            <span style="font-size: 0.875rem; color: #5A5A5A;">Products</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create and display network graph
    try:
        fig = create_network_graph(sna_data)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 Hover over nodes to see details. Graph shows top 5 products and sample customers from fraud rings.")
    except Exception as e:
        st.warning(f"Could not generate network visualization: {str(e)}")

else:
    st.error("**Error:** Cannot load SNA data. Please check the file data/sna_results.json")
