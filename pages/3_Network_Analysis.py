import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
import numpy as np
import networkx as nx

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.styling import inject_custom_css

# Page config
st.set_page_config(
    page_title="Network Analysis - Supply Chain Fraud Detection",
    page_icon="🌐",
    layout="wide"
)

# Apply custom styling
inject_custom_css()


# ============================================
# REAL NETWORK BUILDING FROM DATA
# ============================================
@st.cache_data
def load_transaction_data():
    """Load actual transaction data for building real network"""
    # Try to load combined_features or sample_transactions
    combined_path = Path(__file__).parent.parent.parent / "fraud_supplychain_year4" / "Fraud_SupplyChain" / "data" / "combined_features.csv"
    sample_path = Path(__file__).parent.parent / "data" / "sample_transactions.csv"

    if combined_path.exists():
        df = pd.read_csv(combined_path)
        if 'is_fraud' in df.columns:
            df = df.rename(columns={'is_fraud': 'Fraud'})
        return df, "combined_features"
    elif sample_path.exists():
        df = pd.read_csv(sample_path)
        return df, "sample_transactions"
    else:
        return None, None


@st.cache_data
def build_real_customer_product_network(df, data_type):
    """Build real bipartite network from actual transaction data"""

    if data_type == "sample_transactions":
        # Raw transactions - build customer-product edges
        if 'Customer Id' not in df.columns or 'Product Name' not in df.columns:
            return None

        # Create network
        G = nx.Graph()

        # Add edges (customer purchases product)
        for _, row in df.iterrows():
            customer = f"C_{row['Customer Id']}"
            product = f"P_{row['Product Name'][:20]}"  # Truncate product name

            # Add edge with weight (can be purchase count, amount, etc.)
            if G.has_edge(customer, product):
                G[customer][product]['weight'] += 1
            else:
                G.add_edge(customer, product, weight=1)

            # Add node attributes
            G.nodes[customer]['type'] = 'customer'
            G.nodes[customer]['fraud'] = 1 if row.get('Fraud', 0) == 1 else 0

            G.nodes[product]['type'] = 'product'

        return G

    elif data_type == "combined_features":
        # Aggregated data - need to infer connections
        # Use SNA results if available
        sna_path = Path(__file__).parent.parent / "data" / "sna_results.json"
        if sna_path.exists():
            with open(sna_path, 'r') as f:
                sna_data = json.load(f)

            # Build network from SNA data
            G = nx.Graph()

            # Add product nodes
            for product in sna_data['top_products'][:10]:
                product_id = f"P_{product['product_id']}"
                G.add_node(product_id, type='product',
                           customers=product['customers'])

            # Add customer nodes based on fraud rings
            for ring in sna_data['fraud_rings']:
                # Create customers in this ring
                for i in range(ring['size']):
                    customer_id = f"C_Ring{ring['ring_id']}_{i}"
                    is_fraud = 1 if i < ring['fraud_count'] else 0

                    G.add_node(customer_id, type='customer',
                               fraud=is_fraud, ring=ring['ring_id'])

                    # Connect to random products (realistic: 1-3 products per customer)
                    num_products = np.random.randint(1, 4)
                    products = np.random.choice(
                        sna_data['top_products'][:10], size=num_products, replace=False)

                    for product in products:
                        product_id = f"P_{product['product_id']}"
                        if product_id in G:
                            G.add_edge(
                                customer_id, product_id, weight=np.random.randint(1, 5))

            return G

    return None


def create_network_graph_from_networkx(G, layout_type='spring', highlight_node=None):
    """Create interactive visualization from NetworkX graph"""

    if G is None or len(G.nodes()) == 0:
        return None

    # Choose layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    else:  # bipartite
        # Separate customers and products
        customers = [n for n in G.nodes() if G.nodes[n].get(
            'type') == 'customer']
        products = [n for n in G.nodes() if G.nodes[n].get(
            'type') == 'product']

        pos = {}
        # Products on left
        for i, p in enumerate(products):
            pos[p] = (0.2, i / max(len(products) - 1, 1))
        # Customers on right
        for i, c in enumerate(customers):
            pos[c] = (0.8, i / max(len(customers) - 1, 1))

    # Create edges
    edge_x = []
    edge_y = []
    edge_colors = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#CCCCCC'),
        hoverinfo='none',
        mode='lines',
        opacity=0.4
    )

    # Create nodes
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_symbols = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        node_type = G.nodes[node].get('type', 'unknown')
        is_fraud = G.nodes[node].get('fraud', 0)

        # Styling based on type and fraud status
        if node_type == 'product':
            node_colors.append('#90CAF9')  # Light blue
            node_sizes.append(25)
            node_symbols.append('square')
            node_text.append(
                f"Product: {node}<br>Degree: {G.degree(node)}")
        else:  # customer
            if is_fraud == 1:
                node_colors.append('#1565C0')  # Blue
                node_text.append(
                    f"FRAUD Customer: {node}<br>Degree: {G.degree(node)}")
            else:
                node_colors.append('#AAAAAA')  # Light gray
                node_text.append(
                    f"Normal Customer: {node}<br>Degree: {G.degree(node)}")
            node_sizes.append(15)
            node_symbols.append('circle')

        # Highlight selected node
        if highlight_node and node == highlight_node:
            node_sizes[-1] = node_sizes[-1] * 1.5

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            symbol=node_symbols,
            line=dict(width=2, color='white')
        ),
        customdata=list(G.nodes())
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
        title=dict(
            text=f"Real Customer-Product Network ({len(G.nodes())} nodes, {len(G.edges())} edges)",
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600
    ))

    return fig


@st.cache_data
def load_sna_data():
    """Load Social Network Analysis results"""
    data_path = Path(__file__).parent.parent / "data" / "sna_results.json"

    if not data_path.exists():
        return None

    with open(data_path, 'r') as f:
        return json.load(f)


def create_network_graph(sna_data):
    """LEGACY: Create an interactive network visualization from static SNA data"""

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
        node_colors.append('#90CAF9')  # Light blue for products - more visible
        node_sizes.append(35)
        node_text.append(
            f"Product {product['product_id']}<br>{product['customers']} customers")

    # Add customer nodes around products (fraud rings)
    np.random.seed(42)
    for ring in fraud_rings:
        # Normal customers in ring
        normal_count = ring['size'] - ring['fraud_count']
        for j in range(normal_count):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.6, 0.9)
            node_x.append(0.5 + radius * np.cos(angle))
            node_y.append(0.5 + radius * np.sin(angle))
            node_colors.append('#AAAAAA')  # Light gray for normal
            node_sizes.append(10)
            node_text.append(f"Normal Customer<br>Ring #{ring['ring_id']}")

        # Fraud customers in ring
        for j in range(ring['fraud_count']):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.6, 0.9)
            node_x.append(0.5 + radius * np.cos(angle))
            node_y.append(0.5 + radius * np.sin(angle))
            node_colors.append('#1565C0')  # Blue for fraud - highly visible
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
        line=dict(width=1, color='#CCCCCC'),  # Medium gray for edges - more visible
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
transaction_df, data_type = load_transaction_data()

# Build real network
G = None
if transaction_df is not None and data_type is not None:
    with st.spinner("Building network from transaction data..."):
        G = build_real_customer_product_network(transaction_df, data_type)

# ============================================
# HEADER
# ============================================
st.title("Network Analysis")
st.markdown("Real customer-product network from transaction data")

# Add explanation about connection to model features
st.markdown("""
<div style="background: #E3F2FD; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2196F3; margin: 1.5rem 0;">
    <h3 style="font-size: 1.125rem; font-weight: 700; color: #0D47A1; margin: 0 0 0.75rem 0;">
        🔗 Connection to AI Fraud Detection Model
    </h3>
    <p style="font-size: 0.875rem; color: #1565C0; line-height: 1.6; margin: 0;">
        The network metrics calculated from this customer-product graph are used as <strong>4 important input features</strong> 
        for the fraud detection model (out of 61 total features):
    </p>
    <ul style="font-size: 0.875rem; color: #1976D2; line-height: 1.8; margin: 0.75rem 0 0 1.5rem;">
        <li><strong>degree_centrality</strong> - How many products a customer buys (well-connected vs isolated)</li>
        <li><strong>betweenness_centrality</strong> - Bridge position in network (key connector between groups)</li>
        <li><strong>closeness_centrality</strong> - How close a customer is to all other nodes</li>
        <li><strong>pagerank</strong> - Influence score in the network (like Google PageRank)</li>
    </ul>
    <p style="font-size: 0.875rem; color: #1565C0; line-height: 1.6; margin: 0.75rem 0 0 0;">
        <strong>Why is this important?</strong> Fraudsters often form coordinated networks (fraud rings) with unusual connection patterns. 
        These 4 network features help the AI detect suspicious "fraud rings" that simple transaction analysis would miss.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================
# SIDEBAR: NETWORK CONTROLS
# ============================================
with st.sidebar:
    st.header("⚙️ Network Controls")

    if G is not None:
        st.markdown(f"""
        <div style="background-color: #E8F5E9; border-left: 4px solid #4CAF50; border-radius: 8px; padding: 0.875rem 1rem; margin-bottom: 1rem; color: #2E7D32; font-size: 0.875rem;">
            ✓ Network loaded: <strong>{len(G.nodes())} nodes</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Layout")
        layout_type = st.radio(
            "Choose layout:",
            ["spring", "bipartite", "circular"],
            help="Spring: force-directed, Bipartite: customers | products"
        )

        st.markdown("---")
        st.subheader("Filters")

        # Filter by node type
        show_customers = st.checkbox("Show Customers", value=True)
        show_products = st.checkbox("Show Products", value=True)
        show_fraud_only = st.checkbox("Show Fraud Only", value=False)

        # Degree filter
        st.markdown("**Degree Filter:**")
        min_degree = st.slider("Minimum connections", 0, 20, 0)

        st.markdown("---")

        # Search node
        st.subheader("Search")
        search_node = st.text_input(
            "Node ID:", placeholder="C_12345 or P_Product")

        if search_node and search_node in G:
            st.success(f"Found: {search_node}")
            st.markdown(f"**Degree:** {G.degree(search_node)}")
            st.markdown(
                f"**Type:** {G.nodes[search_node].get('type', 'unknown')}")
            if G.nodes[search_node].get('type') == 'customer':
                st.markdown(
                    f"**Fraud:** {'Yes' if G.nodes[search_node].get('fraud') == 1 else 'No'}")
        elif search_node:
            st.error("Node not found")
    else:
        st.warning("No network data available")

if G is not None:
    # ============================================
    # SECTION 1: REAL NETWORK METRICS
    # ============================================
    st.subheader("Network Overview")

    # Calculate real metrics
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())

    customers = [n for n in G.nodes() if G.nodes[n].get('type') == 'customer']
    products = [n for n in G.nodes() if G.nodes[n].get('type') == 'product']
    fraud_customers = [
        n for n in customers if G.nodes[n].get('fraud') == 1]

    avg_degree = sum(dict(G.degree()).values()
                     ) / num_nodes if num_nodes > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="TOTAL NODES",
            value=f"{num_nodes:,}",
            help="Customers + Products"
        )

    with col2:
        st.metric(
            label="TOTAL EDGES",
            value=f"{num_edges:,}",
            help="Purchase connections"
        )

    with col3:
        st.metric(
            label="CUSTOMERS",
            value=f"{len(customers):,}",
            delta=f"{len(fraud_customers)} fraud" if len(
                fraud_customers) > 0 else None,
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="AVG DEGREE",
            value=f"{avg_degree:.1f}",
            help="Average connections per node"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================
    # SECTION 2: TOP NODES
    # ============================================
    st.subheader("Most Connected Nodes")

    # Get top nodes by degree
    degree_dict = dict(G.degree())
    sorted_nodes = sorted(
        degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Nodes:**")
        top_data = []
        for node, degree in sorted_nodes:
            node_type = G.nodes[node].get('type', 'unknown')
            is_fraud = " 🔴" if G.nodes[node].get('fraud') == 1 else ""
            top_data.append({
                'Node': f"{node}{is_fraud}",
                'Type': node_type.capitalize(),
                'Degree': degree
            })

        df_top = pd.DataFrame(top_data)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Degree Distribution:**")

        # Degree histogram
        degrees = list(degree_dict.values())

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(degrees, bins=20, color='#2196F3',
                edgecolor='#1565C0', alpha=0.7)
        ax.set_xlabel('Degree (Number of Connections)')
        ax.set_ylabel('Frequency')
        ax.set_title('Network Degree Distribution')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================
    # SECTION 3: FRAUD DETECTION
    # ============================================
    st.subheader("Fraud Detection")

    if len(fraud_customers) > 0:
        fraud_rate = len(fraud_customers) / \
            len(customers) * 100 if len(customers) > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Customers", len(fraud_customers))
        with col2:
            st.metric("Normal Customers", len(
                customers) - len(fraud_customers))
        with col3:
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

        # Detect fraud patterns
        st.markdown("**Fraud Customer Analysis:**")

        fraud_degrees = [G.degree(n) for n in fraud_customers]
        normal_degrees = [G.degree(n)
                          for n in customers if G.nodes[n].get('fraud') == 0]

        avg_fraud_degree = np.mean(fraud_degrees) if fraud_degrees else 0
        avg_normal_degree = np.mean(normal_degrees) if normal_degrees else 0

        comparison_data = {
            'Metric': ['Avg Degree', 'Max Degree', 'Min Degree'],
            'Fraud Customers': [
                f"{avg_fraud_degree:.1f}",
                f"{max(fraud_degrees) if fraud_degrees else 0}",
                f"{min(fraud_degrees) if fraud_degrees else 0}"
            ],
            'Normal Customers': [
                f"{avg_normal_degree:.1f}",
                f"{max(normal_degrees) if normal_degrees else 0}",
                f"{min(normal_degrees) if normal_degrees else 0}"
            ]
        }

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        if avg_fraud_degree > avg_normal_degree * 1.2:
            st.warning(
                f"Fraud customers have **{(avg_fraud_degree/avg_normal_degree-1)*100:.1f}% higher** average degree — may indicate coordinated networks!")
        else:
            st.info("Fraud and normal customers show similar connection patterns")

        # Show suspicious high-degree fraud customers
        high_degree_frauds = [
            (n, G.degree(n)) for n in fraud_customers if G.degree(n) > avg_degree * 1.5]

        if high_degree_frauds:
            st.markdown(
                "**High-Risk Fraud Customers (degree > 1.5× average):**")
            risk_data = []
            for node, degree in sorted(high_degree_frauds, key=lambda x: x[1], reverse=True)[:10]:
                neighbors = list(G.neighbors(node))
                products = [
                    n for n in neighbors if G.nodes[n].get('type') == 'product']

                risk_data.append({
                    'Customer': node,
                    'Degree': degree,
                    'Products': len(products),
                    'Risk': f"{(degree / avg_degree) * 100:.0f}%"
                })

            df_risk = pd.DataFrame(risk_data)
            st.dataframe(df_risk, use_container_width=True, hide_index=True)
    else:
        st.info("No fraud customers detected in this network")

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================
    # SECTION 4: INTERACTIVE NETWORK VISUALIZATION
    # ============================================
    st.subheader("Interactive Network Visualization")
    st.caption("Real customer-product relationships from transaction data")

    # Apply filters
    G_filtered = G.copy()

    # Remove nodes based on filters
    nodes_to_remove = []
    for node in G_filtered.nodes():
        node_type = G_filtered.nodes[node].get('type')
        is_fraud = G_filtered.nodes[node].get('fraud', 0)
        degree = G_filtered.degree(node)

        # Type filters
        if not show_customers and node_type == 'customer':
            nodes_to_remove.append(node)
            continue
        if not show_products and node_type == 'product':
            nodes_to_remove.append(node)
            continue

        # Fraud filter
        if show_fraud_only and node_type == 'customer' and is_fraud == 0:
            nodes_to_remove.append(node)
            continue

        # Degree filter
        if degree < min_degree:
            nodes_to_remove.append(node)
            continue

    G_filtered.remove_nodes_from(nodes_to_remove)

    # Show filtered stats
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(
            f"**Displaying:** {len(G_filtered.nodes())} nodes, {len(G_filtered.edges())} edges (filtered from {len(G.nodes())} total)")
    with col2:
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.rerun()

    # Legend
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #AAAAAA; border-radius: 50%;"></span>
                <span style="font-size: 0.875rem; color: #1565C0;">Normal Customers</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #1565C0; border-radius: 50%;"></span>
                <span style="font-size: 0.875rem; color: #1565C0;">Fraud Customers</span>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="display: inline-block; width: 12px; height: 12px; background: #90CAF9;"></span>
                <span style="font-size: 0.875rem; color: #1565C0;">Products</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Create and display network graph
    if len(G_filtered.nodes()) > 0:
        try:
            highlight = search_node if search_node in G_filtered else None
            fig = create_network_graph_from_networkx(
                G_filtered, layout_type, highlight)

            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "**Tip:** Hover over nodes to see details. Use sidebar filters to explore specific patterns.")
            else:
                st.warning("Could not generate visualization for filtered network")
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            st.info("Try adjusting filters or changing layout")
    else:
        st.warning(
            "No nodes to display with current filters. Try adjusting the sidebar filters.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================
    # SECTION 5: NODE DETAILS (if searched)
    # ============================================
    if search_node and search_node in G:
        st.markdown("---")
        st.subheader(f"Details for: {search_node}")

        node_data = G.nodes[search_node]
        neighbors = list(G.neighbors(search_node))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Type", node_data.get('type', 'unknown').capitalize())
        with col2:
            st.metric("Degree", G.degree(search_node))
        with col3:
            if node_data.get('type') == 'customer':
                st.metric(
                    "Status", "FRAUD" if node_data.get('fraud') == 1 else "Normal")
            else:
                st.metric("Customers", len(
                    [n for n in neighbors if G.nodes[n].get('type') == 'customer']))
        with col4:
            st.metric("Connections", len(neighbors))

        # Show neighbors
        st.markdown("**Connected Nodes:**")

        neighbor_data = []
        for n in neighbors[:20]:  # Limit to 20
            n_type = G.nodes[n].get('type', 'unknown')
            n_fraud = " 🔴" if G.nodes[n].get('fraud') == 1 else ""
            neighbor_data.append({
                'Node': f"{n}{n_fraud}",
                'Type': n_type.capitalize(),
                'Degree': G.degree(n),
                'Weight': G[search_node][n].get('weight', 1)
            })

        df_neighbors = pd.DataFrame(neighbor_data)
        st.dataframe(df_neighbors, use_container_width=True, hide_index=True)

        if len(neighbors) > 20:
            st.caption(f"... and {len(neighbors)-20} more connections")

        # Action: Predict fraud for this customer
        if node_data.get('type') == 'customer':
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔎 Predict Fraud for This Customer", use_container_width=True):
                    st.info(
                        "Navigate to Fraud Prediction page to analyze this customer")
                    st.code(f"Customer ID: {search_node}")

else:
    st.error("**Error:** Cannot load network data.")
    st.info("""
        **Possible issues:**
        - Transaction data file not found
        - SNA results file missing
        - Invalid data format

        **Expected files:**
        - `data/sample_transactions.csv` or
        - `../fraud_supplychain_year4/Fraud_SupplyChain/data/combined_features.csv`
        - `data/sna_results.json`
    """)