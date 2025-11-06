"""
Page 2 - Phân tích Mạng (SNA)
Network analysis với visualizations sử dụng Chart.js và Cytoscape
"""

import streamlit as st
import streamlit.components.v1 as components
import json

# Page configuration
st.set_page_config(
    page_title="Phân tích Mạng",
    page_icon="🕸️",
    layout="wide"
)

# Header
st.title("🕸️ Phân tích Mạng lưới (Social Network Analysis)")
st.markdown("Phân tích cấu trúc mạng khách hàng-sản phẩm và phát hiện fraud rings")

# Load SNA results
@st.cache_data
def load_sna_results():
    try:
        with open('data/sna_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ File sna_results.json không tìm thấy!")
        return None

sna_data = load_sna_results()

if sna_data:
    # Section 1: Key Metrics
    st.markdown("---")
    st.subheader("📊 Thống kê Mạng lưới")
    
    metrics = sna_data['network_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tổng số Nodes",
            f"{metrics['total_nodes']:,}",
            help="Tổng số thực thể trong mạng (khách hàng + sản phẩm)"
        )
        st.caption(f"👥 {metrics['customer_nodes']:,} khách hàng")
        st.caption(f"📦 {metrics['product_nodes']:,} sản phẩm")
    
    with col2:
        st.metric(
            "Tổng số Edges",
            f"{metrics['total_edges']:,}",
            help="Tổng số mối quan hệ (giao dịch) giữa khách hàng và sản phẩm"
        )
        st.caption(f"📈 Avg degree: {metrics['avg_customer_degree']:.2f}")
    
    with col3:
        st.metric(
            "Communities",
            f"{metrics['components']}",
            help="Số connected components trong network"
        )
        st.caption(f"🔷 Largest: {metrics['largest_component']:,}")
    
    with col4:
        st.metric(
            "Density",
            f"{metrics['density']:.6f}",
            help="Mật độ kết nối trong network (sparse network)"
        )
        st.caption("📉 Sparse network")
    
    # Section 2: Fraud Analysis Comparison
    st.markdown("---")
    st.subheader("🎯 So sánh Fraud vs Normal")
    
    fraud_analysis = sna_data['fraud_analysis']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Degree (Fraud)",
            f"{fraud_analysis['fraud_avg_degree']:.2f}",
            delta=f"+{fraud_analysis['fraud_degree_multiplier']:.2f}x",
            help="Trung bình số sản phẩm khác nhau mà fraudsters mua"
        )
    
    with col2:
        st.metric(
            "Avg Degree (Normal)",
            f"{fraud_analysis['normal_avg_degree']:.2f}",
            help="Trung bình số sản phẩm khác nhau mà khách hàng thường mua"
        )
    
    with col3:
        st.metric(
            "Fraud Multiplier",
            f"{fraud_analysis['fraud_degree_multiplier']:.2f}x",
            help="Fraudsters mua nhiều sản phẩm hơn bao nhiêu lần so với normal"
        )
    
    # Section 3: Charts
    st.markdown("---")
    st.subheader("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 So sánh Centrality")
        
        # Chart.js Bar Chart for Centrality Comparison
        centrality_data = sna_data['centrality_comparison']
        
        chart_html = f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <canvas id="centralityChart"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx = document.getElementById('centralityChart').getContext('2d');
            const chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Degree', 'Betweenness', 'Closeness', 'PageRank'],
                    datasets: [
                        {{
                            label: 'Fraud',
                            data: [{centrality_data['degree']['fraud_avg']}, 
                                   {centrality_data['betweenness']['fraud_avg']}, 
                                   {centrality_data['closeness']['fraud_avg']}, 
                                   {centrality_data['pagerank']['fraud_avg']}],
                            backgroundColor: 'rgba(239, 83, 80, 0.7)',
                            borderColor: 'rgba(239, 83, 80, 1)',
                            borderWidth: 2
                        }},
                        {{
                            label: 'Normal',
                            data: [{centrality_data['degree']['normal_avg']}, 
                                   {centrality_data['betweenness']['normal_avg']}, 
                                   {centrality_data['closeness']['normal_avg']}, 
                                   {centrality_data['pagerank']['normal_avg']}],
                            backgroundColor: 'rgba(102, 187, 106, 0.7)',
                            borderColor: 'rgba(102, 187, 106, 1)',
                            borderWidth: 2
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Centrality Metrics: Fraud vs Normal',
                            font: {{
                                size: 16,
                                weight: 'bold'
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Average Value'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """
        
        components.html(chart_html, height=400)
    
    with col2:
        st.markdown("#### 📊 Top 5 Sản phẩm Phổ biến")
        
        # Chart.js Doughnut Chart for Top Products
        top_products = sna_data['top_products']
        
        labels = [p['product_id'] for p in top_products]
        values = [p['customers'] for p in top_products]
        
        doughnut_html = f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <canvas id="productsChart"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const ctx2 = document.getElementById('productsChart').getContext('2d');
            const chart2 = new Chart(ctx2, {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        data: {json.dumps(values)},
                        backgroundColor: [
                            'rgba(0, 104, 201, 0.8)',
                            'rgba(102, 187, 106, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(239, 83, 80, 0.8)',
                            'rgba(156, 39, 176, 0.8)'
                        ],
                        borderColor: 'white',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Top 5 Products by Customer Count',
                            font: {{
                                size: 16,
                                weight: 'bold'
                            }}
                        }},
                        legend: {{
                            display: true,
                            position: 'bottom'
                        }}
                    }}
                }}
            }});
        </script>
        """
        
        components.html(doughnut_html, height=400)
    
    # Section 4: Fraud Rings Visualization
    st.markdown("---")
    st.subheader("🎯 Fraud Rings Detection")
    
    st.info("💡 Các fraud rings là các nhóm khách hàng có hành vi gian lận tương tự nhau, được phát hiện qua community detection")
    
    # Fraud Rings Table
    fraud_rings = sna_data['fraud_rings']
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("#### 📋 Danh sách Fraud Rings")
        
        for ring in fraud_rings:
            with st.expander(f"🔴 Ring #{ring['ring_id']} - Fraud Rate: {ring['fraud_rate']*100:.1f}%"):
                st.metric("Kích thước", ring['size'])
                st.metric("Số gian lận", ring['fraud_count'])
                st.metric("Tỷ lệ gian lận", f"{ring['fraud_rate']*100:.1f}%")
                st.caption(ring['description'])
    
    with col2:
        st.markdown("#### 🕸️ Network Visualization")
        
        # Cytoscape.js Network Visualization
        cytoscape_html = """
        <div id="cy" style="width: 100%; height: 500px; background-color: #f9fafb; border-radius: 10px; border: 1px solid #e5e7eb;"></div>
        <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
        <script>
            var cy = cytoscape({
                container: document.getElementById('cy'),
                
                elements: [
                    // Nodes (Customers - Red for fraud, Green for normal)
                    { data: { id: 'c1', label: 'C1', type: 'fraud' } },
                    { data: { id: 'c2', label: 'C2', type: 'fraud' } },
                    { data: { id: 'c3', label: 'C3', type: 'fraud' } },
                    { data: { id: 'c4', label: 'C4', type: 'normal' } },
                    { data: { id: 'c5', label: 'C5', type: 'normal' } },
                    { data: { id: 'c6', label: 'C6', type: 'fraud' } },
                    { data: { id: 'c7', label: 'C7', type: 'fraud' } },
                    { data: { id: 'c8', label: 'C8', type: 'normal' } },
                    
                    // Products (Blue)
                    { data: { id: 'p1', label: 'P1', type: 'product' } },
                    { data: { id: 'p2', label: 'P2', type: 'product' } },
                    { data: { id: 'p3', label: 'P3', type: 'product' } },
                    { data: { id: 'p4', label: 'P4', type: 'product' } },
                    
                    // Edges (Connections)
                    { data: { source: 'c1', target: 'p1' } },
                    { data: { source: 'c1', target: 'p2' } },
                    { data: { source: 'c2', target: 'p1' } },
                    { data: { source: 'c2', target: 'p3' } },
                    { data: { source: 'c3', target: 'p2' } },
                    { data: { source: 'c3', target: 'p3' } },
                    { data: { source: 'c4', target: 'p4' } },
                    { data: { source: 'c5', target: 'p4' } },
                    { data: { source: 'c6', target: 'p1' } },
                    { data: { source: 'c6', target: 'p3' } },
                    { data: { source: 'c7', target: 'p2' } },
                    { data: { source: 'c8', target: 'p4' } }
                ],
                
                style: [
                    {
                        selector: 'node',
                        style: {
                            'label': 'data(label)',
                            'width': 40,
                            'height': 40,
                            'font-size': '12px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#000'
                        }
                    },
                    {
                        selector: 'node[type="fraud"]',
                        style: {
                            'background-color': '#ef5350',
                            'border-width': 3,
                            'border-color': '#c62828'
                        }
                    },
                    {
                        selector: 'node[type="normal"]',
                        style: {
                            'background-color': '#66bb6a',
                            'border-width': 3,
                            'border-color': '#2e7d32'
                        }
                    },
                    {
                        selector: 'node[type="product"]',
                        style: {
                            'background-color': '#0068C9',
                            'border-width': 3,
                            'border-color': '#004c8c',
                            'shape': 'rectangle'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'width': 2,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'curve-style': 'bezier'
                        }
                    }
                ],
                
                layout: {
                    name: 'cose',
                    idealEdgeLength: 100,
                    nodeOverlap: 20,
                    refresh: 20,
                    fit: true,
                    padding: 30,
                    randomize: false,
                    componentSpacing: 100,
                    nodeRepulsion: 400000,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                }
            });
        </script>
        """
        
        components.html(cytoscape_html, height=550)
        
        st.caption("🔴 Red: Fraud customers | 🟢 Green: Normal customers | 🔵 Blue: Products")
    
    # Section 5: Centrality Interpretation
    st.markdown("---")
    st.subheader("💡 Giải thích Các chỉ số Centrality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Degree Centrality**")
        st.info(centrality_data['degree']['interpretation'])
        
        st.markdown("**🌉 Betweenness Centrality**")
        st.info(centrality_data['betweenness']['interpretation'])
    
    with col2:
        st.markdown("**📍 Closeness Centrality**")
        st.info(centrality_data['closeness']['interpretation'])
        
        st.markdown("**⭐ PageRank**")
        st.info(centrality_data['pagerank']['interpretation'])
    
    # Section 6: Feature Importance
    st.markdown("---")
    st.subheader("🔧 Mức độ Quan trọng của Network Features")
    
    importance = sna_data['network_features_importance']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Degree Centrality", f"{importance['degree_centrality']*100:.0f}%")
    
    with col2:
        st.metric("Betweenness", f"{importance['betweenness_centrality']*100:.0f}%")
    
    with col3:
        st.metric("Closeness", f"{importance['closeness_centrality']*100:.0f}%")
    
    with col4:
        st.metric("Community", f"{importance['community_detection']*100:.0f}%")
    
    # Summary
    st.markdown("---")
    st.success("""
    ### ✅ Kết luận
    
    - **Network structure**: Mạng lưới bipartite với 20K+ customers và 118 products
    - **Fraud pattern**: Fraudsters có degree cao hơn 1.57x so với normal customers
    - **Detection method**: Kết hợp centrality metrics và community detection
    - **Effectiveness**: Network features đóng góp 4/61 features quan trọng cho model
    """)

else:
    st.error("❌ Không thể load dữ liệu SNA. Vui lòng kiểm tra file `data/sna_results.json`")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/network.png", width=80)
    st.title("Network Analysis")
    
    st.markdown("""
    ### 🕸️ Phân tích Mạng
    
    **Phương pháp:**
    - Bipartite Network (Customer ↔ Product)
    - Community Detection (Louvain)
    - Centrality Analysis
    
    ---
    
    ### 📊 Metrics
    - **Degree**: Số kết nối
    - **Betweenness**: Vai trò cầu nối
    - **Closeness**: Độ gần trung tâm
    - **PageRank**: Tầm quan trọng
    
    ---
    
    ### 🎯 Phát hiện
    - Fraud rings qua communities
    - Unusual patterns qua centrality
    - Coordinated attacks
    """)
    
    st.markdown("---")
    st.caption("📈 20,770 nodes analyzed")
    st.caption("🔗 101,196 relationships")
