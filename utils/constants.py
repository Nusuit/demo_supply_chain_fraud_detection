"""
Utility module for constants and configuration
"""

# Total number of features (61 total: 57 transaction + 4 network)
TOTAL_FEATURES = 61

# Feature columns that need preprocessing (simplified list for form input)
# Full 61 features are handled in preprocessing.py
FEATURE_COLUMNS = [
    'Days for shipping (real)', 'Days for shipment (scheduled)', 
    'Benefit per order', 'Sales per customer', 'Late_delivery_risk',
    'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price',
    'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total',
    'Order Profit Per Order', 'product_popularity', 'product_profit_margin',
    'product_avg_discount', 'customer_order_count', 'customer_total_spent',
    'customer_avg_order_value', 'customer_fraud_history', 'recency_days',
    'order_count_last_30d', 'total_spent_last_30d', 'time_since_last_order',
    'is_new_customer', 'rush_order', 'unusual_quantity', 'high_discount_flag',
    'negative_benefit', 'international_order', 'high_value_order',
    'high_risk_combination', 'degree_centrality', 'betweenness_centrality',
    'closeness_centrality', 'pagerank'
]

# Categorical features for form input
CATEGORICAL_FEATURES = {
    'Type': ['DEBIT', 'TRANSFER', 'PAYMENT', 'CASH'],
    'Delivery Status': [
        'Advance shipping', 'Late delivery', 'Shipping canceled',
        'Shipping on hold', 'On time'
    ],
    'Category Name': [
        'Sporting Goods', 'Electronics', 'Apparel', 'Health & Beauty',
        'Computers', 'Home & Garden', 'Sports', 'Office Supplies',
        'Books', 'Toys', 'Automotive', 'Food & Beverage'
    ],
    'Customer Country': ['USA', 'Mexico', 'Canada', 'UK', 'France', 'Germany', 'Spain'],
    'Customer Segment': ['Consumer', 'Corporate', 'Home Office'],
    'Department Name': [
        'Fan Shop', 'Technology', 'Apparel', 'Health', 'Garden',
        'Office', 'Books', 'Fitness', 'Outdoors', 'Automotive'
    ],
    'Market': ['US', 'LATAM', 'Europe', 'Pacific Asia', 'Africa', 'USCA'],
    'Order Region': ['West', 'East', 'Central', 'South'],
    'Shipping Mode': [
        'Standard Class', 'First Class', 'Second Class', 'Same Day'
    ],
    'Product Category Name': [
        'Sporting Goods', 'Electronics', 'Apparel', 'Health & Beauty',
        'Computers', 'Home & Garden', 'Sports', 'Office Supplies',
        'Books', 'Toys', 'Automotive', 'Food & Beverage'
    ],
    'Order Status': [
        'COMPLETE', 'PENDING', 'CLOSED', 'PENDING_PAYMENT',
        'CANCELED', 'PROCESSING', 'SUSPECTED_FRAUD', 'ON_HOLD'
    ],
    'product_value_category': ['low', 'medium', 'high']
}

# Model configuration
MODEL_THRESHOLD = 0.20
ENSEMBLE_SEEDS = [42, 123, 456]
N_PCA_COMPONENTS = 45

# Risk levels
RISK_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.6),
    'high': (0.6, 1.0)
}

def get_risk_level(score):
    """Get risk level based on fraud score"""
    if score < 0.3:
        return "⚪ Rủi ro Thấp", "#EDEDED"
    elif score < 0.6:
        return "⚫ Rủi ro Trung bình", "#5A5A5A"
    else:
        return "⬛ Rủi ro Cao", "#0B0B0B"

def get_fraud_prediction(score, threshold=MODEL_THRESHOLD):
    """Get fraud prediction based on score and threshold"""
    return "GIAN LẬN" if score >= threshold else "HỢP LỆ"
