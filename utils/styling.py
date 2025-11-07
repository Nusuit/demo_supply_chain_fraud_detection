"""
Custom CSS Styling for Monochrome Design
Black/White/Gray color scheme with Inter font
No icons, minimalist approach
"""

import streamlit as st


def inject_custom_css():
    """
    Inject custom CSS to override Streamlit defaults
    Implements monochrome design system from HTML mockups
    """
    st.markdown("""
    <style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ============================================
       GLOBAL STYLES
       ============================================ */

    * {
        font-family: 'Inter', sans-serif !important;
    }

    body {
        color: #0B0B0B;
        background-color: #F5F5F5;
    }

    /* Main container */
    .main {
        background-color: #F5F5F5;
    }

    /* Remove default Streamlit branding colors */
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* ============================================
       HEADINGS & TEXT
       ============================================ */

    h1 {
        color: #0B0B0B;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    h2 {
        color: #0B0B0B;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    h3 {
        color: #0B0B0B;
        font-weight: 600;
    }

    p {
        color: #5A5A5A;
        line-height: 1.6;
    }

    /* ============================================
       METRIC CARDS (KPI Cards)
       ============================================ */

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #EDEDED;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stMetric"] label {
        color: #5A5A5A !important;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0B0B0B !important;
        font-size: 2rem;
        font-weight: 700;
    }

    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        display: none; /* Hide delta arrows for monochrome look */
    }

    /* ============================================
       BUTTONS
       ============================================ */

    /* Primary Button (Black) */
    .stButton>button {
        background-color: #0B0B0B;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #2A2A2A;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:active {
        background-color: #0B0B0B;
        transform: scale(0.98);
    }

    /* Secondary Button (Outline) */
    .stButton>button[kind="secondary"] {
        background-color: white;
        color: #0B0B0B;
        border: 1px solid #EDEDED;
    }

    .stButton>button[kind="secondary"]:hover {
        background-color: #F5F5F5;
        border-color: #5A5A5A;
    }

    /* ============================================
       TABS
       ============================================ */

    [data-baseweb="tab-list"] {
        background-color: white;
        border: 1px solid #EDEDED;
        border-radius: 8px;
        padding: 0.25rem;
        gap: 0.25rem;
    }

    [data-baseweb="tab"] {
        background-color: transparent;
        color: #5A5A5A;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }

    [data-baseweb="tab"]:hover {
        background-color: #F5F5F5;
        color: #0B0B0B;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0B0B0B !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Make sure tab text is bright white when active */
    [data-baseweb="tab"][aria-selected="true"] p,
    [data-baseweb="tab"][aria-selected="true"] span,
    [data-baseweb="tab"][aria-selected="true"] div {
        color: #FFFFFF !important;
    }

    /* ============================================
       INPUT FIELDS
       ============================================ */

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select {
        border: 1px solid #EDEDED;
        border-radius: 8px;
        padding: 0.625rem 0.875rem;
        color: #0B0B0B;
        background-color: white;
        font-size: 0.9375rem;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] select:focus {
        border-color: #0B0B0B;
        box-shadow: 0 0 0 3px rgba(11, 11, 11, 0.05);
        outline: none;
    }

    /* Input labels */
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label {
        color: #0B0B0B;
        font-weight: 500;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }

    /* ============================================
       CARDS / CONTAINERS
       ============================================ */

    [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        background-color: white;
        border: 1px solid #EDEDED;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #EDEDED;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stExpander"] summary {
        color: #0B0B0B;
        font-weight: 600;
    }

    /* ============================================
       PROGRESS BARS (Monochrome)
       ============================================ */

    [data-testid="stProgress"] > div > div {
        background-color: #EDEDED;
    }

    [data-testid="stProgress"] > div > div > div {
        background-color: #0B0B0B;
    }

    /* ============================================
       TABLES
       ============================================ */

    [data-testid="stTable"],
    .dataframe {
        border: 1px solid #EDEDED;
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stTable"] th,
    .dataframe thead th {
        background-color: #0B0B0B;
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 0.875rem;
        border: none;
    }

    [data-testid="stTable"] td,
    .dataframe tbody td {
        padding: 0.75rem 0.875rem;
        border-bottom: 1px solid #EDEDED;
        color: #0B0B0B;
    }

    [data-testid="stTable"] tr:hover,
    .dataframe tbody tr:hover {
        background-color: #F5F5F5;
    }

    /* ============================================
       FILE UPLOADER
       ============================================ */

    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #EDEDED;
        border-radius: 8px;
        padding: 2rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #0B0B0B;
    }

    [data-testid="stFileUploader"] label {
        color: #0B0B0B;
        font-weight: 600;
    }

    /* ============================================
       ALERTS / INFO BOXES
       ============================================ */

    [data-testid="stAlert"] {
        background-color: white;
        border-left: 4px solid #0B0B0B;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #0B0B0B;
    }

    [data-testid="stAlert"][data-baseweb="notification"] {
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    /* Success - use dark gray instead of green */
    .success-alert {
        border-left-color: #2A2A2A;
        background-color: #F5F5F5;
    }

    /* Warning - use medium gray */
    .warning-alert {
        border-left-color: #5A5A5A;
        background-color: #EDEDED;
    }

    /* ============================================
       CHARTS (Plotly)
       ============================================ */

    .js-plotly-plot {
        background-color: white;
        border: 1px solid #EDEDED;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ============================================
       SIDEBAR
       ============================================ */

    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #EDEDED;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #0B0B0B;
    }

    /* ============================================
       CUSTOM UTILITY CLASSES
       ============================================ */

    /* Card container */
    .monochrome-card {
        background: white;
        border: 1px solid #EDEDED;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    /* Section header */
    .section-header {
        color: #0B0B0B;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* Subsection header */
    .subsection-header {
        color: #0B0B0B;
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    /* Muted text */
    .text-muted {
        color: #5A5A5A;
        font-size: 0.875rem;
    }

    /* Monochrome badge */
    .monochrome-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #0B0B0B;
        color: white;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Outline badge */
    .outline-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: transparent;
        color: #0B0B0B;
        border: 1px solid #EDEDED;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Divider */
    .monochrome-divider {
        height: 1px;
        background-color: #EDEDED;
        margin: 2rem 0;
    }

    /* ============================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ============================================ */

    /* Hide hamburger menu */
    #MainMenu {
        visibility: hidden;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    /* Hide "Made with Streamlit" */
    .viewerBadge_container__1QSob {
        display: none;
    }

    /* ============================================
       RESPONSIVE DESIGN
       ============================================ */

    @media (max-width: 768px) {
        h1 {
            font-size: 1.75rem;
        }

        h2 {
            font-size: 1.5rem;
        }

        [data-testid="stMetric"] {
            padding: 1rem;
        }

        .monochrome-card {
            padding: 1rem;
        }
    }

    </style>
    """, unsafe_allow_html=True)


def create_metric_card(label, value, description=""):
    """
    Create a custom monochrome metric card

    Args:
        label: Metric label (e.g., "Recall")
        value: Metric value (e.g., "74.83%")
        description: Optional description text
    """
    return f"""
    <div class="monochrome-card">
        <div style="color: #5A5A5A; font-size: 0.875rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            {label}
        </div>
        <div style="color: #0B0B0B; font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
            {value}
        </div>
        {f'<div class="text-muted">{description}</div>' if description else ''}
    </div>
    """


def create_section_header(title, subtitle=""):
    """
    Create a section header with optional subtitle

    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    return f"""
    <div style="margin-bottom: 2rem;">
        <h2 class="section-header">{title}</h2>
        {f'<p class="text-muted">{subtitle}</p>' if subtitle else ''}
    </div>
    """


def create_badge(text, outline=False):
    """
    Create a monochrome badge

    Args:
        text: Badge text
        outline: If True, creates outline badge instead of filled
    """
    badge_class = "outline-badge" if outline else "monochrome-badge"
    return f'<span class="{badge_class}">{text}</span>'


def create_divider():
    """Create a monochrome divider line"""
    return '<div class="monochrome-divider"></div>'
