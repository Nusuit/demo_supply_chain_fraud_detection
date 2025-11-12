"""
Custom CSS Styling for Light Blue Theme
White/Light Blue color scheme with Inter font
Clean and friendly design
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
    
    /* Import Material Icons (for Streamlit expander icons) */
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

    /* ============================================
       GLOBAL STYLES
       ============================================ */

    * {
        font-family: 'Inter', sans-serif !important;
    }

    body {
        color: #1565C0;
        background-color: #E3F2FD;
    }

    /* Main container */
    .main {
        background-color: #E3F2FD;
    }

    /* Remove default Streamlit branding colors */
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* ============================================
       HEADINGS & TEXT
       ============================================ */

    h1 {
        color: #1565C0;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    h2 {
        color: #1976D2;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    h3 {
        color: #1976D2;
        font-weight: 600;
    }

    p {
        color: #1565C0;
        line-height: 1.6;
    }

    /* ============================================
       METRIC CARDS (KPI Cards)
       ============================================ */

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(33, 150, 243, 0.1);
    }

    [data-testid="stMetric"] label {
        color: #1976D2 !important;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1565C0 !important;
        font-size: 2rem;
        font-weight: 700;
    }

    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        display: none; /* Hide delta arrows for monochrome look */
    }

    /* ============================================
       BUTTONS
       ============================================ */

    /* Primary Button (Blue) */
    .stButton>button {
        background-color: #2196F3;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #1976D2;
        color: white !important;  /* Force white text on hover */
        box-shadow: 0 4px 6px rgba(33, 150, 243, 0.3);
    }

    .stButton>button:active {
        background-color: #1565C0;
        color: white !important;  /* Force white text on active */
        transform: scale(0.98);
    }

    /* Secondary Button (Outline) - Make text much darker */
    .stButton>button[kind="secondary"] {
        background-color: white;
        color: #01579B !important;  /* Very dark blue - much more visible */
        border: 2px solid #1976D2;  /* Thicker, darker border */
        font-weight: 700 !important;  /* Extra bold */
        font-size: 0.9375rem;
    }

    .stButton>button[kind="secondary"]:hover {
        background-color: #E3F2FD;
        border-color: #01579B !important;  /* Very dark border on hover */
        color: #004D8D !important;  /* Even darker on hover */
        font-weight: 700 !important;
    }

    /* ============================================
       TABS
       ============================================ */

    [data-baseweb="tab-list"] {
        background-color: white;
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        padding: 0.25rem;
        gap: 0.25rem;
    }

    [data-baseweb="tab"] {
        background-color: transparent;
        color: #1976D2;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }

    [data-baseweb="tab"]:hover {
        background-color: #E3F2FD;
        color: #1565C0;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2196F3 !important;
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
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        padding: 0.625rem 0.875rem;
        color: #1565C0;
        background-color: white;
        font-size: 0.9375rem;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] select:focus {
        border-color: #2196F3;
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.15);
        outline: none;
    }

    /* Input labels */
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label {
        color: #1976D2;
        font-weight: 500;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }

    /* ============================================
       CARDS / CONTAINERS
       ============================================ */

    [data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
        background-color: white;
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(33, 150, 243, 0.1);
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(33, 150, 243, 0.1);
    }

    [data-testid="stExpander"] summary {
        color: #1976D2;
        font-weight: 600;
    }

    /* Hide the arrow icon (keyboard_arrow_right/down) in expander */
    [data-testid="stExpander"] summary::before,
    [data-testid="stExpander"] details summary::before,
    [data-testid="stExpander"] svg,
    [data-testid="stExpander"] summary svg {
        display: none !important;
    }

    /* Hide text fallback when Material Icons don't load (hides "keyboard_arrow_right" text) */
    [data-testid="stExpander"] summary .material-icons,
    [data-testid="stExpander"] summary span[class*="material"],
    [data-testid="stExpander"] summary [class*="icon"],
    [data-testid="stExpander"] [class*="Icon"],
    [data-testid="stExpanderToggleIcon"] {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Nuclear option: Hide all direct children except the text label */
    [data-testid="stExpander"] summary > span:first-child,
    [data-testid="stExpander"] summary > div:first-child {
        display: none !important;
    }

    /* Keep only the text label visible */
    [data-testid="stExpander"] summary {
        display: flex;
        align-items: center;
    }

    /* Adjust padding since icon is hidden */
    [data-testid="stExpander"] summary {
        padding-left: 1rem;
    }

    /* ============================================
       PROGRESS BARS (Monochrome)
       ============================================ */

    [data-testid="stProgress"] > div > div {
        background-color: #BBDEFB;
    }

    [data-testid="stProgress"] > div > div > div {
        background-color: #2196F3;
    }

    /* ============================================
       TABLES
       ============================================ */

    [data-testid="stTable"],
    .dataframe {
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stTable"] th,
    .dataframe thead th {
        background-color: #2196F3;
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
        border-bottom: 1px solid #BBDEFB;
        color: #1565C0;
    }

    [data-testid="stTable"] tr:hover,
    .dataframe tbody tr:hover {
        background-color: #E3F2FD;
    }

    /* ============================================
       FILE UPLOADER
       ============================================ */

    [data-testid="stFileUploader"] {
        background-color: white;
        border: 2px dashed #BBDEFB;
        border-radius: 8px;
        padding: 2rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2196F3;
    }

    [data-testid="stFileUploader"] label {
        color: #1976D2;
        font-weight: 600;
    }

    /* ============================================
       ALERTS / INFO BOXES
       ============================================ */

    /* Fix for st.info, st.success, st.warning - prevent black text */
    [data-testid="stAlert"],
    [data-testid="stNotification"],
    .stAlert {
        background-color: white;
        border-left: 4px solid #2196F3;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #1565C0 !important;
        box-shadow: 0 1px 3px rgba(33, 150, 243, 0.1);
    }

    /* Ensure text inside alerts is not black */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span,
    [data-testid="stNotification"] p,
    [data-testid="stNotification"] div,
    [data-testid="stNotification"] span,
    .stAlert p,
    .stAlert div,
    .stAlert span {
        color: #1565C0 !important;
    }

    /* Success - use green */
    [data-testid="stAlert"][data-baseweb="notification"]:has(.success),
    .success-alert {
        border-left-color: #4CAF50;
        background-color: #E8F5E9;
    }

    [data-testid="stAlert"][data-baseweb="notification"]:has(.success) p,
    [data-testid="stAlert"][data-baseweb="notification"]:has(.success) div,
    .success-alert p,
    .success-alert div {
        color: #2E7D32 !important;
    }

    /* Warning - use orange */
    [data-testid="stAlert"][data-baseweb="notification"]:has(.warning),
    .warning-alert {
        border-left-color: #FF9800;
        background-color: #FFF3E0;
    }

    [data-testid="stAlert"][data-baseweb="notification"]:has(.warning) p,
    [data-testid="stAlert"][data-baseweb="notification"]:has(.warning) div,
    .warning-alert p,
    .warning-alert div {
        color: #E65100 !important;
    }

    /* Error - use red */
    [data-testid="stAlert"][data-baseweb="notification"]:has(.error) {
        border-left-color: #F44336;
        background-color: #FFEBEE;
    }

    [data-testid="stAlert"][data-baseweb="notification"]:has(.error) p,
    [data-testid="stAlert"][data-baseweb="notification"]:has(.error) div {
        color: #C62828 !important;
    }

    /* ============================================
       CHARTS (Plotly)
       ============================================ */

    .js-plotly-plot {
        background-color: white;
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1rem;
    }

    /* ============================================
       SIDEBAR
       ============================================ */

    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #BBDEFB;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
        color: #1976D2;
    }

    /* ============================================
       CUSTOM UTILITY CLASSES
       ============================================ */

    /* Card container */
    .monochrome-card {
        background: white;
        border: 1px solid #BBDEFB;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(33, 150, 243, 0.1);
        margin-bottom: 1rem;
    }

    /* Section header */
    .section-header {
        color: #1976D2;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }

    /* Subsection header */
    .subsection-header {
        color: #1976D2;
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    /* Muted text */
    .text-muted {
        color: #424242;
        font-size: 0.875rem;
    }

    /* Monochrome badge */
    .monochrome-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #2196F3;
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
        color: #2196F3;
        border: 1px solid #BBDEFB;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Divider */
    .monochrome-divider {
        height: 1px;
        background-color: #BBDEFB;
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
    Create a custom light blue metric card

    Args:
        label: Metric label (e.g., "Recall")
        value: Metric value (e.g., "74.83%")
        description: Optional description text
    """
    return f"""
    <div class="monochrome-card">
        <div style="color: #1976D2; font-size: 0.875rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            {label}
        </div>
        <div style="color: #1565C0; font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem;">
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
