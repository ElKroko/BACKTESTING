# container.py
import streamlit as st

# --- Page configuration ---
# This MUST be the first Streamlit command
st.set_page_config(
    page_title='Crypto Backtesting App',
    page_icon='💹',
    layout='wide',
    initial_sidebar_state='collapsed'
)

import config
from analysis_tab import render_analysis
from smartmoney_tab import render_smart_money
# Importamos los utilitarios
from utils import tooltip, apply_tooltip_css

# --- Top navbar: title + palette selector ---
bar = st.container()
col_title, col_palette = bar.columns([5, 1])
with col_title:
    st.markdown("## Crypto Backtesting App")
with col_palette:
    palette_options = list(config.PALETTES.keys())
    default_index = palette_options.index(config.DEFAULT_PALETTE)
    selected = st.selectbox(
        '',
        options=palette_options,
        index=default_index,
        label_visibility='collapsed'
    )
    # Apply selected palette globally
    config.PALETTE = config.PALETTES[selected]

# --- Apply theme colors based on selected palette ---
def apply_custom_css(palette):
    # Custom CSS to override Streamlit's default styles
    custom_css = f"""
    <style>
        /* Main background and text colors */
        .stApp {{
            background-color: {palette['background']};
            color: {palette['text']};
        }}
        
        /* Active tab indicator */
        .stTabs [aria-selected="true"] {{
            background-color: {palette['accent']} !important;
            color: white !important;
        }}
        
        /* Tab hover */
        .stTabs [role="tab"]:hover {{
            background-color: {palette['accent']}88 !important;
            color: white !important;
        }}
        
        /* Containers, cards, expanders */
        .stExpander, div.block-container {{
            border-color: {palette['border']} !important;
        }}
        
        /* Button colors */
        .stButton>button {{
            background-color: {palette['accent']};
            color: white;
        }}
        
        .stButton>button:hover {{
            background-color: {palette['accent']}CC;
        }}
        
        /* Metric text and background */
        [data-testid="stMetricLabel"] {{
            color: {palette['metric_text']} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {palette['metric_value']} !important;
            background-color: {palette['metric_bg']} !important;
            padding: 5px !important;
            border-radius: 5px !important;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {palette['secondary']};
        }}
        
        /* Table header */
        thead tr th {{
            background-color: {palette['table_header_bg']} !important;
            color: {palette['table_text']} !important;
        }}
        
        tbody tr {{
            background-color: {palette['table_bg']} !important;
            color: {palette['table_text']} !important;
        }}
        
        /* DataFrames and tables */
        .stDataFrame {{
            border-color: {palette['border']} !important;
        }}
        
        /* Input elements */
        .stSelectbox label, .stTextInput label, .stNumberInput label, .stDateInput label {{
            color: {palette['text']} !important;
        }}
        
        /* Plot and chart backgrounds */
        .js-plotly-plot, .plotly, .js-plotly-plot .plot-container .svg-container {{
            background-color: {palette['chart_bg']} !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {palette['text']} !important;
        }}
        
        /* Expander content */
        .streamlit-expanderContent {{
            background-color: {palette['secondary']} !important;
            color: {palette['text']} !important;
        }}
        
        /* Warning messages */
        .stAlert {{
            background-color: {palette['secondary']} !important;
            color: {palette['text']} !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the CSS for the selected palette
apply_custom_css(config.PALETTE)

# Aplicar el CSS para los tooltips
apply_tooltip_css()

# --- Main tabs ---
tabs = st.tabs([
    '📊 Analysis',
    '🔄 Backtests',
    '📈 Leveraged Backtests',
    '💡 Smart Money Concepts'
])

with tabs[0]:
    render_analysis()

with tabs[1]:
    # Importar aquí para evitar importación circular
    from backtest_tab import render_backtests
    render_backtests()
    
with tabs[2]:
    # Importar aquí para evitar importación circular
    from leveraged_backtest import render_leveraged_backtest
    render_leveraged_backtest(tooltip_func=tooltip)

with tabs[3]:
    render_smart_money()
