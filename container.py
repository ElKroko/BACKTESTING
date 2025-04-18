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
from tabs.improved_analysis_tab import render_improved_analysis
from tabs.smartmoney_tab import render_smart_money
from tabs.backtest_tab import render_backtests
from tabs.leveraged_backtest import render_leveraged_backtest
# Importar la nueva vista unificada de backtesting
from tabs.unified_backtest_tab import render_unified_backtest

# Importamos las utilidades desde sus nuevas ubicaciones
from utils.html_utils import tooltip, apply_tooltip_css, apply_theme_css, apply_backtest_summary_css

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

# Apply the CSS with the selected palette
apply_theme_css(config.PALETTE)

# Apply tooltip CSS
apply_tooltip_css()

# Apply backtest summary CSS
apply_backtest_summary_css()

# --- Main tabs ---
tabs = st.tabs([
    '📊 Centro de Mando',
    '🔄 Unified Backtests',
    '🔄 Backtests (Legacy)',
    '📈 Leveraged Backtests (Legacy)',
    '💡 Smart Money Concepts'
])

with tabs[0]:
    render_improved_analysis()

with tabs[1]:
    render_unified_backtest()

with tabs[2]:
    render_backtests()
    
with tabs[3]:
    render_leveraged_backtest(tooltip_func=tooltip)

with tabs[4]:
    render_smart_money()
