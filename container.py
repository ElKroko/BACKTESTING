# app_container.py
import streamlit as st
from analysis_tab import render_analysis
from backtest_tab import render_backtests
from smartmoney_tab import render_smart_money

st.set_page_config(
    page_title='Crypto Backtesting App',
    page_icon='💹',
    layout='wide'
)
st.title('Crypto Backtesting App')

tabs = st.tabs([
    '📊 Analysis',
    '🔄 Backtests',
    '💡 Smart Money Concepts'
])

with tabs[0]:
    render_analysis()

with tabs[1]:
    render_backtests()

with tabs[2]:
    render_smart_money()
