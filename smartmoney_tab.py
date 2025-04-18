# smartmoney_tab.py
import streamlit as st
import pandas as pd
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=500"
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    return df[['Open','High','Low','Close','Volume']].astype(float)

def detect_order_blocks(df: pd.DataFrame, pct: float = 0.01) -> list:
    moves = df['Close'].pct_change().shift(-1)
    return df.index[moves.abs() > pct].tolist()

def detect_fvg(df: pd.DataFrame) -> list:
    gaps = []
    for i in range(len(df) - 2):
        h1, l3 = df['High'].iat[i], df['Low'].iat[i+2]
        if h1 < l3:
            gaps.append((df.index[i+1], h1, l3))
    return gaps

def detect_liquidity_pools(df: pd.DataFrame) -> list:
    counts = df['High'].round(2).value_counts() + df['Low'].round(2).value_counts()
    return counts[counts >= 2].index.tolist()

def render_smart_money():
    st.header('💡 Smart Money Concepts for BTCUSDT')
    symbol = 'BTCUSDT'
    intervals = [('1 Hour','1h'), ('4 Hours','4h'), ('Daily','1d')]

    style = mpf.make_mpf_style(base_mpf_style='nightclouds')
    mc = mpf.make_marketcolors(up='lime', down='red', inherit=True)
    style['marketcolors'] = mc

    for title, interval in intervals:
        df = get_data(symbol, interval)
        st.subheader(title)

        if df.empty:
            st.warning(f"No data available for {title} ({interval}). Skipping visualization.")
            continue

        obs   = detect_order_blocks(df)
        gaps  = detect_fvg(df)
        pools = detect_liquidity_pools(df)

        cols = st.columns(3)

        with cols[0]:
            st.markdown('**Order Blocks (>1% next candle)**')
            fig, axlist = mpf.plot(df, type='candle', style=style, mav=(50,100),
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            y_vals = df.loc[obs, 'Close']
            ax.scatter(obs, y_vals, color='cyan', s=20, label='Order Block')
            ax.set_xlim(df.index[0], df.index[-1])
            ax.legend(loc='upper left', fontsize=6)
            st.pyplot(fig)

        with cols[1]:
            st.markdown('**Fair Value Gaps**')
            fig, axlist = mpf.plot(df, type='candle', style=style,
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            for _, h1, l3 in gaps:
                ax.axhspan(l3, h1, color='orange', alpha=0.3)
            st.pyplot(fig)

        with cols[2]:
            st.markdown('**Liquidity Pools (touched ≥2)**')
            fig, axlist = mpf.plot(df, type='candle', style=style,
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            start, end = df.index[0], df.index[-1]
            for lvl in pools:
                ax.hlines(lvl, start, end, linestyles='--', linewidth=1)
            st.pyplot(fig)

        st.markdown('---')

    st.write(
        'These charts use **mplfinance** in dark mode to illustrate '
        '**Order Blocks**, **Fair Value Gaps**, and **Liquidity Pools** for BTCUSDT.'
    )
