# analysis_tab.py
import streamlit as st
import pandas as pd
import requests
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from streamlit.components.v1 import html as components_html

def get_data(symbol: str, interval: str = '1h') -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=500"
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    return df.astype(float)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    macd = ta.trend.MACD(df['Close'])
    df['MACD']   = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI']    = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['MFI']    = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 14).money_flow_index()
    return df

def detect_support_resistance(df: pd.DataFrame):
    highs = df['High']; lows = df['Low']
    piv_high = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    piv_low  = lows[(lows.shift(1) > lows)   & (lows.shift(-1) > lows)]
    res = piv_high.iloc[-2:].sort_values(ascending=False).tolist() if len(piv_high) >= 2 else []
    sup = piv_low.iloc[-2:].sort_values().tolist()             if len(piv_low)  >= 2 else []
    return res, sup

def detect_pivots(df: pd.DataFrame):
    hh = df['High'].rolling(20).max().dropna().iloc[-2:].tolist()[::-1]
    ll = df['Low'].rolling(20).min().dropna().iloc[-2:].tolist()[::-1]
    return hh, ll

def indicator_recommendation(ind: str, value: float, extra=None) -> str:
    if ind == 'SMA50' and extra is not None:
        return 'Bullish' if value > extra else 'Bearish'
    if ind == 'MACD'  and extra is not None:
        return 'Bullish' if value > extra else 'Bearish'
    if ind == 'RSI':
        if value > 70: return 'Overbought'
        if value < 30: return 'Oversold'
        return 'Neutral'
    if ind == 'MFI':
        if value > 80: return 'Overbought'
        if value < 20: return 'Oversold'
        return 'Neutral'
    return ''

def generate_pdf(symbol: str, data_dict: dict, pdf_filename: str = 'analysis_report.pdf') -> str:
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, f"Analysis Report: {symbol}")
    y = height - 80
    for label, df in data_dict.items():
        if df.empty: continue
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, label)
        y -= 20
        last = df.iloc[-1]
        for name in ['SMA50','SMA100','MACD','Signal','RSI','MFI']:
            c.setFont("Helvetica", 12)
            c.drawString(60, y, f"{name}: {round(last[name],2)}")
            y -= 15
            if y < 60:
                c.showPage()
                y = height - 50
        y -= 10
    c.save()
    return pdf_filename

def render_analysis():
    symbol = st.text_input('Symbol (e.g. BTCUSDT)', 'BTCUSDT', key='analysis_symbol').strip().upper()
    st.title(f'Analysis: {symbol}')

    timeframes = [
        ('Weekly', '1w', 'W'),
        ('Daily',  '1d', 'D'),
        ('Hourly','1h', '60')
    ]

    data_dict, levels_dict, pivots_dict = {}, {}, {}
    for label, api_int, tv_int in timeframes:
        df_raw = get_data(symbol, api_int)
        df = calculate_indicators(df_raw).dropna()
        data_dict[label] = df
        if not df.empty:
            res, sup = detect_support_resistance(df)
            hh, ll = detect_pivots(df)
            levels_dict[label] = {'res': res, 'sup': sup}
            pivots_dict[label] = {'hh': hh, 'll': ll}
        else:
            levels_dict[label] = {'res': [], 'sup': []}
            pivots_dict[label] = {'hh': [], 'll': []}

    st.subheader('Combined View by Timeframe')
    cols = st.columns(3)
    hide = '&hide_top_toolbar=true&hide_side_toolbar=true&hide_legend=true&allow_symbol_change=false'

    for col, (label, _, tv_int) in zip(cols, timeframes):
        df     = data_dict[label]
        levels = levels_dict[label]
        pivots = pivots_dict[label]
        with col:
            st.markdown(f'### {label}')
            src = f'https://s.tradingview.com/widgetembed/?symbol=BINANCE:{symbol}&interval={tv_int}&theme=Dark{hide}'
            components_html(f'<iframe src="{src}" width="100%" height="200" frameborder="0"></iframe>', height=210)

            if df.empty:
                st.warning(f'Not enough data for {label}')
            else:
                last = df.iloc[-1]
                rows = []
                for ind in ['SMA50','SMA100','MACD','Signal','RSI','MFI']:
                    extra = last['SMA100'] if ind=='SMA50' else (last['Signal'] if ind=='MACD' else None)
                    rec   = indicator_recommendation(ind, last[ind], extra)
                    rows.append({'Indicator': ind, 'Value': round(last[ind],2), 'Recommendation': rec})
                for i, r in enumerate(levels['res']): rows.append({'Indicator': f'R{i+1}','Value': round(r,2),'Recommendation':''})
                for i, s in enumerate(levels['sup']): rows.append({'Indicator': f'S{i+1}','Value': round(s,2),'Recommendation':''})
                for i, h in enumerate(pivots['hh']): rows.append({'Indicator': f'HH{i+1}','Value': round(h,2),'Recommendation':''})
                for i, l in enumerate(pivots['ll']): rows.append({'Indicator': f'LL{i+1}','Value': round(l,2),'Recommendation':''})
                st.table(pd.DataFrame(rows))

                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(df.index, df['Close'], label='Close')
                for price in levels['res']: ax.hlines(price, df.index[0], df.index[-1], linestyles='--', alpha=0.7)
                for price in levels['sup']: ax.hlines(price, df.index[0], df.index[-1], linestyles='--', alpha=0.7)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.legend(fontsize='small')
                st.pyplot(fig)

    if st.button('Generate and Download PDF', key='analysis_download'):
        pdf_file = generate_pdf(symbol, data_dict, pdf_filename=f'analysis_{symbol}.pdf')
        with open(pdf_file, 'rb') as f:
            st.download_button('Download PDF', f, file_name=pdf_file, mime='application/pdf', key='download_pdf')
