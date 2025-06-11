import ccxt, pandas as pd
import pandas_ta as ta

# 1. Descargar datos de Binance (ejemplo)
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', since=exchange.parse8601('2020-01-01T00:00:00Z'))
df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
df.set_index('ts', inplace=True)

# 2. Calcular EMAs
df['ema9'] = ta.ema(df['close'], length=9)
df['ema21'] = ta.ema(df['close'], length=21)

# 3. Señales
df['long'] = df['ema9'] > df['ema21']
df['signal'] = df['long'].diff().fillna(0)

# 4. Backtest básico
entry_points = df[df['signal'] == 1]
exit_points  = df[df['signal'] == -1]
print("Entradas:", entry_points.index)
print("Salidas:",  exit_points.index)

# 5. Calcular ganancias
def calculate_profit(entry_points, exit_points, df):
    total_profit = 0
    for entry in entry_points.index:
        if entry in exit_points.index:
            exit = exit_points.loc[entry]
            profit = df.loc[exit, 'close'] - df.loc[entry, 'close']
            total_profit += profit
    return total_profit

total_profit = calculate_profit(entry_points, exit_points, df)
print("Ganancia total:", total_profit)
# 6. Mostrar resultados
print("Resultados del backtest:")
print(df[['close', 'ema9', 'ema21', 'long', 'signal']].tail(10))
# 7. Guardar resultados en CSV
df.to_csv('backtest_results.csv')


