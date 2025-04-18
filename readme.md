# Crypto Backtesting App

This repository contains a Streamlit application for:

1. **Technical Indicator Analysis** (SMA, MACD, RSI, MFI, support & resistance levels).
2. **Strategy Backtesting** (MA Crossover, Bollinger Breakout, RSI Reversion, MACD Momentum, SR Breakout).
3. **Smart Money Concepts Visualization** (Order Blocks, Fair Value Gaps, Liquidity Pools).

## File Structure

- `strategies.py`: Defines five strategy functions that accept an OHLC DataFrame and return a list of trades.
- `analysis_tab.py`: Calculates indicators, detects levels and pivots, generates a summary PDF, and embeds TradingView charts.
- `backtest_tab.py`: Implements `run_backtest()` to fetch Binance data, run a chosen strategy, simulate orders, and compute metrics (net profit, max drawdown, Sharpe ratio).
- `smartmoney_tab.py`: Detects and plots Smart Money Concepts on BTCUSDT using mplfinance.
- `app_container.py`: Entry point; sets up the Streamlit page and creates three tabs: “Analysis”, “Backtests”, and “Smart Money Concepts”.

## Installation

1. Clone this repository:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_FOLDER>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setting Up the Virtual Environment

To set up the virtual environment and ensure all necessary dependencies are installed, follow these steps:

1. Make sure Python is installed on your system.
2. Run the `setup_venv.bat` file by double-clicking it or executing the following command in the terminal:

   ```
   setup_venv.bat
   ```

This script will:

- Create a virtual environment in the `venv` folder.
- Activate the virtual environment.
- Install the dependencies listed in the `requirements.txt` file.

Once completed, the virtual environment will be ready to use.

## Usage

Run the app with Streamlit:

```bash
streamlit run app_container.py
```

The app will open in your browser with three tabs:

- **📊 Analysis**: Enter a symbol (e.g. BTCUSDT) to view technical indicators, pivots, and levels; download a PDF report of current values.
- **🔄 Backtests**: Select symbol, timeframe, strategy, and initial capital; click “Run Backtest” for metrics, equity curve, and trade log.
- **💡 Smart Money Concepts**: Visualize Order Blocks, Fair Value Gaps, and Liquidity Pools across different timeframes for BTCUSDT.

## Notes

- The app uses Binance’s public API for OHLC data.
- Ensure you have an internet connection.
- You can tweak strategy parameters in `strategies.py` as needed.