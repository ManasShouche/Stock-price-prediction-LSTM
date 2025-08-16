## Stock price prediction LSTM

This project compares **classical ARIMA** and **deep learning LSTM** models for stock price prediction using Apple (AAPL) historical data.

## Features
- Fetch stock data using `yfinance`
- ARIMA model for time-series forecasting
- LSTM model for deep learning forecasting
- Visualization of actual vs predicted prices

## Project Structure
- `src/` : All source code
- `data/` : Stock data (auto-downloaded)
- `main.py` : Run the whole pipeline

## Use
pip install -r requirements.txt
python main.py
