import pandas as pd
from src.fetch_data import get_stock_data
from src.arima_model import run_arima
from src.lstm_model import run_lstm

def main():
    print(" Fetching stock data...")
    df = get_stock_data("AAPL", "2018-01-01", "2023-12-31")

    print("\n Running ARIMA...")
    arima_metrics = run_arima(df)

    print("\n Running LSTM...")
    lstm_metrics = run_lstm(df)

    print("\nModel Performance Comparison:")
    results = pd.DataFrame([arima_metrics, lstm_metrics])
    print(results)

if __name__ == "__main__":
    main()
