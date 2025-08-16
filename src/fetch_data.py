from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import yfinance as yf

def evaluate_forecast(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "Model": model_name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2)
    }

def get_stock_data(ticker="AAPL", start="2018-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv("data/stock_data.csv")
    return df

if __name__ == "__main__":
    print(get_stock_data())