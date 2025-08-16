import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from src.utils import evaluate_forecast

def run_arima(df, column="Close", order=(5,1,0)):
    """
    Train ARIMA model and evaluate forecast.

    Parameters:
        df (pd.DataFrame): Stock data
        column (str): Column to forecast
        order (tuple): ARIMA order (p,d,q)

    Returns:
        dict: Metrics dictionary from evaluate_forecast
    """
    print("\nRunning ARIMA...")
    
    # Use only the target column
    data = df[column]
    
    # Split into train/test
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    
    # Ensure forecast is 1D
    forecast = np.ravel(forecast)
    
    # Evaluate
    metrics = evaluate_forecast(np.ravel(test.values), forecast, model_name="ARIMA")
    
    print("ARIMA metrics:", metrics)
    return metrics
