from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_forecast(y_true, y_pred, model_name="Model"):
    """
    Calculate common regression metrics for forecasts.

    Parameters:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        model_name (str): Name of the model (for reference)

    Returns:
        dict: Dictionary with RMSE, MAE, MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "Model": model_name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2)
    }
