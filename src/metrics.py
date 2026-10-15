import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(true, predicted):
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(true) != len(predicted) or len(true) < 2:
        raise ValueError("Metrics need equally sized arrays with at least two samples.")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite target or prediction.")
    pcc = float(np.corrcoef(true, predicted)[0, 1]) if (
        np.std(true) > 0 and np.std(predicted) > 0) else None
    mse = float(mean_squared_error(true, predicted))
    return {"PCC": pcc, "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(true, predicted)),
            "MSE": mse, "R2": float(r2_score(true, predicted))}
