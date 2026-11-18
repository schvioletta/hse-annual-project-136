"""Метрики качества прогноза цены."""

import numpy as np

PEAK = 300  # EUR/MWh, порог скачка из EDA


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE и MAE отдельно на часах с ценой <= 0 и > 300.

    MAPE не считаем: цена бывает нулевой и отрицательной.
    """
    y_true = np.asarray(y_true, dtype=float)
    err = y_true - np.asarray(y_pred, dtype=float)

    def mae(mask) -> float:
        return float(np.abs(err[mask]).mean()) if mask.any() else float("nan")

    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAE_le0": mae(y_true <= 0),
        "MAE_gt300": mae(y_true > PEAK),
    }
