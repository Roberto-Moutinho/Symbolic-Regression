

from __future__ import annotations

import math

import numpy as np


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida e normaliza os vetores de valores reais e preditos.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred devem possuir o mesmo tamanho."
        )

    if y_true.ndim != 1:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.all(valid):
        raise ValueError(
            "y_true ou y_pred contém valores NaN ou infinitos."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Os vetores de avaliação não podem estar vazios."
        )

    return y_true, y_pred


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula o Mean Squared Error (MSE).
    """

    y_true, y_pred = _validate_inputs(
        y_true,
        y_pred,
    )

    return float(
        np.mean((y_true - y_pred) ** 2)
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula o Root Mean Squared Error (RMSE).
    """

    return float(
        math.sqrt(
            mse(y_true, y_pred)
        )
    )


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula o Mean Absolute Error (MAE).
    """

    y_true, y_pred = _validate_inputs(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula o coeficiente de determinação R².
    """

    y_true, y_pred = _validate_inputs(
        y_true,
        y_pred,
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return 0.0

    return float(
        1.0 - (ss_res / ss_tot)
    )


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calcula o Mean Absolute Percentage Error (MAPE).

    Valores reais iguais a zero são ignorados.
    """

    y_true, y_pred = _validate_inputs(
        y_true,
        y_pred,
    )

    non_zero = y_true != 0

    if not np.any(non_zero):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true[non_zero] - y_pred[non_zero])
                / y_true[non_zero]
            )
        )
        * 100
    )


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    

    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
