

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .metricas import calcular_metricas


def validar_predicoes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula as métricas de uma série de predições.
    """

    return calcular_metricas(
        y_true=y_true,
        y_pred=y_pred,
    )


def avaliar_train(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    

    metricas = validar_predicoes(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "MSE_train": metricas["MSE"],
        "RMSE_train": metricas["RMSE"],
        "MAE_train": metricas["MAE"],
        "R2_train": metricas["R2"],
        "MAPE_train": metricas["MAPE"],
    }


def avaliar_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
   

    metricas = validar_predicoes(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "MSE_test": metricas["MSE"],
        "RMSE_test": metricas["RMSE"],
        "MAE_test": metricas["MAE"],
        "R2_test": metricas["R2"],
        "MAPE_test": metricas["MAPE"],
    }


def avaliar_modelo(
    train_predictor: Callable[..., np.ndarray],
    test_predictor: Callable[..., np.ndarray],
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> dict[str, float]:
    

    y_pred_train = train_predictor(X_train)
    y_pred_test = test_predictor(X_test)

    resultados_train = avaliar_train(
        y_true=y_train,
        y_pred=y_pred_train,
    )

    resultados_test = avaliar_test(
        y_true=y_test,
        y_pred=y_pred_test,
    )

    return {
        **resultados_train,
        **resultados_test,
    }
