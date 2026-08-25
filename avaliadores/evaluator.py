

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna

from .metricas import calcular_metricas


@dataclass
class EvaluationResult:
    """
    Resultado da avaliação de um modelo.
    """

    metrics: dict[str, float]
    predictions: np.ndarray


class Evaluator:
    

    def __init__(
        self,
        runner: Callable[
            [dict[str, Any], Any],
            np.ndarray,
        ],
        metric: str = "MSE",
    ) -> None:
       

        self.runner = runner
        self.metric = metric.upper()

    def evaluate(
        self,
        params: dict[str, Any],
        X: Any,
        y: np.ndarray,
        trial: optuna.Trial | None = None,
    ) -> EvaluationResult:
       

        predictions = self.runner(
            params,
            X,
        )

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        metrics = calcular_metricas(
            y_true=y,
            y_pred=predictions,
        )

        if trial is not None:
            for name, value in metrics.items():
                trial.set_user_attr(
                    name,
                    value,
                )

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
        )

    def objective(
        self,
        params: dict[str, Any],
        X: Any,
        y: np.ndarray,
        trial: optuna.Trial | None = None,
    ) -> float:
       

        result = self.evaluate(
            params=params,
            X=X,
            y=y,
            trial=trial,
        )

        if self.metric not in result.metrics:
            raise ValueError(
                f"Métrica '{self.metric}' não encontrada. "
                f"Métricas disponíveis: "
                f"{list(result.metrics.keys())}"
            )

        return result.metrics[self.metric]
