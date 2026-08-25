from __future__ import annotations

from typing import Any, Callable, Mapping

import optuna

from .search_space import build_search_space


Evaluator = Callable[
    [dict[str, Any], optuna.Trial],
    float,
]


def create_objective(
    parameters: Mapping[str, Mapping[str, Any]],
    evaluator: Evaluator,
) -> Callable[[optuna.Trial], float]:
   

    def objective(trial: optuna.Trial) -> float:
        params = build_search_space(
            trial=trial,
            parameters=parameters,
        )

        value = evaluator(
            params,
            trial,
        )

        if not isinstance(value, (int, float)):
            raise TypeError(
                "A função evaluator deve retornar um valor numérico."
            )

        if value != value:
            raise ValueError(
                "A função evaluator retornou NaN."
            )

        return float(value)

    return objective
