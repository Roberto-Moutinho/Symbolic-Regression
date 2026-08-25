"""
Criação e execução de estudos Optuna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import optuna

from .sampler import create_sampler


def create_study(
    study_name: str,
    direction: str = "minimize",
    seed: Optional[int] = 42,
    storage: Optional[str] = None,
    sampler_name: str = "tpe",
    load_if_exists: bool = True,
) -> optuna.Study:
   

    direction = direction.lower()

    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "direction deve ser 'minimize' ou 'maximize'."
        )

    sampler = create_sampler(
        seed=seed,
        sampler_name=sampler_name,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=load_if_exists,
    )

    return study


def optimize_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    timeout: Optional[int] = None,
) -> optuna.Study:
   

    if n_trials <= 0:
        raise ValueError("n_trials deve ser maior que zero.")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    return study


def save_best_parameters(
    study: optuna.Study,
    output_path: str | Path,
) -> None:
  

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(f"Study: {study.study_name}\n")
        file.write(f"Best value: {study.best_value}\n\n")
        file.write("Best parameters:\n")

        for name, value in study.best_params.items():
            file.write(f"{name} = {value}\n")
