

from __future__ import annotations

from typing import Optional

import optuna


def create_sampler(
    seed: Optional[int] = 42,
    sampler_name: str = "tpe",
) -> optuna.samplers.BaseSampler:
   

    name = sampler_name.lower()

    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
        )

    if name == "random":
        return optuna.samplers.RandomSampler(
            seed=seed,
        )

    raise ValueError(
        f"Sampler desconhecido: '{sampler_name}'. "
        "Opções disponíveis: 'tpe' e 'random'."
    )
