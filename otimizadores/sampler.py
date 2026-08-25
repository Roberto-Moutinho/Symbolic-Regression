"""
Configuração dos samplers utilizados pelo Optuna.
"""

from __future__ import annotations

from typing import Optional

import optuna


def create_sampler(
    seed: Optional[int] = 42,
    sampler_name: str = "tpe",
) -> optuna.samplers.BaseSampler:
    """
    Cria o sampler utilizado pelo estudo Optuna.

    Parâmetros
    ----------
    seed:
        Semente aleatória para reprodutibilidade.

    sampler_name:
        Nome do sampler.

        Atualmente suportados:
        - "tpe"
        - "random"

    Retorna
    -------
    optuna.samplers.BaseSampler
        Sampler configurado.
    """

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
