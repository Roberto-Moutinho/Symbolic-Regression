

from __future__ import annotations

from typing import Any, Mapping

import optuna


def suggest_parameter(
    trial: optuna.Trial,
    name: str,
    config: Mapping[str, Any],
) -> Any:
    

    parameter_type = str(config.get("type", "")).lower()

    if parameter_type == "int":
        if "low" not in config or "high" not in config:
            raise ValueError(
                f"O hiperparâmetro '{name}' precisa de 'low' e 'high'."
            )

        return trial.suggest_int(
            name=name,
            low=int(config["low"]),
            high=int(config["high"]),
            step=int(config.get("step", 1)),
            log=bool(config.get("log", False)),
        )

    if parameter_type == "float":
        if "low" not in config or "high" not in config:
            raise ValueError(
                f"O hiperparâmetro '{name}' precisa de 'low' e 'high'."
            )

        return trial.suggest_float(
            name=name,
            low=float(config["low"]),
            high=float(config["high"]),
            step=config.get("step"),
            log=bool(config.get("log", False)),
        )

    if parameter_type == "categorical":
        choices = config.get("choices")

        if not choices:
            raise ValueError(
                f"O hiperparâmetro '{name}' precisa de 'choices'."
            )

        return trial.suggest_categorical(
            name=name,
            choices=list(choices),
        )

    if parameter_type == "bool":
        return trial.suggest_categorical(
            name=name,
            choices=[True, False],
        )

    raise ValueError(
        f"Tipo de hiperparâmetro desconhecido para '{name}': "
        f"'{parameter_type}'."
    )


def build_search_space(
    trial: optuna.Trial,
    parameters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
   

    suggested_parameters: dict[str, Any] = {}

    for name, config in parameters.items():
        suggested_parameters[name] = suggest_parameter(
            trial=trial,
            name=name,
            config=config,
        )

    return suggested_parameters
