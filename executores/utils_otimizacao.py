from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna não está instalado. Execute: pip install optuna pandas"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "hiperparametros_treinamento.csv"
DEFAULT_OUTPUT_DIR = ROOT / "resultados_otimizacao"


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_number(value: Any) -> float | int | None:
    s = _clean(value)
    if not s:
        return None

    # Ex.: 14400(4h) -> 14400
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = s.replace(",", ".")
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", s)
    if not m:
        return None

    n = float(m.group())
    return int(n) if n.is_integer() else n


def _parse_range(value: Any) -> tuple[float | int, float | int] | None:
    s = _clean(value)
    if not s or s in {"—", "-", "–"}:
        return None

    # Remove espaços e colchetes.
    s = s.replace("[", "").replace("]", "").strip()
    # Captura números com ponto ou vírgula decimal. O sinal negativo só é
    # aceito no início de um número; assim, `10-100` é interpretado como
    # intervalo e `1e-4` continua sendo notação científica.
    nums = re.findall(
        r"(?<![\\d.])[+\\-]?(?:\\d+(?:[.,]\\d*)?|[.,]\\d+)"
        r"(?:[eE][+\\-]?\\d+)?",
        s,
    )
    if len(nums) < 2:
        return None

    a, b = float(nums[0].replace(",", ".")), float(nums[1].replace(",", "."))
    if a > b:
        a, b = b, a

    def normalize(x: float):
        return int(x) if x.is_integer() else x

    return normalize(a), normalize(b)


def _infer_type(row: pd.Series) -> str:
    declared = _clean(row.get("Tipo"))
    current = _parse_number(row.get("Valor atual"))
    low_high = _parse_range(row.get("Faixa proposta"))

    if declared.lower() == "inteiro":
        # Corrige inconsistências evidentes da tabela, como 0.1 marcado
        # como inteiro. Se houver parte decimal, trata como contínuo.
        if current is not None and not float(current).is_integer():
            return "float"
        return "int"

    if declared.lower() in {"contínuo", "continuo", "float", "real"}:
        return "float"

    if low_high:
        lo, hi = low_high
        if isinstance(lo, int) and isinstance(hi, int):
            if current is None or float(current).is_integer():
                return "int"
        return "float"

    return "fixed"


def load_algorithm_space(csv_path: Path, algorithm: str) -> tuple[dict, list[str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["Algoritmo"] = df["Algoritmo"].replace("", pd.NA).ffill()

    subset = df[df["Algoritmo"].str.strip().str.lower() == algorithm.lower()].copy()
    if subset.empty:
        available = ", ".join(df["Algoritmo"].dropna().unique())
        raise ValueError(
            f"Algoritmo '{algorithm}' não encontrado. Disponíveis: {available}"
        )

    search_space: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for _, row in subset.iterrows():
        name = _clean(row["Hiperparâmetro"])
        if not name or name.lower() == "nan":
            continue

        current = _parse_number(row["Valor atual"])
        bounds = _parse_range(row["Faixa proposta"])
        kind = _infer_type(row)

        # Parâmetros categóricos/listas sem faixa: permanecem fixos.
        if bounds is None:
            search_space[name] = {
                "kind": "fixed",
                "value": _clean(row["Valor atual"]),
                "current": current if current is not None else _clean(row["Valor atual"]),
            }
            continue

        low, high = bounds
        if current is not None and (current < low or current > high):
            warnings.append(
                f"{name}: valor atual {current} está fora da faixa [{low}, {high}]. "
                "A otimização usará a faixa proposta."
            )

        if kind == "int":
            search_space[name] = {
                "kind": "int",
                "low": int(low),
                "high": int(high),
                "current": current,
            }
        else:
            search_space[name] = {
                "kind": "float",
                "low": float(low),
                "high": float(high),
                "current": current,
            }

    return search_space, warnings


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict[str, Any]:
    params = {}
    for name, spec in search_space.items():
        kind = spec["kind"]

        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        else:
            params[name] = spec["value"]

    return params


def _load_callable(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("O avaliador deve estar no formato modulo:funcao")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{spec} não é uma função chamável.")
    return fn


def _metric_from_result(result: Any, metric: str) -> float:
    if isinstance(result, (int, float)):
        value = float(result)
    elif isinstance(result, dict):
        if metric not in result:
            raise KeyError(
                f"A métrica '{metric}' não foi encontrada no resultado. "
                f"Chaves disponíveis: {list(result)}"
            )
        value = float(result[metric])
    else:
        raise TypeError(
            "O avaliador deve retornar float/int ou dict contendo a métrica."
        )

    if not math.isfinite(value):
        raise ValueError(f"Métrica inválida: {value}")
    return value


def _run_command(
    command_template: str,
    params: dict,
    trial_number: int,
    work_dir: Path,
    metric: str,
) -> float:
    trial_dir = work_dir / f"trial_{trial_number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    params_file = trial_dir / "params.json"
    result_file = trial_dir / "result.json"

    params_file.write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    command = command_template.format(
        params_json=json.dumps(params, ensure_ascii=False),
        params_file=str(params_file),
        result_file=str(result_file),
        trial_number=trial_number,
    )

    completed = subprocess.run(
        command,
        shell=True,
        cwd=work_dir,
        text=True,
        capture_output=True,
    )

    (trial_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (trial_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Comando do trial {trial_number} terminou com código "
            f"{completed.returncode}.\nSTDERR:\n{completed.stderr}"
        )

    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        return _metric_from_result(result, metric)

    # Também aceita que o programa imprima JSON no stdout.
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        return _metric_from_result(result, metric)
    except Exception as exc:
        raise RuntimeError(
            "O comando terminou sem produzir result.json nem JSON válido no stdout."
        ) from exc


def run_optimization(
    algorithm: str,
    csv_path: Path = DEFAULT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_trials: int = 50,
    seed: int = 42,
    direction: str = "minimize",
    metric: str = "mse",
    evaluator: str | None = None,
    command: str | None = None,
) -> Path:
    if evaluator is None and command is None:
        raise ValueError(
            "Informe --evaluator modulo:funcao ou --command '...'. "
            "A tabela CSV fornece os hiperparâmetros, mas não a função objetivo."
        )
    if evaluator is not None and command is not None:
        raise ValueError("Use --evaluator OU --command, não os dois.")

    search_space, warnings = load_algorithm_space(csv_path, algorithm)

    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_dir = output_dir / algorithm.lower().replace("-", "_")
    algorithm_dir.mkdir(parents=True, exist_ok=True)

    (algorithm_dir / "warnings.txt").write_text(
        "\n".join(warnings) if warnings else "Nenhum aviso.\n",
        encoding="utf-8",
    )

    evaluator_fn = _load_callable(evaluator) if evaluator else None

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"hp_{algorithm}",
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space)
        trial.set_user_attr("parameters", params)

        if evaluator_fn is not None:
            result = evaluator_fn(params)
            return _metric_from_result(result, metric)

        return _run_command(
            command, params, trial.number, algorithm_dir, metric
        )

    study.optimize(objective, n_trials=n_trials)

    best = {
        "algorithm": algorithm,
        "metric": metric,
        "direction": direction,
        "n_trials": n_trials,
        "seed": seed,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "fixed_params": {
            k: v["value"]
            for k, v in search_space.items()
            if v["kind"] == "fixed"
        },
    }

    best_path = algorithm_dir / "melhores_hiperparametros.json"
    best_path.write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    study.trials_dataframe().to_csv(
        algorithm_dir / "historico_trials.csv",
        index=False,
    )

    print(f"\n[{algorithm}] otimização concluída.")
    print(f"Melhor {metric}: {study.best_value}")
    print(json.dumps(study.best_params, ensure_ascii=False, indent=2))
    print(f"Resultado: {best_path}")

    if warnings:
        print("\nAvisos da tabela:")
        for warning in warnings:
            print(f"- {warning}")

    return best_path


def build_parser(algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Otimização de hiperparâmetros para {algorithm}."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default="minimize",
    )
    parser.add_argument("--metric", default="mse")
    parser.add_argument(
        "--evaluator",
        help="Função Python no formato modulo:funcao, recebendo params e retornando métrica.",
    )
    parser.add_argument(
        "--command",
        help=(
            "Comando externo. Placeholders disponíveis: "
            "{params_json}, {params_file}, {result_file}, {trial_number}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser
