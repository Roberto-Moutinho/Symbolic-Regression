from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optuna não está instalado. Execute: pip install optuna pandas"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "dados" / "hiperparametros_treinamento_corrigido.csv"
DEFAULT_OUTPUT_DIR = ROOT / "resultados_otimizacao"

Evaluator = Callable[[dict[str, Any], optuna.Trial], Any]
PostprocessParams = Callable[[dict[str, Any]], dict[str, Any]]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _parse_number(value: Any) -> float | int | None:
    s = _clean(value)
    if not s:
        return None
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

    s = s.replace("[", "").replace("]", "").strip()
    nums = re.findall(
        r"(?<![\d.])[+\-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+\-]?\d+)?",
        s,
    )
    if len(nums) < 2:
        return None

    a = float(nums[0].replace(",", "."))
    b = float(nums[1].replace(",", "."))
    if a > b:
        a, b = b, a

    return (int(a) if a.is_integer() else a, int(b) if b.is_integer() else b)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aceita a tabela antiga e a tabela atual do projeto."""
    aliases = {
        "algoritmo": "algorithm",
        "Algoritmo": "algorithm",
        "hiperparametro": "parameter",
        "Hiperparâmetro": "parameter",
        "tipo": "type",
        "Tipo": "type",
        "valor_padrao": "default",
        "Valor atual": "default",
        "limite_inferior": "lower",
        "limite_superior": "upper",
        "Faixa proposta": "range",
    }
    rename = {c: aliases[c] for c in df.columns if c in aliases}
    df = df.rename(columns=rename).copy()

    required = {"algorithm", "parameter", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Tabela de hiperparâmetros sem colunas obrigatórias: {sorted(missing)}. "
            f"Colunas encontradas: {list(df.columns)}"
        )

    if "default" not in df.columns:
        df["default"] = ""
    if "lower" not in df.columns:
        df["lower"] = ""
    if "upper" not in df.columns:
        df["upper"] = ""
    if "range" not in df.columns:
        df["range"] = ""

    return df


def _infer_type(row: pd.Series) -> str:
    declared = _clean(row.get("type")).lower()
    current = _parse_number(row.get("default"))

    if declared in {"integer", "inteiro", "int"}:
        return "int"
    if declared in {"float", "real", "continuous", "contínuo", "continuo"}:
        return "float"
    if declared in {"categorical", "categórico", "categorico"}:
        return "categorical"

    bounds = _parse_range(row.get("range"))
    if bounds is None:
        lo = _parse_number(row.get("lower"))
        hi = _parse_number(row.get("upper"))
        bounds = (lo, hi) if lo is not None and hi is not None else None

    if bounds and all(isinstance(x, int) for x in bounds):
        return "int"
    if bounds:
        return "float"
    if current is not None and float(current).is_integer():
        return "int"
    return "fixed"


def load_algorithm_space(
    csv_path: Path,
    algorithm: str,
    exclude_parameters: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = _normalise_columns(df)
    df["algorithm"] = df["algorithm"].replace("", pd.NA).ffill()

    subset = df[
        df["algorithm"].astype(str).str.strip().str.lower() == algorithm.lower()
    ].copy()
    if subset.empty:
        available = ", ".join(sorted(subset["algorithm"].dropna().unique()))
        raise ValueError(f"Algoritmo '{algorithm}' não encontrado no CSV.")

    excluded = {x.lower() for x in (exclude_parameters or set())}
    search_space: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for _, row in subset.iterrows():
        name = _clean(row["parameter"])
        if not name or name.lower() in excluded:
            continue

        kind = _infer_type(row)
        current = _parse_number(row.get("default"))
        low = _parse_number(row.get("lower"))
        high = _parse_number(row.get("upper"))

        if (low is None or high is None) and _clean(row.get("range")):
            parsed = _parse_range(row.get("range"))
            if parsed:
                low, high = parsed

        if kind == "categorical":
            raw = _clean(row.get("default"))
            try:
                choices = json.loads(raw)
                if not isinstance(choices, list) or not choices:
                    raise ValueError
            except Exception:
                choices = [raw]
            search_space[name] = {
                "kind": "categorical",
                "choices": choices,
                "current": raw,
            }
            continue

        if low is None or high is None:
            search_space[name] = {
                "kind": "fixed",
                "value": current if current is not None else _clean(row.get("default")),
                "current": current,
            }
            continue

        if low > high:
            low, high = high, low

        if current is not None and not (float(low) <= float(current) <= float(high)):
            warnings.append(
                f"{name}: valor padrão {current} está fora de [{low}, {high}]."
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


def suggest_params(trial: optuna.Trial, search_space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["kind"]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            params[name] = spec["value"]
    return params


def _metric_from_result(result: Any, metric: str) -> float:
    if isinstance(result, (int, float)):
        value = float(result)
    elif isinstance(result, dict):
        if metric not in result:
            raise KeyError(
                f"A métrica '{metric}' não foi encontrada. Chaves: {list(result)}"
            )
        value = float(result[metric])
    else:
        raise TypeError("O avaliador deve retornar número ou dict contendo a métrica.")

    if not math.isfinite(value):
        raise ValueError(f"Métrica inválida: {value}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _call_evaluator(evaluator: Callable, params: dict[str, Any], trial: optuna.Trial):
    """Permite avaliadores antigos (params) e novos (params, trial)."""
    try:
        signature = inspect.signature(evaluator)
        positional = [
            p for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return evaluator(params, trial)
    except (TypeError, ValueError):
        pass
    return evaluator(params)


def _run_command(command_template: str, params: dict, trial_number: int, work_dir: Path, metric: str) -> float:
    trial_dir = work_dir / f"trial_{trial_number:05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    params_file = trial_dir / "params.json"
    result_file = trial_dir / "result.json"
    params_file.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    command = command_template.format(
        params_json=json.dumps(params, ensure_ascii=False),
        params_file=str(params_file),
        result_file=str(result_file),
        trial_number=trial_number,
    )
    completed = subprocess.run(command, shell=True, cwd=work_dir, text=True, capture_output=True)
    (trial_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (trial_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Comando do trial {trial_number} terminou com código {completed.returncode}.\n"
            f"STDERR:\n{completed.stderr}"
        )

    if result_file.exists():
        return _metric_from_result(json.loads(result_file.read_text(encoding="utf-8")), metric)

    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        return _metric_from_result(result, metric)
    except Exception as exc:
        raise RuntimeError("O comando não produziu result.json nem JSON válido no stdout.") from exc


def _select_best_trial(study: optuna.Study, direction: str, tie_break_key: str = "mean_complexity") -> optuna.trial.FrozenTrial:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Nenhum trial terminou com sucesso.")

    if direction == "maximize":
        best_value = max(t.value for t in completed if t.value is not None)
        candidates = [t for t in completed if t.value is not None and abs(t.value - best_value) <= 1e-12]
    else:
        best_value = min(t.value for t in completed if t.value is not None)
        candidates = [t for t in completed if t.value is not None and abs(t.value - best_value) <= 1e-12]

    if len(candidates) == 1:
        return candidates[0]

    def complexity(t):
        value = t.user_attrs.get(tie_break_key, float("inf"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    return min(candidates, key=complexity)


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
    evaluator_fn: Callable | None = None,
    exclude_parameters: set[str] | None = None,
    postprocess_params: PostprocessParams | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if evaluator_fn is not None and (evaluator is not None or command is not None):
        raise ValueError("Use evaluator_fn ou evaluator/command, não ambos.")
    if evaluator_fn is None and evaluator is None and command is None:
        raise ValueError("Informe evaluator_fn, --evaluator ou --command.")
    if evaluator is not None and command is not None:
        raise ValueError("Use --evaluator OU --command, não os dois.")
    if n_trials <= 0:
        raise ValueError("n_trials deve ser maior que zero.")

    search_space, warnings = load_algorithm_space(
        csv_path, algorithm, exclude_parameters=exclude_parameters
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_dir = output_dir / algorithm.lower().replace("-", "_")
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    (algorithm_dir / "warnings.txt").write_text(
        "\n".join(warnings) if warnings else "Nenhum aviso.\n", encoding="utf-8"
    )

    evaluator_callable = evaluator_fn or (_load_callable(evaluator) if evaluator else None)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"hp_{algorithm}_{seed}",
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space)
        if postprocess_params is not None:
            params = postprocess_params(dict(params))

        trial.set_user_attr("parameters", _jsonable(params))

        if evaluator_callable is not None:
            result = _call_evaluator(evaluator_callable, params, trial)
            value = _metric_from_result(result, metric)
            if isinstance(result, dict):
                for key in ("mean_complexity", "std_r2", "fold_results", "mean_r2"):
                    if key in result:
                        trial.set_user_attr(key, _jsonable(result[key]))
            return value

        return _run_command(command, params, trial.number, algorithm_dir, metric)

    study.optimize(objective, n_trials=n_trials)

    selected_trial = _select_best_trial(study, direction)
    selected_params = dict(selected_trial.params)
    if postprocess_params is not None:
        selected_params = postprocess_params(selected_params)

    fixed_params = {
        k: v["value"] for k, v in search_space.items() if v["kind"] == "fixed"
    }
    all_best_params = {**fixed_params, **selected_params}

    best = {
        "algorithm": algorithm,
        "metric": metric,
        "direction": direction,
        "n_trials": n_trials,
        "seed": seed,
        "best_value": selected_trial.value,
        "best_trial_number": selected_trial.number,
        "best_params": all_best_params,
        "trial_user_attrs": _jsonable(dict(selected_trial.user_attrs)),
        "fixed_params": fixed_params,
        "metadata": _jsonable(metadata or {}),
        "optimization": "5-fold cross-validation when evaluator implements it",
    }

    best_path = algorithm_dir / "melhores_hiperparametros.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    study.trials_dataframe().to_csv(algorithm_dir / "historico_trials.csv", index=False)

    print(f"\n[{algorithm}] otimização concluída.")
    print(f"Melhor {metric}: {selected_trial.value}")
    print(json.dumps(all_best_params, ensure_ascii=False, indent=2))
    print(f"Trial selecionado: {selected_trial.number}")
    print(f"Resultado: {best_path}")

    return best_path


def _load_callable(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("O avaliador deve estar no formato modulo:funcao")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{spec} não é uma função chamável.")
    return fn


def build_parser(algorithm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Otimização de hiperparâmetros para {algorithm}.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--evaluator", help="Função Python no formato modulo:funcao.")
    parser.add_argument("--command", help="Comando externo com placeholders de trial.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser
