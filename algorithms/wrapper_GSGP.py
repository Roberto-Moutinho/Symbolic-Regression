
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from slim_gsgp.main_slim import slim
except ImportError as exc:
    raise SystemExit(
        "Não foi possível importar slim_gsgp. Instale/ative o ambiente Python "
        "que contém a implementação SLIM-GSGP antes de executar este wrapper."
    ) from exc


OPERATOR_SETS = {
    "C1": ["add", "subtract", "multiply", "divide"],
    "C2": ["add", "subtract", "multiply", "divide", "power", "exp", "log", "sqrt"],
}


def _protected_power(base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """Numerically protected power: |base|**clipped_exponent."""
    return torch.pow(torch.abs(base) + 1e-12, torch.clamp(exponent, min=-10.0, max=10.0))


def _protected_exp(value: torch.Tensor) -> torch.Tensor:
    """Avoid overflow while preserving exp over a broad practical range."""
    return torch.exp(torch.clamp(value, min=-20.0, max=20.0))


def _protected_log(value: torch.Tensor) -> torch.Tensor:
    """Protected natural logarithm, defined for every real input."""
    return torch.log(torch.abs(value) + 1e-12)


def _protected_sqrt(value: torch.Tensor) -> torch.Tensor:
    """Protected square root, defined for every real input."""
    return torch.sqrt(torch.abs(value))


def configure_operator_registry(scenario: str) -> None:
    """Register operators in the exact globals used by the imported ``slim``.

    ``main_slim.py`` uses ``from slim_gsgp.config.slim_config import *``.
    Patching only ``slim_config.FUNCTIONS`` can miss the dictionary that the
    already-imported ``slim`` function actually reads. Therefore patch the
    function's own ``__globals__`` mapping and its initializer configuration.
    """
    if scenario != "C2":
        return

   
    slim_globals = slim.__globals__
    functions = slim_globals.get("FUNCTIONS")
    if not isinstance(functions, dict):
        raise RuntimeError(
            "Não encontrei o dicionário FUNCTIONS no escopo de slim_gsgp.main_slim.slim."
        )

    extra_functions = {
        "power": {"function": _protected_power, "arity": 2},
        "exp": {"function": _protected_exp, "arity": 1},
        "log": {"function": _protected_log, "arity": 1},
        "sqrt": {"function": _protected_sqrt, "arity": 1},
    }
    functions.update(extra_functions)

    pi_init = slim_globals.get("slim_gsgp_pi_init")
    if not isinstance(pi_init, dict):
        raise RuntimeError("Não encontrei slim_gsgp_pi_init no escopo da função slim().")
    pi_init["FUNCTIONS"] = functions

   
    try:
        from slim_gsgp.config import slim_config
        slim_config.FUNCTIONS.update(extra_functions)
        slim_config.slim_gsgp_pi_init["FUNCTIONS"] = functions
    except (ImportError, AttributeError):
        pass

    missing = set(OPERATOR_SETS[scenario]) - set(functions)
    if missing:
        raise RuntimeError(
            "Não foi possível registrar todos os operadores C2: "
            + ", ".join(sorted(missing))
        )
    print("Operadores ativos em slim(): " + ", ".join(OPERATOR_SETS[scenario]), flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_dataset_files(dataset: str, data_root: Path) -> tuple[Path, Path]:
    train_path = data_root / f"{dataset}_train.csv"
    test_path = data_root / f"{dataset}_test.csv"
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path
    raise FileNotFoundError(
        f"Não encontrei os arquivos do dataset {dataset!r}: {train_path} e {test_path}"
    )


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"O arquivo {path} não contém a coluna obrigatória 'target'.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise ValueError(f"O arquivo {path} ficou vazio após remover valores ausentes/infinito.")
    feature_columns = [c for c in df.columns if c != "target"]
    if not feature_columns:
        raise ValueError(f"O arquivo {path} não contém colunas de atributos.")
    X = df.loc[:, feature_columns].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)
    return X, y, feature_columns


def safe_metric(metric_fn, y_true, y_pred) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0] or not np.all(np.isfinite(y_pred)):
        return float("inf") if metric_fn is mean_squared_error else float("-inf")
    try:
        value = float(metric_fn(y_true, y_pred))
        return value if math.isfinite(value) else (
            float("inf") if metric_fn is mean_squared_error else float("-inf")
        )
    except Exception:
        return float("inf") if metric_fn is mean_squared_error else float("-inf")


def as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def normalize_expression(value) -> str:
    """Return a stable textual expression; never expose NumPy repr wrappers."""
    try:
        expr = str(value).replace("\n", " ").replace("\r", " ").strip()
        expr = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\2", expr)
        return expr or "NA"
    except Exception:
        return "NA"


def write_result(path: Path, row: dict) -> None:
    """Write exactly one JSON file containing a one-element model list."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_safe(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value

    function_set = row.get("function_set", [])
    if isinstance(function_set, str):
        function_set = [op for op in function_set.split(",") if op]

    record = {
        "model_id": 0,
        "expression": row.get("Expression", "NA"),
        "complexity": row.get("size"),
        "mse_train": row.get("MSE_train"),
        "mse_test": row.get("MSE_test"),
        "r2_train": row.get("R2_train"),
        "r2_test": row.get("R2_test"),
        "seed": row.get("seed"),
        "algorithm": row.get("algorithm", "GSGP"),
        "dataset": row.get("dataset"),
        "scenario": row.get("scenario"),
        "fold": int(row.get("fold")) if str(row.get("fold", "")).isdigit() else row.get("fold"),
        "runtime_seconds": row.get("runtime_seconds"),
        "nll_train": row.get("nll_train", row.get("MSE_train")),
        "nll_test": row.get("nll_test", row.get("MSE_test")),
        "n_samples": row.get("n_samples"),
        "function_set": function_set,
        "pop_size": row.get("pop_size"),
        "n_iter": row.get("n_iter"),
        "n_train": row.get("n_train"),
        "n_test": row.get("n_test"),
        "n_features": row.get("n_features"),
        "source_dataset": row.get("source_dataset"),
        "status": "success" if row.get("status") == "ok" else row.get("status", "error"),
        "error": row.get("error"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump([json_safe(record)], f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa SLIM-GSGP em um fold SRBench.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", choices=["C1", "C2"], default="C1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=str, required=True)
    parser.add_argument("--output", required=True, help="Caminho do único arquivo JSON de saída.")
    parser.add_argument("--data-root", default="dados/datasets")
    parser.add_argument("--pop-size", type=int, default=500)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--max-evaluations", type=int, default=None,
                        help="Compatibilidade com CLI comum; SLIM-GSGP usa n_iter, não avaliações exatas.")
    parser.add_argument("--ms-lower", type=float, default=0.0)
    parser.add_argument("--ms-upper", type=float, default=1.0)
    parser.add_argument("--p-inflate", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--simulate", action="store_true",
                        help="Gera uma linha de status simulate sem executar o algoritmo.")
    args = parser.parse_args()

    output_path = Path(args.output)
    start = time.perf_counter()
    if args.simulate:
        write_result(output_path, {
            "index": 0, "Id": "simulate", "Expression": "NA", "size": 0,
            "MSE_train": "", "MSE_test": "", "nll_train": "", "nll_test": "",
            "R2_train": "", "R2_test": "", "algorithm": "GSGP",
            "dataset": args.dataset, "scenario": args.scenario, "seed": args.seed,
            "fold": args.fold, "runtime_seconds": 0.0, "status": "simulate",
            "n_samples": 0, "function_set": OPERATOR_SETS[args.scenario],
            "pop_size": args.pop_size, "n_iter": args.n_iter, "n_train": 0,
            "n_test": 0, "n_features": 0, "source_dataset": None, "error": None,
        })
        return 0

    if args.pop_size < 1 or args.n_iter < 1:
        parser.error("--pop-size e --n-iter devem ser positivos.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda foi solicitado, mas CUDA não está disponível.")

    set_seed(args.seed)
    train_path, test_path = find_dataset_files(args.dataset, Path(args.data_root))
    X_all_np, y_all_np, feature_names = load_xy(train_path)
    X_test_np, y_test_np, _ = load_xy(test_path)

    # Reproduce the project's five-fold protocol: each fold's held-out portion
    # is validation for model selection; the official test file is only scored
    # after fitting. The seed controls the fold assignment and algorithm RNG.
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(kfold.split(X_all_np))
    try:
        fit_idx, val_idx = splits[int(args.fold) - 1]
    except (ValueError, IndexError):
        raise ValueError("--fold deve estar entre 1 e 5")
    X_fit_np, y_fit_np = X_all_np[fit_idx], y_all_np[fit_idx]
    X_val_np, y_val_np = X_all_np[val_idx], y_all_np[val_idx]
    function_set = list(OPERATOR_SETS[args.scenario])
    configure_operator_registry(args.scenario)

    dtype = torch.float64
    device = torch.device(args.device)
    X_train_all = torch.as_tensor(X_fit_np, dtype=dtype, device=device)
    y_train_all = torch.as_tensor(y_fit_np, dtype=dtype, device=device)
    X_test = torch.as_tensor(X_test_np, dtype=dtype, device=device)
    y_test = torch.as_tensor(y_test_np, dtype=dtype, device=device)

    X_val = torch.as_tensor(X_val_np, dtype=dtype, device=device)
    y_val = torch.as_tensor(y_val_np, dtype=dtype, device=device)

    reg = slim(
        X_train=X_train_all,
        y_train=y_train_all,
        X_test=X_val,
        y_test=y_val,
        dataset_name=args.dataset,
        slim_version="SLIM*ABS",
        pop_size=args.pop_size,
        n_iter=args.n_iter,
        tree_functions=function_set,
        ms_lower=args.ms_lower,
        ms_upper=args.ms_upper,
        p_inflate=args.p_inflate,
        seed=args.seed,
        verbose=0,
    )

    pred_train = as_numpy(reg.predict(torch.as_tensor(X_all_np, dtype=dtype, device=device))).reshape(-1)
    pred_test = as_numpy(reg.predict(X_test)).reshape(-1)
    mse_train = safe_metric(mean_squared_error, y_all_np, pred_train)
    mse_test = safe_metric(mean_squared_error, y_test_np, pred_test)
    r2_train = safe_metric(r2_score, y_all_np, pred_train)
    r2_test = safe_metric(r2_score, y_test_np, pred_test)

    try:
        expression = normalize_expression(reg.get_tree_representation())
    except Exception:
        expression = "NA"
    try:
        size = int(reg.nodes_count)
    except Exception:
        size = -1

    runtime = time.perf_counter() - start
    status = "ok" if np.isfinite(mse_test) and np.isfinite(r2_test) else "non_finite_metrics"
    write_result(output_path, {
        "index": 0, "Id": 0, "Expression": expression, "size": size,
        "MSE_train": mse_train, "MSE_test": mse_test,
     
        "nll_train": mse_train, "nll_test": mse_test,
        "R2_train": r2_train, "R2_test": r2_test,
        "algorithm": "GSGP", "dataset": args.dataset, "scenario": args.scenario,
        "seed": args.seed, "fold": args.fold, "runtime_seconds": runtime,
        "status": status, "n_samples": int(len(y_all_np)),
        "function_set": function_set, "pop_size": args.pop_size,
        "n_iter": args.n_iter, "n_train": int(len(X_fit_np)),
        "n_test": int(len(X_val_np)), "n_features": int(X_all_np.shape[1]),
        "source_dataset": str(train_path), "error": None if status == "ok" else "Métricas não finitas",
    })
    print(f"Resultado salvo em: {output_path}")
    print(f"status={status} MSE_test={mse_test} R2_test={r2_test} size={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
