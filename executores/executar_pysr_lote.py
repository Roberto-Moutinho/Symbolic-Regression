from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

EXECUTORES_DIR = Path(__file__).resolve().parent
ROOT = EXECUTORES_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHM = "PySR"
N_FOLDS = 5
MAX_EVALUATIONS = 200_000
MAX_TIME_SECONDS = 3_600
SEEDS = [42, 43, 44, 45, 46]

DATASETS = [
    "chemical_2_competition",
    "friction_dyn_one-hot",
    "nasa_battery_1_10min",
    "nikuradse_1",
]

OPERATOR_SETS = {
    "C1": {
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": [],
    },
    "C2": {
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": ["exp", "log", "sqrt"],
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dataset_path(dataset: str) -> Path:
    return ROOT / "dados" / "datasets" / f"{dataset}_train.csv"


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset de treino não encontrado: {path}")

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"Coluna 'target' não encontrada em {path}")

    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(f"Os atributos de {path} contêm NaN ou infinito.")
    if not np.isfinite(y).all():
        raise ValueError(f"O target de {path} contém NaN ou infinito.")

    return X, y


def create_model(scenario: str, seed: int):
    import juliapkg

    juliapkg.require_julia("~1.10")
    from pysr import PySRRegressor

    ops = OPERATOR_SETS[scenario]
    kwargs: dict[str, Any] = {
        "niterations": 1_000_000,  
        "max_evals": MAX_EVALUATIONS,
        "timeout_in_seconds": MAX_TIME_SECONDS,
        "populations": 10,
        "population_size": 200,
        "binary_operators": ops["binary_operators"],
        "unary_operators": ops["unary_operators"],
        "maxsize": 50,
        "maxdepth": 10,
        "crossover_probability": 0.9,
        "optimizer_iterations": 50,
        "optimizer_nrestarts": 2,
        "tournament_selection_n": 5,
        "model_selection": "best",
        "parallelism": "serial",
        "random_state": seed,
        "deterministic": True,
        "verbosity": 0,
    }
    return PySRRegressor(**kwargs)


def run_one(
    dataset: str,
    scenario: str,
    seed: int,
    fold: int,
    output_file: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_file.exists() and not overwrite:
        try:
            existing = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                return {"status": "existing", "file": output_file.name}
        except Exception:
            pass

    X, y = load_dataset(dataset)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(splitter.split(X))
    train_idx, validation_idx = splits[fold - 1]
    X_train, X_val = X[train_idx], X[validation_idx]
    y_train, y_val = y[train_idx], y[validation_idx]

    set_seed(seed)
    model = create_model(scenario, seed)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    equations = model.equations_
    results: list[dict[str, Any]] = []
    invalid_predictions = 0

    for idx in range(len(equations)):
        try:
            pred_train = np.asarray(model.predict(X_train, index=idx), dtype=float).reshape(-1)
            pred_val = np.asarray(model.predict(X_val, index=idx), dtype=float).reshape(-1)
            if not np.isfinite(pred_train).all() or not np.isfinite(pred_val).all():
                invalid_predictions += 1
                continue

            expression = str(equations.iloc[idx]["equation"])
            complexity = int(equations.iloc[idx]["complexity"])
            results.append({
                "model_id": int(idx),
                "expression": expression,
                "complexity": complexity,
                "mse_train": float(mean_squared_error(y_train, pred_train)),
                "mse_test": float(mean_squared_error(y_val, pred_val)),
                "r2_train": float(r2_score(y_train, pred_train)),
                "r2_test": float(r2_score(y_val, pred_val)),
                "seed": seed,
                "algorithm": ALGORITHM,
                "dataset": dataset,
                "scenario": scenario,
                "fold": fold,
                "max_evaluations": MAX_EVALUATIONS,
                "max_time_seconds": MAX_TIME_SECONDS,
                "runtime_seconds": round(elapsed, 6),
                "operator_set": OPERATOR_SETS[scenario],
            })
        except Exception:
            invalid_predictions += 1

    if not results:
        raise RuntimeError(
            "PySR terminou, mas nenhuma equação produziu previsões finitas. "
            f"Equações inválidas/ignoradas: {invalid_predictions}."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
   
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "file": output_file.name,
        "models_saved": len(results),
        "invalid_models_skipped": invalid_predictions,
        "runtime_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa as 200 combinações do protocolo PySR (C1/C2 × 4 datasets × 5 seeds × 5 folds)."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results_tuning" / "PySR"),
        help="Diretório dos arquivos de resultado (padrão: results_tuning/PySR).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds a executar (padrão: 42 43 44 45 46).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Refaz mesmo arquivos válidos já existentes.")
    parser.add_argument("--simulate", action="store_true", help="Lista as 200 combinações sem executar PySR.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    combinations = [
        (scenario, dataset, seed, fold)
        for scenario in ["C1", "C2"]
        for dataset in DATASETS
        for seed in args.seeds
        for fold in range(1, N_FOLDS + 1)
    ]

    report_path = output_dir / "relatorio_execucao_pysr.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "algorithm": ALGORITHM,
        "expected_experiments": len(combinations),
        "scenarios": ["C1", "C2"],
        "datasets": DATASETS,
        "seeds": args.seeds,
        "folds": list(range(1, N_FOLDS + 1)),
        "max_evaluations_per_run": MAX_EVALUATIONS,
        "max_time_seconds_per_run": MAX_TIME_SECONDS,
        "parameters": {
            "population_size": 200,
            "populations": 10,
            "niterations_ceiling": 1_000_000,
            "maxsize": 50,
            "maxdepth": 10,
            "crossover_probability": 0.9,
            "optimizer_iterations": 50,
            "optimizer_nrestarts": 2,
            "tournament_selection_n": 5,
            "parallelism": "serial",
        },
        "failures": [],
        "successful_or_existing_files": 0,
    }

    if args.simulate:
        for scenario, dataset, seed, fold in combinations:
            print(f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv")
        print(f"Total: {len(combinations)} execuções planejadas.")
        return 0

    for number, (scenario, dataset, seed, fold) in enumerate(combinations, start=1):
        filename = f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
        output_file = output_dir / filename
        print(f"[{number}/{len(combinations)}] {filename}", flush=True)
        try:
            result = run_one(dataset, scenario, seed, fold, output_file, args.overwrite)
            print(f"  -> {result['status']}", flush=True)
            if result["status"] in {"success", "existing"}:
                report["successful_or_existing_files"] += 1
        except Exception as exc:
            failure = {"file": filename, "error": repr(exc)}
            report["failures"].append(failure)
            print(f"  FALHOU: {exc}", flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    valid_csvs = 0
    for scenario, dataset, seed, fold in combinations:
        p = output_dir / f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
        if p.exists():
            try:
                value = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(value, list) and value:
                    valid_csvs += 1
            except Exception:
                pass
    report["successful_or_existing_files"] = valid_csvs
    report["csv_files_found"] = len(list(output_dir.glob("*.csv")))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nExecução em lote finalizada.")
    print(f"Arquivos CSV válidos: {valid_csvs}/{len(combinations)}")
    print(f"Falhas nesta execução: {len(report['failures'])}")
    print(f"Relatório: {report_path}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
