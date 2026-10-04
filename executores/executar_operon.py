from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Permite executar este arquivo diretamente: python executores/executar_operon.py
EXECUTORES_DIR = Path(__file__).resolve().parent
ROOT = EXECUTORES_DIR.parent
if str(EXECUTORES_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTORES_DIR))

from utils_otimizacao import run_optimization
from algorithms.wrapper_operon import run_operon


ALGORITHM = "Operon"
N_FOLDS = 5
MAX_EVALUATIONS = 200_000

DATASETS = {
    "chemical_2_competition": {
        "train": ROOT / "dados/datasets/chemical_2_competition_train.csv",
        "test": ROOT / "dados/datasets/chemical_2_competition_test.csv",
    },
    "friction_dyn_one-hot": {
        "train": ROOT / "dados/datasets/friction_dyn_one-hot_train.csv",
        "test": ROOT / "dados/datasets/friction_dyn_one-hot_test.csv",
    },
    "nasa_battery_1_10min": {
        "train": ROOT / "dados/datasets/nasa_battery_1_10min_train.csv",
        "test": ROOT / "dados/datasets/nasa_battery_1_10min_test.csv",
    },
    "nikuradse_1": {
        "train": ROOT / "dados/datasets/nikuradse_1_train.csv",
        "test": ROOT / "dados/datasets/nikuradse_1_test.csv",
    },
}

OPERATORS = {
    "C1": "add,sub,mul,div,constant,variable",
    "C2": "add,sub,mul,div,pow,exp,logabs,sqrtabs,constant,variable",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_train_data(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset}' não reconhecido. "
            f"Disponíveis: {', '.join(DATASETS)}"
        )

    path = DATASETS[dataset]["train"]
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(f"'target' não encontrada em {path}")

    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    return X, y


def load_test_data(dataset: str):
    path = DATASETS[dataset]["test"]
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(f"'target' não encontrada em {path}")

    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()
    return X, y


def operon_params_for_trial(params: dict[str, Any]) -> dict[str, Any]:
    """Mantém o orçamento de 200.000 avaliações.

    generations não é otimizado independentemente: ele é derivado de
    max_evaluations // population_size.
    """
    params = dict(params)

    if "population_size" not in params:
        raise ValueError("population_size precisa estar presente no espaço de busca.")

    population_size = int(params["population_size"])
    if population_size <= 0:
        raise ValueError("population_size deve ser > 0.")

    params["population_size"] = population_size
    params["generations"] = max(1, MAX_EVALUATIONS // population_size)

    # O wrapper atual usa pool_size explicitamente. Se ele estiver sendo
    # otimizado, respeitamos o valor sugerido pelo Optuna.
    if "pool_size" in params:
        params["pool_size"] = int(params["pool_size"])

    return params


def _select_fold_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Escolhe a solução da fronteira de Pareto usada para pontuar o trial.

    Primeiro maximiza R² de validação; em empate, prefere menor complexidade.
    """
    valid = [
        r for r in results
        if np.isfinite(float(r["r2_test"]))
        and np.isfinite(float(r["complexity"]))
    ]
    if not valid:
        raise RuntimeError("O Operon não produziu nenhum modelo válido no fold.")

    return max(
        valid,
        key=lambda r: (float(r["r2_test"]), -float(r["complexity"])),
    )


def evaluate_trial_5fold(
    params: dict[str, Any],
    trial,
    *,
    X: np.ndarray,
    y: np.ndarray,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    """Avalia UM conjunto de hiperparâmetros nos cinco folds.

    O valor retornado em R2 é a média dos cinco R² de validação.
    O conjunto de teste externo nunca é usado aqui.
    """
    params = operon_params_for_trial(params)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_results: list[dict[str, Any]] = []

    for fold, (train_idx, validation_idx) in enumerate(kfold.split(X), start=1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_validation = X[validation_idx]
        y_validation = y[validation_idx]

        print(
            f"    Trial {trial.number:03d} | Fold {fold}/5 | "
            f"pop={params['population_size']} | gen={params['generations']}",
            flush=True,
        )

        results = run_operon(
            X_train=X_train,
            y_train=y_train,
            X_test=X_validation,
            y_test=y_validation,
            seed=seed,
            operators=OPERATORS[scenario],
            population_size=params["population_size"],
            pool_size=params.get("pool_size", params["population_size"]),
            generations=params["generations"],
            max_depth=params["max_depth"],
            max_length=params["max_length"],
            crossover_probability=params["crossover_probability"],
            mutation_probability=params["mutation_probability"],
            tournament_size=params["tournament_size"],
            optimizer=params.get("optimizer", "lm"),
            optimizer_iterations=params["optimizer_iterations"],
        )

        selected = _select_fold_model(results)

        fold_result = {
            "fold": fold,
            "r2_validation": float(selected["r2_test"]),
            "mse_validation": float(selected["mse_test"]),
            "complexity": float(selected["complexity"]),
            "model_id": int(selected["model_id"]),
            "expression": str(selected["expression"]),
        }
        fold_results.append(fold_result)

        print(
            f"        R²={fold_result['r2_validation']:.8f} | "
            f"complexity={fold_result['complexity']:.0f}",
            flush=True,
        )

    r2_values = np.array([r["r2_validation"] for r in fold_results], dtype=float)
    complexity_values = np.array([r["complexity"] for r in fold_results], dtype=float)

    mean_r2 = float(np.mean(r2_values))
    std_r2 = float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0
    mean_complexity = float(np.mean(complexity_values))

    print(
        f"    Trial {trial.number:03d} → mean R²={mean_r2:.8f} ± {std_r2:.8f} | "
        f"mean complexity={mean_complexity:.2f}",
        flush=True,
    )

    return {
        "R2": mean_r2,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_complexity": mean_complexity,
        "fold_results": fold_results,
    }


def run_tuning(
    dataset: str,
    scenario: str,
    seed: int,
    n_trials: int,
    output_dir: Path,
    csv_path: Path,
    direction: str = "maximize",
    metric: str = "R2",
) -> Path:
    if scenario not in OPERATORS:
        raise ValueError(f"Scenario inválido: {scenario}. Use C1 ou C2.")

    X, y = load_train_data(dataset)
    set_seed(seed)

    run_dir = output_dir / ALGORITHM / dataset / scenario / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # O arquivo CSV contém generations, mas o protocolo do projeto define
    # generations = 200000 // population_size. Por isso generations não entra
    # no espaço independente do Optuna.
    def evaluator(params, trial):
        return evaluate_trial_5fold(
            params,
            trial,
            X=X,
            y=y,
            scenario=scenario,
            seed=seed,
        )

    metadata = {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "n_folds": N_FOLDS,
        "n_trials": n_trials,
        "selection_metric": "mean_validation_R2",
        "tie_break": "lower_mean_validation_complexity",
        "max_evaluations_per_fold": MAX_EVALUATIONS,
        "test_set_used_during_tuning": False,
        "operator_set": OPERATORS[scenario],
    }

    best_path = run_optimization(
        algorithm=ALGORITHM,
        csv_path=csv_path,
        output_dir=run_dir,
        n_trials=n_trials,
        seed=seed,
        direction=direction,
        metric=metric,
        evaluator_fn=evaluator,
        exclude_parameters={"generations"},
        postprocess_params=operon_params_for_trial,
        metadata=metadata,
    )

    # Também salva um resumo explícito do protocolo daquela execução.
    summary = {
        "dataset": dataset,
        "scenario": scenario,
        "seed": seed,
        "n_folds": N_FOLDS,
        "n_trials": n_trials,
        "best_params_file": str(best_path),
        "test_used": False,
        "note": "Cada trial foi avaliado nos 5 folds; o Optuna otimizou a média dos R² de validação.",
    }
    (run_dir / "protocolo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return best_path


def run_single_fold(
    dataset: str,
    scenario: str,
    seed: int,
    fold: int,
    output: Path,
    params: dict[str, Any],
) -> None:
    """Modo legado: uma única execução de um fold com HP já definidos."""
    if not 1 <= fold <= N_FOLDS:
        raise ValueError("fold deve estar entre 1 e 5.")

    X, y = load_train_data(dataset)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(kfold.split(X))
    train_idx, validation_idx = splits[fold - 1]

    p = operon_params_for_trial(params)
    results = run_operon(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[validation_idx],
        y_test=y[validation_idx],
        seed=seed,
        operators=OPERATORS[scenario],
        population_size=p["population_size"],
        pool_size=p.get("pool_size", p["population_size"]),
        generations=p["generations"],
        max_depth=p["max_depth"],
        max_length=p["max_length"],
        crossover_probability=p["crossover_probability"],
        mutation_probability=p["mutation_probability"],
        tournament_size=p["tournament_size"],
        optimizer=p.get("optimizer", "lm"),
        optimizer_iterations=p["optimizer_iterations"],
    )

    for result in results:
        result.update({
            "algorithm": ALGORITHM,
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "fold": fold,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Resultado salvo em: {output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Operon com tuning por Optuna usando 5-fold cross-validation "
            "dentro de cada trial."
        )
    )

    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--scenario", required=True, choices=["C1", "C2"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--hyperparameters", type=Path, default=ROOT / "dados/hiperparametros_treinamento_corrigido.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "resultados_tuning_cv")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--metric", default="R2")

    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.fold is not None:
        raise SystemExit(
            "O modo --fold pertence ao protocolo antigo. Para o novo tuning, "
            "não informe --fold: cada trial já executa os 5 folds."
        )

    best = run_tuning(
        dataset=args.dataset,
        scenario=args.scenario,
        seed=args.seed,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        csv_path=args.hyperparameters,
        direction=args.direction,
        metric=args.metric,
    )

    print(f"\nMelhores hiperparâmetros: {best}")


if __name__ == "__main__":
    main()

