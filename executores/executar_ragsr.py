from __future__ import annotations

import argparse
import json
import traceback
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.wrapper_rag import RAGSRRegressor

ALGORITHM = "RAG-SR"
OUTPUT_FOLDER = "RAG"
N_FOLDS = 5
MAX_EVALUATIONS = 200_000
DEFAULT_POPULATION = 200
DEFAULT_GENERATIONS = MAX_EVALUATIONS // DEFAULT_POPULATION
SEEDS = [42, 43, 44, 45, 46]

DATASETS = {
    "chemical_2_competition": ROOT / "dados/datasets/chemical_2_competition_train.csv",
    "friction_dyn_one-hot": ROOT / "dados/datasets/friction_dyn_one-hot_train.csv",
    "nasa_battery_1_10min": ROOT / "dados/datasets/nasa_battery_1_10min_train.csv",
    "nikuradse_1": ROOT / "dados/datasets/nikuradse_1_train.csv",
}

# C1: operadores aritméticos básicos. C2: conjunto estendido de primitivas
# suportadas pelo wrapper original do RAG-SR/EvolutionaryForest.
PRIMITIVES = {
    "C1": "Add,Sub,Mul,AQ",
    "C2": "Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset not in DATASETS:
        raise ValueError(f"Dataset desconhecido: {dataset}. Opções: {list(DATASETS)}")
    path = DATASETS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {path}")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"Coluna 'target' ausente em {path}")
    X = df.drop(columns=["target"]).to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64).reshape(-1)
    if len(X) < N_FOLDS:
        raise ValueError(f"Dataset {dataset} tem menos de {N_FOLDS} amostras.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Dataset {dataset} contém NaN ou valores infinitos.")
    return X, y


def experiment_params(seed: int, scenario: str, n_pop: int, max_evaluations: int) -> dict[str, Any]:
    if n_pop < 1 or max_evaluations < 1:
        raise ValueError("n_pop e max_evaluations precisam ser positivos.")
    n_gen = max(1, max_evaluations // n_pop)
    return {
        "n_pop": int(n_pop),
        "n_gen": int(n_gen),
        "max_evaluations_requested": int(max_evaluations),
        "effective_population_generations": int(n_pop * n_gen),
        "gene_num": 10,
        "neural_pool": 0.1,
        "neural_pool_num_of_functions": 5,
        "weight_of_contrastive_learning": 0.05,
        "neural_pool_dropout": 0.1,
        "neural_pool_transformer_layer": 1,
        "neural_pool_hidden_size": 64,
        "neural_pool_mlp_layers": 3,
        "selective_retrain": True,
        "negative_data_augmentation": True,
        "select": "AutomaticLexicase",
        "cross_pb": 0.9,
        "mutation_pb": 0.1,
        "max_height": 10,
        "basic_primitives": PRIMITIVES[scenario],
        "base_learner": "RidgeCV",
        "normalize": "MinMax",
        "categorical_encoding": "Target",
        "random_seed": int(seed),
    }


def run_one(dataset: str, scenario: str, seed: int, fold: int,
            output_dir: Path, n_pop: int, max_evaluations: int,
            overwrite: bool = False) -> Path:
    if scenario not in PRIMITIVES:
        raise ValueError("Cenário inválido; use C1 ou C2.")
    if not 1 <= fold <= N_FOLDS:
        raise ValueError("fold deve estar entre 1 e 5.")

    output = output_dir / f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
    if output.exists() and not overwrite:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                print(f"Já existe; pulando: {output.name}", flush=True)
                return output
        except (OSError, json.JSONDecodeError):
            pass

    X, y = load_dataset(dataset)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    train_idx, validation_idx = list(splitter.split(X))[fold - 1]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[validation_idx], y[validation_idx]

    params = experiment_params(seed, scenario, n_pop, max_evaluations)
    set_seed(seed)
    model = RAGSRRegressor(**params)
    start = time.perf_counter()
    # O protocolo do projeto trata o fold reservado como teste da execução.
    model.fit(
    X_train,
    y_train,
    categorical_features=np.zeros(X_train.shape[1], dtype=np.float64)
)
    pred_train = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
    pred_val = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
    runtime = time.perf_counter() - start

    if pred_train.shape != y_train.shape or not np.isfinite(pred_train).all():
        raise RuntimeError("RAG-SR produziu predições de treino inválidas.")
    if pred_val.shape != y_val.shape or not np.isfinite(pred_val).all():
        raise RuntimeError("RAG-SR produziu predições de validação inválidas.")

    expression = str(model.model())
    try:
        complexity = int(model.complexity())
    except Exception:
        # Fallback informativo caso a expressão gerada não seja parseável pelo SymPy.
        complexity = len(expression.split())

    result = {
        "model_id": 0,
        "expression": expression,
        "complexity": complexity,
        "mse_train": float(mean_squared_error(y_train, pred_train)),
        "mse_test": float(mean_squared_error(y_val, pred_val)),
        "r2_train": float(r2_score(y_train, pred_train)),
        "r2_test": float(r2_score(y_val, pred_val)),
        "seed": int(seed),
        "algorithm": ALGORITHM,
        "dataset": dataset,
        "scenario": scenario,
        "fold": int(fold),
        "n_samples": int(len(X)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(validation_idx)),
        "n_features": int(X.shape[1]),
        "n_pop": params["n_pop"],
        "n_gen": params["n_gen"],
        "effective_population_generations": params["effective_population_generations"],
        "basic_primitives": params["basic_primitives"],
        "status": "success",
        "error": None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    # Mantém o formato de lista JSON usado pelos resultados já existentes no projeto,
    # apesar da extensão .csv.
    temp.write_text(json.dumps([result], ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output)
    print(f"Salvo: {output} | R²={result['r2_test']:.6f} | MSE={result['mse_test']:.6g} | {runtime:.1f}s", flush=True)
    return output


def run_all(output_dir: Path, seeds: list[int], n_pop: int,
            max_evaluations: int, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(DATASETS) * len(PRIMITIVES) * len(seeds) * N_FOLDS
    current = 0
    failures: list[dict[str, str]] = []
    for scenario in ("C1", "C2"):
        for dataset in DATASETS:
            for seed in seeds:
                for fold in range(1, N_FOLDS + 1):
                    current += 1
                    print(f"\n[{current}/{total}] {ALGORITHM} | {scenario} | {dataset} | seed={seed} | fold={fold}", flush=True)
                    try:
                        run_one(dataset, scenario, seed, fold, output_dir, n_pop, max_evaluations, overwrite)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()

                        failure = {
                            "dataset": dataset,
                            "scenario": scenario,
                            "seed": str(seed),
                            "fold": str(fold),
                            "error": repr(exc),
                        }
                        failures.append(failure)
                        print(f"FALHA: {failure}", flush=True)
                    
    report = {
        "algorithm": ALGORITHM,
        "output_folder": str(output_dir),
        "expected_experiments": total,
        "datasets": list(DATASETS),
        "scenarios": list(PRIMITIVES),
        "seeds": seeds,
        "folds": list(range(1, N_FOLDS + 1)),
        "max_evaluations_requested_per_run": max_evaluations,
        "n_pop": n_pop,
        "n_gen": max(1, max_evaluations // n_pop),
        "failures": failures,
        "csv_files_present": len(list(output_dir.glob("*.csv"))),
    }
    (output_dir / "relatorio_execucao_RAG.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLote finalizado: {report['csv_files_present']} arquivos .csv encontrados; {len(failures)} falhas.", flush=True)
    if failures:
        print(f"Consulte {output_dir / 'relatorio_execucao_RAG.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o protocolo experimental RAG-SR (4 datasets × 2 cenários × 5 seeds × 5 folds).")
    parser.add_argument("--all-experiments", action="store_true", help="Executa as 200 combinações do protocolo.")
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--scenario", choices=list(PRIMITIVES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_tuning" / OUTPUT_FOLDER)
    parser.add_argument("--n-pop", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--max-evaluations", type=int, default=MAX_EVALUATIONS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all_experiments:
        run_all(args.output_dir, args.seeds, args.n_pop, args.max_evaluations, args.overwrite)
        return
    if args.dataset is None or args.scenario is None or args.seed is None or args.fold is None:
        parser.error("Use --all-experiments ou informe --dataset, --scenario, --seed e --fold.")
    run_one(args.dataset, args.scenario, args.seed, args.fold, args.output_dir,
            args.n_pop, args.max_evaluations, args.overwrite)


if __name__ == "__main__":
    main()
