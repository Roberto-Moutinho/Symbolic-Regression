import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "chemical_2_competition",
    "friction_dyn_one-hot",
    "nasa_battery_1_10min",
    "nikuradse_1",
]

FUNCTION_SETS = {
    "C1": ["add", "sub", "mul", "div", "const", "poly"],
    "C2": ["add", "sub", "mul", "div", "exp", "log", "sqrt", "const", "poly"],
}

# Cada execução grava um arquivo JSON contendo uma lista com um registro.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa uma configuração uDSR no benchmark."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--scenario", choices=["C1", "C2"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--gp-population-size", type=int, default=25)
    parser.add_argument("--gp-generations", type=int, default=20)
    parser.add_argument("--no-run-gp-meld", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mantido por compatibilidade; a saída já é silenciosa por padrão.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Exibe os logs internos do DSO (desativado por padrão).",
    )
    return parser.parse_args()


@contextlib.contextmanager
def suppress_process_output():
    """Redireciona stdout/stderr no nível do processo, incluindo extensões C/Cython."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout, 1)
                os.dup2(saved_stderr, 2)
    finally:
        os.close(saved_stdout)
        os.close(saved_stderr)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_fold(dataset: str, fold: int, seed: int):
    path = ROOT / "dados" / "datasets" / "{}_train.csv".format(dataset)
    if not path.is_file():
        raise FileNotFoundError("Dataset de treino não encontrado: {}".format(path))

    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError("A coluna 'target' não existe em {}".format(path))

    X = frame.drop(columns=["target"]).to_numpy(dtype=np.float64)
    y = frame["target"].to_numpy(dtype=np.float64)

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("Dimensões inválidas em {}".format(path))
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contém NaN ou valores infinitos: {}".format(path))
    if len(y) < 5:
        raise ValueError("O dataset precisa ter pelo menos 5 linhas para KFold.")

    splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))
    train_idx, test_idx = splits[fold - 1]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], path


def build_config(args, train_csv: Path, config_path: Path):
    config = {
        "task": {
            "task_type": "regression",
            "dataset": str(train_csv),
            "function_set": FUNCTION_SETS[args.scenario],
            "metric": "inv_nrmse",
            "metric_params": [1.0],
            "extra_metric_test": None,
            "extra_metric_test_params": [],
            "threshold": 1e-12,
            "protected": False,
            "reward_noise": 0.0,
            "reward_noise_type": "r",
            "normalize_variance": False,
            "decision_tree_threshold_set": [],
            "poly_optimizer_params": {
                "degree": args.degree,
                "coef_tol": 1e-6,
                "regressor": "dso_least_squares",
                "regressor_params": {
                    "cutoff_p_value": 1.0,
                    "n_max_terms": None,
                    "coef_tol": 1e-6,
                },
            },
        },
        "gp_meld": {
            "run_gp_meld": not args.no_run_gp_meld,
            "population_size": args.gp_population_size,
            "generations": args.gp_generations,
            "crossover_operator": "cxOnePoint",
            "p_crossover": 0.5,
            "mutation_operator": "multi_mutate",
            "p_mutate": 0.5,
            "tournament_size": 5,
            "train_n": 50,
            "mutate_tree_max": 3,
            "verbose": False,
            "parallel_eval": False,
        },
        "training": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "epsilon": 0.02,
            "n_cores_batch": 1,
            "verbose": False,
        },
        "controller": {
            "learning_rate": 0.0025,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
            "pqt": True,
            "pqt_k": 10,
            "pqt_batch_size": 1,
            "pqt_weight": 200.0,
            "pqt_use_pg": False,
        },
        "prior": {
            "length": {"min_": 4, "max_": 100, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": True},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
            "domain_range": {"on": True},
        },
        "experiment": {
            "logdir": None,
            "exp_name": "uDSR_{}_{}_fold{}_seed{}".format(
                args.dataset, args.scenario, args.fold, args.seed
            ),
            "seed": args.seed,
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def create_record(args, expression=None, size=None, mse_train=None,
                  mse_test=None, r2_train=None, r2_test=None,
                  runtime_seconds=None, status="success", n_train=None,
                  n_test=None, n_features=None, source_dataset=None, error=None):
    """Monta um registro JSON padronizado para uma execução do benchmark."""
    return {
        "model_id": 0,
        "expression": expression,
        "complexity": size,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "seed": args.seed,
        "algorithm": "uDSR",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "fold": args.fold,
        "runtime_seconds": runtime_seconds,
        "nll_train": mse_train,
        "nll_test": mse_test,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "function_set": FUNCTION_SETS[args.scenario],
        "degree": args.degree,
        "run_gp_meld": not args.no_run_gp_meld,
        "gp_population_size": args.gp_population_size,
        "gp_generations": args.gp_generations,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "source_dataset": str(source_dataset) if source_dataset else None,
        "status": status,
        "error": error,
    }


def write_json(output: Path, record: dict) -> None:
    """Grava uma lista JSON com um registro, mantendo formato consistente."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump([record], file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\\n")


def main() -> int:
    args = parse_args()
    if args.n_samples < 1 or args.batch_size < 1:
        raise SystemExit("--n-samples e --batch-size devem ser positivos.")
    if args.gp_population_size < 1 or args.gp_generations < 1:
        raise SystemExit("Os parâmetros de população e gerações devem ser positivos.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.simulate:
        record = create_record(
            args,
            expression="SIMULATION_ONLY",
            runtime_seconds=0.0,
            status="simulated",
        )
        write_json(args.output, record)
        return 0

    set_seed(args.seed)
    X_train, y_train, X_test, y_test, dataset_path = load_fold(
        args.dataset, args.fold, args.seed
    )

    # A importação, construção do otimizador e treinamento ficam silenciosos.
    # --verbose pode ser usado para depuração manual.
    if args.verbose and not args.quiet:
        from dso import DeepSymbolicOptimizer

        with tempfile.TemporaryDirectory(prefix="udsr_benchmark_") as tmp:
            train_csv = Path(tmp) / "train_fold.csv"
            np.savetxt(
                train_csv,
                np.column_stack((X_train, y_train)),
                delimiter=",",
            )
            config_path = Path(tmp) / "config.json"
            build_config(args, train_csv, config_path)
            optimizer = DeepSymbolicOptimizer(str(config_path))
            train_result = optimizer.train()
            program = train_result["program"]
    else:
        with suppress_process_output():
            from dso import DeepSymbolicOptimizer

            with tempfile.TemporaryDirectory(prefix="udsr_benchmark_") as tmp:
                train_csv = Path(tmp) / "train_fold.csv"
                np.savetxt(
                    train_csv,
                    np.column_stack((X_train, y_train)),
                    delimiter=",",
                )
                config_path = Path(tmp) / "config.json"
                build_config(args, train_csv, config_path)
                optimizer = DeepSymbolicOptimizer(str(config_path))
                train_result = optimizer.train()
                program = train_result["program"]

    if program is None:
        raise RuntimeError("O DSO terminou sem retornar um programa.")

    pred_train = np.asarray(program.execute(X_train), dtype=float).reshape(-1)
    pred_test = np.asarray(program.execute(X_test), dtype=float).reshape(-1)

    finite = (
        pred_train.shape == y_train.shape
        and pred_test.shape == y_test.shape
        and np.isfinite(pred_train).all()
        and np.isfinite(pred_test).all()
    )

    if finite:
        mse_train = float(mean_squared_error(y_train, pred_train))
        mse_test = float(mean_squared_error(y_test, pred_test))
        r2_train = float(r2_score(y_train, pred_train)) if len(y_train) > 1 else None
        r2_test = float(r2_score(y_test, pred_test)) if len(y_test) > 1 else None
        status = "success"
    else:
        mse_train = mse_test = r2_train = r2_test = None
        status = "non_finite_predictions"

    try:
        expression = str(getattr(program, "sympy_expr", program))
    except Exception:
        expression = str(program)

    try:
        size = int(len(program.traversal))
    except Exception:
        size = None

    record = create_record(
        args,
        expression=expression,
        size=size,
        mse_train=mse_train,
        mse_test=mse_test,
        r2_train=r2_train,
        r2_test=r2_test,
        runtime_seconds=round(time.perf_counter() - started, 6),
        status=status,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        n_features=int(X_train.shape[1]),
        source_dataset=dataset_path,
    )
    write_json(args.output, record)

    # O executor de lote lê o JSON impresso em stdout para identificar o status.
    # O treinamento continua silencioso; apenas este resumo final é exibido.
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
