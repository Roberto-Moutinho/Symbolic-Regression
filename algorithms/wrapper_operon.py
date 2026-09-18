
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pyoperon.sklearn import SymbolicRegressor


DATASETS = {
    "chemical_2_competition": {
        "train": "dados/datasets/chemical_2_competition_train.csv",
        "test": "dados/datasets/chemical_2_competition_test.csv",
    },
    "friction_dyn_one-hot": {
        "train": "dados/datasets/friction_dyn_one-hot_train.csv",
        "test": "dados/datasets/friction_dyn_one-hot_test.csv",
    },
    "nasa_battery_1_10min": {
        "train": "dados/datasets/nasa_battery_1_10min_train.csv",
        "test": "dados/datasets/nasa_battery_1_10min_test.csv",
    },
    "nikuradse_1": {
        "train": "dados/datasets/nikuradse_1_train.csv",
        "test": "dados/datasets/nikuradse_1_test.csv",
    },
}


DEFAULT_OPERATORS = (
    "add,sub,mul,div,exp,logabs,pow,sqrtabs,abs,sin,constant,variable"
)

DEFAULT_POPULATION_SIZE = 2000
DEFAULT_GENERATIONS = 100
DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_LENGTH = 33
DEFAULT_CROSSOVER_PROBABILITY = 0.9
DEFAULT_MUTATION_PROBABILITY = 0.3
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_OPTIMIZER = "lm"
DEFAULT_OPTIMIZER_ITERATIONS = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path):
    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(
            f"O arquivo {path} não possui a coluna 'target'."
        )

    X = df.loc[:, df.columns != "target"].values
    y = df["target"].values

    return X, y


def get_dataset_paths(dataset):
    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset}' não reconhecido. "
            f"Datasets disponíveis: {', '.join(DATASETS.keys())}"
        )

    return DATASETS[dataset]["train"], DATASETS[dataset]["test"]


def run_operon(
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
    operators=DEFAULT_OPERATORS,
    population_size=DEFAULT_POPULATION_SIZE,
    generations=DEFAULT_GENERATIONS,
    max_depth=DEFAULT_MAX_DEPTH,
    max_length=DEFAULT_MAX_LENGTH,
    crossover_probability=DEFAULT_CROSSOVER_PROBABILITY,
    mutation_probability=DEFAULT_MUTATION_PROBABILITY,
    tournament_size=DEFAULT_TOURNAMENT_SIZE,
    optimizer=DEFAULT_OPTIMIZER,
    optimizer_iterations=DEFAULT_OPTIMIZER_ITERATIONS,
):
    set_seed(seed)

    evaluation_budget = population_size * generations

    reg = SymbolicRegressor(
        allowed_symbols=operators,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        female_selector="tournament",
        male_selector="tournament",
        generations=generations,
        optimizer_iterations=optimizer_iterations,
        optimizer=optimizer,
        max_depth=max_depth,
        max_length=max_length,
        objectives=["mse", "length"],
        pool_size=population_size,
        population_size=population_size,
        reinserter="keep-best",
        tournament_size=tournament_size,
        max_evaluations=evaluation_budget,
        random_state=seed,
    )

    reg.fit(X_train, y_train)

    results = []

    for model_id, model in enumerate(reg.pareto_front_):
        tree = model["tree"]

        y_hat_train = reg.evaluate_model(
            tree,
            X_train
        )

        y_hat_test = reg.evaluate_model(
            tree,
            X_test
        )

        mse_train = mean_squared_error(
            y_train,
            y_hat_train
        )

        mse_test = mean_squared_error(
            y_test,
            y_hat_test
        )

        r2_train = r2_score(
            y_train,
            y_hat_train
        )

        r2_test = r2_score(
            y_test,
            y_hat_test
        )

        results.append(
            {
                "model_id": model_id,
                "expression": model["model"],
                "complexity": model["complexity"],
                "mse_train": mse_train,
                "mse_test": mse_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "seed": seed,
            }
        )

    return results


def run_from_files(
    dataset,
    scenario,
    fold,
    seed,
    operators=DEFAULT_OPERATORS,
    population_size=DEFAULT_POPULATION_SIZE,
    generations=DEFAULT_GENERATIONS,
):
    train_path, test_path = get_dataset_paths(dataset)

    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    results = run_operon(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        operators=operators,
        population_size=population_size,
        generations=generations,
    )

    for result in results:
        result["algorithm"] = "Operon"
        result["dataset"] = dataset
        result["scenario"] = scenario
        result["fold"] = fold

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys()
    )

    parser.add_argument(
        "--scenario",
        required=True
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int
    )

    parser.add_argument(
        "--operators",
        default=DEFAULT_OPERATORS
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS
    )

    parser.add_argument(
        "--output",
        default=None
    )

    args = parser.parse_args()

    results = run_from_files(
        dataset=args.dataset,
        scenario=args.scenario,
        fold=args.fold,
        seed=args.seed,
        operators=args.operators,
        population_size=args.population_size,
        generations=args.generations,
    )

    if args.output:
        output_dir = os.path.dirname(args.output)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        with open(args.output, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
                default=str
            )
    else:
        print(
            json.dumps(
                results,
                indent=2,
                default=str
            )
        )


if __name__ == "__main__":
    main()

