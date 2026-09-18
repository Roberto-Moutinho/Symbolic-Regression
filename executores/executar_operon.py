
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from algorithms.wrapper_operon import run_operon


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

OPERATORS = {
    "C1": "add,sub,mul,div,constant,variable",
    "C2": "add,sub,mul,div,pow,exp,logabs,sqrtabs,constant,variable",
}


def load_data(dataset):
    train_path = DATASETS[dataset]["train"]
    test_path = DATASETS[dataset]["test"]

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "target" not in train.columns:
        raise ValueError(f"'target' não encontrada em {train_path}")

    if "target" not in test.columns:
        raise ValueError(f"'target' não encontrada em {test_path}")

    X_train = train.drop(columns=["target"]).values
    y_train = train["target"].values

    X_test = test.drop(columns=["target"]).values
    y_test = test["target"].values

    return X_train, y_train, X_test, y_test


def run_experiment(dataset, scenario, seed, fold):
    X, y, X_test, y_test = load_data(dataset)

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    splits = list(kfold.split(X))

    train_idx, validation_idx = splits[fold - 1]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_validation = X[validation_idx]
    y_validation = y[validation_idx]

    results = run_operon(
        X_train=X_train,
        y_train=y_train,
        X_test=X_validation,
        y_test=y_validation,
        seed=seed,
        operators=OPERATORS[scenario],
        population_size=500,
        generations=400,
        max_depth=10,
        max_length=50,
        crossover_probability=0.9,
        mutation_probability=0.3,
        tournament_size=5,
        optimizer="lm",
        optimizer_iterations=100,
    )

    for result in results:
        result["algorithm"] = "Operon"
        result["dataset"] = dataset
        result["scenario"] = scenario
        result["seed"] = seed
        result["fold"] = fold

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=["C1", "C2"],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=[1, 2, 3, 4, 5],
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = run_experiment(
        dataset=args.dataset,
        scenario=args.scenario,
        seed=args.seed,
        fold=args.fold,
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=str,
        )

    print(f"Resultado salvo em: {output}")


if __name__ == "__main__":
    main()

