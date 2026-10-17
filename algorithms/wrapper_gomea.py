
import random

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pygpg.sk import GPGRegressor
from pygpg.complexity import compute_complexity


FSETS = {
    "C1": "+,-,*,/",
    "C2": "+,-,*,/,^,exp,log,sqrt",
}


def run_gomea(
    X_train,
    y_train,
    X_test,
    y_test,
    seed=42,
    scenario="C1",
    pop=500,
    generations=None,
    max_evaluations=10000,
    depth=5,
):
    random.seed(seed)
    np.random.seed(seed)

    if scenario not in FSETS:
        raise ValueError(f"Scenario inválido: {scenario}")

    pop = int(pop)
    if pop <= 0:
        raise ValueError("pop deve ser maior que zero")

    generations = int(
        generations or max(1, int(max_evaluations) // pop)
    )

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

    reg = GPGRegressor(
        fset=FSETS[scenario],
        d=int(depth),
        pop=pop,
        g=generations,
        t=-1,
        e=-1,
    )

    reg.fit(X_train, y_train)

    pred_train = np.asarray(
        reg.predict(X_train), dtype=float
    ).reshape(-1)

    pred_test = np.asarray(
        reg.predict(X_test), dtype=float
    ).reshape(-1)

    valid_train = (
        pred_train.shape == y_train.shape
        and np.all(np.isfinite(pred_train))
    )

    valid_test = (
        pred_test.shape == y_test.shape
        and np.all(np.isfinite(pred_test))
    )

    mse_train = (
        float(mean_squared_error(y_train, pred_train))
        if valid_train
        else float("inf")
    )

    mse_test = (
        float(mean_squared_error(y_test, pred_test))
        if valid_test
        else float("inf")
    )

    r2_train = (
        float(r2_score(y_train, pred_train))
        if valid_train
        else float("-inf")
    )

    r2_test = (
        float(r2_score(y_test, pred_test))
        if valid_test
        else float("-inf")
    )

    expression = str(reg.model)

    try:
        complexity = int(
            compute_complexity(
                reg.model,
                complexity_metric="node_count",
            )
        )
    except Exception:
        complexity = int(compute_complexity(reg.model))

    return [
        {
            "model_id": 0,
            "expression": expression,
            "complexity": complexity,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "seed": int(seed),
            "population_size": pop,
            "generations": generations,
            "max_evaluations": pop * generations,
            "operators": FSETS[scenario],
        }
    ]

