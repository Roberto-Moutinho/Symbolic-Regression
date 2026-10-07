#!/usr/bin/env python3

import os
import csv
import random
import argparse

import numpy as np
import pandas as pd
import torch

from llmsr import pipeline
from llmsr import config as llmsr_config
from llmsr import sampler
from llmsr import evaluator


# ============================================================
# Argumentos
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Wrapper SRBench para LLMSR"
    )

    parser.add_argument(
        "--dataset",
        choices=["chemical_2_competition"],
        required=True
    )

    parser.add_argument(
        "--scenario",
        required=True
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--spec_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--use_api",
        action="store_true"
    )

    parser.add_argument(
        "--api_model",
        type=str,
        default="gpt-3.5-turbo"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None
    )

    parser.add_argument(
        "--simulate",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

def load_dataset(dataset, scenario, fold):

    """
    Estrutura esperada:

    data/
        <dataset>/
            <scenario>/
                fold<fold>/
                    train.csv
                    test.csv
    """

    base_dir = os.path.join(
        "data",
        dataset,
        scenario,
        f"fold{fold}"
    )

    train_path = os.path.join(
        base_dir,
        "train.csv"
    )

    test_path = os.path.join(
        base_dir,
        "test.csv"
    )

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train não encontrado: {train_path}"
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test não encontrado: {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1)

    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1)

    return X_train, y_train, X_test, y_test


# ============================================================
# Métricas
# ============================================================

def mse(y_true, y_pred):

    return float(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def r2(y_true, y_pred):

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return 0.0

    return float(
        1.0 - ss_res / ss_tot
    )


# ============================================================
# Simulação
# ============================================================

def simulate(args):

    return {
        "algorithm": "LLMSR",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "mse_train": np.nan,
        "mse_test": np.nan,
        "r2_train": np.nan,
        "r2_test": np.nan,
        "expression": "SIMULATION",
        "complexity": np.nan,
        "time": 0.0,
        "status": "simulated"
    }


# ============================================================
# LLMSR
# ============================================================

def run_llmsr(args, X_train, y_train, X_test, y_test):

    # --------------------------------------------------------
    # Configuração
    # --------------------------------------------------------

    class_config = llmsr_config.ClassConfig(
        llm_class=sampler.LocalLLM,
        sandbox_class=evaluator.LocalSandbox
    )

    config = llmsr_config.Config(
        use_api=args.use_api,
        api_model=args.api_model
    )

    # --------------------------------------------------------
    # Specification
    # --------------------------------------------------------

    with open(
        args.spec_path,
        encoding="utf-8"
    ) as f:

        specification = f.read()

    # --------------------------------------------------------
    # Tensor / NumPy
    # --------------------------------------------------------

    if "torch" in args.spec_path:

        X = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y = torch.tensor(
            y_train,
            dtype=torch.float32
        )

    else:

        X = X_train
        y = y_train

    dataset = {
        "data": {
            "inputs": X,
            "outputs": y
        }
    }

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    log_dir = os.path.join(
        "logs",
        "srbench",
        args.dataset,
        args.scenario,
        f"fold{args.fold}",
        f"seed{args.seed}"
    )

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Execução
    # --------------------------------------------------------

    import time

    start = time.time()

    result = pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=args.max_samples,
        class_config=class_config,
        log_dir=log_dir
    )

    elapsed = time.time() - start

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    expression = result

    return {
        "algorithm": "LLMSR",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "mse_train": np.nan,
        "mse_test": np.nan,
        "r2_train": np.nan,
        "r2_test": np.nan,
        "expression": str(expression),
        "complexity": np.nan,
        "time": elapsed,
        "status": "success"
    }


# ============================================================
# Salvar CSV
# ============================================================

def save_result(result, output):

    output_dir = os.path.dirname(
        os.path.abspath(output)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    fields = [
        "algorithm",
        "dataset",
        "scenario",
        "fold",
        "seed",
        "mse_train",
        "mse_test",
        "r2_train",
        "r2_test",
        "expression",
        "complexity",
        "time",
        "status"
    ]

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerow(result)


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    if args.simulate:

        result = simulate(args)

    else:

        X_train, y_train, X_test, y_test = load_dataset(
            args.dataset,
            args.scenario,
            args.fold
        )

        result = run_llmsr(
            args,
            X_train,
            y_train,
            X_test,
            y_test
        )

    save_result(
        result,
        args.output
    )

    print(result)


if __name__ == "__main__":
    main()
