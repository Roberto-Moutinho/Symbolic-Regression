
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch


# ============================================================
# Argumentos
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wrapper SRBench para TPSR"
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
        "--simulate",
        action="store_true",
        help="Executa apenas uma simulação sem carregar o TPSR."
    )

    return parser.parse_args()


# ============================================================
# Seeds
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Loader
# ============================================================

def load_dataset(dataset, scenario, fold):
    """
    Carrega o dataset preparado para o experimento.

    Espera-se que os dados estejam disponíveis no diretório:

        data/<dataset>/<scenario>/fold<fold>/

    com:

        X_train.npy
        y_train.npy
        X_test.npy
        y_test.npy

    Caso sua estrutura de diretórios seja diferente, altere
    somente esta função.
    """

    base_dir = os.path.join(
        "data",
        dataset,
        scenario,
        f"fold{fold}"
    )

    x_train_path = os.path.join(base_dir, "X_train.npy")
    y_train_path = os.path.join(base_dir, "y_train.npy")
    x_test_path = os.path.join(base_dir, "X_test.npy")
    y_test_path = os.path.join(base_dir, "y_test.npy")

    required = [
        x_train_path,
        y_train_path,
        x_test_path,
        y_test_path
    ]

    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arquivo do dataset não encontrado: {path}"
            )

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)

    return X_train, y_train, X_test, y_test


# ============================================================
# Métricas
# ============================================================

def compute_mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_nmse(y_true, y_pred):
    eps = 1e-9

    return float(
        np.sqrt(
            np.mean((y_true - y_pred) ** 2)
            / (np.mean(y_true ** 2) + eps)
        )
    )


# ============================================================
# Simulação
# ============================================================

def simulate(args):
    """
    Simulação para testar a interface do wrapper.

    Não representa resultado científico do TPSR.
    """

    result = {
        "algorithm": "TPSR",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "mse_train": np.nan,
        "mse_test": np.nan,
        "r2_train": np.nan,
        "r2_test": np.nan,
        "nmse_train": np.nan,
        "nmse_test": np.nan,
        "expression": "SIMULATION",
        "complexity": np.nan,
        "time": 0.0,
        "status": "simulated"
    }

    return result


# ============================================================
# TPSR
# ============================================================

def run_tpsr(args, X_train, y_train, X_test, y_test):

    # Imports do projeto TPSR original.
    #
    # Eles ficam aqui para que --simulate não dependa da
    # instalação completa do TPSR.
    from parsers import get_parser
    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules
    from symbolicregression.trainer import Trainer
    from symbolicregression.e2e_model import (
        Transformer,
        pred_for_sample_no_refine,
        refine_for_sample_test,
    )
    from dyna_gym.agents.uct import UCT
    from dyna_gym.agents.mcts import update_root
    from rl_env import RLEnv
    from default_pi import E2EHeuristic

    # --------------------------------------------------------
    # Configuração do TPSR
    # --------------------------------------------------------

    # Utilizamos o parser original do TPSR para preservar
    # todas as configurações internas do algoritmo.
    parser = get_parser()

    # Não conseguimos simplesmente passar os argumentos
    # SRBench para o parser interno, portanto usamos apenas
    # os argumentos que o TPSR espera.
    #
    # O wrapper deve ser chamado a partir do ambiente original
    # do TPSR.

    sys.argv = [
        sys.argv[0]
    ]

    params = parser.parse_args()

    params.seed = args.seed
    params.debug = False
    params.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------
    # Ambiente TPSR
    # --------------------------------------------------------

    equation_env = build_env(params)
    modules = build_modules(
        equation_env,
        params
    )

    if not params.cpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TPSR foi configurado para CUDA, "
                "mas nenhuma GPU CUDA está disponível."
            )

    # Mantém comportamento original.
    import symbolicregression

    symbolicregression.utils.CUDA = not params.cpu

    Trainer(
        modules,
        equation_env,
        params
    )

    # --------------------------------------------------------
    # Formato de samples utilizado pelo TPSR original
    # --------------------------------------------------------

    samples = {
        "x_to_fit": [X_train],
        "y_to_fit": [y_train.reshape(-1, 1)],
        "x_to_pred": [X_test],
        "y_to_pred": [y_test.reshape(-1, 1)]
    }

    model = Transformer(
        params=params,
        env=equation_env,
        samples=samples
    )

    model.to(params.device)



    generations_ref, gen_len_ref = respond_to_batch(
        model,
        max_target_length=200,
        top_p=1.0,
        sample_temperature=None
    )

    sequence_ref = generations_ref[0][
        :gen_len_ref - 1
    ].tolist()

    rl_env = RLEnv(
        params=params,
        samples=samples,
        equation_env=equation_env,
        model=model
    )

    dp = E2EHeuristic(
        equation_env=equation_env,
        rl_env=rl_env,
        model=model,
        k=params.width,
        num_beams=params.num_beams,
        horizon=params.horizon,
        device=params.device,
        use_seq_cache=not params.no_seq_cache,
        use_prefix_cache=not params.no_prefix_cache,
        length_penalty=params.beam_length_penalty,
        train_value_mode=params.train_value,
        debug=params.debug
    )


    agent = UCT(
        action_space=[],
        gamma=1.0,
        ucb_constant=1.0,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base
    )

    if params.sample_only:
        horizon = 1
    else:
        horizon = 200

    done = False
    state = rl_env.state

    start_time = time.time()

 

    for _ in range(horizon):

        if len(state) >= params.horizon:
            break

        if done:
            break

        action = agent.act(
            rl_env,
            done
        )

        state, reward, done, _ = rl_env.step(action)

        update_root(
            agent,
            action,
            state
        )

        dp.update_cache(state)

    elapsed = time.time() - start_time



    y_pred_train, expression, _ = pred_for_sample_no_refine(
        model,
        equation_env,
        state,
        X_train
    )

    (
        y_pred_train_refined,
        y_pred_test_refined,
        _,
        _
    ) = refine_for_sample_test(
        model,
        equation_env,
        state,
        X_train,
        y_train,
        X_test
    )

    y_pred_train_refined = np.asarray(
        y_pred_train_refined
    ).reshape(-1)

    y_pred_test_refined = np.asarray(
        y_pred_test_refined
    ).reshape(-1)



    mse_train = compute_mse(
        y_train,
        y_pred_train_refined
    )

    mse_test = compute_mse(
        y_test,
        y_pred_test_refined
    )

    r2_train = compute_r2(
        y_train,
        y_pred_train_refined
    )

    r2_test = compute_r2(
        y_test,
        y_pred_test_refined
    )

    nmse_train = compute_nmse(
        y_train,
        y_pred_train_refined
    )

    nmse_test = compute_nmse(
        y_test,
        y_pred_test_refined
    )

    complexity = len(state)

    return {
        "algorithm": "TPSR",
        "dataset": args.dataset,
        "scenario": args.scenario,
        "fold": args.fold,
        "seed": args.seed,
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "nmse_train": nmse_train,
        "nmse_test": nmse_test,
        "expression": str(expression),
        "complexity": complexity,
        "time": elapsed,
        "status": "success"
    }




def save_result(result, output_path):

    output_dir = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    fieldnames = [
        "algorithm",
        "dataset",
        "scenario",
        "fold",
        "seed",
        "mse_train",
        "mse_test",
        "r2_train",
        "r2_test",
        "nmse_train",
        "nmse_test",
        "expression",
        "complexity",
        "time",
        "status"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerow(result)




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

        result = run_tpsr(
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

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str
        )
    )


if __name__ == "__main__":
    main()
