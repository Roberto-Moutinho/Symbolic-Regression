import os
import re
import json
import copy
import subprocess
import argparse
import numpy as np
import pandas as pd
import optuna

from pathlib import Path


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

ARQUIVO_PLANILHA = "hiperparametros.csv"

PASTA_RESULTADOS = Path("resultados_otimizacao")

N_TRIALS = 50

SEMENTES = [42, 43, 44]

ALGORITMO = "uDSR"

DATASET = "dataset.csv"

CONFIG_BASE = "config_base.json"


# ============================================================
# LEITURA DA PLANILHA
# ============================================================

def carregar_hiperparametros(algoritmo):

    df = pd.read_excel(
        ARQUIVO_PLANILHA,
        sheet_name="Espaço de busca Hiperparâmetros"
    )

    # A planilha utiliza células mescladas.
    df["Algoritmo"] = df["Algoritmo"].ffill()

    df = df[
        df["Algoritmo"].astype(str).str.strip() == algoritmo
    ].copy()

    if df.empty:
        raise ValueError(
            f"Algoritmo '{algoritmo}' não encontrado na planilha."
        )

    return df


# ============================================================
# INTERPRETAÇÃO DAS FAIXAS
# ============================================================

def interpretar_faixa(valor):

    if pd.isna(valor):
        return None

    texto = str(valor).strip()

    # Remove espaços
    texto = texto.replace(" ", "")

    # Converte separador decimal brasileiro
    texto = texto.replace(",", ".")

    # Remove colchetes
    texto = texto.replace("[", "").replace("]", "")

    # Caso: 10-100
    match = re.match(
        r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$",
        texto
    )

    if match:
        return float(match.group(1)), float(match.group(2))

    return None


# ============================================================
# CONVERSÃO DO HIPERPARÂMETRO PARA OPTUNA
# ============================================================

def sugerir_parametro(trial, linha):

    nome = str(linha["Hiperparâmetro"]).strip()

    tipo = str(linha["Tipo"]).strip()

    escala = str(linha["Escala"]).strip()

    faixa = interpretar_faixa(
        linha["Faixa proposta"]
    )

    if faixa is None:
        return None

    minimo, maximo = faixa

    # --------------------------------------------------------
    # INTEIROS
    # --------------------------------------------------------

    if tipo.lower() == "inteiro":

        if escala.lower() == "log":
            return trial.suggest_int(
                nome,
                int(minimo),
                int(maximo),
                log=True
            )

        return trial.suggest_int(
            nome,
            int(minimo),
            int(maximo)
        )

    # --------------------------------------------------------
    # CONTÍNUOS
    # --------------------------------------------------------

    if tipo.lower() == "contínuo":

        if escala.lower() == "log":
            return trial.suggest_float(
                nome,
                minimo,
                maximo,
                log=True
            )

        return trial.suggest_float(
            nome,
            minimo,
            maximo
        )

    return None


# ============================================================
# GERAÇÃO DOS HIPERPARÂMETROS
# ============================================================

def gerar_parametros(trial, df):

    parametros = {}

    for _, linha in df.iterrows():

        nome = str(
            linha["Hiperparâmetro"]
        ).strip()

        valor = sugerir_parametro(
            trial,
            linha
        )

        if valor is not None:
            parametros[nome] = valor

    return parametros


# ============================================================
# ATUALIZAÇÃO DO JSON DO uDSR
# ============================================================

def atualizar_config_udsr(
    config_base,
    parametros,
    dataset,
    seed
):

    config = copy.deepcopy(config_base)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    if "task" not in config:
        config["task"] = {}

    config["task"]["dataset"] = dataset

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    if "experiment" not in config:
        config["experiment"] = {}

    config["experiment"]["seed"] = seed

    # --------------------------------------------------------
    # PARÂMETROS DO GP MELD
    # --------------------------------------------------------

    if "gp_meld" not in config:
        config["gp_meld"] = {}

    gp = config["gp_meld"]

    if "generations" in parametros:
        gp["generations"] = int(
            parametros["generations"]
        )

    if "p_crossover" in parametros:
        gp["p_crossover"] = float(
            parametros["p_crossover"]
        )

    if "p_mutate" in parametros:
        gp["p_mutate"] = float(
            parametros["p_mutate"]
        )

    if "tournament_size" in parametros:
        gp["tournament_size"] = int(
            parametros["tournament_size"]
        )

    if "train_n" in parametros:
        gp["train_n"] = int(
            parametros["train_n"]
        )

    if "mutate_tree_max" in parametros:
        gp["mutate_tree_max"] = int(
            parametros["mutate_tree_max"]
        )

    return config


# ============================================================
# EXECUÇÃO DO uDSR
# ============================================================

def executar_udsr(config_path):

    comando = [
        "python",
        "-m",
        "dso.run",
        str(config_path)
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(resultado.stdout)

    if resultado.returncode != 0:

        raise RuntimeError(
            "Erro durante a execução do uDSR."
        )

    return resultado.stdout


# ============================================================
# EXTRAÇÃO DO RMSE
# ============================================================

def extrair_rmse(log):

    # Procura padrões como:
    #
    # RMSE: 0.123
    # rmse = 0.123
    # validation_rmse: 0.123

    padroes = [
        r"RMSE\s*[:=]\s*([0-9eE\+\-\.]+)",
        r"rmse\s*[:=]\s*([0-9eE\+\-\.]+)",
        r"validation_rmse\s*[:=]\s*([0-9eE\+\-\.]+)"
    ]

    valores = []

    for padrao in padroes:

        encontrados = re.findall(
            padrao,
            log
        )

        for valor in encontrados:

            try:
                valores.append(
                    float(valor)
                )
            except ValueError:
                pass

    if not valores:

        raise RuntimeError(
            "Não foi possível encontrar o RMSE no resultado do uDSR."
        )

    return valores[-1]


# ============================================================
# FUNÇÃO OBJETIVO DO OPTUNA
# ============================================================

def criar_objective(
    df,
    config_base,
    dataset,
    seed
):

    def objective(trial):

        parametros = gerar_parametros(
            trial,
            df
        )

        print("\n")
        print("=" * 70)
        print(
            f"TRIAL {trial.number}"
        )
        print("=" * 70)

        print(
            "Hiperparâmetros:"
        )

        for nome, valor in parametros.items():

            print(
                f"{nome}: {valor}"
            )

        # ----------------------------------------------------
        # CONFIGURAÇÃO
        # ----------------------------------------------------

        config = atualizar_config_udsr(
            config_base,
            parametros,
            dataset,
            seed
        )

        pasta_trial = (
            PASTA_RESULTADOS /
            f"trial_{trial.number}"
        )

        pasta_trial.mkdir(
            parents=True,
            exist_ok=True
        )

        config_path = (
            pasta_trial /
            "config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                config,
                arquivo,
                indent=4
            )

        # ----------------------------------------------------
        # EXECUTA uDSR
        # ----------------------------------------------------

        log = executar_udsr(
            config_path
        )

        # Salva log
        with open(
            pasta_trial / "log.txt",
            "w",
            encoding="utf-8"
        ) as arquivo:

            arquivo.write(log)

        # ----------------------------------------------------
        # OBTÉM MÉTRICA
        # ----------------------------------------------------

        rmse = extrair_rmse(
            log
        )

        print(
            f"RMSE = {rmse}"
        )

        # Guarda informações no trial
        trial.set_user_attr(
            "seed",
            seed
        )

        trial.set_user_attr(
            "rmse",
            rmse
        )

        return rmse

    return objective


# ============================================================
# EXECUÇÃO DA OTIMIZAÇÃO
# ============================================================

def otimizar():

    print(
        f"\nAlgoritmo selecionado: {ALGORITMO}"
    )

    # --------------------------------------------------------
    # CARREGA HIPERPARÂMETROS
    # --------------------------------------------------------

    df = carregar_hiperparametros(
        ALGORITMO
    )

    print("\nHiperparâmetros encontrados:")

    print(
        df[
            [
                "Hiperparâmetro",
                "Valor atual",
                "Faixa proposta",
                "Escala"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # CONFIGURAÇÃO BASE
    # --------------------------------------------------------

    with open(
        CONFIG_BASE,
        "r",
        encoding="utf-8"
    ) as arquivo:

        config_base = json.load(
            arquivo
        )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    todos_resultados = []

    # --------------------------------------------------------
    # DIFERENTES SEMENTES
    # --------------------------------------------------------

    for seed in SEMENTES:

        print("\n")
        print("#" * 70)
        print(
            f"SEMENTE = {seed}"
        )
        print("#" * 70)

        sampler = optuna.samplers.TPESampler(
            seed=seed
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        objective = criar_objective(
            df,
            config_base,
            DATASET,
            seed
        )

        study.optimize(
            objective,
            n_trials=N_TRIALS
        )

        # ----------------------------------------------------
        # MELHOR RESULTADO
        # ----------------------------------------------------

        print("\nMelhor resultado:")
        print(
            study.best_params
        )

        print(
            f"RMSE = {study.best_value}"
        )

        # ----------------------------------------------------
        # RESULTADOS DOS TRIALS
        # ----------------------------------------------------

        resultados = study.trials_dataframe()

        resultados[
            "seed"
        ] = seed

        todos_resultados.append(
            resultados
        )

    # --------------------------------------------------------
    # JUNTA TODOS OS RESULTADOS
    # --------------------------------------------------------

    resultados_finais = pd.concat(
        todos_resultados,
        ignore_index=True
    )

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_saida = (
        PASTA_RESULTADOS /
        "resultados_hiperparametros.xlsx"
    )

    resultados_finais.to_excel(
        arquivo_saida,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("OTIMIZAÇÃO FINALIZADA")
    print("=" * 70)

    print(
        f"Resultados salvos em:\n{arquivo_saida}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    otimizar()
