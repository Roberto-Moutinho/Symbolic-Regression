#!/bin/bash
set -e

ALGORITHMS=(
    "operon"
    "gomea"
    "gpzdg"
    "qlattice"
    "pysr"
    "gsgp"
    "udsr"
    "tpsr"
    "dysymnet"
    "ragsr"
    "llmsr"
)

DATASETS=(
    "chemical_2_competition_train.csv"
    "friction_dyn_one-hot_train.csv"
    "nasa_battery_1_10min_train.csv"
    "nikuradse_1_train.csv"
)

SEEDS=(42 43 44 45 46)

for ALGORITHM in "${ALGORITHMS[@]}"; do

    echo "=========================================="
    echo "Algoritmo: $ALGORITHM"
    echo "=========================================="

    for DATASET in "${DATASETS[@]}"; do

        DATASET_NAME="${DATASET%.csv}"

        for FOLD in {1..5}; do

            for SEED in "${SEEDS[@]}"; do

                echo "------------------------------------------"
                echo "Algorithm: $ALGORITHM"
                echo "Dataset:   $DATASET_NAME"
                echo "Fold:      $FOLD"
                echo "Seed:      $SEED"
                echo "------------------------------------------"

                python "executores/executar_${ALGORITHM}.py" \
                    --hyperparameters "dados/hiperparametros_treinamento_corrigido.csv" \
                    --dataset "dados/${DATASET}" \
                    --fold "$FOLD" \
                    --output-dir "resultados/${ALGORITHM}/${DATASET_NAME}/fold_${FOLD}/seed_${SEED}" \
                    --n-trials 50 \
                    --seed "$SEED" \
                    --direction maximize \
                    --metric R2

            done
        done
    done
done

python scripts/consolidar_resultados.py
