#!/bin/bash

set -e

ALGORITHMS=(
    "operon"
    "gomea"
    "qlattice"
    "gsgp"
    "gpzdg"
    "pysr"
    "ragsr"
    "dysymnet"
    "llmsr"
    "ragsr"
    "tpsr"
    "udsr"
)

for ALGORITHM in "${ALGORITHMS[@]}"; do

    echo "=========================================="
    echo "Otimizando: $ALGORITHM"
    echo "=========================================="

    python "EXECUTORES/executar_${ALGORITHM}.py" \
        --csv dados/hiperparametros_treinamento.csv \
        --output-dir "resultados/${ALGORITHM}" \
        --n-trials 50 \
        --seed 42 \
        --direction minimize \
        --metric MSE

done


python scripts/consolidar_resultados.py
