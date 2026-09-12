# Symbolic Regression Benchmark #

Benchmark experimental de algoritmos de **Regressão Simbólica (Symbolic Regression)**, com foco na comparação de diferentes abordagens evolutivas, probabilísticas, neurais e baseadas em Large Language Models (LLMs).

## Objetivo

Dado um conjunto de dados

$$
D = \{(x_i,y_i)\}_{i=1}^{n}, \qquad x_i \in \mathbb{R}^{d},\; y_i \in \mathbb{R},
$$

o objetivo da Regressão Simbólica é encontrar uma expressão matemática

$$
f:\mathbb{R}^{d}\rightarrow\mathbb{R}
$$

que aproxime os valores observados, isto é,

$$
f(x_i) \approx y_i.
$$

O projeto busca comparar os algoritmos sob um protocolo experimental padronizado, considerando desempenho preditivo, complexidade das expressões e custo computacional.

## Algoritmos

O benchmark inclui:

* **Operon**
* **GOMEA**
* **GPZDG**
* **QLattice**
* **PySR**
* **GSGP**
* **uDSR**
* **TPSR**
* **DySymNet**
* **RAG-SR**
* **LLM-SR**

## Estrutura

```text
Symbolic-Regression/
│
├── dados/
│   ├── *.csv
│   └── hiperparametros_treinamento_corrigido.csv
│
├── EXECUTORES/
│   └── executar_*.py
│
├── avaliadores/
│   └── ...
│
├── configs/
│   └── ...
│
├── otimizadores/
│   └── ...
│
├── scripts/
│   └── consolidar_resultados.py
│
├── resultados/
│   └── resultados_consolidados.csv
│
├── run_tuning.sh
├── run_final.sh
└── README.md
```

### Diretórios

* `dados/`: datasets e espaço de busca dos hiperparâmetros.
* `EXECUTORES/`: execução individual de cada algoritmo.
* `avaliadores/`: cálculo das métricas e avaliação das expressões.
* `configs/`: configurações dos experimentos.
* `otimizadores/`: otimização de hiperparâmetros.
* `scripts/`: scripts auxiliares e consolidação dos resultados.
* `resultados/`: resultados consolidados dos experimentos.

## Datasets

### Ajuste de hiperparâmetros

Os seguintes datasets são utilizados na etapa de tuning:

* `chemical_2_competition_train.csv`
* `friction_dyn_one-hot_train.csv`
* `nasa_battery_1_10min_train.csv`
* `nikuradse_1_train.csv`

### Avaliação final

A avaliação final utiliza datasets independentes de primeiros princípios:

* Ideal Gas
* Kepler
* Planck
* Rydberg

Os datasets utilizados na etapa final não participam da seleção dos hiperparâmetros.

## Protocolo experimental

O experimento é dividido em duas etapas.

### 1. Hyperparameter Tuning

Para cada algoritmo são avaliadas diferentes configurações de hiperparâmetros utilizando **Optuna**.

O protocolo considera:

* 4 datasets;
* 5 folds;
* 5 repetições independentes por fold;
* seeds `42–46`;
* 50 trials por execução;
* orçamento máximo de **200.000 avaliações** ou **60 minutos**, o que ocorrer primeiro.

Os resultados dos trials não são descartados, permitindo análises posteriores de **sensibilidade aos hiperparâmetros**.

### 2. Avaliação Final

Após o tuning, os melhores hiperparâmetros são utilizados nos datasets de avaliação final.

São realizadas **10 repetições independentes**, com seeds `100–109`.

## Cenários de operadores

O benchmark considera diferentes conjuntos de operadores matemáticos. Entre eles:

**Cenário C1**

```text
+, -, *, /
```

**Cenário C2**

```text
+, -, *, /, exp, log, sqrt, ^
```

O mesmo protocolo experimental é aplicado aos diferentes cenários para permitir uma comparação justa.

## Métricas

Os algoritmos são avaliados considerando:

* **R²**
* **RMSE / NRMSE**
* **MSE**
* **Complexidade da expressão**
* **Tempo de execução**
* **Número efetivo de avaliações**
* **Recuperação simbólica**, quando aplicável

Os resultados são armazenados juntamente com informações sobre dataset, fold, seed, algoritmo, configuração e expressão encontrada.

## Resultados

Os resultados consolidados são armazenados em:

```text
resultados/resultados_consolidados.csv
```

O arquivo contém os resultados dos experimentos e permite análises estatísticas e de sensibilidade posteriores.

## Reprodutibilidade

Cada experimento registra sua seed e configuração utilizada. Os experimentos são executados de forma independente para permitir a reprodução dos resultados.

Quando aplicável, os algoritmos são executados em ambientes isolados por **Docker**.

## Execução

### Otimização de hiperparâmetros

```bash
chmod +x run_tuning.sh
./run_tuning.sh
```

### Avaliação final

```bash
chmod +x run_final.sh
./run_final.sh
```

> Os scripts devem ser executados a partir da raiz do projeto.





