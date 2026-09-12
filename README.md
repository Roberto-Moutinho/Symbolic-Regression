# Symbolic-Regression

A reproducible research project for the systematic evaluation of
symbolic regression (SR) algorithms under controlled experimental
conditions.

The project compares evolutionary, genetic-programming, neural,
probabilistic, language-model-based, and hybrid symbolic regression
methods using common datasets, operator sets, computational budgets,
cross-validation procedures, and statistical evaluation protocols.

Overview

Symbolic regression aims to discover an explicit mathematical expression
that approximates an unknown relationship between input variables and a
target variable.

Given a dataset

[ \mathcal{D}{=tex} = {(\mathbf{x}{=tex}i,y_i)}{i=1}^{n}, ]

where

[ \mathbf{x}{=tex}_i \in {=tex}\mathbb{R}{=tex}^{d},
\qquad {=tex}y_i \in {=tex}\mathbb{R}{=tex}, ]

the objective is to find a function

[ f:\mathbb{R}{=tex}^{d}\rightarrow{=tex}\mathbb{R}{=tex} ]

such that

[ f(\mathbf{x}{=tex}_i)\approx {=tex}y_i. ]

In the ideal case,

[ f(\mathbf{x}{=tex}_i)=y_i. ]

Unlike conventional regression, symbolic regression searches
simultaneously for the mathematical structure of the model and its
numerical parameters.

This project investigates how different SR algorithms behave under a
common experimental framework, with particular attention to:

predictive accuracy;

expression complexity;

computational cost;

robustness across independent repetitions;

hyperparameter sensitivity;

operator-set effects;

and, when possible, recovery of the underlying generating equation.

Algorithms

The benchmark currently considers the following algorithms:

Algorithm      Category

Operon     Genetic programming / evolutionary symbolic regression
PySR       Evolutionary symbolic regression
GOMEA      Gene-pool-based evolutionary optimization
GSGP       Geometric Semantic Genetic Programming
GPZDG      Genetic programming symbolic regression
QLattice   Symbolic regression based on model enumeration/search
uDSR       Deep / neural-guided symbolic regression
TPSR       Tree-based probabilistic symbolic regression
DySymNet   Neural symbolic regression
RAG-SR     Retrieval-augmented symbolic regression
LLM-SR     Large-language-model-based symbolic regression

Each algorithm is treated according to its own native implementation and
hyperparameterization. The benchmark standardizes the experimental
protocol, not the internal algorithmic implementation.

Repository Structure

A recommended project structure is:

Symbolic-Regression/
│
├── README.md
├── dataset.csv
├── hiperparametros_treinamento.csv
├── hiperparametros_consolidada.xlsx
├── config_base.json
│
├── resultados_otimizacao/
│
├── Operon/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── PySR/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── GOMEA/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── GSGP/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── GPZDG/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── QLattice/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── uDSR/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── TPSR/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── DySymNet/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
├── RAG-SR/
│   ├── Dockerfile
│   ├── config_base.json
│   └── ...
│
└── LLM-SR/
    ├── Dockerfile
    ├── config_base.json
    └── ...

The exact implementation-specific structure may differ between
algorithms. Dockerfiles are used to isolate dependencies and improve
reproducibility.

Experimental Design

Operator Scenarios

Two operator scenarios are used.

C1 --- Common Operator Set

The minimum common set is:

+, -, *, /

C1 is intended to provide the most comparable setting across all
algorithms.

C2 --- Extended Operator Set

The broader scenario is:

+, -, *, /, exp, log, sqrt, ^

Not every algorithm supports every operator. Therefore, C2 must be
interpreted according to the documented capabilities of each
implementation.

When an algorithm does not support an operator, the unavailable operator
is not artificially introduced. Its effective operator set must be
recorded.

This distinction is important because C1 primarily evaluates algorithmic
behavior under a common search space, whereas C2 evaluates behavior
under a richer symbolic search space.

Datasets

Hyperparameter-Tuning Datasets

Hyperparameter tuning is performed using the training datasets only.

The current tuning datasets are:

Dataset                                Samples   Variables Target

chemical_2_competition_train.csv         711          57 target
friction_dyn_one-hot_train.csv          1009          17 target
nasa_battery_1_10min_train.csv           504           6 target
nikuradse_1_train.csv                    230           2 target

The corresponding test sets are not used to select hyperparameters.

Final Evaluation Datasets

The final evaluation uses separate external datasets:

first_principles_ideal_gas

first_principles_kepler

first_principles_planck

first_principles_rydberg

These datasets are intentionally kept separate from the
hyperparameter-selection process to reduce the risk of overfitting the
benchmark to the tuning data.

Hyperparameter Tuning

Hyperparameter tuning is performed before the final evaluation.

The tuning protocol uses:

4 tuning datasets;

5-fold cross-validation;

5 independent repetitions per fold;

fixed random seeds;

both C1 and C2 operator scenarios;

preservation of all tuning results.

For each dataset, scenario, and algorithm, the experimental structure
is:

5 folds × 5 repetitions = 25 executions

Thus, for one algorithm:

4 datasets × 2 scenarios × 5 folds × 5 repetitions
= 200 tuning executions

For all 11 algorithms:

200 × 11 = 2,200 tuning executions

The exact number of executions may change if an algorithm does not
support a particular experimental scenario.

Selection Criterion

Hyperparameters are selected primarily according to validation
performance.

The intended selection criterion is:

maximize mean validation (R^2);

in the case of a tie or practically equivalent performance, prefer
lower expression complexity.

The selected configuration is then fixed for the final evaluation.

Important

Hyperparameter-tuning results must not be discarded.

All trials should be retained because they can later be used to analyze:

parameter sensitivity;

robustness;

interactions between hyperparameters;

performance distributions;

and the stability of each algorithm.

Computational Budget

The target computational budget is:

200,000 evaluations per execution
OR
60 minutes,
whichever occurs first.

The benchmark records the actual number of evaluations completed before
termination.

The purpose of this restriction is to avoid giving one algorithm
substantially more search effort than another.

Population-Based Algorithms

For algorithms where one evaluation corresponds approximately to one
individual evaluated, a population-generation formulation can be used.

For example:

population_size = 500
generations = 400

gives:

500 × 400 = 200,000

The same principle is applied to the corresponding population-based
implementations when their evaluation semantics permit it.

Native Evaluation Counters

Some algorithms expose a native evaluation budget and should use it
directly.

For example, PySR should use its native evaluation limit rather than
assuming that:

population_size × populations × niterations

is equivalent to the number of objective-function evaluations.

For algorithms whose internal computational unit differs from a
conventional symbolic-regression evaluation, the implementation must be
inspected and the effective budget documented.

This is particularly important for:

QLattice;

uDSR;

TPSR;

DySymNet;

RAG-SR;

LLM-SR.

For these methods, internal training steps, model generations, LLM
calls, neural epochs, and symbolic evaluations should not be silently
treated as identical quantities.

Hyperparameter Search Space

The file

hiperparametros_treinamento.csv

contains the consolidated hyperparameter search space.

It records, for each algorithm:

parameter name;

parameter type;

default value;

lower bound;

upper bound;

experimental notes.

The corrected version should be treated as the working specification:

hiperparametros_treinamento_corrigido.csv

Some parameters require implementation-level verification before
large-scale execution, particularly when their relation to the common
200,000-evaluation budget is not direct.

Final Evaluation

After hyperparameter tuning, one winning configuration is selected for
each:

algorithm × operator scenario

The selected configuration must come exclusively from the tuning phase.

The final evaluation uses:

4 external datasets;

C1 and/or C2 according to the benchmark design;

10 independent repetitions;

seeds 100--109.

The final configuration is not retuned on the final datasets.

Metrics

The benchmark records multiple complementary metrics.

Predictive Performance

Coefficient of Determination

[ R^2 = 1 - \frac{\sum_i(y_i-\hat{y}_i)^2}{=tex}
{\sum{=tex}_i(y_i-\bar{=tex}{y})^2}. ]

Higher values indicate better predictive performance.

Root Mean Squared Error

[ RMSE = \sqrt{
\frac{1}{n}
\sum_i(y_i-\hat{y}_i)^2
}{=tex}. ]

Lower values are better.

A normalized version may also be reported when appropriate.

Expression Complexity

Expression complexity measures the structural size or difficulty of the
discovered symbolic expression.

Depending on the algorithm, this may involve quantities such as:

number of nodes;

expression length;

tree depth;

number of operators;

or the implementation's native complexity measure.

Complexity should be recorded alongside predictive performance rather
than using accuracy alone.

Runtime

The elapsed execution time is recorded.

The benchmark terminates an execution when either:

200,000 evaluations

or

60 minutes

is reached.

Actual Evaluations

The experiment should record the actual number of evaluations
performed.

This is essential because an execution can terminate early due to:

the time limit;

convergence;

implementation-specific stopping criteria;

or another termination condition.

Symbolic Recovery

For datasets with a known generating equation, symbolic recovery should
be evaluated whenever possible.

A model may have high predictive accuracy without recovering the
underlying mathematical relationship. Therefore, predictive equivalence
and structural recovery should be considered separate objectives.

Repetitions and Random Seeds

Randomness is explicitly controlled.

Hyperparameter Tuning

The tuning repetitions use:

42, 43, 44, 45, 46

for the five independent repetitions.

Final Evaluation

The final evaluation uses:

100, 101, 102, 103, 104,
105, 106, 107, 108, 109

for ten independent repetitions.

Seeds must be saved with every result.

Cross-Validation

The tuning phase uses 5-fold cross-validation.

For each fold:

the training portion is used to fit the symbolic regression model;

the validation portion is used to evaluate the candidate
configuration;

the procedure is repeated for the specified independent seeds.

The folds should be generated with a controlled randomization procedure
and reproducible seeds.

No final-test data should participate in hyperparameter selection.

Statistical Analysis

Because symbolic regression results can vary substantially between runs,
conclusions should not rely on a single execution.

The benchmark should report:

median performance;

mean where appropriate;

dispersion;

95% confidence intervals;

per-dataset results;

per-algorithm results;

per-scenario results.

When algorithms are statistically compared, paired comparisons should
preferably use the same datasets and random seeds.

Multiple-comparison correction should be applied when multiple
statistical tests are performed.

The analysis should prioritize per-dataset behavior rather than
relying exclusively on a single aggregate ranking.

Reproducibility

Reproducibility is a central requirement of the project.

Every execution should save at least:

algorithm
dataset
scenario
fold
seed
hyperparameters
operator set
random seed
evaluation budget
actual evaluations
runtime
best expression
R²
RMSE / NRMSE
expression complexity
termination condition
hardware information
software/environment information
logs

A recommended result record is conceptually:

results/
└── <algorithm>/
    └── <dataset>/
        └── <scenario>/
            └── fold_<n>/
                └── seed_<n>/
                    ├── config.json
                    ├── result.json
                    ├── expression.txt
                    └── log.txt

Docker

Each algorithm should preferably have its own Docker environment.

A typical workflow is:

docker build -t <algorithm-name> .

and then:

docker run --rm <algorithm-name>

The exact commands depend on the implementation.

Docker is used to reduce dependency conflicts and improve
reproducibility across algorithms.

Configuration

The project uses configuration files to separate experimental settings
from implementation code.

A configuration should distinguish between:

Common experimental settings

dataset
operator scenario
seed
cross-validation
evaluation budget
time limit
metrics
output directory

and:

Algorithm-specific settings

population size
number of generations
mutation probability
crossover probability
learning rate
beam size
LLM parameters
neural-network parameters
etc.

The configuration format should not imply that all algorithms expose
identical parameters.

The experimental protocol is standardized; the algorithms are not.

Workflow

The recommended workflow is:

1. Prepare datasets
        ↓
2. Build algorithm environments
        ↓
3. Validate each implementation
        ↓
4. Verify evaluation-counter semantics
        ↓
5. Run hyperparameter tuning
        ↓
6. Preserve all tuning trials
        ↓
7. Analyze hyperparameter sensitivity
        ↓
8. Select one configuration per algorithm/scenario
        ↓
9. Run final evaluation
        ↓
10. Aggregate results
        ↓
11. Perform statistical analysis
        ↓
12. Generate tables and figures

Before launching the complete benchmark, each algorithm should first be
tested on a small experimental case to verify:

installation;

input format;

operator configuration;

objective function;

evaluation counting;

timeout behavior;

result parsing;

and reproducibility.

Recommended Validation Procedure

Before running thousands of executions, validate one algorithm
end-to-end.

For example:

Algorithm: Operon
Dataset: Nikuradse
Scenario: C1
Fold: 1
Seed: 42

Confirm that:

the algorithm starts correctly;

the dataset is loaded correctly;

the operator set is correct;

the evaluation counter behaves as expected;

the 200,000-evaluation budget is respected;

the 60-minute timeout is respected;

the best expression is saved;

the metrics are calculated correctly;

the configuration is saved;

the run can be reproduced with the same seed.

Only after this validation should the full experiment be launched.

Hyperparameter Sensitivity

The tuning database is also an experimental result.

Rather than keeping only the winning configuration, all trials should be
retained.

This enables later analyses such as:

one-dimensional parameter sensitivity;

partial dependence;

parameter-performance correlations;

robustness of optimal regions;

interaction effects;

variance across folds;

variance across datasets;

and comparison of tuning landscapes between algorithms.

A useful conceptual representation is:

Hyperparameters
       │
       ▼
 ┌───────────────┐
 │ Search space  │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ CV experiments│
 └───────┬───────┘
         │
         ├───────────────┐
         ▼               ▼
 Performance        Complexity
         │               │
         └───────┬───────┘
                 ▼
       Selected configuration

Important Experimental Principles

1. Do not tune on final-test data

Final evaluation datasets must remain unseen during hyperparameter
selection.

2. Do not discard tuning results

All trials are potentially useful for sensitivity analysis.

3. Do not assume identical parameters across algorithms

A parameter called population_size, n_iter, epochs, or models
may represent fundamentally different computational processes.

4. Do not equate all computational units with evaluations

The common evaluation budget must be mapped to the actual
implementation.

5. Keep random seeds explicit

Every result must be traceable to its seed.

6. Preserve failed runs

Failed executions can reveal implementation or resource problems and
should not simply disappear from the experiment database.

7. Report accuracy and complexity together

A more accurate expression is not automatically a better symbolic model
if it is dramatically more complex.

Current Experimental Scope

The benchmark currently consists of:

11 algorithms
×
2 operator scenarios
×
4 tuning datasets
×
5 folds
×
5 tuning repetitions

for the hyperparameter-selection stage.

This corresponds to up to:

2,200 tuning executions

before considering additional validation runs.

The final evaluation consists of:

11 algorithms
×
4 external datasets
×
10 repetitions

per operator scenario evaluated.

Because the total computational cost can be substantial, pilot
validation is strongly recommended before launching the complete
benchmark.

Project Status

The experimental framework is under active development.

The following components are defined at the methodological level:

algorithm set;

tuning datasets;

final datasets;

C1/C2 operator scenarios;

cross-validation structure;

independent repetitions;

random seeds;

computational budget;

evaluation metrics;

statistical-analysis principles;

and hyperparameter-search space.

Implementation-level verification is still required for parameters whose
meaning is specific to each algorithm, especially the mapping between
native algorithmic computation and the common evaluation budget.

References and Implementations

The benchmark uses the original implementations of the algorithms
whenever possible. Each algorithm directory should document:

source repository;

version or commit;

installation procedure;

required dependencies;

license;

citation;

implementation-specific configuration;

and any deviations from the standard benchmark protocol.

For reproducible research, the exact software version or Git commit used
in the experiments should be recorded.

Citation

If this benchmark is used in academic work, cite the project according
to the publication or technical report associated with the study.

A project-specific citation will be added once the corresponding
research manuscript is finalized.

License

Add the appropriate project license here once the licensing terms for
the benchmark code and its components have been defined.

Note that individual third-party algorithms may have their own licenses,
which remain applicable to their respective source code and
dependencies.

Authors

Roberto Moutinho

Federal University of ABC (UFABC)

This project is developed in the context of research on symbolic
regression, evolutionary computation, and computational methods for
scientific modeling.
