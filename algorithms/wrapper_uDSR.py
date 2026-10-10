import numpy as np
from dso import DeepSymbolicRegressor

def run_udsr_benchmark():
   
    np.random.seed(42)
    n_samples = 200
    X = np.random.uniform(-3.3, 3.3, size=(n_samples, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2

   
    config = {
        "experiment": {
            "logdir": "./logs_udsr_benchmark",
            "seed": 42
        },
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
            "metric": "mse",
            "metric_params": {}
        },
        "model": {
            "controller": {
                "learning_rate": 0.0003,
                "entropy_weight": 0.005
            }
        },
      
        "gp_meld": {
            "run_gp_meld": True,       
            "verbose": False,
            "generations": 20,
            "p_crossover": 0.5,
            "p_mutate": 0.5,
            "tournament_size": 5,
            "train_n": n_samples,
            "parallel_eval": True
        }
    }

   
    print("Iniciando o treinamento do uDSR para benchmark...")
    
    model = DeepSymbolicRegressor(config=config)
    
  
    model.fit(X, y)

   
    best_program = model.predict()
    print("\n--- Resultados do Benchmark uDSR ---")
    print(f"Expressão Matemática Encontrada: {best_program}")

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f"Erro Quadrático Médio (MSE) no conjunto: {mse:.6f}")

if __name__ == "__main__":
    run_udsr_benchmark()
