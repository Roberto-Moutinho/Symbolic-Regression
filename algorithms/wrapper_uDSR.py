from dso import DeepSymbolicRegressor
from dso import DeepSymbolicOptimizer

import sys
import os
from tempfile import TemporaryDirectory
import json
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

import tensorflow as tf

print("Num GPUs Available: ", tf.test.is_gpu_available())
print("CUDA Available:     ", tf.test.is_built_with_cuda())

function_set = [
    # ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt"],
    ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "const"],
    # ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"],
    ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "const", "poly"],
]
degree = [2, 3]
run_gp_meld = [True, False]

hyper_params = []
for f in function_set:
    hyper_params.append({
        'function_set' : [f],
        'degree'       : [2],
        'run_gp_meld'  : [True]
    })
        

class uDSRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42,
                 function_set=["add","sub","mul","div","sin","cos","exp","log","sqrt","const","poly"],
                 degree=3, run_gp_meld=True):
        
        self.random_state = random_state
        self.function_set = function_set
        self.degree       = degree
        self.run_gp_meld  = run_gp_meld

    def fit(self, X, y):
        with TemporaryDirectory() as temp_dir:
            X, y = check_X_y(X, y, accept_sparse=False)
            if len(y.shape) == 1:
                Z = np.hstack((X, y[:,None]))
            else:
                Z = np.hstack((X, y))

            fname   = temp_dir + "/tmpdata.csv"
            np.savetxt(f"{fname}", Z, delimiter=",")
            
            config = {
                'task' : {
                    'task_type' : 'regression',
                    'dataset'   : fname,
                    
                    "function_set": self.function_set,
                
                  
                    "metric" : "inv_nrmse",
                    "metric_params" : [1.0],
                
                    "extra_metric_test" : None,
                    "extra_metric_test_params" : [],
                
                   
                    "threshold" : 1e-12,
                
                      
                    "protected" : False,
                
               
                    "reward_noise" : 0.0,
                    "reward_noise_type" : "r",
                    "normalize_variance" : False,
                
                  
                    "decision_tree_threshold_set" : [],
                
                   
                    "poly_optimizer_params" : {
                        "degree": self.degree,
                      
                        "coef_tol": 1e-6,
                     
                        "regressor": "dso_least_squares",
                        "regressor_params": {
                          
                            "cutoff_p_value": 1.0,
                          
                            "n_max_terms": None,
                           
                            "coef_tol": 1e-6
                        }
                    }
                },

                "gp_meld" : {
                    "run_gp_meld" : self.run_gp_meld,
                    "population_size" : 25,
                    "generations" : 25,
                    "crossover_operator" : "cxOnePoint",
                    "p_crossover" : 0.5,
                    "mutation_operator" : "multi_mutate",
                    "p_mutate" : 0.5,   
                    "tournament_size" : 5,
                    "train_n" : 50,
                    "mutate_tree_max" : 3,
                    "verbose" : True,
                    "parallel_eval" : False
                },

              
                "training" : {
                    "n_samples" : 10000,
                    "batch_size" : 500,
                    "epsilon" : 0.02,
           
                    "n_cores_batch" : -1
                },
            
              
                "controller" : {
                    "learning_rate": 0.0025,
                    "entropy_weight" : 0.03,
                    "entropy_gamma" : 0.7,
                    
                    "pqt" : True,
                    "pqt_k" : 10,
                    "pqt_batch_size" : 1,
                    "pqt_weight" : 200.0,
                    "pqt_use_pg" : False
                },
            
               
                "prior": {
                    "length" : {
                        "min_" : 4,
                        "max_" : 100,
                        "on" : True
                    },
                    "inverse" : {
                        "on" : True
                    },
                    "trig" : {
                        "on" : True
                    },
                    "const" : {
                        "on" : True
                    },
                    "no_inputs" : {
                        "on" : True
                    },
                    "uniform_arity" : {
                        "on" : True
                    },
                    "soft_length" : {
                        "loc" : 10,
                        "scale" : 5,
                        "on" : True
                    },
                    "domain_range" : {
                        "on" : True
                    }
                },
                'experiment' : {
                    'logdir' : None,
                }
            }
            
            cname   = temp_dir + "/config.json"
            with open(cname, "w") as config_file:
                json.dump(config, config_file, indent=4)

            model = DeepSymbolicOptimizer(cname)
            train_result = model.train()

        self.program_ = train_result["program"]

        return self

    def predict(self, X):
        check_is_fitted(self, "program_")
        X = check_array(X)

        return self.program_.execute(X)



est = uDSRRegressor()


def model(est, X=None):
  
    return str(est.program_.sympy_expr)
    

if __name__ == "__main__":
    import numpy as np  
    
    np.random.seed(0)
    X = np.random.random((10, 2))
    y = np.sin(X[:,0]) + X[:,1] ** 2

    est.fit(X, y)

    print(est.program_.pretty())
    print(est.program_.sympy_expr)

    print(len(est.program_.traversal))

    est.predict(2 * X)
