from datetime import datetime
import pandas as pd
import numpy as np
import sys
import json
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import *
from xgboost.sklearn import *
from sklearn.linear_model import *
from functools import partial
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import *
import warnings
from pprint import pprint
import os

def bayesian_optimization(model_info, X, y):
    """
        Tries to obtain the best model parameters by using bayesian optimization rather than an exhaustive search
        
        :param dict model_info : The information about the model and the tuning settings
        :param pandas.DataFrame X : The training data
        :param pandas.Series y : The training labels
        :return dict : The hyper parameters and the final score
    """
    
    # Obtain the grid
    grid = model_info["tuning"]["grid"]
    
    # Define a local model assessment function (to be used by scikit-optimize)
    def model_assessment(args, model, X_, y_):
        
        # From the arguments, parse the model parameters
        model_params = {name : value for name, value in zip([element.name for element in grid], args)}
        
        # Fit the model
        model = model(**model_params)
        
        # Compute the score
        score = cross_val_score(model, X_, y_, cv=model_info["tuning"]["cv"]).mean()
        
        if model_info["tuning"]["debug"]:
            print("Parameters considered", model_params, " with score ", score)
        
        # TODO: Have a look at this! This only works for the base metrics the model uses!
        return 1 - score if model_info["task"] == "classification" else score
        
    objective_function = partial(model_assessment, model=model_info["base"], X_=X, y_=y)
    
    # Determine which minimize strategy should be used
    if model_info["tuning"]["model_training_speed"] == "fast":
        results = gp_minimize(objective_function, grid, base_estimator=None, n_calls=model_info["tuning"]["optimization_cycles"], n_random_starts=model_info["tuning"]["optimization_cycles"] - 1)
    if model_info["tuning"]["model_training_speed"] == "slow":
        results = forest_minimize(objective_function, grid, base_estimator=None, n_calls=model_info["tuning"]["optimization_cycles"], n_random_starts=model_info["tuning"]["optimization_cycles"]-1)
    if model_info["tuning"]["model_training_speed"] == "very slow":
        results = gbrt_minimize(objective_function, grid, base_estimator=None, n_calls=model_info["tuning"]["optimization_cycles"], n_random_starts=model_info["tuning"]["optimization_cycles"]-1)
    
    # Obtain the optimal parameters
    return {name : value for name, value in zip([element.name for element in grid], results.x)}, 1 - results.fun if model_info["task"] == "classification" else results.fun

def randomized_optimization(model_info, X, y):
    """
        Tries to obtain the best model parameters by using randomized search
        
        :param dict model_info : The information about the model and the tuning settings
        :param pandas.DataFrame X : The training data
        :param pandas.Series y : The training labels
        :return dict : The hyper parameters and the final score
    """
    
    # Obtain the model and grid
    model = model_info["base"]()
    grid = model_info["tuning"]["grid"]
    
    # Perform the randomized search
    search = RandomizedSearchCV(model, grid, n_jobs=-1, n_iter=model_info["tuning"]["optimization_cycles"], verbose=2, cv=model_info["tuning"]["cv"])
    search = search.fit(X, y)
    
    return search.best_params_, search.best_score_

def exhaustive_optimization(model_info, X, y):
    """
        Tries to obtain the best model parameters by using exhasutive search

        :param dict model_info : The information about the model and the tuning settings
        :param pandas.DataFrame X : The training data
        :param pandas.Series y : The training labels
        :return dict : The hyper parameters and the final score
    """
    
    # Obtain the model and grid
    model = model_info["base"]()
    grid = model_info["tuning"]["grid"]
    
    # Perform the exhaustive search
    search = GridSearchCV(model, grid, n_jobs=-1, verbose=2, cv=model_info["tuning"]["cv"])
    search = search.fit(X, y)
    
    return search.best_params_, search.best_score_
    
    
def tune(model_name, model_info, X, y, tune=False):
    """
        Tunes the given model based on the model information to obtain the best hyper parameters
        
        :param string model_name : The name of the model
        :param dict model_info : The information about the model
        :param pandas.DataFrame X : The training data
        :param pandas.Series y : The training labels
    """
    model = model_info["base"]
    
    if model_info["tuning"]["preference"] == "BayesianOptimization":
        print("( BAYESIAN OPTIMIZATION with {} cycles.)".format(model_info["tuning"]["optimization_cycles"]))
        return bayesian_optimization(model_info, X, y)
    if model_info["tuning"]["preference"] == "Randomized":
        print("( RANDOMIZED SEARCH CV with {} cycles.)".format(model_info["tuning"]["optimization_cycles"]))
        return randomized_optimization(model_info, X, y)
    if model_info["tuning"]["preference"] == "Exhaustive":
        print("( EXHAUSTIVE SEARCH CV )")
        return exhaustive_optimization(model_info, X, y)
    
    # This model does not want to be tuned
    return None

if len(sys.argv) != 4:
    print("ERROR: Not enough parameters given. Usage: tuner.py <path to model info configuration file> <path to training features csv> <path to training labels csv>")
    exit(0)

warnings.filterwarnings("ignore", category=FutureWarning)
    
# Obtain the required information
models = eval(open(sys.argv[1]).read())
X = pd.read_csv(sys.argv[2])
y = pd.read_csv(sys.argv[3])
y = y[y.columns[0]]

model_output = {}

# Go over each model and tune it
for model_name in models:
    
    # Skip models that should not be tuned
    if models[model_name]["tuning"]["preference"] is None:
        continue
    
    print()
    print("TUNING :", model_name)
    optimal_paramaters, score = tune(model_name, models[model_name], X, y)
    print("OPTIMAL PARAMETERS")
    pprint(optimal_paramaters)
    print("OPTIMAL SCORE ", score)
    
    model_output[model_name] = {"optimal_params": optimal_paramaters, "score": score}

# Output the model output
try:
    os.stat("tuning_output")
except:
    os.mkdir("tuning_output")
with open("tuning_output/{}".format(datetime.now().strftime("tuning_output_%d-%m-%Y %H:%M")), "w+") as file:
    file.write(str(model_output))