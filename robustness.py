from ema_workbench.em_framework.optimization import EpsilonProgress
from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, IntegerParameter, RealParameter)
from ema_workbench import ema_logging, MultiprocessingEvaluator, SequentialEvaluator, Samplers
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import functools

from problem_formulation import get_model_for_problem_formulation

model, _ = get_model_for_problem_formulation(1)
convergence = [EpsilonProgress()]

def robustness(direction, threshold, data):
    if direction == SMALLER:
        return np.sum(data<=threshold)/data.shape[0]
    else:
        return np.sum(data>=threshold)/data.shape[0]

SMALLER = 'SMALLER'

Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 1) #not ok
Expected_Annual_Damage = functools.partial(robustness, SMALLER, 2000e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Total_Investment_Costs = functools.partial(robustness, SMALLER, 300e6)#THOSE NUMBERS NEED TO BE SPECIFIED AGAIN

nfe = 20000

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE

funcs = {'Expected Number of Deaths':Expected_Number_of_Deaths,
         'Expected Annual Damage': Expected_Annual_Damage,
         'Total Investment Costs': Total_Investment_Costs}

robustnes_functions = [ScalarOutcome('Expected Number of Deaths', kind=MAXIMIZE,
                                     function=Expected_Number_of_Deaths),
                       ScalarOutcome('Expected Annual Damage', kind=MAXIMIZE,
                                     function=Expected_Annual_Damage),
                       ScalarOutcome('Total Investment Costs', kind=MAXIMIZE,
                                     function=Total_Investment_Costs),
                      ]

epsilons=[0.05,]*len(funcs)  #final value of epsilon should be much lower.Just for experiment purposes is 1

# True, use results in pickle file; False, run MultiprocessingEvaluator
use_pickle1 = False

if __name__ == '__main__':


    if use_pickle1:
        with open('data/formulation_results.pickle', 'rb') as filehandler:
            results = pickle.load(filehandler)

    else:
        scenarios = 10
        with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
            results = evaluator.robust_optimize(robustnes_functions, scenarios,nfe=nfe, 
                                                convergence=convergence, epsilons=epsilons)

        # Save results in Pickle file
        with open('data/formulation_results.pickle', 'wb') as filehandler:
            pickle.dump(results, filehandler)

    archive, convergence = results