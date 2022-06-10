
from ema_workbench.em_framework import sample_uncertainties

from ema_workbench import (Model, CategoricalParameter,
                           ScalarOutcome, IntegerParameter, RealParameter)

from ema_workbench.em_framework.optimization import EpsilonProgress
from ema_workbench import ema_logging, MultiprocessingEvaluator, SequentialEvaluator, Samplers
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import functools

from problem_formulation import get_model_for_problem_formulation

model, _ = get_model_for_problem_formulation(1)

def robustness(direction, threshold, data):
    if direction == SMALLER:
        return np.sum(data<=threshold)/data.shape[0]
    else:
        return np.sum(data>=threshold)/data.shape[0]

SMALLER = 'SMALLER'

def costs(data):
    return data[0]/1e9 # makes numbers nicer

Expected_Number_of_Deaths = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Annual_Damage = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Total_Investment_Costs = costs #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN

n_scenarios = 10
scenarios = sample_uncertainties(model, n_scenarios)
nfe = int(50000)  # Original value: 1000

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE

funcs = {'Expected Number of Deaths':Expected_Number_of_Deaths,
         'Expected Annual Damage': Expected_Annual_Damage,
         'Total Investment Costs': Total_Investment_Costs}

robustnes_functions = [ScalarOutcome('Expected Number of Deaths', kind=MINIMIZE,
                                     function=Expected_Number_of_Deaths),
                       ScalarOutcome('Expected Annual Damage', kind=MINIMIZE,
                                     function=Expected_Annual_Damage),
                       ScalarOutcome('Total Investment Costs', kind=MINIMIZE,
                                     function=Total_Investment_Costs),
                      ]
if __name__ == '__main__':

    use_pickle4 = False
    if use_pickle4:
        with open('data/moro_results5.pickle', 'rb') as filehandler:
            results4 = pickle.load(filehandler)
    else:
        # we have to change the plausible max for total investment costs
        convergence = [EpsilonProgress()]

        epsilons=[0.05,]*len(robustnes_functions)  #final value of epsilon should be much lower.Just for experiment purposes is 1
        with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
            results4 = evaluator.robust_optimize(robustnes_functions, scenarios, nfe=nfe,
                                                            convergence=convergence, epsilons=epsilons)
        # Save results in Pickle file
        with open("data/moro_results5.pickle","wb") as filehandler:
            pickle.dump(results4, filehandler)
