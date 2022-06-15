
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
np.random.seed(0)

model, _ = get_model_for_problem_formulation(3)

def robustness(direction, threshold, data):
    if direction == SMALLER:
        return np.sum(data<=threshold)/data.shape[0]
    else:
        return np.sum(data>=threshold)/data.shape[0]

SMALLER = 'SMALLER'

# def costs(data):
#     return data[0]/1e9 # makes numbers nicer

Expected_Number_of_Deaths1 = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Costs1 = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAINExpected_Number_of_Deaths1 = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Costs2 = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Expected_Number_of_Deaths2 = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Costs3 = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Expected_Number_of_Deaths3 = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Costs4 = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Expected_Number_of_Deaths4 = functools.partial(robustness, SMALLER, 0.001) #not ok
Expected_Costs5 = functools.partial(robustness, SMALLER, 80e6) #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN
Expected_Number_of_Deaths5 = functools.partial(robustness, SMALLER, 0.001) #not ok

rfr_costs = functools.partial(robustness, SMALLER, 80e6) #not ok
evac_costs = functools.partial(robustness, SMALLER, 80e6) #not ok



# Total_Investment_Costs = costs #THOSE NUMBERS NEED TO BE SPECIFIED AGAIN


n_scenarios = 10
scenarios = sample_uncertainties(model, n_scenarios)
# with open('data/scenariosselection.pickle', 'rb') as filehandler:
#         scenarios = pickle.load(filehandler)

nfe = int(100)  # Original value: 1000

MAXIMIZE = ScalarOutcome.MAXIMIZE
MINIMIZE = ScalarOutcome.MINIMIZE

funcs = {'A.1 Total Costs':Expected_Costs1,
'A.1_Expected Number of Deaths': Expected_Number_of_Deaths1,
'A.2 Total Costs':Expected_Costs2,
'A.2_Expected Number of Deaths': Expected_Number_of_Deaths2, 
'A.3 Total Costs':Expected_Costs3,
'A.3_Expected Number of Deaths': Expected_Number_of_Deaths3, 
'A.4 Total Costs':Expected_Costs4,
'A.4_Expected Number of Deaths':Expected_Number_of_Deaths4, 
'A.5 Total Costs':Expected_Costs5,
'A.5_Expected Number of Deaths':Expected_Number_of_Deaths5, 
'RfR Total Costs':rfr_costs,
'Expected Evacuation Costs':evac_costs}

robustnes_functions = [ScalarOutcome('A.1 Total Costs', kind=MAXIMIZE, function=Expected_Costs1),
                        ScalarOutcome('A.2 Total Costs', kind=MAXIMIZE, function=Expected_Costs2),
                        ScalarOutcome('A.3 Total Costs', kind=MAXIMIZE, function=Expected_Costs3),
                        ScalarOutcome('A.4 Total Costs', kind=MAXIMIZE, function=Expected_Costs4),
                        ScalarOutcome('A.5 Total Costs', kind=MAXIMIZE, function=Expected_Costs5),
                        ScalarOutcome('A.1_Expected Number of Deaths', kind=MAXIMIZE, function=Expected_Number_of_Deaths1),
                        ScalarOutcome('A.2_Expected Number of Deaths', kind=MAXIMIZE, function=Expected_Number_of_Deaths2),
                        ScalarOutcome('A.3_Expected Number of Deaths', kind=MAXIMIZE, function=Expected_Number_of_Deaths3),
                        ScalarOutcome('A.4_Expected Number of Deaths', kind=MAXIMIZE, function=Expected_Number_of_Deaths4),
                        ScalarOutcome('A.5_Expected Number of Deaths', kind=MAXIMIZE, function=Expected_Number_of_Deaths5),
                       ScalarOutcome('RfR Total Costs', kind=MAXIMIZE, function=rfr_costs),
                       ScalarOutcome('Expected Evacuation Costs', kind=MAXIMIZE,
                                     function=evac_costs),
                      ]

if __name__ == '__main__':
    np.random.seed(0)
    use_pickle4 = False
    if use_pickle4:
        with open('data/moro_results123.pickle', 'rb') as filehandler:
            results4 = pickle.load(filehandler)
    else:
        # we have to change the plausible max for total investment costs
        convergence = [EpsilonProgress()]

        epsilons=[0.1,]*len(robustnes_functions)  #final value of epsilon should be much lower.Just for experiment purposes is 1
        with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
            results4 = evaluator.robust_optimize(robustnes_functions, scenarios, nfe=nfe,
                                                            convergence=convergence, epsilons=epsilons)
        # Save results in Pickle file
        with open("data/moro_results123.pickle","wb") as filehandler:
            pickle.dump(results4, filehandler)
