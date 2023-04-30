"""
This script wil fit the gradients
"""
from data import DataGrads
from surface import GradientSet

from scipy.optimize import minimize
import numpy as np 


data = DataGrads()

# order for the polynomials used by the effective gradients 
GRADIENT_ORDER = 3


def fit_norms(params):
    """
        For an (effecyive) GradientSet defined by the given parameters,
        runs fits to try fitting effective gradients to the original data gradients 

        We quantify how well a linear combination of these effective gradients will match the analysis gradients 
    """

    eff_grads = GradientSet(GRADIENT_ORDER, *params)

    x0 = np.ones(eff_grads.n_params)
    bounds = [[0, 5] for i in range(eff_grads.n_params)]

    llhs = 0
    for key in data.keys():

        def min_func(*norms):
            prior_penalty = np.sum([(norm-1)**2 for norm in norms])
            eval_grad = eff_grads(data.e_nodes, norms)


            out =  np.sum((data.node(key) - np.sum(eval_grad, axis=0))**2)
            return prior_penalty +out
        
        result = minimize(min_func, x0,bounds=bounds).x 
        llhs+= min_func(result)
    return llhs


def fit_gradients(n_grads):
    """
        This function finds `n_grads` effective gradients to match the true analysis gradients! 
    """
    n_params = (GRADIENT_ORDER+1)*n_grads
    start_values = np.ones(n_params)

    bounds = [[-100,100] for i in range(n_params)]

    res = minimize(fit_norms, 
            x0=start_values,
            bounds=bounds)
    
    return res.x

res = fit_gradients(3)

eff_gradients = GradientSet(GRADIENT_ORDER, *res)
print(eff_gradients)
