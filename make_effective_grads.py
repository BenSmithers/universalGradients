"""
Here we do the fits to the effective gradients for the ice things

it's set up for two gradients, but down the line we can expand it to more... though that'll get complicaded since this will need a more complicated correlation function
though that new correlation function can just use the inverse cholesky decomposition approach I used in the direct fitter in the other repo
"""

from calculate_analysis_gradients import build_gradients

import numpy as np
from math import exp
from scipy.optimize import minimize
import os 


def plot_results(binning, cov, params):
    import matplotlib.pyplot as plt 

    out_folder = os.path.join(os.path.dirname(__file__),"plots")

    b_center = 0.5*(binning[1:] + binning[:-1])

    use_log= False
    if max(b_center)>1:
        use_log = True

    diag_real = np.sqrt(np.diag(cov))
    n_bin = len(cov)
    sigma_g = np.array([[1, params[0]], [params[0], 1]])
    print(sigma_g)
    gvecs = np.reshape(params[1:], (2, n_bin))
    net = np.linalg.multi_dot((np.transpose(gvecs), sigma_g, gvecs))

    plt.stairs(diag_real,binning, label="Original")
    plt.stairs( np.sqrt(np.diag(net)),binning, label="2 Effective Gradients")
    if use_log:
        plt.xlim([500, 10000])
        plt.xscale('log')
    else:
        plt.xscale('linear')
        plt.xlim([-1,0])

    plt.ylabel("Diagonal Uncertainty", size=14)
    plt.legend()
    plt.ylim([0,0.08])
    plt.xlabel("Energy [GeV]" if use_log else "Cos Theta", size=14)
    plt.savefig(out_folder + "/{}_diagnal_unc.png".format("energy" if use_log else "zenith"), dpi=400)
    plt.clf()

    plt.xscale('linear')
    for iv, vec in enumerate(gvecs):
        plt.stairs(vec,binning, label="Eff Grad {}".format(iv))

    if use_log:
        plt.xlim([500, 10000])
        plt.xscale('log')
    else:
        plt.xscale('linear')
        plt.xlim([-1,0])

    plt.ylabel("Gradient", size=14)
    plt.legend()
    plt.xlabel("Energy [GeV]" if use_log else "Cos Theta", size=14)
    plt.savefig(out_folder + "/{}_grads.png".format("energy" if use_log else "zenith"), dpi=400)
    plt.ylim([0,0.08])
    plt.clf()    

def main():
    out_dict = build_gradients()

    axes = ["energy", "zenith"]
    for axis in axes:
        cov = out_dict[axis]["cov"]
        binning = out_dict[axis]["bins"]

        print("Diagonal gradient: {}".format(np.sqrt(np.diag(cov))))

        alpha_0 = [0.1,]
        n_bin = len(binning)-1
        x0 = 0.1*np.ones(n_bin)
        x0 = np.concatenate((alpha_0, x0, x0))

        #bounds = [(0,1)] + [(-np.inf, np.inf) for i in range(len(x0)-1)]

        def min_func(params):
            sigma_g = np.array([[1, params[0]], [params[0], 1]])

            gvecs = np.reshape(params[1:], (2, n_bin))
            net = np.linalg.multi_dot((np.transpose(gvecs), sigma_g, gvecs))

            return np.sum(np.abs(net - cov))*exp(100*params[0]**2)
        

        result = minimize(min_func, x0).x
        plot_results(binning, cov, result)

if __name__ == "__main__":
    main()
    """
                CovMatrixToReturn[i][j]=(FirstGradient[i]*FirstGradient[j]+ SecondGradient[i]*SecondGradient[j]+ alpha*(FirstGradient[i]*SecondGradient[j]+FirstGradient[j]*SecondGradient[i]))

    """