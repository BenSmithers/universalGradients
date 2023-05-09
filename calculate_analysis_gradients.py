"""
    Calculates the analysis level gradients
    Don't need to worry about the widths; those are already baked into the saved gradients from the MC! 
"""

import json
import os 
import h5py as h5
from math import sqrt, pi
import numpy as np

def build_gradients():
    from utils import correlation, modes      

    grad_file = os.path.join(os.path.dirname(__file__), "data_folder", "cascade_ice_gradients.h5")
    grad_data = h5.File(grad_file, 'r')

    grad_vectors = np.zeros( (9, len(grad_data["e_bins"][:])-1) )
    zengrad_vectors = np.zeros( (9, len(grad_data["c_bins"][:])-1) )

    e_bins= grad_data["e_bins"][:]
    z_bins= grad_data["c_bins"][:]

    analysis_dim = -1
    grad_modes = ["amp", "phase"]
    for param in grad_modes:
        for mode in modes:
            if param=="phase" and mode==0:
                print("skipping")
                continue
            index = mode
            if param=="phase":
                index+=4

            key = "{}0{}_energy".format(param, mode)
            zkey = "{}0{}_zenith".format(param, mode)
            grad_vectors[index] = np.array(grad_data[key][:])
            zengrad_vectors[index] =  np.array(grad_data[zkey][:])
            if analysis_dim==-1:
                analysis_dim = len(grad_data[key][:])
            elif analysis_dim!=len(grad_data[key][:]):
                raise Exception("{} - {}".format(analysis_dim, len(grad_data[key][:])))
            
    grad_vectors =grad_vectors*sqrt(pi/2) 

    analysis_grad = np.zeros((analysis_dim,analysis_dim))
    for alpha in range(analysis_dim):
        for beta in range(analysis_dim):
            for i in range(9):
                for j in range(9):
                    analysis_grad[alpha][beta] = grad_vectors[i][alpha]*correlation[i][j]*grad_vectors[j][beta]

    analysis_grad = np.linalg.multi_dot([np.transpose(grad_vectors), correlation, grad_vectors])
    zengrad = np.linalg.multi_dot([np.transpose(zengrad_vectors), correlation, zengrad_vectors])


    out_dict = {
        "energy":{
            "cov":analysis_grad,
            "bins":e_bins
        },
        "zenith":{
            "cov":zengrad,
            "bins":z_bins
        }
    }

    return out_dict

if __name__=="__main__":
    out_dict = build_gradients()
    analysis_grad = out_dict["energy"]["cov"]
    zengrad = out_dict["zenith"]["cov"]
    e_bins = out_dict["energy"]["bins"]
    z_bins = out_dict["zenith"]["bins"]

    import matplotlib.pyplot as plt 
    plt.pcolormesh(e_bins, e_bins, analysis_grad, vmin=-0.005, vmax=0.005,cmap='RdBu')
    plt.colorbar()
    plt.xlim([500, 10000])
    plt.ylim([500, 10000])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Energy Bin No", size=14)
    plt.ylabel("Energy Bin No", size=14)
    plt.show()
    plt.clf()

    plt.pcolormesh(z_bins, z_bins, zengrad, vmin=-0.01, vmax=0.01,cmap='RdBu')
    plt.colorbar()
    plt.xlim([-1, 0])
    plt.ylim([-1, 0])
    plt.xlabel("Zenith Bin No", size=14)
    plt.ylabel("Zenith Bin No", size=14)
    plt.show()