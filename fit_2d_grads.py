
from surface import Surface, SurfaceSample
import numpy as np
import os 
import h5py as h5 
import json 

import matplotlib.pyplot as plt 

from scipy.optimize import minimize
DEGREE = 4 

surface_state = os.path.join(os.path.dirname(__file__),"data_folder", "surface_state.json")


def main():
    
    modes = [0,1,2,3,4]
    params = ["amp", "phase"]

    filename = os.path.join(os.path.dirname(__file__), "data_folder", "cascade_2d_ice_gradients.h5")
    data = h5.File(filename, 'r')

    energy_bin_edges = np.log10(np.array(data["e_bins"][:])) # do this in log-energy 
    cos_bin_edges = np.array(data["c_bins"][:])

    n_sample = 100
    sample_e = np.linspace(min(energy_bin_edges), max(energy_bin_edges), n_sample)
    sample_c = np.linspace(min(cos_bin_edges), max(cos_bin_edges), n_sample+1)
    mesh_sample_c, mesh_sample_e = np.meshgrid(sample_c, sample_e)
    sample_points = np.transpose([mesh_sample_c.flatten(), mesh_sample_e.flatten()])
    

    e_centers = 0.5*(energy_bin_edges[1:] + energy_bin_edges[:-1])
    c_centers = 0.5*(cos_bin_edges[1:] + cos_bin_edges[:-1])

    mesh_c, mesh_e = np.meshgrid(c_centers, e_centers)

    mesh_e = mesh_e.flatten()
    mesh_c = mesh_c.flatten()

    points = np.transpose([mesh_c, mesh_e])

    surfaces = []

    state_dict ={
    }

    def make_plot(surface:Surface, name):
        z_vals = surface.eval(sample_points)
        

        plt.clf()
        plt.pcolormesh(sample_c, np.power(10, sample_e), np.reshape(z_vals, (n_sample , n_sample+1)), vmin=-0.06, vmax=0.06, cmap="RdBu")
        plt.colorbar()
        plt.ylim([500, 10000])
        plt.yscale('log')
        plt.xlim([-1,0])
        plt.xlabel("Cos Theta") 
        plt.ylabel("Log(Energy/GeV)")
        plt.title(name)
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", "grid_2d_{}.png".format(name)), dpi=400)

    for mode in modes:
        for param in params:
            if mode==0 and param=="phase":
                continue

            gkey = "{}0{}_gradient".format(param, mode)
            ukey = "{}0{}_unc".format(param, mode)

            gradients = np.array(data[gkey]).T
            uncertainty = np.array(data[ukey]).T 


#            assert np.shape(gradients) == np.shape(mesh_e)

            gradients = gradients.flatten()
            uncertainty = uncertainty.flatten()

            x0 = np.ones(15)
            bounds = [(-np.inf, np.inf) for i in range(15)]

            def minfunc(values):
                this_surface = Surface(DEGREE, *values)

                #print(np.shape(this_surface.eval(points)))

                llh = np.log10( 1 + ((this_surface.eval(points) - gradients)/uncertainty)**2)
                return np.sum(llh)
            
            min_params = minimize(minfunc, x0, bounds = bounds).x
            #print(min_params)
            sf = Surface(DEGREE, *min_params)
            state_dict["{}0{}".format(param, mode)] = min_params.tolist()
            make_plot(sf, gkey)
            surfaces.append(sf)
            print("Finished "+"{}0{}".format(param, mode))

    _obj = open(surface_state, 'wt')
    json.dump(state_dict, _obj, indent=4)
    _obj.close()


main()