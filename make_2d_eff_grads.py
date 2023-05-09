
import os 
import json
from surface import Surface, SurfaceSample

from scipy.optimize import minimize
import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

from math import exp 
DEGREE = 4 

surface_state = os.path.join(os.path.dirname(__file__),"data_folder", "surface_state.json")

def build_downcast_from_cov():
    pass

def build_tensor():
    from utils import correlation, modes

    _obj = open(surface_state, 'rt')
    data = json.load(_obj)
    _obj.close()

    keyvals = {}
    grad_modes = ["amp", "phase"]
    for param in grad_modes:
        for mode in modes:
            if param=="phase" and mode==0:
                print("skipping")
                continue
            index = mode
            if param=="phase":
                index+=4

            keyvals["{}0{}".format(param, mode)] = index
    
    surfaces = [(key, Surface(DEGREE, *data[key])) for key in data.keys()]
    surfaces = list(sorted(surfaces, key=lambda x:keyvals[x[0]]))
    
    for entry in surfaces:
        print(entry[0])
    surfaces = [ent[1] for ent in surfaces]

    # we're going to get bin values from the 1D gradient file 
    grad_file = os.path.join(os.path.dirname(__file__), "data_folder", "cascade_ice_gradients.h5")
    grad_data = h5.File(grad_file, 'r')
    e_bins= grad_data["e_bins"][:]
    z_bins= grad_data["c_bins"][:]

    e_vals = np.log10(0.5*(e_bins[:-1] + e_bins[1:]))
    c_vals = 0.5*(z_bins[:-1] + z_bins[1:])

    mesh_c, mesh_e = np.meshgrid(c_vals, e_vals)
    points = np.transpose([mesh_c.flatten(), mesh_e.flatten()])

    gvecs = np.array([
        np.reshape(suf.eval(points), (len(e_vals), len(c_vals) )) for suf in surfaces
    ])


    # all_2d_grads is our Gradient Vector
    # we now do a tensor sum (AAAAAHHHH) to fill the 4D elements of 
    # gvecs.T dot correlation dot gvecs

    print(np.shape(gvecs))
    print(np.shape(e_vals))

    full_tensor = np.einsum('iab,ij,jdg->abdg', gvecs , correlation, gvecs)    
    diagonal_tensor = np.sqrt(np.einsum('ijij->ij', full_tensor))


    plt.pcolormesh(z_bins, e_bins, diagonal_tensor, vmin=0, vmax=0.20, cmap="magma")
    cbar = plt.colorbar()
    cbar.set_label(r"Down-cast Gradient, $\sqrt{\Sigma_{ijij}}$")
    plt.yscale('log')
    plt.xlim([-1,0])
    plt.ylim([500, 10000])
    plt.ylabel("Energy [GeV]",size=14)
    plt.xlabel("cos theta")
    plt.savefig(os.path.join(os.path.dirname(__file__),"plots", "downcast_2dgrad_diag.svg"))
    plt.clf()
    return full_tensor

def main():
    full_tensor = build_tensor()

    grad_file = os.path.join(os.path.dirname(__file__), "data_folder", "cascade_ice_gradients.h5")
    grad_data = h5.File(grad_file, 'r')
    e_bins= grad_data["e_bins"][:]
    z_bins= grad_data["c_bins"][:]

    e_vals = np.log10(0.5*(e_bins[:-1] + e_bins[1:]))
    c_vals = 0.5*(z_bins[:-1] + z_bins[1:])

    mesh_c, mesh_e = np.meshgrid(c_vals, e_vals)
    points = np.transpose([mesh_c.flatten(), mesh_e.flatten()])

    alpha_0 = [0.1,]
    x0 = 0.1*np.ones(15)
    x0 = np.concatenate((alpha_0, x0, x0))

    def min_func(params):
        cor = np.array([[1, params[0]], [params[0], 1]])
        surfaces = SurfaceSample(DEGREE, *params[1:])
        gvecs = surfaces.eval(points)
        gvecs= np.array([np.reshape(gvec, (len(e_vals), len(c_vals))) for gvec in gvecs])
        net = np.einsum('iab,ij,jdg->abdg', gvecs , cor, gvecs)    
        return np.sum(np.abs(net - full_tensor))*exp(100*params[0]**2)
    
    
    result = minimize(min_func, x0).x
    

    out_dict = {
        "eff_grads":result.tolist()
    }
    import json
    _obj = open(os.path.join(os.path.dirname(__file__), "data_folder", "effective_tensor_grads.json"), 'wt')
    json.dump(out_dict, _obj, indent=4)
    _obj.close()
        
    n_sample = 100
    e_bins = np.log10(e_bins)
    sample_e = np.linspace(min(e_bins), max(e_bins), n_sample)
    sample_c = np.linspace(min(z_bins), max(z_bins), n_sample+1)
    mesh_sample_c, mesh_sample_e = np.meshgrid(sample_c, sample_e)
    sample_points = np.transpose([mesh_sample_c.flatten(), mesh_sample_e.flatten()])
    

        
    cor = np.array([[1, result[0]], [result[0], 1]])
    surfaces = SurfaceSample(DEGREE, *result[1:])

    print("Cor: {}".format(cor))

    z_vals = surfaces.eval(sample_points)
 
    
    ax = plt.axes(projection="3d")

    #ax.plot3D(mesh_sample_c.flatten(), mesh_sample_e.flatten(), z_vals[0], label="Eff Grad 0")
    #ax.plot3D(mesh_sample_c.flatten(), mesh_sample_e.flatten(), z_vals[1], label="Eff Grad 1")

    ax.plot_surface(mesh_sample_c, mesh_sample_e, np.reshape(z_vals[0], (n_sample , n_sample+1)), label="Eff Grad 0")
    ax.plot_surface(mesh_sample_c, mesh_sample_e, np.reshape(z_vals[1], (n_sample , n_sample+1)), label="Eff Grad 1")


    from math import log10
    #ax.set_xlim([-1,0.0])
    #ax.set_ylim([log10(500),log10(10000)])

    ax.set_xlabel("Cos Theta",size=14)
    ax.set_ylabel("Log Energy",size=14)
    #ax.set_yscale('log')
    ax.set_zlabel("Gradient",size=14)
    #ax.legend()
    plt.show()

if __name__ == "__main__":
    main()


