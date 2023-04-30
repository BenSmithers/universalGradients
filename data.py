import h5py as h5 
import os 
import numpy as np 

from scipy.interpolate import interp1d

class DataGrads:
    def __init__(self):
        """
            Loads in the analysis gradients and the points where those gradients are defined 

            This has the gradients for the zeniths _and_ energies, but only the energies are used right now 
        """
        _data = h5.File(os.path.join(os.path.dirname(__file__), "cascade_ice_gradients.h5"))
        self._e_bins = np.log10(0.5*(_data["e_bins"][:1] + _data["e_bins"][:-1]))
        self._z_bins = 0.5*(_data["c_bins"][:1] + _data["c_bins"][:-1])

        self._node_values= {}

        self._grads = {}
        for key in _data.keys():
            if "amp" in key and "zenith" not in key:
                self._node_values[key.split("_")[0]] = 1.+np.array(_data[key][:])
                self._grads[key.split("_")[0]] = interp1d(self._e_bins,  1.+np.array(_data[key][:]))

    def keys(self):
        return self._grads.keys()
    
    def node(self, key):
        return self._node_values[key]

    @property
    def e_nodes(self):
        return self._e_bins
    
    def eval(self,key:str, values: np.ndarray):

        return self._grads[key](values)
    