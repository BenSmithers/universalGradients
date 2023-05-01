import json     
import os 
import numpy as np
import matplotlib.pyplot as plt 
from math import log10

from surface import GradientSet
from fitgrads import STATE_FILE

def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

assert os.path.exists(STATE_FILE), "Need state file to make plots"

_obj = open(STATE_FILE, 'rt')
data = json.load(_obj)
_obj.close()

gradset = GradientSet.from_state(data)

energies = np.linspace(log10(500), log10(20000), 100)
norms = np.ones(gradset.n_params)
grad_eval = gradset(energies,[norms,])

i = 0
for ig, grad in enumerate(grad_eval):
    plt.plot(energies, grad, label="Eff Grad {}".format(i+1), color=get_color(i+1, len(grad_eval)))
plt.xlabel("Energy [GeV]", size=14)
plt.ylabel("Eff Gradient")
plt.legend()
plt.show()