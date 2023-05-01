from typing import Any
import numpy as np


class GradientSet:
    """
        a set of polynomial effective gradients of arbitrary dimensionality 
    """
    def __init__(self,order, *params):
        assert len(params)%(order+1) == 0, "Can not construct Gradients of order {} with {} params".format(order, len(params))

        self._params = np.reshape( params, newshape=(int(len(params)/(order+1) ), order+1))
        self._default_norm = np.ones(int(len(params)/(order+1)))

        self._n_params = int(len(params)/(order+1))
        self._order = order

    @property
    def state(self):
        return {"order":self._order,
                "params":self._params.flatten().tolist()}
    
    @classmethod
    def from_state(cls, state_dict)->'GradientSet':
        return cls(state_dict["order"],
                   *state_dict["params"])


    @property
    def n_params(self):
        return self._n_params

    def __call__(self, x:np.ndarray, _norm):
        """
            Evaluate all the polynomials and scale them by the normalization we're given 
        """
        norm = _norm[0]
        out = [norm[ip]*np.polyval(self._params[ip], x) for ip in range(self._n_params)]
        return out
    
    def __str__(self):
        outstr = ""
        for params in self._params:
            outstr+="+ {}x^3 + {}x^2 + {}x + {}\n".format(*params)
        return outstr
    
