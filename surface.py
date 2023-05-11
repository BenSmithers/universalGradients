import numpy as np

class SurfaceSample:
    def __init__(self, degree, *params):
        """
            Order is the degree of the polynomials 
        """
        param_per_surface = int((degree+1)*(degree+2)/2)
        n_surfaces = len(params)/param_per_surface
        assert abs(n_surfaces - int(n_surfaces))<1e-15, "Non-integer number of surface. Gave {} params, but need {} params per surface".format(len(params), param_per_surface)

        _params = np.reshape(params, (int(n_surfaces), param_per_surface))
        self._surfaces = [Surface(degree, *sub_params) for sub_params in _params]

    @property
    def surfaces(self):
        return self._surfaces
    

    def __iter__(self):
        return self._surfaces.__iter__()

    def eval(self, points):
        return np.array([sf.eval(points) for sf in self._surfaces])

class Surface:
    def __init__(self,degree=3, *params):
        """
            Creates a surface object 
            for each 
        """

        self._params = params

        self._dim = degree+1
        assert len(params) == int((degree+1)*(degree+2)/2), "Got {} params, expected {}".format(len(params), int((degree+1)*(degree+2)/2) )
        self._paramset = np.zeros(shape=(self._dim , self._dim ))
        self._paramset[np.tril_indices(self._dim , )] = params
        self._xpowers = np.array(range(self._dim ))
        self._ypowers = self._xpowers[::-1]

    @property
    def params(self):
        return self._params

    def eval(self, points):
        """
            Takes a numpy array of points we want to evaluate, do some rotations, and return a numpy array of the Z-coordinates 
        """

        x, y = points[:, 0], points[:, 1]
        powers = np.arange(self._dim).reshape(-1,1).T
        X = np.tile(x, (self._dim, 1)).T
        Y = np.tile(y, (self._dim, 1)).T
        Z = np.power(X, powers) * np.power(Y, powers[::-1])
        return np.sum(np.dot(Z, self._paramset), axis=1)


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

        # sanity check
        flat = self._params.flatten()
        assert [abs(flat[i] - params[i])<1e-15 for i in range(len(params))]

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
        out = [10.0*norm[ip]*np.polyval(self._params[ip], np.log10(x)) for ip in range(self._n_params)]

        return out
    
    def __str__(self):
        outstr = ""
        for params in self._params:
            outstr+="{}x^3 + {}x^2 + {}x + {}\n".format(*params)
        return outstr
    
