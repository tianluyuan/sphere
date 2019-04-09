import numpy as np

def tile_repeat(x, r):
    r = np.asarray(r)
    if r.size == 1:
        return np.tile(x, r.dtype.type(r))
    elif r.size == len(x):
        return np.repeat(x, r)

def fb8_to_ks_params(kappa, nu, beta, eta):
    nu = np.asarray(nu)
    theta = np.array([0.0, -beta, beta*eta])
    gamma = kappa*nu

    alpha = np.concatenate([theta, gamma])
    return alpha

