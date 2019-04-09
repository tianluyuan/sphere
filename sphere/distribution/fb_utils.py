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
    p = len(theta)
    gamma = kappa*nu

    theta, inv_theta, ns = np.unique(theta, return_inverse=True, return_counts=True)
    l = len(theta)

    gamma = np.array([np.sqrt(np.sum(gamma[inv_theta == i]**2.0)) for i in range(l)])

    alpha = np.concatenate([theta, gamma])
    return alpha, ns

