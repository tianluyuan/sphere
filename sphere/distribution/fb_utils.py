import numpy as np

def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

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
    return alpha, ns, inv_theta

def reduce_dim(alpha):
    p = len(alpha) / 2
    theta = alpha[:p]
    gamma = alpha[p:]

    reduced_theta, inv_theta, ns = np.unique(theta, return_inverse=True, return_counts=True)
    l = len(reduced_theta)

    reduced_gamma = np.array([np.sqrt(np.sum(gamma[inv_theta == i]**2.0)) for i in range(l)])

    return np.concatenate([reduced_theta, reduced_gamma]), ns, inv_theta

def increase_dim(alpha, inv_theta=None):
    l = len(alpha) / 2
    ns = [np.sum(inv_theta == i) for i in range(l)]
    if inv_theta is None:
        pass
    p = np.sum(ns)

    increased_theta = np.zeros(p)
    increased_gamma = np.zeros(p)

    down_map = np.zeros(l).astype(int)

    for i in range(l):
        down_map[i] = np.arange(p)[(inv_theta == i)][0]
        increased_theta[inv_theta == i] = alpha[i]
        increased_gamma[inv_theta == i] = alpha[i+l] / np.sqrt(ns[i])

    return np.concatenate([increased_theta, increased_gamma]), down_map

def increase_grad_dim(grad, alpha, original_alpha, inv_theta):
    l = len(alpha) / 2
    ns = [np.sum(inv_theta == i) for i in range(l)]
    if inv_theta is None:
        pass
    p = np.sum(ns)

    increased_theta = np.zeros(p)
    increased_gamma = np.zeros(p)
    increased_dtheta = np.zeros(p)
    increased_dgamma = np.zeros(p)

    for i in range(l):
        increased_theta[inv_theta == i] = alpha[i]
        increased_gamma[inv_theta == i] = alpha[i+l] / np.sqrt(ns[i])
        increased_dtheta[inv_theta == i] = grad[i]
        increased_dgamma[inv_theta == i] = grad[i] * original_alpha[p:][inv_theta == i] * alpha[i+l]

    return np.concatenate([increased_dtheta, increased_dgamma])

