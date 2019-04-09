import numpy as np
import scipy
import scipy.special
import scipy.optimize

if __package__ is None:
    from fb_utils import tile_repeat
else:
    from sphere.distribution.fb_utils import tile_repeat

# Implement the saddle point approximation for FB8
def saddleapprox_FB_revised(L, M=None, dub=3, order=3):
    L = np.tile(L, dub)

    if M is None:
        M = np.zeros(L.shape)

    a = 1.0 / np.sqrt(np.prod(L / np.pi))
    Y = 0

    def KM(t):
        y = np.sum(-0.5*np.log(1.0-t/L) + M**2.0 / (L-t))
        return y

    def KM1(t):
        y = np.sum(0.5 / (L-t) + (M**2.0/(L-t)**2.0))
        return y

    def KM2(t):
        y = np.sum(0.5 / (L-t)**2.0 + 2.0 * M**2.0 / (L-t)**3.0)
        return y

    def KM3(t):
        y = np.sum(1.0 / (L-t)**3.0 + 6.0 * M**2.0 / (L-t)**4.0)
        return y

    def KM4(t):
        y = np.sum(3.0 / (L-t)**4.0 + 24.0 * M**2.0 / (L-t)**5.0)
        return y

    def sol(y):
        f = lambda t: abs(KM1(t) - 1.0)
        fmin = np.amin(L)-len(L)/(4.0*y) - np.sqrt(len(L)**2.0 / 4.0 + len(L)*np.amax(L)**2.0*np.amax(M)**2.0)
        fmax = np.amin(L)
        res = scipy.optimize.minimize_scalar(f, method='brent', bounds=(fmin, fmax))
        return res.x

    that = sol(1.0)

    Y = 2.0*a/np.sqrt(2.0*np.pi*KM2(that)) * np.exp(KM(that) - that)
    if order == 3:
        rho3sq = KM3(that)**2.0 / KM2(that)**3.0

        rho4 = KM4(that)/KM2(that)**(2.0)

        Rhat = 3.0/24.0 * rho4 - 5.0/24.0 * rho3sq

        Y = Y*np.exp(Rhat)
    elif order == 2:
        rho3sq = KM3(that)**2.0 / KM2(that)**3.0

        rho4 = KM4(that)/KM2(that)**(2.0)

        Rhat = 3.0/24.0 * rho4 - 5.0/24.0 * rho3sq

        Y = Y*(1.0 + Rhat)
    return Y

def SPA(alpha, ns=None, withvol=True):
    p = int(len(alpha) / 2)

    if ns is None:
        ns = np.ones(p).astype(int)

    theta = tile_repeat(alpha[:p], ns)

    mu = tile_repeat(alpha[p:] / np.sqrt(ns), ns) / 2.0

    dsum = np.sum(ns)

    v0 = 2.0 * np.pi**(dsum/2.0) / scipy.special.gamma(dsum/2.0)

    coef = 1.0 if withvol else 1.0/v0

    return saddleapprox_FB_revised(theta + 1.0, mu, dub=1, order=3) * np.exp(1.0) / coef

def test():
    alpha1 = np.array([1,2,3,4,7,6,8,5])
    ns1 = np.array([2,2,1,3])
    print 'alpha=' + str(alpha1)
    print 'ns=', ns1
    G4 = SPA(alpha1, ns1)
    print 'SPA:', G4

    print

    print 'Near singular case:'
    alpha2 = np.array([0.067, 0.294, 0.311, 0.405, 0.663, 0.664, 0.667, 0.712, 0.784, 0.933, 0.070, 0.321, 0.288, 0.367, 0.051, 0.338, 0.798, 0.968, 0.506, 0.590])
    ns2 = np.ones(10).astype(int)
    print 'alpha=', alpha2
    print 'ns=', ns2
    G4 = SPA(alpha2, ns2)
    print 'SPA:', G4

if __name__ == '__main__':
    test()
