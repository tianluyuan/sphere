import numpy as np
from scipy.special import erfc


class enct(object):
    def __init__(self, fb8, wd=0.5, wu=3., N=200):
        """
        Perform efficient numerical integration based on arxiv:2004.14660 Eq. 8
        """
        k = fb8.kappa
        b = fb8.beta
        m = fb8.eta

        self.p = len(fb8.nu)

        _ = np.asarray([0, -b, b*m])
        self._c = np.abs(np.min(_))+1
        self._ts = _[np.argsort(_)]+self._c
        assert np.all(self._ts) > 0

        self._gs = (k*fb8.nu)[np.argsort(_)]

        d = np.min(self._ts)/2.
        assert N >= 2*d*(wu+wd)*wu**2/(np.pi*wd**2)

        self._h = np.sqrt(2*np.pi*d*(wd+wu)/(wd**2*N))
        self._p = np.sqrt(N*self._h/wd)
        self._q = np.sqrt(wd*N*self._h/4.)
        self._N = N


    def A(self, t):
        return np.product(np.exp(self._gs[:,None]**2/(4*(self._ts[:,None]-1j*t)))/
                          np.sqrt(self._ts[:,None]-1j*t), axis=0)


    def w(self, x):
        return 0.5*erfc(x/self._p-self._q)
                                              

    def c(self):
        _ = np.arange(-self._N-1, self._N+1)*self._h
        return (np.sqrt(np.pi)*np.exp(self._c)*self._h*np.sum(
            self.w(np.abs(_))*self.A(_)*np.exp(-1j*_))).real
