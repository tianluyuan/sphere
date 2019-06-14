import numpy as np
from scipy.optimize import brentq

class spa(object):
    def __init__(self, fb8):
        k = fb8.kappa
        b = fb8.beta
        m = fb8.eta

        self.p = len(fb8.nu)

        _ = np.asarray([0, -b, b*m])
        self._ls = _[np.argsort(_)]
        self._gs = (k*fb8.nu)[np.argsort(_)]
        self._t_hat = self.solve_t()


    def K0(self, t):
        assert t < self._ls[0]
        return np.sum(-1/2. * np.log(1-t/self._ls) + 1/4. * self._gs**2/(self._ls-t) - self._gs**2/(4*self._ls))


    def K1(self, t):
        return self.Kj(t,1)

    
    def Kj(self, t, j):
        return np.sum(np.math.factorial(j-1)/2. *1/(self._ls - t)**j+
                      np.math.factorial(j)/4. *self._gs**2/(self._ls-t)**(j+1))

    
    def Kj_hat(self, j):
        return self.Kj(self._t_hat, j)


    def rhoj_hat(self, j):
        return self.Kj_hat(j)/(self.Kj_hat(2)**(j/2.))


    def T(self):
        return 1/8.*self.rhoj_hat(4) - 5/24.*self.rhoj_hat(3)**2


    def solve_t(self):
        p = self.p
        lb = self._ls[0] - p/4. - 1/2.*np.sqrt(p**2/4.+p*np.max(self._gs**2))
        ub = self._ls[0] - 1/4. - 1/2.*np.sqrt(1/4.+self._gs[0]**2)
        return brentq(lambda t: self.K1(t)-1, lb, ub)


    def log_c1(self):
        p = self.p
        return (np.log(np.sqrt(2)*np.pi**((p-1)/2.))-1/2.*np.log(self.Kj_hat(2))-
                    1/2.*np.sum(np.log(self._ls-self._t_hat))-self._t_hat+
                    1/4.*np.sum(self._gs**2/(self._ls**2-self._t_hat)))

    def log_c2(self):
        return self.log_c1()+np.log(1+self.T())

    
    def log_c3(self):
        return self.log_c1()+self.T()
