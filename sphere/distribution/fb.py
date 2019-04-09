import scipy
import scipy.integrate
import scipy.optimize
import scipy.special
import numpy as np

def tile_repeat(x, r):
    r = np.asarray(r)
    if r.size == 1:
        return np.tile(x, r.dtype.type(r))
    elif r.size == len(x):
        return np.repeat(x, r)

# Pfaffian for Fisher-Bingham distribution
def dG_fun_FB(alpha, G, ns=None, s=1):
    d = len(alpha)
    p = d / 2
    if ns is None:
        ns = np.ones(p)
    r = len(G) # should be equal to d+1
    assert(r == d+1)

    th = alpha[:p]
    xi = alpha[p:p*2]
    gam = xi**2.0 / 4.0
    Gth = G[1:p+1]
    Gxi = G[p+1:p*2+1]
    dG = np.zeros((d,r))

    dG[:,0] = G[1:d+1]

    # derivative of dC/dth[i] with respect to th[j]
    for i in range(p):
        dG[i, i+1] = -s*Gth[i]
        for j in range(p):
            if j != i:
                a1 = -(ns[j]/2.0 / (th[j]-th[i]) + xi[j]**2.0 / 4.0 / (th[j]-th[i])**2.0)
                a2 = -(ns[i]/2.0 / (th[i]-th[j]) + xi[i]**2.0 / 4.0 / (th[i]-th[j])**2.0)
                a3 = -(ns[j] * xi[i] / 4.0 / (th[j]-th[i])**2.0 + xi[i]*xi[j]**2.0 / 4 / (th[j]-th[i])**3.0)
                a4 = -(ns[i]*xi[j] / 4.0 / (th[i]-th[j])**2.0 + xi[i]**2.0 * xi[j] / 4.0 / (th[i]-th[j])**3.0)
                dG[j,i+1] = a1*Gth[i] + a2*Gth[j] + a3*Gxi[i] + a4*Gxi[j]
                dG[i,i+1] = dG[i,i+1] - dG[j,i+1]

    # derivative of dC/dxi[i] with respect to th[j]
    # derivative of dC/dxi[j] with respect to xi[i]
    for i in range(p):
        dG[i,p+i+1] = -s*Gxi[i]
        for j in range(p):
            if j != i:
                b2 = xi[i] / 2.0 / (th[i]-th[j])
                b3 = -(ns[j] / 2.0 / (th[j]-th[i]) + xi[j]**2.0 / 4.0 / (th[j]-th[i])**2.0)
                b4 = xi[i]*xi[j] / 4.0 / (th[i]-th[j])**2.0
                dG[j,p+i+1] = b2*Gth[j] + b3*Gxi[i] + b4*Gxi[j]
                dG[p+i,j+1] = dG[j,p+i+1]
                dG[i,p+i+1] = dG[i,p+i+1] - dG[j,p+i+1]
        dG[p+i,i+1] = dG[i,p+i+1]

    # derivative of dC/dxi[i] with respect to xi[j]
    for i in range(p):
        for j in range(p):
            if j != i:
                c3 = xi[j] / 2.0 / (th[j]-th[i])
                c4 = -xi[i] / 2.0 / (th[j]-th[i])
                dG[p+j,p+i+1] = c3*Gxi[i] + c4*Gxi[j]
        if ns[i] == 1 or abs(xi[i]) < 1e-10:
            dG[p+i,p+i+1] = -Gth[i] # cheat
        else:
            dG[p+i,p+i+1] = -Gth[i] - (ns[i]-1) / xi[i] * Gxi[i] # singular if xi[i] == 0
    return dG

# Initial vlaue for Pfaffian system (by power series expansion)
# parameterization: exp(th*x**2 + xi*x)
def C_FB_power(alpha, v=None, d=None, Ctol=1e-6, alphatol=1e-10, Nmax=None):
    p = len(alpha) / 2
    if v is None:
        v = np.zeros(len(alpha))
    if d is None:
        d = np.ones(p)
    if Nmax is None:
        Nmax = int(np.ceil(10.0**(10.0/len(alpha))))

    alpha_sum = np.sum(abs(alpha[:p*2]))
    N0 = max(int(np.ceil(alpha_sum)), 1)
    if N0 > Nmax:
        raise ValueError("alpha is too large!")
    for N in range(N0, Nmax):
        logep = N*np.log(alpha_sum) - scipy.special.loggamma(N+1) + np.log((N+1) / (N+1.0-alpha_sum))
        if logep < np.log(Ctol):
            break
    if N > Nmax:
        raise ValueError("alpha is too large!")

    def f(k):
        k1 = k[:p]
        k2 = k[p:p*2]
        v1 = v[:p]
        v2 = v[p:p*2]
        if np.any(((k2 + v2) % 2) == 1):
            return 0.0
        w = k > 0
        a = np.prod(alpha[w] ** k[w])
        b1 = np.sum(-scipy.special.loggamma(k1+1) - scipy.special.loggamma(k2+1) + scipy.special.loggamma(k1 + v1 + (k2 + v2)/2.0 + (d/2.0)) + scipy.special.loggamma((k2+v2)/2.0 + 0.5) - scipy.special.loggamma((k2+v2)/2.0 + d/2.0))
        b2 = -scipy.special.loggamma(np.sum(k1+v1+(k2+v2)/2.0 + (d/2.0)))
        b3 = scipy.special.loggamma(np.sum(d)/2.0) - p*scipy.special.loggamma(0.5)
        return a * np.exp(b1+b2+b3)

    stack = [(np.array([]), None, None, None)]
    res = None
    tot = 0.0
    while len(stack) > 0:
        k, branch, max_index, index = stack.pop()
        kn = len(k)
        if branch is None:
            if kn == p*2:
                res = f(k)
                continue
            else:
                knxt = kn
                if abs(alpha[knxt]) < alphatol:
                    stack.append((np.concatenate((k, [0])), None, None, None))
                    continue
                imax = N - np.sum(k)
                stack.append((k, 2, imax+1, 0))
                continue
        else:
            if branch == 0:
                raise ValueError("There shouldn't be anything that reaches here")
            elif branch == 1:
                raise ValueError("There shouldn't be anything that reaches here")
                pass
            elif branch == 2:
                if index < max_index:
                    stack.append((k, 2, max_index, index+1))
                    if index > 0:
                        tot += res
                    stack.append((np.concatenate((k, [index])), None, None, None))
                    continue
                else:
                    tot += res
                    res = tot
                    tot = 0.0
                    continue
            else:
                raise ValueError("There shouldn't be anything that reaches here")
    return res

def G_FB_power(alpha, ns=None, Ctol=1e-6, alphatol=1e-10):
    # not restricted to Bingham
    p = len(alpha) / 2
    th = alpha[:p]
    xi = alpha[p:p*2]

    alpha_prime = np.concatenate((-th, xi))

    C = C_FB_power(alpha_prime, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization
    dC = np.zeros(p*2)
    for i in range(p):
        e = np.zeros(p*2)
        e[i] = 1
        dC[i] = -C_FB_power(alpha_prime, v=e, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization
        e = np.zeros(p*2)
        e[p+i] = 1
        dC[p+i] = C_FB_power(alpha_prime, v=e, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization

    return np.concatenate(([C],dC))

def rsphere(N, p):
    x = np.random.normal(size=(N,p))
    r = np.sqrt(np.sum(x**2.0, axis=1))
    return x / r[:,None]

def G_FB_MC(alpha, ns=None, N=1e6, t=None):
    N = int(N)
    p = len(alpha) / 2
    th = alpha[:p]
    xi = alpha[p:p*2]
    G = np.zeros(p*2+1)
    if t is None:
        t = rsphere(N, np.sum(ns))
    idx = np.concatenate(([0], np.cumsum(ns)))
    t1 = np.zeros((N,p))
    t2 = np.zeros((N,p))

    for i in range(p):
        t2[:,i] = np.sum(t[:,idx[i]:idx[i+1]]**2.0, axis=1)
        t1[:,i] = t[:,idx[i]]

    itrd = np.exp(t2.dot(-th) + t1.dot(xi)) # integrand
    G[0] = np.mean(itrd)

    for i in range(p):
        G[i+1] = -np.mean(itrd * t2[:,i])
        G[p+i+1] = np.mean(itrd * t1[:,i])
    return G

def G_FB(alpha, ns=None, method='power', withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    dsum = np.sum(ns)
    v0 = 2.0 * np.pi**(dsum/2.0) / scipy.special.gamma(dsum/2.0)
    v = v0 if withvol else 1.0
    if method == 'power':
        return v * G_FB_power(alpha, ns=ns)
    elif method == 'MC':
        return v * G_FB_MC(alpha, ns=ns)
    else:
        raise ValueError('Method (%s) not found!' % method)

def my_ode_hg(tau, G, th=None, dG_fun=None, v=None, ns=None, s=None):
    th = th + tau * v
    dG = dG_fun(th, G, ns=ns, s=s)
    return v.dot(dG)

def hg(th0, G0, th1, dG_fun, max_step=0.01, ns=None, s=None, show_trace=False):
    func = lambda t, y: my_ode_hg(t, y, th=th0, dG_fun=dG_fun, v=th1-th0, ns=ns, s=s)
    t_min = 0.0
    t_max = 1.0

    integrator = scipy.integrate.RK45(func, t_min, G0, t_max, max_step=max_step)

    if show_trace:
        trace = []
        while integrator.t < t_max:
            trace.append((integrator.t, integrator.y))
            try:
                integrator.step()
            except:
                break
        trace.append((integrator.t, integrator.y))
    else:
        trace = None
        while integrator.t < t_max:
            try:
                integrator.step()
            except:
                break

    return integrator.y, trace

def my_ode_hg_mod(tau, G, th0=None, th1=None, dG_fun=None, ns=None, s=None):
    p = len(th0) / 2
    th = np.zeros(p*2)
    v = np.zeros(p*2)
    w = np.arange(0,p)
    th[w] = th0[w] + tau * (th1[w] - th0[w])
    v[w] = th1[w] - th0[w]
    w = np.arange(p, p*2)
    th[w] = np.sign(th1[w]) * np.sqrt(th0[w]**2.0 + tau * (th1[w]**2.0 - th0[w]**2.0))
    wz = (th[w] == 0)
    w = w[~wz]
    v[w] = (th1[w]**2.0 - th0[w]**2.0) / 2.0 / th[w]

    dG = dG_fun(th, G, ns=ns, s=s)
    return v.dot(dG)

# hg main (for square-root transformation)
def hg_mod(th0, G0, th1, dG_fun, max_step=0.01, ns=None, s=None, show_trace=False):
    func = lambda t, y: my_ode_hg_mod(t, y, th0=th0, th1=th1, dG_fun=dG_fun, ns=ns, s=s)

    t_min = 0.0
    t_max = 1.0

    integrator = scipy.integrate.RK45(func, t_min, G0, t_max, max_step=max_step)

    if show_trace:
        trace = []
        while integrator.t < t_max:
            trace.append((integrator.t, integrator.y))
            try:
                integrator.step()
            except:
                break
        trace.append((integrator.t, integrator.y))
    else:
        trace = None
        while integrator.t < t_max:
            try:
                integrator.step()
            except:
                break

    return integrator.y, trace

# Evaluation of FB normalization constant by HGM
def hgm_FB(alpha, ns=None, alpha0=None, G0=None, withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    r = np.sum(abs(alpha))
    N = max(r, 1)**2.0 * 10.0
    if alpha0 is None:
        alpha0 = alpha[:p*2] / N
    if G0 is None:
        G0 = G_FB(alpha0, ns=ns, method='power', withvol=withvol)
    res = hg(alpha0, G0, alpha, dG_fun_FB, ns=ns, s=1)[0]
    return res

# Evaluation of FB normalization constant by HGM (via square-root transformation)
def hgm_FB_2(alpha, ns=None, withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    r = np.sum(abs(alpha))
    N = max(r, 1)**2.0 * 10.0
    alpha0 = np.concatenate((alpha[:p]/N, alpha[p:2*p]/np.sqrt(N)))
    G0 = G_FB(alpha0, ns=ns, method="power", withvol=withvol)
    res = hg_mod(alpha0, G0, alpha, dG_fun_FB, ns=ns, s=1)[0]
    return res

# Implement the saddle point approximation for FB8
def saddleapprox_FB_revised(L, M=None, dub=3, order=3):
    L = np.tile(L, dub)

    if M is None:
        M = np.zeros(L.shape)

    a = 1.0 / np.sqrt(np.prod(L / np.pi))
    Y = 0

    def KM(t):
        y = np.sum(-0.5*np.log(1.0-t/L) + M**2.0 / (L-t) / L)
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
    G1 = hgm_FB(alpha1, ns=ns1) # HGM
    print 'HGM:', G1
    G2 = hgm_FB_2(alpha1, ns=ns1) # HGM via a suitable path (the same result)
    print 'HGM2:', G2
    G3 = G_FB(alpha1, ns=ns1, method='MC') # direct computation by Monte Carlo
    print 'MC:', G3
    G4 = SPA(alpha1, ns1)
    print 'SPA:', G4

    print

    print 'Near singular case:'
    alpha2 = np.array([0.067, 0.294, 0.311, 0.405, 0.663, 0.664, 0.667, 0.712, 0.784, 0.933, 0.070, 0.321, 0.288, 0.367, 0.051, 0.338, 0.798, 0.968, 0.506, 0.590])
    ns2 = np.ones(10).astype(int)
    print 'alpha=', alpha2
    print 'ns=', ns2
    G1 = hgm_FB(alpha2, ns=ns2) # HGM
    print 'HGM:', G1
    G2 = hgm_FB_2(alpha2, ns=ns2) # HGM via a suitable path
    print 'HGM2:', G2
    G3 = G_FB(alpha2, ns=ns2, method='MC') # direct computation by Monte Carlo
    print 'MC:', G3
    G4 = SPA(alpha2, ns2)
    print 'SPA:', G4

if __name__ == '__main__':
    test()
