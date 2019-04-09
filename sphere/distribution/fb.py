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

    #th = alpha[1:p]
    th = alpha[:p]
    #xi = alpha[p+(1:p)]
    xi = alpha[p:p*2]
    #gam = xi^2/4
    gam = xi**2.0 / 4.0
    #Gth = G[1+(1:p)]
    Gth = G[1:p+1]
    #Gxi = G[1+p+(1:p)]
    Gxi = G[p+1:p*2+1]
    #dG = array(0,c(d, r))
    dG = np.zeros((d,r))

    #dG[,1] = G[1+(1:d)]
    dG[:,0] = G[1:d+1]

    # derivative of dC/dth[i] with respect to th[j]
    for i in range(p):
	#dG[i,1+i] = -s*Gth[i]
        dG[i, i+1] = -s*Gth[i]
        for j in range(p):
            if j != i:
                #a1 = - ( ns[j]/2/(th[j]-th[i]) + xi[j]^2/4/(th[j]-th[i])^2 )
                a1 = -(ns[j]/2.0 / (th[j]-th[i]) + xi[j]**2.0 / 4.0 / (th[j]-th[i])**2.0)
                #a2 = - ( ns[i]/2/(th[i]-th[j]) + xi[i]^2/4/(th[i]-th[j])^2 )
                a2 = -(ns[i]/2.0 / (th[i]-th[j]) + xi[i]**2.0 / 4.0 / (th[i]-th[j])**2.0)
                #a3 = - ( ns[j]*xi[i]/4/(th[j]-th[i])^2 + xi[i]*xi[j]^2/4/(th[j]-th[i])^3 )
                a3 = -(ns[j] * xi[i] / 4.0 / (th[j]-th[i])**2.0 + xi[i]*xi[j]**2.0 / 4 / (th[j]-th[i])**3.0)
                #a4 = - ( ns[i]*xi[j]/4/(th[i]-th[j])^2 + xi[i]^2*xi[j]/4/(th[i]-th[j])^3 )
                a4 = -(ns[i]*xi[j] / 4.0 / (th[i]-th[j])**2.0 + xi[i]**2.0 * xi[j] / 4.0 / (th[i]-th[j])**3.0)
                #dG[j,1+i] = a1*Gth[i] + a2*Gth[j] + a3*Gxi[i] + a4*Gxi[j]
                dG[j,i+1] = a1*Gth[i] + a2*Gth[j] + a3*Gxi[i] + a4*Gxi[j]
                #dG[i,1+i] = dG[i,1+i] - dG[j,1+i]
                dG[i,i+1] = dG[i,i+1] - dG[j,i+1]

    # derivative of dC/dxi[i] with respect to th[j]
    # derivative of dC/dxi[j] with respect to xi[i]
    for i in range(p):
	#dG[i,1+p+i] = -s*Gxi[i]
        dG[i,p+i+1] = -s*Gxi[i]
        for j in range(p):
            if j != i:
                #b2 = xi[i]/2/(th[i]-th[j])
                b2 = xi[i] / 2.0 / (th[i]-th[j])
                #b3 = - ( ns[j]/2/(th[j]-th[i]) + xi[j]^2/4/(th[j]-th[i])^2 )
                b3 = -(ns[j] / 2.0 / (th[j]-th[i]) + xi[j]**2.0 / 4.0 / (th[j]-th[i])**2.0)
                #b4 = xi[i]*xi[j]/4/(th[i]-th[j])^2
                b4 = xi[i]*xi[j] / 4.0 / (th[i]-th[j])**2.0
                #dG[j,1+p+i] = b2*Gth[j] + b3*Gxi[i] + b4*Gxi[j]
                dG[j,p+i+1] = b2*Gth[j] + b3*Gxi[i] + b4*Gxi[j]
                #dG[p+i,1+j] = dG[j,1+p+i]
                dG[p+i,j+1] = dG[j,p+i+1]
                #dG[i,1+p+i] = dG[i,1+p+i] - dG[j,1+p+i]
                dG[i,p+i+1] = dG[i,p+i+1] - dG[j,p+i+1]
	#dG[p+i,1+i] = dG[i,1+p+i]
        dG[p+i,i+1] = dG[i,p+i+1]

    # derivative of dC/dxi[i] with respect to xi[j]
    for i in range(p):
        for j in range(p):
            if j != i:
                #c3 = xi[j]/2/(th[j]-th[i])
                c3 = xi[j] / 2.0 / (th[j]-th[i])
                #c4 = -xi[i]/2/(th[j]-th[i])
                c4 = -xi[i] / 2.0 / (th[j]-th[i])
                #dG[p+j,1+p+i] = c3*Gxi[i] + c4*Gxi[j]
                dG[p+j,p+i+1] = c3*Gxi[i] + c4*Gxi[j]
        if ns[i] == 1 or abs(xi[i]) < 1e-10:
	    #dG[p+i,1+p+i] = -Gth[i]  # cheat
            dG[p+i,p+i+1] = -Gth[i] # cheat
        else:
	    #dG[p+i,1+p+i] = -Gth[i] - (ns[i]-1)/xi[i]*Gxi[i]  # singular if xi[i] = 0
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

    #alpha.sum = sum(abs(alpha[1:p])) + sum(abs(alpha[(p+1):(2*p)]))
    alpha_sum = np.sum(abs(alpha[:p*2]))
    N0 = max(int(np.ceil(alpha_sum)), 1)
    if N0 > Nmax:
        raise ValueError("alpha is too large!")
    for N in range(N0, Nmax):
	#logep = N*log(alpha.sum) - lfactorial(N) + log((N+1)/(N+1-alpha.sum))
        logep = N*np.log(alpha_sum) - scipy.special.loggamma(N+1) + np.log((N+1) / (N+1.0-alpha_sum))
        if logep < np.log(Ctol):
            break
    if N > Nmax:
        raise ValueError("alpha is too large!")

    def f(k):
        #k1 = k[1:p]
        k1 = k[:p]
        #k2 = k[(p+1):(2*p)]
        k2 = k[p:p*2]
        #v1 = v[1:p]
        v1 = v[:p]
        #v2 = v[(p+1):(2*p)]
        v2 = v[p:p*2]
	#if(any((k2+v2) %% 2 == 1))
        if np.any(((k2 + v2) % 2) == 1):
            return 0.0
        #w = which(k > 0)
        w = k > 0
        #a = prod( alpha[w]^(k[w]) )
        a = np.prod(alpha[w] ** k[w])
        #b1 = sum( - lfactorial(k1) - lfactorial(k2) + lgamma(k1 + v1 + (k2 + v2)/2 + (d/2)) + lgamma((k2+v2)/2 + (1/2)) - lgamma((k2+v2)/2 + (d/2)))
        b1 = np.sum(-scipy.special.loggamma(k1+1) - scipy.special.loggamma(k2+1) + scipy.special.loggamma(k1 + v1 + (k2 + v2)/2.0 + (d/2.0)) + scipy.special.loggamma((k2+v2)/2.0 + 0.5) - scipy.special.loggamma((k2+v2)/2.0 + d/2.0))
        #b2 = - lgamma(sum(k1 + v1 + (k2 + v2)/2 + (d/2)))
        b2 = -scipy.special.loggamma(np.sum(k1+v1+(k2+v2)/2.0 + (d/2.0)))
        #b3 = lgamma(sum(d)/2) - p*lgamma(1/2)
        b3 = scipy.special.loggamma(np.sum(d)/2.0) - p*scipy.special.loggamma(0.5)
	#return( a * exp(b1+b2+b3) )
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
		#if(abs(alpha[knxt]) < alphatol) return( f(c(k,0)) ) # speed-up
                if abs(alpha[knxt]) < alphatol:
                    stack.append((np.concatenate((k, [0])), None, None, None))
                    continue
                #imax = N - sum(k)
                imax = N - np.sum(k)
                #for(i in 0:imax) a = a + f(c(k,i))
                stack.append((k, 2, imax+1, 0))
                continue
        else:
            if branch == 0:
                raise ValueError("There shouldn't be anything that reaches here")
            elif branch == 1:
                raise ValueError("There shouldn't be anything that reaches here")
                pass
            elif branch == 2:
                #for(i in 0:imax)
                if index < max_index:
                    stack.append((k, 2, max_index, index+1))
                    #a = a + f(c(k,i))
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
    #th = alpha[1:p]
    th = alpha[:p]
    #xi = alpha[(p+1):(2*p)]
    xi = alpha[p:p*2]

    alpha_prime = np.concatenate((-th, xi))

    #C = C.FB.power(c(-th, xi), d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parameterization
    C = C_FB_power(alpha_prime, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization
    #dC = numeric(2*p)
    dC = np.zeros(p*2)
    for i in range(p):
        #e = rep(0,2*p); e[i] = 1
        e = np.zeros(p*2)
        e[i] = 1
        #dC[i] = -C.FB.power(c(-th, xi), v=e, d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parametrerization
        dC[i] = -C_FB_power(alpha_prime, v=e, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization
        #e = rep(0,2*p); e[p+i] = 1
        e = np.zeros(p*2)
        e[p+i] = 1
        #dC[p+i] = C.FB.power(c(-th, xi), v=e, d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parametrerization
        dC[p+i] = C_FB_power(alpha_prime, v=e, d=ns, Ctol=Ctol, alphatol=alphatol) # note: parameterization

    return np.concatenate(([C],dC))

def rsphere(N, p):
    #x = matrix(rnorm(N*p), N, p)
    x = np.random.normal(size=(N,p))
    #r = sqrt(apply(x^2, 1, sum))
    r = np.sqrt(np.sum(x**2.0, axis=1))
    #x / r
    return x / r[:,None]

def G_FB_MC(alpha, ns=None, N=1e6, t=None):
    N = int(N)
    p = len(alpha) / 2
    #th = alpha[1:p]
    th = alpha[:p]
    #xi = alpha[p+(1:p)]
    xi = alpha[p:p*2]
    #G = numeric(2*p+1)
    G = np.zeros(p*2+1)
    #if(is.null(t)) t = rsphere(N, sum(ns))
    if t is None:
        t = rsphere(N, np.sum(ns))
    #idx = c(0, cumsum(ns))
    idx = np.concatenate(([0], np.cumsum(ns)))
    #t2 = t1 = matrix(0, N, p)
    t1 = np.zeros((N,p))
    t2 = np.zeros((N,p))

    for i in range(p):
	#t2[,i] = rowSums(t[,(idx[i]+1):idx[i+1],drop=FALSE]^2)
        t2[:,i] = np.sum(t[:,idx[i]:idx[i+1]]**2.0, axis=1)
	#t1[,i] = t[,idx[i]+1]
        t1[:,i] = t[:,idx[i]]

    #itrd = exp(t2 %*% (-th) + t1 %*% xi)  # integrand
    itrd = np.exp(t2.dot(-th) + t1.dot(xi)) # integrand
    #G[1] = mean(itrd)
    G[0] = np.mean(itrd)

    for i in range(p):
	#G[1+i] = -mean(itrd * t2[,i])
        G[i+1] = -np.mean(itrd * t2[:,i])
	#G[1+p+i] = mean(itrd * t1[,i])
        G[p+i+1] = np.mean(itrd * t1[:,i])
    return G

def G_FB(alpha, ns=None, method='power', withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    #dsum = sum(ns)
    dsum = np.sum(ns)
    #v0 = 2 * pi^(dsum/2)/gamma(dsum/2)
    v0 = 2.0 * np.pi**(dsum/2.0) / scipy.special.gamma(dsum/2.0)
    #v = ifelse(withvol, v0, 1)
    v = v0 if withvol else 1.0
    if method == 'power':
        return v * G_FB_power(alpha, ns=ns)
    elif method == 'MC':
        return v * G_FB_MC(alpha, ns=ns)
    else:
        raise ValueError('Method (%s) not found!' % method)

def my_ode_hg(tau, G, th=None, dG_fun=None, v=None, ns=None, s=None):
    #th = th + tau * v
    th = th + tau * v
    #else dG = dG.fun(th, G, fn.params)
    dG = dG_fun(th, G, ns=ns, s=s)
    #G.rhs = v %*% dG
    return v.dot(dG)

def hg(th0, G0, th1, dG_fun, max_step=0.01, ns=None, s=None, show_trace=False):
    func = lambda t, y: my_ode_hg(t, y, th=th0, dG_fun=dG_fun, v=th1-th0, ns=ns, s=s)
    t_min = 0.0
    t_max = 1.0

    #times=seq(0,1,length=101)
    #params = list(th = th0, dG.fun = dG.fun, v = th1-th0, fn.params = fn.params)
    #rk.res = rk(G0, times, my.ode.hg, params)
    #if(show.trace) trace = rk.res
    #else trace=NULL

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

    #list(G = rk.res[nrow(rk.res), 1+(1:length(G0))], trace = trace)
    return integrator.y, trace

def my_ode_hg_mod(tau, G, th0=None, th1=None, dG_fun=None, ns=None, s=None):
    p = len(th0) / 2
    #th = v = numeric(2*p)
    th = np.zeros(p*2)
    v = np.zeros(p*2)
    #w = 1:p
    w = np.arange(0,p)
    #th[w] = th0[w] + tau * (th1[w] - th0[w])
    th[w] = th0[w] + tau * (th1[w] - th0[w])
    #v[w] = th1[w] - th0[w]
    v[w] = th1[w] - th0[w]
    #w = (p+1):(2*p)
    w = np.arange(p, p*2)
    #th[w] = sign(th1[w]) * sqrt(th0[w]^2 + tau * (th1[w]^2 - th0[w]^2))
    th[w] = np.sign(th1[w]) * np.sqrt(th0[w]**2.0 + tau * (th1[w]**2.0 - th0[w]**2.0))
    #wz = (th[w] == 0)
    wz = (th[w] == 0)
    #w = w[!wz]
    w = w[~wz]
    #v[w] = (th1[w]^2 - th0[w]^2) / 2 / th[w]
    v[w] = (th1[w]**2.0 - th0[w]**2.0) / 2.0 / th[w]

    #dG = dG.fun(th, G, fn.params)
    dG = dG_fun(th, G, ns=ns, s=s)
    #G.rhs = v %*% dG
    return v.dot(dG)

# hg main (for square-root transformation)
def hg_mod(th0, G0, th1, dG_fun, max_step=0.01, ns=None, s=None, show_trace=False):
    func = lambda t, y: my_ode_hg_mod(t, y, th0=th0, th1=th1, dG_fun=dG_fun, ns=ns, s=s)

    t_min = 0.0
    t_max = 1.0

    #rk.res = rk(G0, times, my.ode.hg.mod, params)
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

    #list(G = rk.res[nrow(rk.res), 1+(1:length(G0))], trace = trace)
    return integrator.y, trace

# Evaluation of FB normalization constant by HGM
def hgm_FB(alpha, ns=None, alpha0=None, G0=None, withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    #r = sum(abs(alpha))
    r = np.sum(abs(alpha))
    #N = max(r, 1)^2 * 10
    N = max(r, 1)**2.0 * 10.0
    #if(is.null(alpha0))  alpha0 = alpha[1:(2*p)] / N
    if alpha0 is None:
        alpha0 = alpha[:p*2] / N
    #if(is.null(G0)) G0 = G.FB(alpha0, ns=ns, method="power", withvol=withvol)
    if G0 is None:
        G0 = G_FB(alpha0, ns=ns, method='power', withvol=withvol)
    #as.vector(hg(alpha0, G0, alpha, dG.fun.FB, fn.params=fn.params)$G)
    res = hg(alpha0, G0, alpha, dG_fun_FB, ns=ns, s=1)[0]
    return res

# Evaluation of FB normalization constant by HGM (via square-root transformation)
def hgm_FB_2(alpha, ns=None, withvol=True):
    p = len(alpha) / 2
    if ns is None:
        ns = np.ones(p).astype(int)
    #r = sum(abs(alpha))
    r = np.sum(abs(alpha))
    #N = max(r, 1)^2 * 10
    N = max(r, 1)**2.0 * 10.0
    #alpha0 = c(alpha[1:p] / N, alpha[(p+1):(2*p)] / sqrt(N))
    alpha0 = np.concatenate((alpha[:p]/N, alpha[p:2*p]/np.sqrt(N)))
    #G0 = G.FB(alpha0, ns=ns, method="power", withvol=withvol)
    G0 = G_FB(alpha0, ns=ns, method="power", withvol=withvol)
    #res = hg.mod(alpha0, G0, alpha, dG.fun.FB, fn.params=fn.params)$G
    res = hg_mod(alpha0, G0, alpha, dG_fun_FB, ns=ns, s=1)[0]
    return res

# Implement the saddle point approximation for FB8
def saddleapprox_FB_revised(L, M=None, dub=3, order=3):
    #M=L*0

    #L<-rep(t(L),dub)
    L = np.tile(L, dub)

    #M=L*0
    if M is None:
        M = np.zeros(L.shape)


    #a<-prod(L/pi)^(-1/2)
    a = 1.0 / np.sqrt(np.prod(L / np.pi))

    #Y<-0
    Y = 0

    def KM(t):
        #Y<-sum(-1/2*log(1-t/L)+M^2/(1-t/L)/L)
        y = np.sum(-0.5*np.log(1.0-t/L) + M**2.0 / (L-t) / L)
        #print 't=', t, ', KM=', y
        return y

    def KM1(t):
        #Y<-sum(1/2*1/(L-t)+M^2/(L-t)^2)
        y = np.sum(0.5 / (L-t) + (M**2.0/(L-t)**2.0))
        #print 't=', t, ', KM1=', y
        return y

    def KM2(t):
        #Y<-sum(1/2*1/(L-t)^2+2*M^2/(L-t)^3)
        y = np.sum(0.5 / (L-t)**2.0 + 2.0 * M**2.0 / (L-t)**3.0)
        #print 't=', t, ', KM2=', y
        return y

    def KM3(t):
        #Y<-sum(1/(L-t)^3+6*M^2/(L-t)^4)
        y = np.sum(1.0 / (L-t)**3.0 + 6.0 * M**2.0 / (L-t)**4.0)
        #print 't=', t, ', KM3=', y
        return y

    def KM4(t):
        #Y<-sum(3/(L-t)^4+24*M^2/(L-t)^5)
        y = np.sum(3.0 / (L-t)**4.0 + 24.0 * M**2.0 / (L-t)**5.0)
        #print 't=', t, ', KM4=', y
        return y

    def sol(y):
        #Y<-optimize(loc<-function(t) {abs(KM1(t)-1)},c(min(L)-length(L)/(4*y)-sqrt(length(L)^2/4+length(L)*max(L)^2*max(M)^2),min(L)), tol = .Machine$double.eps^2)$min
        #loc<-function(t) {abs(KM1(t)-1)}
        f = lambda t: abs(KM1(t) - 1.0)
        #fmin<- min(L)-length(L)/(4*y)-sqrt(length(L)^2/4+length(L)*max(L)^2*max(M)^2)
        fmin = np.amin(L)-len(L)/(4.0*y) - np.sqrt(len(L)**2.0 / 4.0 + len(L)*np.amax(L)**2.0*np.amax(M)**2.0)
        #min(L)
        fmax = np.amin(L)
        res = scipy.optimize.minimize_scalar(f, method='brent', bounds=(fmin, fmax))
        return res.x

    #that<-sol(KM1,1)
    that = sol(1.0)

    #print 'that=', that

    #Y<-2*a/sqrt(2*pi*KM2(that))*exp(KM(that)-that)
    Y = 2.0*a/np.sqrt(2.0*np.pi*KM2(that)) * np.exp(KM(that) - that)
    if order == 3:
        #rho3sq<-KM3(that)^2/KM2(that)^3
        rho3sq = KM3(that)**2.0 / KM2(that)**3.0

        #rho4<-KM4(that)/KM2(that)^(4/2)
        rho4 = KM4(that)/KM2(that)**(2.0)

        #Rhat<-3/24 *rho4-5/24*rho3sq
        Rhat = 3.0/24.0 * rho4 - 5.0/24.0 * rho3sq

        #Y<-Y*exp(Rhat)
        Y = Y*np.exp(Rhat)
    elif order == 2:
        #rho3sq<-KM3(that)^2/KM2(that)^3
        rho3sq = KM3(that)**2.0 / KM2(that)**3.0

        #rho4<-KM4(that)/KM2(that)^(4/2)
        rho4 = KM4(that)/KM2(that)**(2.0)

        #Rhat<-3/24 *rho4-5/24*rho3sq
        Rhat = 3.0/24.0 * rho4 - 5.0/24.0 * rho3sq

        #Y<-Y*(1+Rhat)
        Y = Y*(1.0 + Rhat)
    return Y

def SPA(alpha, ns=None, withvol=True):
    #p=length(alpha)/2
    p = int(len(alpha) / 2)

    #ns=rep(1,length(alpha)/2)
    if ns is None:
        ns = np.ones(p).astype(int)

    #theta=rep(alpha[1:(p)],ns)
    theta = tile_repeat(alpha[:p], ns)

    #mu=rep(alpha[-(1:(p))]/sqrt(ns),ns)/2
    mu = tile_repeat(alpha[p:] / np.sqrt(ns), ns) / 2.0

    #dsum = sum(ns)
    dsum = np.sum(ns)

    #v0 = 2 * pi^(dsum/2)/gamma(dsum/2)
    v0 = 2.0 * np.pi**(dsum/2.0) / scipy.special.gamma(dsum/2.0)

    #coef = ifelse(withvol, 1, 1/v0)
    coef = 1.0 if withvol else 1.0/v0

    #print 'theta=', theta
    #print 'mu=', mu
    #print 'coef=', coef

    #return(saddleaprox.FB.revised(theta+1,mu,dub=1,order=3)*exp(1) / coef)
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
