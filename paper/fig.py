import timeit
from itertools import product
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sphere.distribution import fb8, FB8Distribution, fb8_mle, spa


plt.style.use('paper.mplstyle')


def grid(npts):
    return [_.flatten() for _ in np.meshgrid(np.linspace(0, np.pi, npts), np.linspace(0,2*np.pi, npts))]


def make_title(fb8, kbdec=0):
    def FBname(n):
        return r'\rm{{FB}}_{}'.format(n)
    def FBtitle(n, ps):
        return r'${}({})$'.format(FBname(n), ps)

    kapbet = r'\kappa = {:.'+str(kbdec)+r'f}, \beta = {:.'+str(kbdec)+r'f}'
    kapbet = kapbet.format(fb8.kappa, fb8.beta)
    if fb8.nu[0] == 1.:
        if fb8.eta == 1.:
            return FBtitle(5, kapbet)
        if fb8.eta == -1.:
            return FBtitle(4, kapbet)
        return FBtitle(6, kapbet+r', \eta={:.1g}'.format(fb8.eta))
    return FBtitle(8, kapbet+r', \eta={:.1g}, \vec{{\nu}}=({:.3g},{:.3g},{:.3g})'.format(
        fb8.eta, np.round(fb8.nu[0],3), np.round(fb8.nu[1],3), np.round(fb8.nu[2],3)))


def plot_fb8(fb8, npts):
    """
    Plot fb8 on 3D sphere
    """
    xs = fb8.spherical_coordinates_to_nu(*grid(npts))
    pdfs = fb8.pdf(xs)
    z,x,y = xs.T

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.reshape(npts, npts),
                    y.reshape(npts, npts),
                    z.reshape(npts, npts),
                    alpha=0.5,
                    rstride=1, cstride=1,
                    facecolors=cm.gray(pdfs.reshape(npts, npts)/pdfs.max()))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_axis_off()
    ax.set_title(make_title(fb8), fontsize=12, y=0.18)
    plt.tight_layout(-5)


def hp_plot_fb8(fb8, nside):
    import healpy as hp
    npix = hp.nside2npix(nside)
    fb8_map = fb8.pdf(fb8.spherical_coordinates_to_nu(
        *hp.pix2ang(nside, np.arange(npix))))

    plt.figure(figsize=(9,6))
    vmap = cm.gray
    vmap.set_under('w')
    vmap.set_bad('w')
    hp.mollview(fb8_map,
                title=make_title(fb8, 1),
                min=0,
                max=np.round(np.nanmax(fb8_map),2),
                cmap=vmap, hold=True,
                cbar=True,
                xsize=1600)
    hp.graticule()


def build_args(kappa, beta, eta, alpha=0., rho=0.):
    if kappa is None:
        xvals = np.arange(beta/10., 0.8*beta)
        idx = 3
        xlabel='kappa'
        text = rf'$\beta={beta}, \eta={eta:.1g}$'
        textx = 0.03
    elif beta is None:
        xvals = np.arange(kappa/10., 0.8*kappa)
        idx = 4
        xlabel = 'beta'
        text = rf'$\kappa={kappa}, \eta={eta:.1g}$'
        textx = 0.03
    elif eta is None:
        xvals = np.arange(-1., 1.02, 0.02)
        idx = 5
        xlabel = 'eta'
        text = rf'$\kappa={kappa}, \beta={beta}$'
        textx = 0.5
    args = []
    for x in xvals:
        arg = [0.,0.,0.,kappa,beta,eta,alpha,rho]
        arg[idx] = x
        args.append(arg)
    return xvals, xlabel, text, textx, args


def approx_norm(kappa, beta, eta):
    """
    Compare log-c6 vs approx log-c6
    """
    xvals, xlabel, text, textx, args = build_args(kappa, beta, eta)
    plt.figure()
    plt.plot(xvals, [np.log(fb8(*_).normalize()) for _ in args], label='Series', color='k', linewidth=3.5)
    plt.plot(xvals, [fb8(*_)._approx_log_normalize() for _ in args],
             linestyle='--',
             color='gray',
             label='Approximate')
    plt.plot(xvals, [spa(fb8(*_)).log_c3() for _ in args],
             linestyle=':',
             color='gray',
             label='Saddlepoint')
    plt.xlabel(rf'$\{xlabel}$')
    plt.ylabel(rf'$\ln c_6(\{xlabel})$')
    plt.legend()
    plt.text(textx,0.7,text,
             transform=plt.gca().transAxes, fontsize=14)

    plt.tight_layout(0.1)


def numerical_norm(kappa, beta, eta, alpha, rho):
    """
    Compare log-c8 (series) vs numerical integration log-c8
    """
    xvals, xlabel, text, textx, args = build_args(kappa, beta, eta, alpha, rho)
    plt.figure()
    plt.plot(xvals, [np.log(fb8(*_).normalize()) for _ in args],
             label='Series', color='k', linewidth=3.5)
    plt.plot(xvals, [np.log(fb8(*_)._nnormalize()) for _ in args],
             linestyle='--',
             color='gray',
             label='Numerical integration')
    plt.plot(xvals, [spa(fb8(*_[:-2])).log_c3() for _ in args],
             linestyle=':',
             color='gray',
             label='Saddlepoint')
    plt.xlabel(rf'$\{xlabel}$')
    plt.ylabel(rf'$\ln c_8(\{xlabel})$')
    plt.legend()
    plt.text(textx,0.7,text,
             transform=plt.gca().transAxes, fontsize=14)
    _ = fb8(0.,0.,0.,100.,10.,-0.5,alpha,rho)
    textnu = rf'$\vec{{\nu}}=({_.nu[0]:.3g},{_.nu[1]:.3g},{_.nu[2]:.3g})$'
    plt.text(textx,0.6,textnu,
             transform=plt.gca().transAxes, fontsize=14)
    plt.tight_layout(0.1)


def time_norm(kappa, beta, eta, alpha, rho):
    """ Plot execution time of .normalize to ._nnormalize
    """
    xvals, xlabel, text, textx, args = build_args(kappa, beta, eta, alpha, rho)
    tfile = os.path.join('figs', 'time', 'timec8.pkl')
    if os.path.isfile(tfile) and str(args) in pickle.load(open(tfile, 'rb')):
        times_normalize, times_nnormalize = pickle.load(open(tfile, 'rb'))[str(args)]
    else:                  
        times_normalize = []
        times_nnormalize = []
        setup = 'from sphere.distribution import fb8'
        for _ in args:
            times_normalize.append(
                min(timeit.repeat(stmt=('fb8('+','.join(['{}']*8)+').normalize(cache=dict())').format(*_),
                                  setup=setup, repeat=3, number=1)))
            times_nnormalize.append(
                min(timeit.repeat(stmt=('fb8('+','.join(['{}']*8)+')._nnormalize()').format(*_),
                                  setup=setup, repeat=3, number=1)))
        if os.path.isfile(tfile):
            ddd = pickle.load(open(tfile, 'rb'))
        else:
            ddd = {}
        ddd[str(args)] = [times_normalize, times_nnormalize]
        with open(tfile, 'wb') as f:
            pickle.dump(ddd, f)
    plt.figure()
    plt.plot(xvals, times_normalize,
             label='Series', color='k', linewidth=3.5)
    plt.plot(xvals, times_nnormalize,
             linestyle='--',
             color='gray',
             label='Numerical integration')
    plt.xlabel(rf'$\{xlabel}$')
    plt.ylabel(rf'Runtime [s]')
    # plt.yscale('log')
    plt.legend()
    plt.text(0.42,0.38,text,
             transform=plt.gca().transAxes, fontsize=14)
    _ = fb8(0.,0.,0.,100.,10.,-0.5,alpha,rho)
    textnu = rf'$\vec{{\nu}}=({_.nu[0]:.3g},{_.nu[1]:.3g},{_.nu[2]:.3g})$'
    plt.text(0.42,0.28,textnu,
             transform=plt.gca().transAxes, fontsize=14)
    plt.tight_layout(0.1)


def do_fits(ths, phs):
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    xs = FB8Distribution.spherical_coordinates_to_nu(ths, phs)
    z,x,y = xs.T
    fit5 = fb8_mle(xs, True, fb5_only=True)
    plot_fb8(fit5, 200)
    ax = plt.gca()
    # ax.scatter(x*1.05, y*1.05, z*1.05, color='k', depthshade=False, edgecolors='k', linewidth=0.5)
    for (_x, _y, _z) in zip(x,y,z):
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)
    
    fit8 = fb8_mle(xs, True)
    plot_fb8(fit8, 200)
    ax = plt.gca()
    for (_x, _y, _z) in zip(x,y,z):
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)

    
def hp_fits(ths, phs, nside=64):
    import healpy as hp
    xs = FB8Distribution.spherical_coordinates_to_nu(ths, phs)
    z,x,y = xs.T
    fit5 = fb8_mle(xs, True, fb5_only=True)
    hp_plot_fb8(fit5, nside)
    hp.projscatter(ths, phs, marker='.', linewidths=0, s=5, c='k')
    ax = plt.gca()
    ax.annotate(r"$\bf{-180^\circ}$", xy=(1.7, 0.625), size="medium")
    ax.annotate(r"$\bf{180^\circ}$", xy=(-1.95, 0.625), size="medium")
    ax.annotate("Galactic", xy=(0.8, -0.05),
                size="medium", xycoords="axes fraction")
    plt.savefig('figs/Fig5_fb5.png')

    fit8 = fb8_mle(xs, True)
    hp_plot_fb8(fit8, nside)
    hp.projscatter(ths, phs, marker='.', linewidths=0, s=5, c='k')
    ax = plt.gca()
    ax.annotate(r"$\bf{-180^\circ}$", xy=(1.7, 0.625), size="medium")
    ax.annotate(r"$\bf{180^\circ}$", xy=(-1.95, 0.625), size="medium")
    ax.annotate("Galactic", xy=(0.8, -0.05),
                size="medium", xycoords="axes fraction")
    plt.savefig('figs/Fig5_fb8.png')


def yukspor():
    phs, ths = np.radians(np.loadtxt('yukspor.txt'))
    do_fits(ths, phs)


def bsc5(mag_low=6):
    dat = np.loadtxt('bsc5.dat', comments='#', skiprows=43)
    _ = dat[dat[:,-1]<=mag_low]
    phs, ths = np.radians([_[:,1], 90.-_[:,2]])
    hp_fits(ths, phs)


def toy(seed=92518):
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    np.random.seed(seed)
    toyf8 = fb8(np.pi/16, -np.pi/3,0,55,60,-1.,0.07,0.3)
    xs = toyf8.rvs(100)
    fit5 = fb8_mle(xs, fb5_only=True)
    print(fit5, -fit5.log_likelihood(xs))
    plot_fb8(fit5, 200)
    ax = plt.gca()
    for (_z, _x, _y) in xs:
        p = Circle((_x, _y), 0.01, ec='w', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z, zdir='z')
    plt.savefig('figs/Fig4_toyfb5.png')
    
    fit8 = fb8_mle(xs)
    print(fit8, -fit8.log_likelihood(xs))
    plot_fb8(fit8, 200)
    ax = plt.gca()
    for (_z, _x, _y) in xs:
        p = Circle((_x, _y), 0.01, ec='w', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)
    plt.savefig('figs/Fig4_toyfb8.png')
        

def time(eta=1, alpha=0, rho=0, step=10):
    """ Plot ratio of time spent on .normalize to ._nnormalize
    """
    times_normalize = []
    times_nnormalize = []
    kappas = range(1, 200, step)
    betas = range(1, 200, step)
    setup = 'from sphere.distribution import fb8'
    for x in product([0], [0], [0],
                     kappas, betas,
                     [eta], [alpha], [rho]):
        print(x)
        times_normalize.append(
            min(timeit.timeit(stmt=('fb8('+','.join(['{}']*8)+')._normalize(dict())').format(*x),
                              setup=setup, number=1)))
        times_nnormalize.append(
            min(timeit.timeit(stmt=('fb8('+','.join(['{}']*8)+')._nnormalize()').format(*x),
                              setup=setup, number=1)))
    np.reshape(times_normalize, (len(kappas), len(betas)))
    np.reshape(times_nnormalize, (len(kappas), len(betas)))
    return times_normalize, times_nnormalize


def appendix(th, ph, ps):
    for x in product([th], [ph], [ps],
                     [10,], [1,10],
                     [-1, -0.8, 1], [0, np.pi/2], [0]):
        plot_fb8(fb8(*x), 200)
        plt.savefig('figs/appendix/fb8_k{:.0f}_b{:.0f}_e{:.1f}_a{:.2f}.png'.format(*x[3:-1]))

    
def __main__():
    th,ph,ps = (np.pi/16, -np.pi/3, 0)
    # FB4
    plot_fb8(fb8(th,ph,ps,10,10,-1,0,0), 200)
    plt.savefig('figs/Fig1_fb4.png')
    # FB5
    plot_fb8(fb8(th,ph,ps,10,4,1,0,0), 200)
    plt.savefig('figs/Fig1_fb5.png')
    # FB6
    plot_fb8(fb8(th+np.pi/6,ph,ps,10,10,-0.5,0,0), 200)
    plt.savefig('figs/Fig2_fb6.png')
    # FB8
    plot_fb8(fb8(th,ph,ps,10,10,-1,0.5,0.3), 200)
    plt.savefig('figs/Fig2_fb8.png')

    # approx_c6
    approx_norm(None, 100., -0.5)
    plt.savefig('figs/Fig3_approxc6_kappa.pdf')
    approx_norm(100., None, -0.5)
    plt.savefig('figs/Fig3_approxc6_beta.pdf')
    approx_norm(100., 100., None)
    plt.savefig('figs/Fig3_approxc6_eta.pdf')
    # ln_c8
    numerical_norm(None, 100., -0.5, 0.5, 0.3)
    plt.savefig('figs/Fig3_lnc8_kappa.pdf')
    numerical_norm(100., None, -0.5, 0.5, 0.3)
    plt.savefig('figs/Fig3_lnc8_beta.pdf')
    numerical_norm(100., 100., None, 0.5, 0.3)
    plt.savefig('figs/Fig3_lnc8_eta.pdf')
    # time_c8
    time_norm(None, 100., -0.5, 0.5, 0.3)
    plt.savefig('figs/Fig3_timec8_kappa.pdf')
    time_norm(100., None, -0.5, 0.5, 0.3)
    plt.savefig('figs/Fig3_timec8_beta.pdf')
    time_norm(100., 100., None, 0.5, 0.3)
    plt.savefig('figs/Fig3_timec8_eta.pdf')
    
    # toy application
    toy()

    # bright stars catalog
    bsc5()
    
    # appendixfb8s
    appendix(0,0,0)
    

if __name__=='__main__':
    __main__()
