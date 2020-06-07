import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sphere.distribution import fb8, FB8Distribution, fb8_mle, spa
import timeit
from itertools import product


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


def approx_norm(kappa, eta):
    """
    Compare log-c6 vs approx log-c6
    """
    betas = np.arange(kappa/10., 0.8*kappa)
    plt.figure()
    plt.plot(betas, [np.log(fb8(0,0,0,kappa,beta,eta).normalize()) for beta in betas], label='Series', color='k', linewidth=3.5)
    plt.plot(betas, [fb8(0,0,0,kappa,beta,eta)._approx_log_normalize() for beta in betas],
             linestyle='--',
             color='gray',
             label='Approximate')
    plt.plot(betas, [spa(fb8(0,0,0,kappa,beta,eta)).log_c3() for beta in betas],
             linestyle=':',
             color='gray',
             label='Saddlepoint')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\ln c_6(\beta)$')
    plt.legend()
    plt.text(0.03,0.7,
             r'$\kappa={}, \eta={:.1g}$'.format(kappa, eta),
             transform=plt.gca().transAxes, fontsize=14)

    plt.tight_layout(0.1)


def numerical_norm(kappa, eta, alpha, rho):
    """
    Compare log-c8 (series) vs numerical integration log-c8
    """
    betas = np.arange(kappa/10, 0.8*kappa)
    plt.figure()
    plt.plot(betas, [np.log(fb8(0,0,0,kappa,beta,eta,alpha,rho).normalize()) for beta in betas], label='Series', color='k', linewidth=3.5)
    plt.plot(betas, [np.log(fb8(0,0,0,kappa,beta,eta,alpha,rho)._nnormalize()) for beta in betas],
             linestyle='--',
             color='gray',
             label='Numerical integration')
    plt.plot(betas, [spa(fb8(0,0,0,kappa,beta,eta)).log_c3() for beta in betas],
             linestyle=':',
             color='gray',
             label='Saddlepoint')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\ln c_8(\beta)$')
    plt.legend()
    plt.text(0.03,0.7,
             r'$\kappa={}, \eta={:.1g}$'.format(kappa, eta),
             transform=plt.gca().transAxes, fontsize=14)
    _ = fb8(0,0,0,kappa,10,eta,alpha,rho)
    plt.text(0.03,0.6,
             r'$\vec{{\nu}}=({:.3g},{:.3g},{:.3g})$'.format(
                 _.nu[0], _.nu[1], _.nu[2]),
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
            min(timeit.repeat(stmt=('fb8('+','.join(['{}']*8)+').normalize(dict())').format(*x),
                              setup=setup,
                              repeat=1, number=1)))
        times_nnormalize.append(
            min(timeit.repeat(stmt=('fb8('+','.join(['{}']*8)+')._nnormalize()').format(*x),
                              setup=setup,
                              repeat=1, number=1)))
    np.reshape(times_normalize, (len(kappas), len(betas)))
    np.reshape(times_nnormalize, (len(kappas), len(betas)))
    return times_normalize, times_nnormalize


def appendix(th, ph, ps):
    for x in product([th], [ph], [ps],
                     [10,], [1,10],
                     [-1, -0.8, 1], [0, np.pi/2], [0]):
        plot_fb8(fb8(*x), 200)
        # plt.savefig('figs/appendix/fb8_k{:0f}_b{:0f}_e{:1f}_a{:2f}'.format(*x[3:-1]))

    
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
    approx_norm(100, -0.5)
    plt.savefig('figs/Fig3_approxc6.pdf')
    # ln_c8
    numerical_norm(100, -0.5, 0.5, 0.3)
    plt.savefig('figs/Fig3_lnc8.pdf')

    # toy application
    toy()

    # bright stars catalog
    bsc5()
    
    # appendixfb8s
    
if __name__=='__main__':
    __main__()
