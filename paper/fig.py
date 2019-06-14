import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sphere.distribution import fb8, FB8Distribution, fb8_mle, spa

plt.style.use('paper.mplstyle')


def grid(npts):
    return [_.flatten() for _ in np.meshgrid(np.linspace(0, np.pi, npts), np.linspace(0,2*np.pi, npts))]


def make_title(fb8):
    def FBname(n):
        return r'\rm{{FB}}_{}'.format(n)
    def FBtitle(n, ps):
        return r'${}({})$'.format(FBname(n), ps)

    kapbet = r'\kappa = {:.0f}, \beta = {:.0f}'.format(fb8.kappa, fb8.beta)
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
                    facecolors=cm.plasma(pdfs.reshape(npts, npts)/pdfs.max()))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_axis_off()
    ax.set_title(make_title(fb8), fontsize=12, y=0.18)
    plt.tight_layout(-5)


def approx_norm(kappa, eta):
    """
    Compare log-c6 vs approx log-c6
    """
    betas = np.arange(kappa/10., 0.8*kappa)
    plt.figure()
    plt.plot(betas, [np.log(fb8(0,0,0,kappa,beta,eta).normalize()) for beta in betas], label='Series', color='k', linewidth=3.5)
    plt.plot(betas, [fb8(0,0,0,kappa,beta,eta)._approx_log_normalize() for beta in betas],
             linestyle='--',
             label='Approximate')
    plt.plot(betas, [spa(fb8(0,0,0,kappa,beta,eta)).log_c3() for beta in betas],
             linestyle=':',
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
             label='Numerical integration')
    plt.plot(betas, [spa(fb8(0,0,0,kappa,beta,eta)).log_c3() for beta in betas],
             linestyle=':',
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


def yukspor():
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    phs, ths = np.radians(np.loadtxt('yukspor.txt'))
    xs = FB8Distribution.spherical_coordinates_to_nu(ths, phs)
    z,x,y = xs.T
    yuk5 = fb8_mle(xs, True, fb5_only=True)
    plot_fb8(yuk5, 200)
    ax = plt.gca()
    # ax.scatter(x*1.05, y*1.05, z*1.05, color='k', depthshade=False, edgecolors='k', linewidth=0.5)
    for (_x, _y, _z) in zip(x,y,z):
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)
    
    yuk8 = fb8_mle(xs, True)
    plot_fb8(yuk8, 200)
    ax = plt.gca()
    for (_x, _y, _z) in zip(x,y,z):
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)


def toy(seed=92518):
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    np.random.seed(seed)
    toyf8 = fb8(np.pi/16, -np.pi/3,0,55,60,-1.,0.07,0.3)
    xs = toyf8.rvs(100)
    fit5 = fb8_mle(xs, fb5_only=True)
    print fit5, -fit5.log_likelihood(xs)
    plot_fb8(fit5, 200)
    ax = plt.gca()
    for (_z, _x, _y) in xs:
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z, zdir='z')
    plt.savefig('figs/toyfb5.png')
    
    fit8 = fb8_mle(xs)
    print fit8, -fit8.log_likelihood(xs)
    plot_fb8(fit8, 200)
    ax = plt.gca()
    for (_z, _x, _y) in xs:
        p = Circle((_x, _y), 0.01, ec='k', fc="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=_z)
    plt.savefig('figs/toyfb8.png')
        
    
def __main__():
    th,ph,ps = (np.pi/16, -np.pi/3, 0)
    # FB4
    plot_fb8(fb8(th,ph,ps,10,10,-1,0,0), 200)
    plt.savefig('figs/fb4.png')
    # FB5
    plot_fb8(fb8(th,ph,ps,10,4,1,0,0), 200)
    plt.savefig('figs/fb5.png')
    # FB6
    plot_fb8(fb8(th+np.pi/6,ph,ps,10,10,-0.5,0,0), 200)
    plt.savefig('figs/fb6.png')
    # FB8
    plot_fb8(fb8(th,ph,ps,10,10,-1,0.5,0.3), 200)
    plt.savefig('figs/fb8.png')

    # approx_c6
    approx_norm(100, -0.5)
    plt.savefig('figs/approxc6.png')
    # ln_c8
    numerical_norm(100, -0.5, 0.5, 0.3)
    plt.savefig('figs/lnc8.png')

    # toy application
    toy()
    
if __name__=='__main__':
    __main__()
