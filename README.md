[![PyPI version](https://badgen.net/pypi/v/fb8)](https://pypi.org/project/fb8) [![Build Status](https://travis-ci.com/tianluyuan/sphere.svg?branch=master)](https://travis-ci.com/tianluyuan/sphere) [![Python versions](https://img.shields.io/pypi/pyversions/fb8)](https://pypi.org/project/fb8)

Getting started
=================
`pip install fb8`

```Python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sphere.distribution import fb8


def grid(npts):
    return [_.flatten() for _ in np.meshgrid(np.linspace(0, np.pi, npts), np.linspace(0,2*np.pi, npts))]


def plot_fb8(fb8, npts):
    """
    Plot fb8 on 3D sphere
    """
    xs = fb8.spherical_coordinates_to_nu(*grid(npts))
    pdfs = fb8.pdf(xs)
    z,x,y = xs.T #!!! Note the ordering for xs here is used consistently throughout. Follows Kent's 1982 paper.

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.reshape(npts, npts),
                    y.reshape(npts, npts),
                    z.reshape(npts, npts),
                    alpha=0.5,
                    rstride=1, cstride=1,
                    facecolors=cm.plasma(pdfs.reshape(npts, npts)/pdfs.max()))
    ax.set_axis_off()
    plt.tight_layout(-5)
    plt.show()


plot_fb8(fb8(np.pi/16,-np.pi/3,0,10,10,-1,0.5,0.3), 200)
```

Basic information
=================
Implements the FB8 distribution on a sphere, which is a generalization of the FB6, FB5 (Kent), and FB4 (Bingham-Mardia) distributions described below.

Implements the FB6 distribution that is first introduced in Rivest ([1984](https://www.doi.org/10.1214/aos/1176346724)).

Implements calculation of the density and fitting (using maximum likelihood estimate) of the Kent distribution based on Kent ([1982](https://doi.org/10.1111/j.2517-6161.1982.tb01189.x)). A unittest is performed if distribution.py is called from the command line.

Implements the Bingham-Mardia distribution whose mode is a small-circle on the sphere based on Bingham, Mardia ([1978](https://doi.org/10.1093/biomet/65.2.379)).

Also calculates directional, percentile levels which can be used to indicate the N% highest-posterior-density regions in the sky.

![maps](https://github.com/tianluyuan/sphere/blob/master/fig/example.png?raw=true)

Additional references
=================
Kent, Hussein, Jah, [_Directional distributions in tracking of space debris_](https://ieeexplore.ieee.org/abstract/document/7528139) 

Terdik, Jammalamadaka, Wainwright, [_Simulation and visualization of spherical distributions_](https://www.researchgate.net/profile/Gyorgy_Terdik/publication/324605982_Simulation_and_Visualization_of_Spherical_Distributions/links/5ad8edceaca272fdaf81fe04/Simulation-and-Visualization-of-Spherical-Distributions.pdf)

Mardia, Jupp, [_Directional statistics_](https://www.doi.org/10.1002/9780470316979)

Notes
=================
Currently the `scipy.special.hyp2f1` is used and may exhibit inaccuracies for large parameters. See github [issues](https://github.com/scipy/scipy/issues?utf8=%E2%9C%93&q=is%3Aissue+is%3Aopen+hyp2f1).

Contributors
=================

This project was originally developed for the FB5 (Kent) distribution [here](https://github.com/edfraenkel/kent_distribution).

_Tianlu Yuan_
