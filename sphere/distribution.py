#!/usr/bin/env python
"""
The algorithms here are partially based on methods described in:
[The Fisher-Bingham Distribution on the Sphere, John T. Kent
Journal of the Royal Statistical Society. Series B (Methodological)
Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
Article Stable URL: http://www.jstor.org/stable/2984712]

The example code in example.py serves not only as an example
but also as a test. It performs some higher level tests but it also 
generates example plots if called directly from the shell.
"""

from numpy import *
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fun
from scipy.special import gammaln as gammaln_fun
from scipy.special import iv as modified_bessel_1stkind
from scipy.special import ivp as modified_bessel_1stkind_derivative
from scipy.stats import uniform
from scipy.special import comb
# to avoid confusion with the norm of a vector we give the normal distribution a less confusing name here
from scipy.stats import norm as gauss
from scipy.linalg import eig
import sys
import warnings


# helper function
def MMul(A, B):
    return inner(A, transpose(B))


def norm(x, axis=None):
    """
    helper function to compute the L2 norm. scipy.linalg.norm is not used because this function does not allow to choose an axis
    """
    if isinstance(x, list) or isinstance(x, tuple):
        x = array(x)
    return sqrt(sum(x * x, axis=axis))


def kent(theta, phi, psi, kappa, beta, eta=1.):
    """
    Generates the Kent distribution based on the spherical coordinates theta, phi, psi
    with the concentration parameter kappa and the ovalness beta
    """
    gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(
        theta, phi, psi)
    k = KentDistribution(gamma1, gamma2, gamma3, kappa, beta, eta)
    return k


def kent2(gamma1, gamma2, gamma3, kappa, beta, eta=1.):
    """
    Generates the Kent distribution using the orthonormal vectors gamma1, 
    gamma2 and gamma3, with the concentration parameter kappa and the ovalness beta
    """
    assert abs(inner(gamma1, gamma2)) < 1E-10
    assert abs(inner(gamma2, gamma3)) < 1E-10
    assert abs(inner(gamma3, gamma1)) < 1E-10
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta, eta)


def kent3(A, B, eta=1.):
    """
    Generates the Kent distribution using the orthogonal vectors A and B
    where A = gamma1*kappa and B = gamma2*beta (gamma3 is inferred)
    A may have not have length zero but may be arbitrarily close to zero
    B may have length zero however. If so, then an arbitrary value for gamma2
    (orthogonal to gamma1) is chosen
    """
    kappa = norm(A)
    beta = norm(B)
    gamma1 = A / kappa
    if beta == 0.0:
        gamma2 = __generate_arbitrary_orthogonal_unit_vector(gamma1)
    else:
        gamma2 = B / beta
    theta, phi, psi = KentDistribution.gammas_to_spherical_coordinates(
        gamma1, gamma2)
    gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(
        theta, phi, psi)
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta, eta)


def kent4(Gamma, kappa, beta, eta=1.):
    """
    Generates the kent distribution
    """
    gamma1 = Gamma[:, 0]
    gamma2 = Gamma[:, 1]
    gamma3 = Gamma[:, 2]
    return kent2(gamma1, gamma2, gamma3, kappa, beta, eta)


def __generate_arbitrary_orthogonal_unit_vector(x):
    v1 = cross(x, array([1.0, 0.0, 0.0]))
    v2 = cross(x, array([0.0, 1.0, 0.0]))
    v3 = cross(x, array([0.0, 0.0, 1.0]))
    v1n = norm(v1)
    v2n = norm(v2)
    v3n = norm(v3)
    v = [v1, v2, v3][argmax([v1n, v2n, v3n])]
    return v / norm(v)


class KentDistribution(object):
    minimum_value_for_kappa = 1E-6

    @staticmethod
    def create_matrix_H(theta, phi):
        return array([
            [cos(theta),          -sin(theta),         0.0],
            [sin(theta) * cos(phi), cos(theta) * cos(phi), -sin(phi)],
            [sin(theta) * sin(phi), cos(theta) * sin(phi), cos(phi)]
        ])

    @staticmethod
    def create_matrix_Ht(theta, phi):
        return transpose(KentDistribution.create_matrix_H(theta, phi))

    @staticmethod
    def create_matrix_K(psi):
        return array([
            [1.0, 0.0,      0.0],
            [0.0, cos(psi), -sin(psi)],
            [0.0, sin(psi), cos(psi)]
        ])

    @staticmethod
    def create_matrix_Kt(psi):
        return transpose(KentDistribution.create_matrix_K(psi))

    @staticmethod
    def create_matrix_Gamma(theta, phi, psi):
        H = KentDistribution.create_matrix_H(theta, phi)
        K = KentDistribution.create_matrix_K(psi)
        return MMul(H, K)

    @staticmethod
    def create_matrix_Gammat(theta, phi, psi):
        return transpose(KentDistribution.create_matrix_Gamma(theta, phi, psi))

    @staticmethod
    def spherical_coordinates_to_gammas(theta, phi, psi):
        Gamma = KentDistribution.create_matrix_Gamma(theta, phi, psi)
        gamma1 = Gamma[:, 0]
        gamma2 = Gamma[:, 1]
        gamma3 = Gamma[:, 2]
        return gamma1, gamma2, gamma3

    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1):
        theta = arccos(gamma1[0])
        phi = arctan2(gamma1[2], gamma1[1])
        return theta, phi

    @staticmethod
    def gammas_to_spherical_coordinates(gamma1, gamma2):
        theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
        Ht = KentDistribution.create_matrix_Ht(theta, phi)
        u = MMul(Ht, reshape(gamma2, (3, 1)))
        psi = arctan2(u[2][0], u[1][0])
        return theta, phi, psi

    def __init__(self, gamma1, gamma2, gamma3, kappa, beta, eta=1.):
        self.gamma1 = array(gamma1, dtype=float64)
        self.gamma2 = array(gamma2, dtype=float64)
        self.gamma3 = array(gamma3, dtype=float64)
        self.kappa = float(kappa)
        self.beta = float(beta)
        # Bingham-Mardia, 4-param, small-circle distribution has eta=-1
        self.eta = eta

        self.theta, self.phi, self.psi = KentDistribution.gammas_to_spherical_coordinates(
            self.gamma1, self.gamma2)

        for gamma in gamma1, gamma2, gamma3:
            assert len(gamma) == 3

        self._cached_rvs = array([], dtype=float64)
        self._cached_rvs.shape = (0, 3)

        # save rvs used to calculated level contours to keep levels self-consistent
        self._level_log_pdf = array([], dtype=float64)

    @property
    def Gamma(self):
        return self.create_matrix_Gamma(self.theta, self.phi, self.psi)

    def normalize(self, cache=dict(), return_num_iterations=False):
        """
        Returns the normalization constant of the Kent distribution.
        The proportional error may be expected not to be greater than 
        1E-11.


        >>> gamma1 = array([1.0, 0.0, 0.0])
        >>> gamma2 = array([0.0, 1.0, 0.0])
        >>> gamma3 = array([0.0, 0.0, 1.0])
        >>> tiny = KentDistribution.minimum_value_for_kappa
        >>> abs(kent2(gamma1, gamma2, gamma3, tiny, 0.0).normalize() - 4*pi) < 4*pi*1E-12
        True
        >>> for kappa in [0.01, 0.1, 0.2, 0.5, 2, 4, 8, 16]:
        ...     print abs(kent2(gamma1, gamma2, gamma3, kappa, 0.0).normalize() - 4*pi*sinh(kappa)/kappa) < 1E-15*4*pi*sinh(kappa)/kappa,
        ... 
        True True True True True True True True
        """
        k, b, m = self.kappa, self.beta, self.eta
        if not (k, b, m) in cache:
            G = gamma_fun
            LG = gammaln_fun
            I = modified_bessel_1stkind
            result = 0.0
            j = 0
            if b == 0.0:
                result = (
                    ((0.5 * k)**(-2 * j - 0.5)) *
                    (I(2 * j + 0.5, k))
                )
                result /= G(j + 1)
                result *= G(j + 0.5)

            else:
                while True:
                    # int sin(theta) dtheta
                    a = (
                        exp(
                            log(b) * j +
                            log(0.5 * k) * (-j - 0.5)
                        ) * I(j + 0.5, k)
                    )
                    # int dphi
                    irange = arange(j + 1)
                    aj = ((-m)**irange * exp(LG(irange + 0.5) + LG(j -
                                                                   irange + 0.5) - LG(irange + 1) - LG(j - irange + 1))).sum()
                    a /= sqrt(pi)
                    a *= aj
                    result += a

                    j += 1
                    if (j % 2 and abs(a) < abs(result) * 1E-12 and j > 5):
                        break

            cache[k, b, m] = 2 * pi * result
        if return_num_iterations:
            return cache[k, b, m], j
        else:
            return cache[k, b, m]

    def log_normalize(self):
        """
        Returns the logarithm of the normalization constant.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                lnormalize = log(self.normalize())
            except (OverflowError, RuntimeWarning) as e:
                k = self.kappa
                b = self.beta
                m = self.eta
                I = modified_bessel_1stkind
                if k < 0 or b < 0:
                    lnormalize = 1e10
                elif k > 2 * b:
                    lnormalize = log(2 * pi) + k - \
                        log((k - 2 * b) * (k + 2 * b)) / 2.
                else:
                    # c = sqrt(pi/b)*4*pi/exp(-b*(1+(k/2*b)**2))
                    # this is the approximation in Bingham-Mardia (1978), converting F to c, where c is the normalization in the Kent paper, with a correction factor for floating eta
                    lnormalize = (
                        0.5 * (log(pi) - log(b)) + log(4 * pi) + b * (1 + (k / (2 * b))**2) +
                        log(I(0, 0.5 * (1 + m) * pi * b)) -
                        0.5 * (1 + m) * pi * b
                    )
            return lnormalize

    def max(self):
        if self.beta == 0.0:
            x1 = 1
        else:
            x1 = self.kappa / (2 * self.beta)
        if x1 > 1:
            x1 = 1
        x2 = sqrt(1 - x1**2)
        x3 = 0
        x = dot(self.Gamma, asarray((x1, x2, x3)))
        return KentDistribution.gamma1_to_spherical_coordinates(x)

    def pdf_max(self, normalize=True):
        return exp(self.log_pdf_max(normalize))

    def log_pdf_max(self, normalize=True):
        """
        Returns the maximum value of the log(pdf)
        """
        if self.beta == 0.0:
            x = 1
        else:
            x = self.kappa * 1.0 / (2 * self.beta)
        if x > 1.0:
            x = 1
        fmax = self.kappa * x + self.beta * (1 - x**2)
        if normalize:
            return fmax - self.log_normalize()
        else:
            return fmax

    def pdf(self, xs, normalize=True):
        """
        Returns the pdf of the kent distribution for 3D vectors that
        are stored in xs which must be an array of N x 3 or N x M x 3
        N x M x P x 3 etc.

        The code below shows how points in the pdf can be evaluated. An integral is
        calculated using random points on the sphere to determine wether the pdf is
        properly normalized.

        >>> from numpy.random import seed
        >>> from scipy.stats import norm as gauss
        >>> seed(666)
        >>> num_samples = 400000
        >>> xs = gauss(0, 1).rvs((num_samples, 3))
        >>> xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
        >>> assert abs(4*pi*average(kent(1.0, 1.0, 1.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 4.0,  8.0).pdf(xs)) - 1.0) < 0.01
        >>> assert abs(4*pi*average(kent(1.0, 2.0, 3.0, 16.0, 8.0).pdf(xs)) - 1.0) < 0.01
        """
        return exp(self.log_pdf(xs, normalize))

    def log_pdf(self, xs, normalize=True):
        """
        Returns the log(pdf) of the kent distribution.
        """
        axis = len(shape(xs)) - 1
        g1x = sum(self.gamma1 * xs, axis)
        g2x = sum(self.gamma2 * xs, axis)
        g3x = sum(self.gamma3 * xs, axis)
        k, b, m = self.kappa, self.beta, self.eta

        f = k * g1x + b * (g2x**2 - m * g3x**2)
        if normalize:
            return f - self.log_normalize()
        else:
            return f

    def log_likelihood(self, xs):
        """
        Returns the log likelihood for xs.
        """
        retval = self.log_pdf(xs)
        return sum(retval, len(shape(retval)) - 1)

    def _rvs_helper(self):
        num_samples = 10000
        xs = gauss(0, 1).rvs((num_samples, 3))
        xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
        lpvalues = self.log_pdf(xs, normalize=False)
        lfmax = self.log_pdf_max(normalize=False)
        shifted = lpvalues - lfmax
        return xs[uniform(0, 1).rvs(num_samples) < exp(shifted)]

    def rvs(self, n_samples=None):
        """
        Returns random samples from the Kent distribution by rejection sampling. 
        May become inefficient for large kappas.

        The returned random samples are 3D unit vectors.
        If n_samples == None then a single sample x is returned with shape (3,)
        If n_samples is an integer value N then N samples are returned in an array with shape (N, 3)
        """
        num_samples = 1 if n_samples == None else n_samples
        rvs = self._cached_rvs
        while len(rvs) < num_samples:
            new_rvs = self._rvs_helper()
            rvs = concatenate([rvs, new_rvs])
        if n_samples == None:
            self._cached_rvs = rvs[1:]
            return rvs[0]
        else:
            self._cached_rvs = rvs[num_samples:]
            retval = rvs[:num_samples]
            return retval

    def level(self, percentile=50, n_samples=10000):
        """
        Returns the -log_pdf level at percentile by generating a set of rvs and their log_pdfs
        """
        if 0 <= percentile < 100:
            curr_len = self._level_log_pdf.size
            if curr_len < n_samples:
                new_rvs = self.rvs(n_samples)
                self._level_log_pdf = -self.log_pdf(new_rvs)
                self._level_log_pdf.sort()
            log_pdf = self._level_log_pdf
            loc = (log_pdf.size - 1) * percentile / 100.
            idx, frac = int(loc), loc - int(loc)
            return log_pdf[idx] + frac * (log_pdf[idx + 1] - log_pdf[idx])
        else:
            print '{} percentile out of bounds'.format(percentile)
            return nan

    def contour(self, percentile=50):
        """
        Returns the (spherical) coordinates that correspond to a contour percentile

        Solution is based on Eq 1.4 (Kent 1982)
        """
        k = self.kappa
        b = self.beta
        m = self.eta
        ln = self.log_normalize()
        lev = self.level(percentile)

        # work in coordinate system x' = Gamma'*x
        # range over which x1 remains real is [-x2_max, x2_max]
        # assume -1 < m < 1
        x2_max = min(abs(sqrt(k**2 + 4 * b * m * (b * m - lev + ln)
                              ) / (2 * sqrt(m * (1 + m)) * b)), 1)
        if isnan(x2_max):
            x2_max = 1
        x2 = linspace(-x2_max, x2_max, 10000)
        if abs(m) < 1E-4:
            x1_0 = -(lev - ln + b * x2**2) / k
            x1_1 = x1_0
        else:
            x1_0 = (-k + sqrt(k**2 + 4 * m * b * (m * b - lev +
                                                  ln - (1 + m) * b * x2**2))) / (2 * b * m)
            x1_1 = (-k - sqrt(k**2 + 4 * m * b * (m * b - lev +
                                                  ln - (1 + m) * b * x2**2))) / (2 * b * m)
        x3_0 = sqrt(1 - x1_0**2 - x2**2)
        x3_1 = sqrt(1 - x1_1**2 - x2**2)
        x2 = concatenate([x2, -x2, x2, -x2])
        x1 = concatenate([x1_0, x1_0, x1_1, x1_1])
        x3 = concatenate([x3_0, -x3_0, x3_1, -x3_1])

        # Since Kent distribution is well-defined for points not on a sphere,
        # possible solutions for L=-log_pdf(kent) extend beyond surface of
        # sphere. For the contour evaluation, only use points that lie on sphere.
        ok = x1**2 + x2**2 <= 1
        x123 = asarray((x1[ok], x2[ok], x3[ok]))

        # rotate back into x coordinates
        x = dot(self.Gamma, x123)
        return KentDistribution.gamma1_to_spherical_coordinates(x)

    def __repr__(self):
        return "kent(%s, %s, %s, %s, %s, %s)" % (self.theta, self.phi, self.psi, self.kappa, self.beta, self.eta)


def kent_me(xs):
    """Generates and returns a KentDistribution based on a moment estimation."""
    lenxs = len(xs)
    xbar = average(xs, 0)  # average direction of samples from origin
    # dispersion (or covariance) matrix around origin
    S = average(xs.reshape((lenxs, 3, 1)) * xs.reshape((lenxs, 1, 3)), 0)
    # has unit length and is in the same direction and parallel to xbar
    gamma1 = xbar / norm(xbar)
    theta, phi = KentDistribution.gamma1_to_spherical_coordinates(gamma1)

    H = KentDistribution.create_matrix_H(theta, phi)
    Ht = KentDistribution.create_matrix_Ht(theta, phi)
    B = MMul(Ht, MMul(S, H))

    eigvals, eigvects = eig(B[1:, 1:])
    eigvals = real(eigvals)
    if eigvals[0] < eigvals[1]:
        eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
        eigvects = eigvects[:, ::-1]
    K = diag([1.0, 1.0, 1.0])
    K[1:, 1:] = eigvects

    G = MMul(H, K)
    Gt = transpose(G)
    T = MMul(Gt, MMul(S, G))

    r1 = norm(xbar)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33

    # kappa and beta can be estimated but may not lie outside their permitted ranges
    min_kappa = KentDistribution.minimum_value_for_kappa
    kappa = max(min_kappa, 1.0 / (2.0 - 2.0 * r1 - r2) +
                1.0 / (2.0 - 2.0 * r1 + r2))
    beta = 0.5 * (1.0 / (2.0 - 2.0 * r1 - r2) - 1.0 / (2.0 - 2.0 * r1 + r2))

    return kent4(G, kappa, beta)


def __kent_mle_output1(k_me, callback):
    print
    print "******** Maximum Likelihood Estimation ********"
    print "Initial moment estimates are:"
    print "theta =", k_me.theta
    print "phi   =", k_me.phi
    print "psi   =", k_me.psi
    print "kappa =", k_me.kappa
    print "beta  =", k_me.beta
    print "eta   =", k_me.eta
    print "******** Starting the Gradient Descent ********"
    print "[iteration]   theta   phi   psi   kappa   beta   eta   -L"


def __kent_mle_output2(x, minusL, output_count, verbose):
    interval = verbose if isinstance(verbose, int) else 1
    str_values = list()
    for value in (tuple(x) + (minusL,)):
        str_value = "%- 8g" % value
        while len(str_value) < 12:
            str_value += " "
        str_values.append(str_value)
    if output_count[0] % interval == 0:
        print ("[%3i]       " + " %s" * 3) % tuple(output_count + str_values)
    output_count[0] = output_count[0] + 1


def kent_mle(xs, verbose=False, return_intermediate_values=False, warning='warn'):
    """
    Generates a KentDistribution fitted to xs using maximum likelihood estimation
    For a first approximation kent_me() is used. The function 
    -k.log_likelihood(xs)/len(xs) (where k is an instance of KentDistribution) is 
    minimized.

    Input:
      - xs: values on the sphere to be fitted by MLE
      - verbose: if True, output is given for every step
      - return_intermediate_values: if true the values of all intermediate steps
        are returned as well
      - warning: choices are 
        - "warn": issues any warning via warning.warn
        - a file object: which results in any warning message being written to a file 
          (e.g. stdout) 
        - "none": or any other value for this argument results in no warnings to be issued
    Output:
      - an instance of the fitted KentDistribution
    Extra output:
      - if return_intermediate_values is specified then
      a tuple is returned with the KentDistribution argument as the first element
      and containing the extra requested values in the rest of the elements.
    """
    # method that generates the minus L to be minimized
    def minus_log_likelihood(x):
        return -kent(*x).log_likelihood(xs) / len(xs)

    # callback for keeping track of the values
    intermediate_values = list()

    def callback(x, output_count=[0]):
        kx = kent(*x)
        minusL = -kx.log_likelihood(xs)
        imv = intermediate_values
        imv.append((x, minusL))
        if verbose:
            print len(imv), kx.theta, kx.phi, kx.psi, kx.kappa, kx.beta, kx.eta, minusL

    # first get estimated moments
    # these don't depend on BM4
    k_me = kent_me(xs)
    theta, phi, psi, kappa, beta, eta = k_me.theta, k_me.phi, k_me.psi, k_me.kappa, k_me.beta, k_me.eta
    min_kappa = KentDistribution.minimum_value_for_kappa

    # here the mle is done
    # starting parameters (small value is subtracted from kappa and add in generatke k)
    x_start = array([theta, phi, psi, kappa, beta])
    y_start = array([theta, phi, psi, beta, kappa, -0.99])
    if verbose:
        __kent_mle_output1(k_me, callback)

    # constrain kappa, beta >= 0 and 2*beta <= kappa for FB5 (Kent 1982)
    cons = ({"type": "ineq",
             "fun": lambda x: x[3] - 2 * x[4]},
            {"type": "ineq",
             "fun": lambda x: x[3]},
            {"type": "ineq",
             "fun": lambda x: x[4]})
    _x = minimize(minus_log_likelihood,
                  x_start,
                  method="SLSQP",
                  constraints=cons,
                  callback=callback,
                  options={"disp": False, "eps": 1e-08,
                           "maxiter": 100, "ftol": 1e-08})
    if verbose:
        __kent_mle_output1(k_me, callback)
    # constrain kappa, beta >= 0 while allowing eta to float to allow for multiple modes
    # note eta=-1 with 2*beta >= kappa is the small-circle distribution (Bingham-Mardia 1978)
    cons = ({"type": "ineq",
             "fun": lambda x: x[3]},
            {"type": "ineq",
             "fun": lambda x: x[4]},
            {"type": "ineq",
             "fun": lambda x: 1 - abs(x[-1])})
    _y = minimize(minus_log_likelihood,
                  y_start,
                  method="SLSQP",
                  constraints=cons,
                  callback=callback,
                  options={"disp": False, "eps": 1e-08,
                           "maxiter": 100, "ftol": 1e-08})

    all_values = _x if _x.fun < _y.fun else _y
    warnflag = all_values.status
    if not all_values.success:
        warning_message = all_values.message
        if warning == "warn":
            warnings.warn(warning_message, RuntimeWarning)
        if hasattr(warning, "write"):
            warning.write("Warning: " + warning_message + "\n")

    k = (kent(*all_values.x),)
    if return_intermediate_values:
        k += (intermediate_values,)
    if len(k) == 1:
        k = k[0]
    return k


if __name__ == "__main__":
    __doc__ += """
>>> from sphere.example import test_example_normalization, test_example_mle, test_example_mle2
>>> from numpy.random import seed
>>> test_example_normalization(gridsize=10)
Calculating the matrix M_ij of values that can be calculated: kappa=100.0*i+1, beta=100.0+j*1
Calculating normalization factor for combinations of kappa and beta: 
Iterations necessary to calculate normalize(kappa, beta):
8   x   x   x   x   x   x   x   x   x
6  81 150 172 172   x   x   x   x   x
6  51 134 172 172 172 172 172 172   x
6  27 108 172 172 172 172 172 172   x
6  19  71 159 172 172 172 172   x   x
6  15  42 126 172 172 172   x   x   x
6  13  29  86 172 172   x   x   x   x
6  12  23  55 141   x   x   x   x   x
x   x   x   x   x   x   x   x   x   x
x   x   x   x   x   x   x   x   x   x
>>> seed(888)
>>> test_example_mle()
Original Distribution: k = kent(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(0.007720841890575154, -1.5740824920888568, -1.4375584223239652, 1.43492053209, 0.0044675770317, 1.0)
Fitted with MLE:   k_mle = kent(0.007459189777403647, -1.5740811153814367, -1.437558656621312, 0.964197999109, 0.0354161152778, 1.0)
Original Distribution: k = kent(0.75, 2.391592653589793, 2.3915926535897936, 20.0, 0.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(0.7474012488747638, 2.3956473758798102, -1.5107404958092123, 20.1062024974, 0.158640323669, 1.0)
Fitted with MLE:   k_mle = kent(0.7474018360417302, 2.3956506602230427, -1.5113301600900508, 20.3358029783, 0.316186752336, 0.1951429860460265)
Original Distribution: k = kent(0.7853981633974483, 2.356194490192345, -2.827433388230814, 20.0, 2.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(0.7830806816854289, 2.3566994796348273, 0.30374453765385184, 20.2392003233, 1.75374534349, 1.0)
Fitted with MLE:   k_mle = kent(0.7829809342195447, 2.3567108200918363, 0.3038350185089338, 20.2872562797, 2.05942839445, 1.0)
Original Distribution: k = kent(0.7853981633974483, 2.356194490192345, -2.945243112740431, 20.0, 5.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(0.7878716936286756, 2.358370795028891, 0.17634802796746255, 19.5541770417, 3.8469672309, 1.0)
Fitted with MLE:   k_mle = kent(0.7880385025649629, 2.358491916826943, 0.17625765017119674, 19.9372091782, 4.77383865091, 1.0)
Original Distribution: k = kent(1.0995574287564276, 2.356194490192345, -3.043417883165112, 50.0, 25.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(1.102718086827844, 2.3558887019102808, 0.09774818228688721, 37.5233247341, 14.8542512779, 1.0)
Fitted with MLE:   k_mle = kent(1.1017455790512998, 2.3558270872470137, 0.09777101616795454, 50.1918525143, 24.9807555374, 1.0)
Original Distribution: k = kent(0.0, 0.0, 0.09817477042468103, 50.0, 25.0, 1.0)
Drawing 10000 samples from k
Moment estimation:  k_me = kent(0.004005040503695006, 0.2396182325912848, -0.14364867043005725, 37.7138963117, 14.9491168355, 1.0)
Fitted with MLE:   k_mle = kent(0.005098617598271794, 0.20858549213401353, -0.11261422537952466, 50.5466339851, 25.1819351371, 1.0)
>>> seed(2323)
>>> assert test_example_mle2(300)
Testing various combinations of kappa and beta for 300 samples.
MSE of MLE is higher than 0.7 times the moment estimate for beta/kappa <= 0.2
MSE of MLE is higher than moment estimate for beta/kappa >= 0.3
MSE of MLE is five times higher than moment estimates for beta/kappa >= 0.5

A test to ensure that the vectors gamma1 ... gamma3 are orthonormal
>>> for k in [
...   kent(0.0,      0.0,      0.0,    20.0, 0.0),
...   kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 0.0),
...   kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 5.0),
...   kent(0.0,      0.0,      0.5*pi, 10.0, 7.0),
...   kent(0.0,      0.0,      0.5*pi, 0.1,  0.0),
...   kent(0.0,      0.0,      0.5*pi, 0.1,  0.1),
...   kent(0.0,      0.0,      0.5*pi, 0.1,  8.0),
... ]:
...   assert(abs(sum(k.gamma1 * k.gamma2)) < 1E-14)
...   assert(abs(sum(k.gamma1 * k.gamma3)) < 1E-14)
...   assert(abs(sum(k.gamma3 * k.gamma2)) < 1E-14)
...   assert(abs(sum(k.gamma1 * k.gamma1) - 1.0) < 1E-14)
...   assert(abs(sum(k.gamma2 * k.gamma2) - 1.0) < 1E-14)
...   assert(abs(sum(k.gamma3 * k.gamma3) - 1.0) < 1E-14)

A test to ensure that the pdf() and the pdf_max() are calculated
correctly.
>>> from numpy.random import seed
>>> from scipy.stats import norm as gauss
>>> seed(666)
>>> for k, pdf_value in [
...   (kent(0.0,      0.0,      0.0,    20.0, 0.0), 3.18309886184),
...   (kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 0.0), 0.00909519370),
...   (kent(-0.25*pi, -0.25*pi, 0.0,    20.0, 5.0), 0.09865564569),
...   (kent(0.0,      0.0,      0.5*pi, 10.0, 7.0), 0.59668931662),
...   (kent(0.0,      0.0,      0.5*pi, 0.1,  0.0), 0.08780030026),
...   (kent(0.0,      0.0,      0.5*pi, 0.1,  0.1), 0.08768344462),
...   (kent(0.0,      0.0,      0.5*pi, 0.1,  8.0), 0.00063128997),
... ]:
...   assert abs(k.pdf(array([1.0, 0.0, 0.0])) - pdf_value) < 1E-8
...   assert abs(k.log_pdf(array([1.0, 0.0, 0.0])) - log(pdf_value)) < 1E-8
...   num_samples = 100000
...   xs = gauss(0, 1).rvs((num_samples, 3))
...   xs = divide(xs, reshape(norm(xs, 1), (num_samples, 1)))
...   values = k.pdf(xs, normalize=False)
...   fmax = k.pdf_max(normalize=False)
...   assert all(values <= fmax)
...   assert any(values > fmax*0.999)
...   values = k.pdf(xs)
...   fmax = k.pdf_max()
...   assert all(values <= fmax)
...   assert any(values > fmax*0.999)

These are tests to ensure that the coordinate transformations are done correctly
that the functions that generate instances of KentDistribution are consistent and
that the derivatives are calculated correctly. In addition some more orthogonality 
testing is done.

>>> from distribution import *
>>> from numpy.random import seed
>>> from scipy.stats import uniform 
>>> def test_orth(k):
...   # a bit more orthonormality testing for good measure 
...   assert(abs(sum(k.gamma1 * k.gamma2)) < 1E-14)
...   assert(abs(sum(k.gamma1 * k.gamma3)) < 1E-14)
...   assert(abs(sum(k.gamma3 * k.gamma2)) < 1E-14)
...   assert(abs(sum(k.gamma1 * k.gamma1) - 1.0) < 1E-14)
...   assert(abs(sum(k.gamma2 * k.gamma2) - 1.0) < 1E-14)
...   assert(abs(sum(k.gamma3 * k.gamma3) - 1.0) < 1E-14)
... 
>>> # generating some specific boundary values and some random values
>>> seed(666)
>>> upi, u2pi = uniform(0, pi), uniform(-pi, 2*pi)
>>> thetas, phis, psis = list(upi.rvs(925)), list(u2pi.rvs(925)), list(u2pi.rvs(925))
>>> for a in (0.0, 0.5*pi, pi):
...   for b in (-pi, -0.5*pi, 0, 0.5*pi, pi):
...     for c in (-pi, -0.5*pi, 0, 0.5*pi, pi):
...       thetas.append(a)
...       phis.append(b)
...       psis.append(c)
... 
>>> # testing consintency of angles (specifically kent())
>>> for theta, phi, psi in zip(thetas, phis, psis):
...   k = kent(theta, phi, psi, 1.0, 1.0)
...   assert abs(theta - k.theta) < 1E-12
...   a = abs(phi - k.phi)
...   b = abs(psi - k.psi)
...   if theta != 0 and theta != pi:
...     assert a < 1E-12 or abs(a-2*pi) < 1E-12
...     assert b < 1E-12 or abs(b-2*pi) < 1E-12
...   test_orth(k)
... 
>>> # testing consistency of gammas and consistency of back and forth
>>> # calculations between gammas and angles (specifically kent2(), kent3() and kent4())
>>> kappas = gauss(0, 2).rvs(1000)**2
>>> betas = gauss(0, 2).rvs(1000)**2
>>> for theta, phi, psi, kappa, beta in zip(thetas, phis, psis, kappas, betas):
...   gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
...   theta, phi, psi = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)
...   gamma1a, gamma2a, gamma3a = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
...   assert all(abs(gamma1a-gamma1) < 1E-12) 
...   assert all(abs(gamma2a-gamma2) < 1E-12) 
...   assert all(abs(gamma3a-gamma3) < 1E-12) 
...   k2 = kent2(gamma1, gamma2, gamma3, kappa, beta)
...   assert all(abs(gamma1 - k2.gamma1) < 1E-12) 
...   assert all(abs(gamma2 - k2.gamma2) < 1E-12) 
...   assert all(abs(gamma3 - k2.gamma3) < 1E-12)
...   A = gamma1*kappa
...   B = gamma2*beta
...   k3 = kent3(A, B)
...   assert all(abs(gamma1 - k3.gamma1) < 1E-12) 
...   assert all(abs(gamma2 - k3.gamma2) < 1E-12) 
...   assert all(abs(gamma3 - k3.gamma3) < 1E-12)
...   test_orth(k)
...   gamma = array([
...     [gamma1[0], gamma2[0], gamma3[0]], 
...     [gamma1[1], gamma2[1], gamma3[1]], 
...     [gamma1[2], gamma2[2], gamma3[2]], 
...   ])
...   k4 = kent4(gamma, kappa, beta)
...   assert all(k2.gamma1 == k4.gamma1) 
...   assert all(k2.gamma2 == k4.gamma2) 
...   assert all(k2.gamma3 == k4.gamma3) 
... 
>>> # testing special case for B with zero length (kent3())
>>> for theta, phi, psi, kappa, beta in zip(thetas, phis, psis, kappas, betas):
...   gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(theta, phi, psi)
...   A = gamma1*kappa
...   B = gamma2*0.0
...   k = kent3(A, B)
...   assert all(abs(gamma1 - k.gamma1) < 1E-12) 
...   test_orth(k)
...  
"""

    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
