#!/usr/bin/env python
"""
The example code below shows various examples of the Fb8 distribution
some random samples are drawn from the distribution and new estimates
are generated using a moment estimation and a maximum likelihood fit.
Plots will be shown unless this script is called with the --no-plots
option.
"""

from __future__ import print_function
import numpy as np
import sphere.distribution
from sphere.distribution import fb8
import warnings
import sys


def test_example_normalization(showplots=False, verbose=False, gridsize=100, print_grid=True,
                               eta=1., alpha=0., rho=0.):
    scale = (1000.0 / gridsize)
    print("Calculating the matrix M_ij of values that can be calculated: kappa=%.1f*i+1, beta=%.1f*j+1" %
          (scale, scale))
    print(f"with eta={eta:.1f}, alpha={alpha:.1f}, rho={rho:.1f}")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c_grid = np.zeros((gridsize, gridsize)) - 1.0
        cnum_grid = np.zeros((gridsize, gridsize), dtype=np.int32) - 1
        print("Calculating normalization factor for combinations of kappa and beta:", end='')
        for i in range(gridsize):
            if verbose:
                print(f"{i * gridsize}/{gridsize * gridsize} ", end=' ', flush=True)
            kappa = scale * i + 1.0
            for j in range(gridsize):
                beta = scale * j + 1.0
                f = sphere.distribution.fb8(0.0, 0.0, 0.0, kappa, beta, eta, alpha, rho)
                try:
                    c, cnum = f.normalize(return_num_iterations=True)
                    c_grid[i, j] = np.log(c)
                    cnum_grid[i, j] = cnum
                except (OverflowError, RuntimeWarning):
                    pass
        if showplots:
            from matplotlib.pyplot import figure
            from matplotlib.ticker import FuncFormatter
            for name, grid in zip(
                [
                    r"$\mathrm{Calculated\ values\ of\ }c(\kappa,\beta)$",
                    r"$\mathrm{Iterations\ necessary\ to\ calculate\ }c(\kappa,\beta)$",
                ],
                [c_grid, cnum_grid]
            ):
                f = figure()
                ax = f.add_subplot(111)
                cb = ax.imshow(grid, interpolation="nearest")
                f.colorbar(cb)
                ax.set_title(name + r" $(-1=\mathrm{overflow}$)")
                ax.xaxis.set_major_formatter(FuncFormatter(lambda tv, tp: str(int(tv * scale + 1))))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda tv, tp: str(int(tv * scale + 1))))
                ax.set_ylabel(r"$\kappa$")
                ax.set_xlabel(r"$\beta$")
    print()
    if print_grid:
        for message, grid in [
            ("Iterations necessary to calculate normalize(kappa, beta):", cnum_grid),
        ]:
            print(message)
            for i, line in enumerate(grid):
                print(" ".join(['  x' if n == -1 else '%3i' % n for n in line]))


def test_example_mle(showplots=False, verbose=False, seed=3):
    def similar(a, b):
        for attr in 'theta phi psi kappa beta eta alpha rho'.split():
            if abs((a.__getattribute__(attr) - b.__getattribute__(attr))) > 0.02:
                return False
        return True

    # expected for seed=3
    expected_mes = [fb8(0.01, -2.36, -2.12, 1.45, 0.00, 1.00, 0.00, 0.00),
                    fb8(0.75, 2.39, -1.91, 20.25, 0.17, 1.00, 0.00, 0.00),
                    fb8(0.79, 2.36, 0.34, 20.16, 1.63, 1.00, 0.00, 0.00),
                    fb8(0.79, 2.36, 0.19, 19.81, 4.03, 1.00, 0.00, 0.00),
                    fb8(1.10, 2.35, 0.10, 37.29, 14.81, 1.00, 0.00, 0.00),
                    fb8(0.00, 0.30, -0.20, 37.84, 15.07, 1.00, 0.00, 0.00)]

    expected_mles = [fb8(0.01, -2.36, -2.12, 0.98, 0.04, 1.00, 0.00, 0.00),
                     fb8(0.75, 2.39, -1.91, 20.25, 0.20, 1.00, 0.00, 0.00),
                     fb8(0.79, 2.36, 0.34, 20.20, 1.91, 1.00, 0.00, 0.00),
                     fb8(0.79, 2.36, 0.19, 20.23, 5.01, 1.00, 0.00, 0.00),
                     fb8(1.10, 2.36, 0.10, 50.31, 25.16, 1.00, 0.00, 0.00),
                     fb8(0.00, 0.77, -0.67, 51.11, 25.56, 1.00, 0.00, 0.00)]

    for idx, k in enumerate([
        fb8(0.0, 0.0, 0.0, 1.0, 0.0),
        fb8(-0.75, -0.75, -0.75, 20.0, 0.0),
        fb8(-0.25 * np.pi, -0.25 * np.pi, np.pi / 10, 20.0, 2.0),
        fb8(-0.25 * np.pi, -0.25 * np.pi, np.pi / 16, 20.0, 5.0),
        fb8(-0.35 * np.pi, -0.25 * np.pi, np.pi / 32, 50.0, 25.0),
        fb8(0.0, 0.0, np.pi / 32, 50.0, 25.0),
    ]):
        gridsize = 200
        u = np.linspace(0, 2 * np.pi, gridsize)
        v = np.linspace(0, np.pi, gridsize)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        keys = list()
        points = list()
        for i in range(gridsize):
            for j in range(gridsize):
                points.append([x[i, j], y[i, j], z[i, j]])
                keys.append((i, j))
        points = np.array(points)

        xs = k.rvs(10000, seed)
        k_me = sphere.distribution.kent_me(xs)
        k_mle = sphere.distribution.fb8_mle(xs, warning=sys.stdout, fb5_only=True)
        if seed == 3 and sys.platform == "linux":
            assert similar(k_me, expected_mes[idx])
            assert similar(k_mle, expected_mles[idx])
        assert k_me.log_likelihood(xs) < k_mle.log_likelihood(xs)
        if verbose:
            print(f"Original Distribution: k = {k}")
            print(f"Moment estimation:  k_me = {k_me}")
            print(f"Fitted with MLE:   k_mle = {k_mle}")
            print("***")

        value_for_color = k_mle.pdf(points)
        value_for_color /= max(value_for_color)
        colors = np.empty((gridsize, gridsize), dtype=tuple)
        for (i, j), v in zip(keys, value_for_color):
            colors[i, j] = (1.0 - v, 1.0 - v, 1.0, 1.0)

        if showplots:
            import matplotlib.pyplot as plt

            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
            xx, yy, zz = list(zip(*xs[:100]))  # plot only a portion of these values
            ax.scatter(1.05 * np.array(xx), 1.05 *
                       np.array(yy), 1.05 * np.array(zz), c='b')
            ax.plot_surface(x, y, z, rstride=4, cstride=4,
                            facecolors=colors, linewidth=0)
            values_t = r"$\theta=%.2f^\circ,\ \phi=%.2f^\circ,\ \psi=%.2f^\circ,\ \kappa=%.3f,\ \beta=%.3f$"
            f.text(0.12, 0.99 - 0.025, r"$\mathrm{Original\ Values:}$")
            f.text(0.12, 0.99 - 0.055, r"$\mathrm{Moment\ estimates:}$")
            f.text(0.12, 0.99 - 0.080, r"$\mathrm{MLE\ (shown):}$")
            f.text(0.30, 0.99 - 0.025, values_t % (k.theta * 180 / np.pi,
                                                   k.phi * 180 / np.pi, k.psi * 180 / np.pi, k.kappa, k.beta))
            f.text(0.30, 0.99 - 0.055, values_t % (k_me.theta * 180 / np.pi,
                                                   k_me.phi * 180 / np.pi, k_me.psi * 180 / np.pi, k_me.kappa, k_me.beta))
            f.text(0.30, 0.99 - 0.080, values_t % (k_mle.theta * 180 / np.pi,
                                                   k_mle.phi * 180 / np.pi, k_mle.psi * 180 / np.pi, k_mle.kappa, k_mle.beta))
            ax.set_xlabel(r"$Q\rightarrow$")
            ax.set_ylabel(r"$U\rightarrow$")
            ax.set_zlabel(r"$V\rightarrow$")


def calculate_bias_var_and_mse(x, y):
    bias = np.average(x) - np.average(y)
    variance = np.var(x - y)
    mse = np.average((x - y)**2)
    assert abs((bias**2 + variance) - mse) < 1E-12 * (bias**2 + variance + mse)
    return bias, variance, mse


def test_example_mle2(num_samples, showplots=False, verbose=False, stepsize=1.0):
    rng = np.random.default_rng(3)
    max_kappa = 50.0
    real_kappas = np.arange(1.0, max_kappa, stepsize)
    print("Testing various combinations of kappa and beta for", num_samples, "samples.")
    bias_var_mse_kappa_me, bias_var_mse_kappa_mle, bias_var_mse_beta_me, bias_var_mse_beta_mle = [
        list() for i in range(4)]
    beta_ratios = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    for beta_ratio in beta_ratios:
        real_betas = beta_ratio * real_kappas
        kappas_me, kappas_mle, betas_me, betas_mle = [
            list() for i in range(4)]
        if verbose:
            print(f"beta (max 2.0) = {beta_ratio}*kappa : kappa (max {max_kappa:.1f}) = ", end=' ')
        for kappa in real_kappas:
            if verbose:
                print(f"{kappa:.1f}", end=' ', flush=True)
            beta = kappa * beta_ratio
            k = sphere.distribution.fb8(rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi),
                                        rng.uniform(0, 2 * np.pi), kappa, beta)
            samples = k.rvs(num_samples, rng)
            k_me = sphere.distribution.kent_me(samples)
            k_mle = sphere.distribution.fb8_mle(samples, warning=sys.stdout, fb5_only=True, verbose=verbose)
            assert k_me.log_likelihood(samples) - k_mle.log_likelihood(samples) < 1e-5
            kappas_me.append(k_me.kappa)
            betas_me.append(k_me.beta)
            kappas_mle.append(k_mle.kappa)
            betas_mle.append(k_mle.beta)
        bias_var_mse_kappa_me.append(
            calculate_bias_var_and_mse(real_kappas, kappas_me))
        bias_var_mse_beta_me.append(
            calculate_bias_var_and_mse(real_betas, betas_me))
        bias_var_mse_kappa_mle.append(
            calculate_bias_var_and_mse(real_kappas, kappas_mle))
        bias_var_mse_beta_mle.append(
            calculate_bias_var_and_mse(real_betas, betas_mle))
        if verbose:
            print()
        if showplots:
            from matplotlib.pyplot import figure
            f = figure(figsize=(12.0, 5))
            ax = f.add_subplot(121)
            ax.plot(real_kappas, kappas_me,
                    label=r"$\kappa\ \mathrm{Moment\ Est.}$")
            ax.plot(real_kappas, kappas_mle, label=r"$\kappa\ \mathrm{MLE}$")
            ax.plot(real_kappas, real_kappas, '-.k')
            ax.set_xlabel(r"$\mathrm{Real}\ \kappa$")
            ax.set_xlabel(r"$\mathrm{Fitted}\ \kappa$")
            ax.set_title(r"$\mathrm{Number of Samples = %s}$" % num_samples)
            ax.legend()
            ax = f.add_subplot(122)
            ax.plot(real_betas, betas_me,
                    label=r"$\beta\ \mathrm{Moment\ Est.}$")
            ax.plot(real_betas, betas_mle, label=r"$\beta\ \mathrm{MLE}$")
            ax.plot(real_betas, real_betas, '-.k')
            ax.set_xlabel(r"$\mathrm{Real}\ \beta$")
            ax.set_xlabel(r"$\mathrm{Fitted}\ \beta$")
            ax.set_title(r"$\beta = %.1f\kappa$" % beta_ratio)
            ax.legend()
    if showplots:
        for (name, bias_var_mse_mle, bias_var_mse_me) in [
            ("kappa", bias_var_mse_kappa_mle, bias_var_mse_kappa_me),
            ("beta", bias_var_mse_beta_mle, bias_var_mse_beta_me),
        ]:
            f = figure()
            ax = f.add_subplot(111)
            bias_me, var_me, mse_me = list(zip(*bias_var_mse_me))
            bias_mle, var_mle, mse_mle = list(zip(*bias_var_mse_mle))
            ax.plot(bias_me, bias_mle, label='bias')
            ax.plot(var_me, var_mle, label='var')
            ax.plot(mse_me, mse_mle, label='mse')
            for me, mle in [
                (bias_me, bias_mle),
                (var_me, var_mle),
                (mse_me, mse_mle),
            ]:
                for x, y, br in zip(me, mle, beta_ratios):
                    ax.text(x, y, f"{br:.2f}")
            ax.set_xlabel("Moment Estimate")
            ax.set_ylabel("MLE")
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            mxl = (min(xl[0], yl[0]), max(xl[1], yl[1]))
            ax.plot(mxl, mxl, '-.k')
            ax.set_xlim(mxl)
            ax.set_ylim(mxl)
            ax.set_title(r"$\%s$" % name)
            ax.legend()

    for (name, bias_var_mse_mle, bias_var_mse_me, beta_ratio) in [
        ("kappa", bias_var_mse_kappa_mle, bias_var_mse_kappa_me, beta_ratios),
        ("beta", bias_var_mse_beta_mle, bias_var_mse_beta_me, beta_ratios),
    ]:
        biass_me, vars_me, mses_me = list(zip(*bias_var_mse_me))
        biass_mle, vars_mle, mses_mle = list(zip(*bias_var_mse_mle))
        for mse_me, mse_mle, beta_ratio in zip(mses_me, mses_mle, beta_ratios):
            if mse_me < mse_mle * 0.65:
                print(f"MSE of moment estimate is lower than 0.65 times the MLE for {name}")
                return False
            if beta_ratio >= 0.3:
                if mse_me < mse_mle:
                    print(f"MSE of moment estimate is lower than MLE for {name} with beta/kappa >= 0.3")
                    return False
            if beta_ratio > 0.5:
                if mse_me < 5 * mse_mle:
                    print(f"MSE of moment estimate is lower than five times the MLE {name} with beta/kappa > 0.5")
                    return False

    print("MSE of ME is higher than 0.65 times the MLE for beta/kappa < 0.3")
    print("MSE of ME is higher than MLE for beta/kappa >= 0.3")
    print("MSE of ME is five times higher than MLE for beta/kappa > 0.5")
    return True


if __name__ == "__main__":
    from sys import argv
    showplots = False if len(argv) > 1 and '--no-plots' in argv[1:] else True

    if not (len(argv) > 1 and '--no-normalization' in argv[1:]):
        test_example_normalization(
            showplots=showplots, verbose=True, print_grid=False)
    if not (len(argv) > 1 and '--no-mle' in argv[1:]):
        test_example_mle(showplots=showplots)
    if not (len(argv) > 1 and '--no-mle2' in argv[1:]):
        test_example_mle2(300, showplots=showplots, verbose=True)
        test_example_mle2(10000, showplots=showplots, verbose=True)
    if showplots:
        from matplotlib.pyplot import show
        show()
    sys.exit()
