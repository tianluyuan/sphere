13 May 2021 v1.1.1
-----------------
Patch an edge case in `FB8Distribution.contour` calculation when `beta` is 0.

9 May 2021 v1.1.0
-----------------
Faster implementation of the FB8 normalization calculation using `heapq`. The idea is to first run an increasingly coarser grid search over the indices to find the approximate-maximum summand, then start with a 3D cube around that point. Contributions from its six sides and their next-step coordinates are placed in a heap such that the next-largest contribution is summed next.

27 Oct 2020 v1.0.1
------------------
Optimize series summations by reducing repeated special function calls. Fix a bug and catch an edge.

26 Oct 2020 v1.0.0
------------------
Implement gradient calculations for mle fitting. This includes series computation of derivatives of the FB8 normalization wrt its parameters.

24 Oct 2020 v0.3.0
------------------
Minor optimizations by switching to `np.mgrid` and `np.moveaxis`.

10 Oct 2020 v0.2.3
------------------
Run Travis.ci tests for multiple python versions. Updates and additions to example scripts in `paper/fig.py`

29 Nov 2019 v0.2.1
------------------
Python 3 support.

19 Jun 2019 v0.2.0
------------------
Implement series calculation for FB8 normalization and make that the default. Fall back is numerical integration. A closed form approximation for the FB6 normalization constant is also implemented, which allows for computing the log-pdf for large values of kappa and beta.

12 May 2019 v0.1.0
------------------
First working version with FB8 distribution implemented. Its normalization constant is computed using numerical integration.
