sphere
=================

Implments the FB8 distribution on a sphere, which is a generalization of the FB6, FB5 (Kent), and FB4 (Bingham-Mardia) distributions described below.

Implements the FB6 distribution that is first introduced in Rivest ([1984](https://www.doi.org/10.1214/aos/1176346724)).

Implements calculation of the density and fitting (using maximum likelihood estimate) of the Kent distribution based on Kent ([1982](https://doi.org/10.1111/j.2517-6161.1982.tb01189.x)). A unittest is performed if distribution.py is called from the command line.

Implements the Bingham-Mardia distribution whose mode is a small-circle on the sphere based on Bingham, Mardia ([1978](https://doi.org/10.1093/biomet/65.2.379)).

Also calculates directional, percentile contours which can be used to indicate the N% highest-posterior-density regions in the sky.

![maps](https://github.com/tianluyuan/sphere/blob/master/fig/example.png?raw=true)

Additional references
=================
Kent, Hussein, Jah, [_Directional distributions in tracking of space debris_](https://ieeexplore.ieee.org/abstract/document/7528139) 

Terdik, Jammalamadaka, Wainwright, [_Simulation and visualization of spherical distributions_](https://www.researchgate.net/profile/Gyorgy_Terdik/publication/324605982_Simulation_and_Visualization_of_Spherical_Distributions/links/5ad8edceaca272fdaf81fe04/Simulation-and-Visualization-of-Spherical-Distributions.pdf)

Mardia, Jupp, [_Directional statistics_](https://www.doi.org/10.1002/9780470316979)

Contributors
=================

This project was originally developed for the FB5 (Kent) distribution [here](https://github.com/edfraenkel/kent_distribution).

_Tianlu Yuan_
