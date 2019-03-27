sphere
=================

Implements calculation of the density and fitting (using maximum likelihood estimate) of the Kent distribution based on Kent ([1982](https://doi.org/10.1111/j.2517-6161.1982.tb01189.x)). A unittest is performed if distribution.py is called from the command line.

Implements the Bingham-Mardia distribution whose mode is a small-circle on the sphere based on Bingham, Mardia ([1978](https://doi.org/10.1093/biomet/65.2.379)).

Implements a mixture model that allows for an additional parameter to tune between FB5 and BM4.

Also calculates directional, percentile contours which can be used to indicate the N% highest-posterior-density regions in the sky.

![maps](/fig/example.png?raw=true)

This project was originally developed for only the Kent distribution [here](https://github.com/edfraenkel/kent_distribution).
