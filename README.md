# Channel capacity with Monte Carlo integration (chancapmc)
This module computes channel capacity between a discrete input random variable *Q* and a multivariate, continuous output random variable **Y**, using the Blahut-Arimoto and Monte Carlo integration for the multivariate integrals involved. Currently, only multivariate normal conditional distributions P(**Y**|*Q*) are supported for the output variable, but the module could easily be extended to other distributions. It was developed for the paper
> Achar, S., Bourassa, F.X.P., Rademaker, T.J., ..., François, P. and Altan-Bonnet, G. "Universal antigen encoding of T cell activation from high dimensional cytokine dynamics", under review, 2021.

The algorithm is coded in pure C and wrapped as a Python C extension; the integration is not currently perfect (e.g., the stdout is not always properly redirected), but it allows one to import and run the algorithm from Python with NumPy arrays as inputs.

## Requirements
- Python 3 (tested on Python 3.7.6)
- Unix OS (tested on macOS 10.15.7 Catalina and Linux 3.2.84-amd64-sata x86_64)
- C compiler supporting the C99 standard for variable-length arrays (tested with gcc 4.9.4 on Linux and Apple clang 11.0.3)
- Python C-API

## Installation
Download the code in a subfolder of your Python project. Then, compile the Python C-API module by running `setup_chancapmc.py` with Python, and move the .so object to the main folder:
```bash
>>> cd chancapmc/
>>> python setup_chancapmc.py build_ext --inplace
>>> mv build/lib*/chancapmc*.so .
```

For testing, you can then run `test_chancapmc.py` with Python. Other Python scripts for testing and visualizing sampling distributions are in the tests/ folder. More unit tests are coded in pure C in `unittests.c`. Compile them with a compiler supporting the C99 standard following the instructions given in `unittests.c`. Some compiler options may be platform-dependent; if a bug arises, it's a good idea to look at what commands `setup_chancapmc.py` prints when the module is set up, and try to use the compiler arguments automatically selected there.

## Usage
Import the module as follows in your Python script:

```python
import chancapmc.chancapmc as chancapmc
```

Launch the Blahut-Arimoto algorithm with the function `chancapmc.ba_discretein_gaussout` for an output random variable which has a multivariate normal distribution. The expected arguments are a 2D array giving the mean vector of each conditional distribution, a 3D array giving the covariance matrices of each conditional distribution, a 1D array giving the input variable values. Optional arguments are the relative tolerance on the computed channel capacity -– for convergence criterion of the B.-A. algorithm -– and the relative tolerance on Monte Carlo integrals accuracy (recommended at least as small as the tolerance on capacity).

Other main functions will need to be coded in the future to extend the algorithm to  different output distributions, since the call arguments are distribution-dependent.

## Copyright note
This code is licensed under the BSD-3-Clause License. It uses the (unmodified) source code of the [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/ "SFMT project homepage") package in C, which provides a fast and reliable variant of the  Mersenne Twister algorithm for random number generation in C. The source code and copyright license of this package can be found in the dSFMT/ folder. They are also licensed under the BSD-3-Clause License.

This code also uses the Python C-API module in C, which comes with some Python installations such as Anaconda.
