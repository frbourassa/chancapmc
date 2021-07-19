/* Main module for the Blahut-Arimoto algorithm with multivariate continuous
* output, discrete input, and Monte Carlo integration on the outputs.
*
* WARNING: This module uses the C99 standard, esp. variable-length arrays.
*
* It also uses the Python-C API because it is part of a C extension and must
* deal with exceptions and Ctrl-C exits, but it can easily be made pure C:
* just remove the call to PyErr_CheckSignals and to PyErr_SetString.
*
*/
#ifndef BLAHUT_ARIMOTO_H
#define BLAHUT_ARIMOTO_H

// Need to include because prototypes below use this. 
#include <stddef.h>

// Prototypes
double amax(size_t n, double arr[n]);

double cj_integrand_gaussout(
    size_t dim, size_t nin, size_t argj, double x[dim], double means[nin][dim],
    double invLs[nin][dim][dim], double p_vec[nin]);

double blahut_arimc_gaussout(size_t dim, size_t nin, double pvec[nin],
    double means[nin][dim], double covmats[nin][dim][dim],
    double rtol, int seed);

#endif
