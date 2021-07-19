/* Monte Carlo integration module.
*/
#ifndef MCINT_H
#define MCINT_H

// Need to include because the prototypes below use types defined in dSFMT
#include "dSFMT/dSFMT.h"
#include <stddef.h>

// Prototypes
// Sampling from gaussian distribution.
size_t mcint_gaussdist(size_t dim, size_t nin, size_t argj,
    double integrand(size_t d, size_t ni, size_t aj, double x[d],
        double means[ni][d], double invLs[ni][d][d], double pv[ni]),
    double pvec[nin], double mat1[nin][dim], double mat2[nin][dim][dim],
    double matinv2[nin][dim][dim], dsfmt_t rand_state, size_t nsmp,
    double tol, double *integ_res, double *abserror);

#endif
