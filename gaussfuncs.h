/*
* Module containing functions for a multivariate gaussian distribution
* and sampling from such a distribution, including some linear algebra methods.
*
* WARNING: this code uses the C99 standard for variable-length array, because
* this is more efficient than pointers-of-pointer arrays (double **)
*/
#ifndef GAUSSFUNCS_H
#define GAUSSFUNCS_H

// Need to include because the prototypes below use types defined in dSFMT
#include <stddef.h>
#include "dSFMT/dSFMT.h"

// Prototypes
int matprod(size_t m, size_t n, size_t p, double matA[m][n],
            double matB[n][p], double matO[m][p]);

int ldot(size_t m, size_t n, double matA[m][n], double vecU[n], double vecV[m]);

int rdot(size_t m, size_t n, double vecU[m], double matA[m][n], double vecV[n]);

int cholesky(size_t n, double matA[n][n], double cholL[n][n]);

int transpose_mat(size_t m, size_t n, double matA[m][n], double matT[n][m]);

int inv_triang(size_t m, double matT[m][m], double invT[m][m]);

double * gen_univariate_normal01_samples(size_t, double *, dsfmt_t);

size_t gen_multivariate_normal_samples(size_t nsamp, size_t dim,
    double mean[dim], double cholL[dim][dim], double sampvecs[nsamp][dim],
    dsfmt_t rand_state);

double pdf_multivariate_normal(size_t n, double x[n], double mu[n],
    double cholL[n][n]);

double pdf_multivariate_normal_fast(size_t n, double x[n], double mu[n],
    double invL[n][n]);

#endif
