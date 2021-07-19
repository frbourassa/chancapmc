/*
Module containing functions for a multivariate gaussian distribution
and sampling from such a distribution, including some linear algebra methods.

Uses dSFMT Mersenne Twister random number generator, because its output
is higher quality compared to C's default random module.

We could use the Python C API to access Numpy's generator, but I don't
want to bother with this API, and keep the code pure C as much as possible.

WARNING: this code uses the C99 standard for variable-length array, because
this is more efficient than pointers-of-pointer arrays (double **)
*/
#define DSFMT_MEXP 19937
#include "dSFMT/dSFMT.h"
#include <math.h>
#include "gaussfuncs.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* LINEAR ALGEBRA */

// 2D matrix sum

// 2D matrix dot product: A_{mxn} dot B_{nxp} = O_{mxp}.
// Both should be 2D arrays, but either can have one or two dimensions == 1
// matO supposed to be malloc'd already.
int
matprod(size_t m, size_t n, size_t p, double matA[m][n], double matB[n][p],
    double matO[m][p])
{
    if (m <= 0 || n <= 0 || p <= 0)
        return 1;
    size_t i = 0, j = 0, k = 0;
    for (i = 0; i < m; i++){
        for (j = 0; j < p; j++){
            matO[i][j] = 0.;  // initialize
            // Sum elements A_{ik} B_{kj} sum over k
            for (k = 0; k < n; k++){
                matO[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
    return 0;
}

// Matrix applied on the left to a 1D vector: A_{mxn} u_n = v_m. Useful when
// the vector is not a 2d array but a 1d array. vecV supposed malloc'd already
int
ldot(size_t m, size_t n, double matA[m][n], double vecU[n], double vecV[m])
{
    if (m <= 0 || n <= 0)
        return 1;
    size_t i = 0, j = 0;
    for (i = 0; i < m; i++){
        vecV[i] = 0.;  // initialize
        // Sum on j in A_{ij} u_j = v_i
        for (j = 0; j < n; j++){
            vecV[i] += matA[i][j] * vecU[j];
        }
    }
    return 0;
}

// Matrix applied on the right to a 1D vector: u_m A_{mxn}. Useful when the
// vector is not a 2d array but a 1d array.
int
rdot(size_t m, size_t n, double vecU[m], double matA[m][n], double vecV[n])
{
    if (m <= 0 || n <= 0)
        return 1;
    size_t i = 0, j = 0;
    for (i = 0; i < n; i++){
        vecV[i] = 0.;  // initialize
        // Sum on j in u_j A_{ji} = v_i
        for (j = 0; j < m; j++){
            vecV[i] += vecU[j] * matA[j][i];
        }
    }
    return 0;
}


/* Cholesky decomposition of a square, symmetric, positive-definite matrix A
Reference to understand: Numerical Recipes in C, section 2.9, p. 96.
Will fill the upper triangular part of cholL with zeros.
Return codes:
    0: correct input, decomposition succeeded;
    1: input A is not positive definite within numerical roundoff error;
    2: input A is not symmetric
*/
int
cholesky(size_t n, double matA[n][n], double cholL[n][n])
{
    size_t i = 0, j = 0, k = 1;
    double partialsum = 0.;
    for (i = 0; i < n; i++){
        for (j = i; j < n; j++){
            // Check symmetry; if not, return 2
            if (matA[i][j] != matA[j][i]) return 2;
            // Compute the sum subtracted from a_ij
            partialsum = matA[i][j];
            for (k = i; k >= 1; k--)
                partialsum -= cholL[i][k-1] * cholL[j][k-1];
            // Compute the L_ji entry of L
            if (i == j){
                // Check that the matrix is indeed positive symmetric.
                if (partialsum < 0.) return 1;
                else cholL[j][i] = sqrt(partialsum);
            }
            else cholL[j][i] = partialsum / cholL[i][i];
        }
        // Filling the upper triangular part of cholL with zeros
        for (j = 0; j < i; j++)
            cholL[j][i] = 0;
    }
    return 0;
}

// 2D matrix transpose, input matrix of size m x n (args in that order)
int
transpose_mat(size_t m, size_t n, double matA[m][n], double matT[n][m]){
    size_t i = 0, j = 0;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++)
            matT[j][i] = matA[i][j];
    }
    return 0;
}
/* Inverting the Cholesky matrix L of A=LL^T to compute e.g. L^-T L^-1 = A^-1
* Don't bother storing the result in place: I will deal with small matrices
* Works to invert any invertible triangular matrix in fact.
* Assumes matT is lower triangular; no check on the terms above diagonal.
*/
int
inv_triang(size_t m, double matT[m][m], double invT[m][m])
{
    size_t i = 0, j = 0, k = 0;
    double partsum = 0.;
    // Loop over columns of invT first
    for (j = 0; j < m; j++){
        // Diagonal term
        if (matT[j][j] == 0.)
            return 1;  // found that T is not invertible.
        else
            invT[j][j] = 1/matT[j][j];
        // Terms below diagonal, ith row based on all previous rows
        for (i = j + 1; i < m; i++){
            partsum = 0.;
            for (k = j; k < i; k++)
                partsum -= matT[i][k] * invT[k][j];
            invT[i][j] = partsum / matT[i][i];
        }
        // Set entries above diagonal to zero for correct matrix products
        for (i = 0; i < j; i++){
            invT[i][j] = 0.;
        }
    }
    return 0;
}



/* Generating univariate normal(0, 1) samples
* Adapted from Numerical Recipes in C, 2nd ed. to use dSFMT and pre-generate
* sample. Works in-place for memory efficiency. Because of rejection, we need
* on average 4n/pi uniform random samples to create n normal samples.
* Need to input the random state address of an already initialized generator.
*
* TODO: Allocate on the heap with malloc, because VLAs go on the stack
* (8MB max) and will cause overflow if we want on the order of 10^6 samples
* of 64-bit doubles (8 MB precisely). Won't be a significant access performance
* issue, because the whole list of samples will be attributed once per integral
* to compute, then accessed in close succession */
double *
gen_univariate_normal01_samples(size_t nvari, double *container, dsfmt_t rand_state)
{
    size_t min_nvari;
    min_nvari = get_min_array_size();  //  at least (SFMT_MEXP / 128) * 2
    if (nvari % 2 != 0)
        return NULL;  // there must be an even number of samples.

    // Indices for looping and working in-place on the container array.
    size_t i_u = 0; // index to next uniform samples to consider
    size_t nsaved = 0;  // next position where to save new normal variate

    // Variables to hold the pairs of variates being transformed
    double v1, v2, radius2, factor;
    double y1, y2;
    int loopflag = 0;

    while (nvari - nsaved > 0) {
        /* Fill the rest of the array with fresh new uniform samples
        * Give the address of the next unfilled position
        * as the start of the array to fill */
        if (nvari - nsaved >= min_nvari)
            dsfmt_fill_array_open_close(&rand_state, &container[nsaved], nvari-nsaved);
        else  // if we need only a few more samples
            for (i_u = nsaved; i_u < nvari; i_u++){
                container[i_u] = dsfmt_genrand_open_close(&rand_state);
            }
        /* Start considering the new uniform samples to generate normal ones
        * Start i_u at the first position not yet filled with a normal sample
        * Work one pair of samples at a time, so i_u += 2. */
        for (i_u = nsaved; i_u < nvari; i_u += 2){
            v1 = 2. * container[i_u] - 1.;
            v2 = 2. * container[i_u+1] - 1.;
            radius2 = v1*v1 + v2*v2;
            if (radius2 <= 1. && radius2 > 0.){
                factor = sqrt(-2.*log(radius2)/radius2);
                y1 = factor * v1;
                y2 = factor * v2;
                container[nsaved] = y1;
                container[nsaved+1] = y2;
                // After the normal samples are kept and saved, increment nsaved!
                nsaved += 2;
            }
            else {
                continue;  // do not update nsaved, just i_u in the for loop
            }
        }
        loopflag += 1;
    }
    // printf("Number of loops: %d\n", loopflag);
    return container;
}


/* Filling an array with multivariate normal samples,
 given the Cholesky decomposition L of the covariance matrix E=LL^T.
We don't want to compute this decomposition every time.
Need to input the random state address of an already initialized generator.

Return the number of samples, in case it is adjusted because not even.
*/
size_t
gen_multivariate_normal_samples(size_t nsamp, size_t dim, double mean[dim],
    double cholL[dim][dim], double sampvecs[nsamp][dim], dsfmt_t rand_state)
{
    // Check that nsamp*dim is even, to guarantee dSFMT array-filling can work
    double *container1d;
    size_t ns1d = nsamp*dim;
    if (ns1d % 2 != 0)
        ns1d += 1;
    else if (ns1d == 0)
        return 0;
    container1d = malloc(ns1d * sizeof(double));

    /* Could pass &sampvecs[0][0] as a double * being the start of an array,
    * rather than creating another array, to avoid duplicating memory.
    * But that's undefined behaviour, platform-dependent. So avoid doing this.
    * Thus I need to create a 1d container of the same size as sampvecs */
    /*
    if ((nsamp*dim) % 2 != 0){
        if (nsamp > 1){
            nsamp -= 1;
            ns1d = nsamp * dim;
            container1d = &sampvecs[0][0];
            printf("Please use an even number of samples next time\n");
        }
        else{
            container1d = malloc((nsamp*dim + 1) * sizeof(double));
            ns1d = nsamp*dim + 1;
        }
    }
    else if (nsamp == 0)
        return 0;
    else {
        container1d = &sampvecs[0][0];
        ns1d = nsamp*dim;
    }
    */

    // Get nsamp * dim normal(0, 1) variates
    container1d = gen_univariate_normal01_samples(ns1d, container1d, rand_state);

    // Linear combination
    int reslt;
    size_t i = 0, j = 0;
    for (i = 0; i < nsamp; i++){
        // TODO: make this more efficient, use the fact that L_{ij}=0 for i > j
        // Take the product of cholL and the ith tuple of dim univariate normals
        reslt = ldot(dim, dim, cholL, &container1d[i*dim], &sampvecs[i][0]);
        if (reslt != 0)
            return i-1;  // Number of good samples that could be built.
        // Add the mean vector
        for (j = 0; j < dim; j++){
            sampvecs[i][j] += mean[j];
        }
    }

    // Free the container of 1D univariates
    free(container1d);
    return nsamp;
}

/* TODO: similar function to gen_multivariate_normal_samples, but for the
* conditional input-output. For each generated sample, first pick one input,
* then use the corresponding covariance and mean (composition method to sample
* from the joint distribution, then keep only the variable of interest).
*/
// IS IT NECESSARY FOR BLAHUT-ARIMOTO, OR JUST FOR MI?



/* Probability density function of a multivariate normal distribution.
    Inputs: dimension n, sample x (vector of len n), mean vector mu,
    Cholesky decomposition L of the covariance matrix, Sigma=LL^T.
*/
double
pdf_multivariate_normal(size_t n, double x[n], double mu[n], double cholL[n][n])
{
    size_t i = 0;
    int success = 0;
    // Prefactor: first put the pi^(n/2), then divide by the det(Sigma)^1/2,
    //equal to the product of diagonal entries of cholL.
    double prefactor = 1. / pow(2.*M_PI, ((double) n)/2.);
    for (i = 0; i < n; i++){
        prefactor /= cholL[i][i];
    }

    // Argument of the exponent. First, get L^{-1}
    double invL [n][n];
    double ytemp[n], yfin[n];
    double exp_arg = 0.;
    success = inv_triang(n, cholL, invL);

    if (success != 0) return -1.;  // this is an error

    // Then, compute x - mu
    for (i = 0; i < n; i++)
        ytemp[i] = x[i] - mu[i];

    // Then, take the dot product with L^{-1}
    success = ldot(n, n, invL, ytemp, yfin);
    if (success != 0) return -1.;

    // Finally, take -1/2 y^T dot y as the argument
    for (i = 0; i < n; i++)
        exp_arg += yfin[i] * yfin[i];
    exp_arg *= -0.5;

    // Finally, combine the prefactor and the exponential
    return prefactor * exp(exp_arg);
}

// PDF for a multivariate normal, when the inverse of the Cholesky decomposition,
// L^-1, is given, for higher performance in repeated use.
double
pdf_multivariate_normal_fast(size_t n, double x[n], double mu[n], double invL[n][n])
{
    size_t i = 0;
    int success = 0;
    // Prefactor: first put the pi^(n/2), then divide by the det(Sigma)^1/2,
    //equal to the product of diagonal entries of cholL.
    double prefactor = 1. / pow(2.*M_PI, ((double) n)/2.);
    for (i = 0; i < n; i++){
        prefactor *= invL[i][i];
    }

    // Argument of the exponent. First, compute x - mu
    double ytemp[n], yfin[n];
    double exp_arg = 0.;
    for (i = 0; i < n; i++)
        ytemp[i] = x[i] - mu[i];

    // Then, take the dot product with L^{-1}
    success = ldot(n, n, invL, ytemp, yfin);
    if (success != 0) return -1.;

    // Finally, take -1/2 y^T dot y as the argument
    for (i = 0; i < n; i++)
        exp_arg += yfin[i] * yfin[i];
    exp_arg *= -0.5;

    // Finally, combine the prefactor and the exponential
    return prefactor * exp(exp_arg);
}
