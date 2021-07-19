/* A function in this package will create the array to be filled with samples
by the sampling function, so expect malloc here. Never forget to free
at the end of MC integration!

Consider using PyMem_Malloc, and so including Python extension
in this header as well? See doc to find how to do this again.
*/
#include <math.h>
#include "gaussfuncs.h"  // includes dSFMT.h
#include "mcint.h"

// When mcint'ing, use gen_multivariate_normal_samples with cholLs[j] and
// means[j] pre-sliced; this way, no need for a wrapper.
// This is why we include gaussfuncs.h here.

/* Monte Carlo integration for the capacity of a channel with
discrete input distribution p_vec, and multivariate normal output
with means mat1, covariance matrix Cholesky decomposition mat2,
and their inverses matinv2.

Could be another integrand, but the sampling distribution is always
a multivariate gaussian with the means and covariances given as arguments.

Directly uses gen_multivariate_normal_samples with proper arg_j.
Receives the integrand function, which has a very specific signature.

Args:
    dim: dimension of the channel output vector
    n_ins: number of input values
    argj: which input to evaluate the integrand for
    integrand: the integrand function, with call signature as below
    pvec: current probability mass function of the inputs
    mat1: mean vector for each input value
    mat2: Cholesky L of the covariance matrix for each input value
    matinv2: L^-1 of the covariance matrix for each input value
    rand_state: random number generator state
    nsmp: number of samples to generate, minimally
    tol: relative error tolerated on the integral result.
*/
size_t
mcint_gaussdist(size_t dim, size_t n_ins, size_t argj,
    double integrand(size_t d, size_t ni, size_t aj, double x[d],
        double means[ni][d], double invLs[ni][d][d], double pv[ni]),
    double pvec[n_ins], double mat1[n_ins][dim], double mat2[n_ins][dim][dim],
    double matinv2[n_ins][dim][dim], dsfmt_t rand_state, size_t nsmp,
    double tol, double *integ_res, double *abserror)
{
    // a single sample makes no sense
    if (nsmp <= 1)
        nsmp = 2;

    double intval = 0., densval = 0.;  // g(x) and f_X(x)
    double intratio = 0.;  // ratio of intval / densval for the current sample
    // estimator of the integral result and its second moment
    double integral_estim = 0., integral_delta = 0.;
    double integralm2_estim = 0., integralm2_delta = 0.;
    // estimator of the integral variance and error
    double s_g2 = 0., abserr = 0.;

    size_t k = 0;
    size_t di = 0;
    int niter = 0;   // count iterations
    size_t nsmp_alloc = nsmp;  // to know the size of the array we malloc'ed
    long nsmp_add = nsmp;  // new samples added in the current iteration
    // don't make this size_t because it will never be negative!
    size_t nsmp_add_compare = nsmp;  // a version for comparison, zero when nsmp_add <= 0
    double (*samples)[dim] = malloc(sizeof(double[nsmp_alloc][dim]));
    nsmp = 0;  // Preparing for the first iteration where nsmp is incremented

    // Loop until tolerance is satisfied
    do {
        // Generate samples with gen_multivariate_normal_samples, mat1, mat2
        gen_multivariate_normal_samples(nsmp_add, dim, mat1[argj],
            mat2[argj], samples, rand_state);

        // Prepare the running estimates to the addition of nsmp_add samples
        // G_new = G_old * nsmp_old / nsmp_tot + delta / nsmp_tot
        nsmp += nsmp_add;
        integral_estim *= ((double) (nsmp - nsmp_add)) / ((double) nsmp);
        integralm2_estim *= ((double) (nsmp - nsmp_add)) / ((double) nsmp);
        integral_delta = 0.;
        integralm2_delta = 0.;

        // Evaluate the integrand at each sample with mat1 and matinv2 and argj
        // Add g(x)/f_X(x) each time, then divide by nsmp to increment the estimate
        for (k = 0; k < nsmp_add_compare; k++){
            intval = integrand(dim, n_ins, argj, samples[k], mat1, matinv2, pvec);
            densval = pdf_multivariate_normal_fast(dim, samples[k], mat1[argj], matinv2[argj]);
            if (densval == 0.){
                printf("Zero probability point for j = %ld:", argj);
                for (di = 0; di < dim; di++){
                    printf("%f ", samples[k][di]);
                }
                printf("\n");
                continue;  // just adding zero
            }
            intratio = intval / densval;
            integral_delta += intratio;
            integralm2_delta += intratio*intratio;
        }
        integral_estim += integral_delta / nsmp;
        integralm2_estim += integralm2_delta / nsmp;

        // Estimate the error, sqrt(s_g^2 / N)
        s_g2 = ( ((double) nsmp) / (((double) nsmp) - 1.)
                * (integralm2_estim - integral_estim*integral_estim) );
        abserr = sqrt(s_g2 / nsmp);

        /* Add samples if needed to achieve tolerance.
        * Find the next number of samples to add, nsmp_add
        * Total number of samples needed: s_g2 / (tol*tol*integral_estim)
        * because tol = sqrt(s_g2 / nsmp_needed) / integral_estim
        * So add the following: */
        nsmp_add = (s_g2/(tol*tol*integral_estim*integral_estim) - nsmp)*1.1;
        // Update the non-negative (for comparisons) version of nsmp_add
        if (nsmp_add <= 0)
            nsmp_add_compare = 0;
        else
            nsmp_add_compare = nsmp_add;

        // If comparing nsmp_add to 10000000/dim, which is unsigned int,
        // nsmp_add is converted to unsigned, so a large positive number.
        // So also check it is positive first.
        if (nsmp_add_compare > 100000000/dim){ // over 1e8 doubles fills memory
            nsmp_add = 100000000/dim;
            nsmp_add_compare = nsmp_add;
            printf("Need more than 10^8 new MC samples; consider aborting\n");
        }

        // Allocate more memory if needed for the next round
        // Make nsmp_alloc a long (not size_t) so it can be compared to negative nsmp_add
        if (nsmp_add_compare > nsmp_alloc){
            free(samples);
            samples = malloc(sizeof(double[nsmp_add][dim]));
            nsmp_alloc = nsmp_add;  // new mallocation
        }
        // If this is getting too long, exit and store what we have
        if (nsmp >= 1e9 || niter > 100){
            printf("MC integration failed to converge to desired accuracy\n");
            nsmp_add = 0;
        }
        niter += 1;
    // nsmp updated in the next iteration, if we do need more samples
    } while (nsmp_add > 1);

    // Free memory
    free(samples);

    // Return number of samples, store the result and error at given location
    *integ_res = integral_estim;
    *abserror = abserr;
    return nsmp;
}
