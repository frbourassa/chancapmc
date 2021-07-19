/* Main module for the Blahut-Arimoto algorithm with multivariate continuous
output, discrete input, and Monte Carlo integration on the outputs.

WARNING: This module uses the C99 standard, especially variable-length arrays.

It also uses the Python-C API because it is part of a C extension and must
deal with exceptions and Ctrl-C exits, but it can easily be made pure C:
just remove the call to PyErr_CheckSignals and to PyErr_SetString.

Must be included first in any other C code calling it, so the Python headers
come before any other header.
*/

/* CAN'T DEFINE FUNCTIONS IN FUNCTIONS IN C
* https://stackoverflow.com/questions/957592/functions-inside-functions-in-c
* FIND ANOTHER SOLUTION TO PASS ARRAY OF CHOLESKY L (OR L^-1 TO DENSITY)
* AND MEANS.
*
* WORKAROUND 1 (Python- and GSL-like): STRUCTS OF ARBITRARY PARAMETERS.
* In particular, malloc the VLAs at the end of the struct:
* https://stackoverflow.com/questions/49879219/2-dimensional-array-in-a-struct-in-c
* then pass a pointer to the struct, make mcint and blahut accept void *
* so any kind of struct with arbitrary parameters in it can be accepted.
* Keeps the methods (mcint, blahut-arimoto) generic, but less efficient
*
* WORKAROUND 2 (more C-like, less flexible): define specific blahut and
* mc integration methods taking the VLAs for L/L^-1 and means as args.
* Not generic, and also more cumbersome in function argument lists,
* but faster (VLAs are not copied on the heap in the structs) and less
* opaque.
CHOSE WORKAROUND 2 IN THIS TRUNK FOSSIL BRANCH.
*/


#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "mcint.h"  // includes math.h and gaussfuncs.h and dSFMT.h
#include "gaussfuncs.h"
#include "blahut_arimoto.h"


// Small function to find the max of an array
double amax(size_t n, double arr[n]){
    size_t i = 0;
    double mx = arr[0];
    for (i = 1; i < n; i++){
        if (arr[i] > mx) mx = arr[i];
    }
    return mx;
}


/* Integrand for the Blahut-Arimoto algorithm with discrete input,
multivariate normal output channel*/
double
cj_integrand_gaussout(size_t dim, size_t n_ins, size_t argj, double x[dim],
    double means[n_ins][dim], double invLs[n_ins][dim][dim],
    double p_vec[n_ins])
{
    double densj = pdf_multivariate_normal_fast(
                dim, x, means[argj], invLs[argj]);
    double densmarg = 0.;
    size_t loc_i = 0;  // local looping variable
    for (loc_i = 0; loc_i < n_ins; loc_i++){
        densmarg += p_vec[loc_i] * pdf_multivariate_normal_fast(
            dim, x, means[loc_i], invLs[loc_i]);
    }
    return densj * log(densj / densmarg);
}


/* Main Blahut-Arimoto algorithm for continuous multivariate outputs,
* discrete inputs.
Args:
    size_t dim
    size_t ninputs
    double pvec[ninputs]: initial probability distrbution vector
        Is this really necessary? Yes because this is where the optimized
        input distribution can be recovered.

Returns:
    lowbound (double): lower bound to  the capacity, within rtol
        of the true value. Units: nits. Convert to bits by dividing by log(2).
    (Access the optimal pvec through the inputted pvec array,
        which is modified in-place. )
*/
double
blahut_arimc_gaussout(size_t dimension, size_t ninputs,
    double pvec[ninputs], double means[ninputs][dimension],
    double covmats[ninputs][dimension][dimension], double rtol, int seed)
{
    /* 0. Initialization */
    double cvec[ninputs];  // vector of c_j^r
    double lowbound = 0., upbound = 1., rerr = rtol + 1.;  // bounds and error
    double avg_c = 0.;  // sum_j p_j^r c_j^r
    double res_mcint = 0.;   // result of mcint integration
    double err_mcint = 0.;   // abs error of mcint -- not needed but returned.
    int niter = 0;  // Number of iterations (r+1)
    size_t j = 0;  // Index to loop on inputs
    int success = 0;
    size_t n_samples = 100000; //1e5 samples to begin with
    double mctol = rtol / sqrt((double) ninputs);  // MC integration tolerance

    // Initialize the random number generator with a seed for reproducibility
    dsfmt_t rand_state;
    size_t totsamp = 0;   // samples needed for MC int
    dsfmt_init_gen_rand(&rand_state, seed);

    /* Find the Cholesky decomposition of each covariance matrix,
    * and the inverse L as well for fast pdf computation
    */
    double cholLs [ninputs][dimension][dimension];
    double invcholLs[ninputs][dimension][dimension];

    for (j = 0; j < ninputs; j++){
        success = cholesky(dimension, covmats[j], cholLs[j]);
        if (success == 1){
            printf("Covariance %ld not positive def.; returning C = -1.\n", j);
            PyErr_SetString(PyExc_ValueError,
                "Covariance not positive def.; returning C = -1.");
            return -1.;
        }
        else if (success == 2){
            printf("Covariance %ld not symmetric; returning C = -1.\n", j);
            PyErr_SetString(PyExc_ValueError,
                "Covariance not symmetric; returning C = -1.");
            return -1.;
        }
        success = inv_triang(dimension, cholLs[j], invcholLs[j]);
        if (success != 0){
            printf("Covariance %ld not invertible; returning C = -1.\n", j);
            PyErr_SetString(PyExc_ValueError,
                "Covariance not invertible; returning C = -1.");
            return -1.;
        }
    }

    /* Iterations */
    while (rerr > rtol){
        // Check whether the code was stopped during the last iteration
        if (PyErr_CheckSignals() != 0){
            PyErr_SetString(PyExc_ValueError, "Exit signal was received\n");
            return -1.;
        }

        niter += 1;  // the p_j being used are p^{r = niter - 1}

        // 1. Compute the c_j(p^r)
        /* THIS COULD EASILY BE PARALLELIZED; but how to do this in C?
        * And is dSFMT thread-safe? Maybe would need an array of rand_states?.
        * This "jump" function can also help:
        * See http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/SFMT/JUMP/dsfmt-jump.html
        * And potential issues with accessing cholLs, invcholLs, means, pvec.
        * but it should be safe if the data is truly read-only:
        * https://stackoverflow.com/questions/5643060/is-it-wise-to-access-read-only-data-from-multiple-threads-simultaneously
        * But it can degrade performance because this data is accessed at every function call.
        * OpenMP could help with those.
        * And might run into issues of memory usage if we have many samples
        * arrays in parallel.
        */
        for (j = 0; j < ninputs; j++){
            totsamp = mcint_gaussdist(dimension, ninputs, j, cj_integrand_gaussout,
                pvec, means, cholLs, invcholLs, rand_state, n_samples, mctol,
                &res_mcint, &err_mcint);
            if (totsamp < 0){  // This means an error happened
                printf("MC integ. failed to converge; returning C = -1.\n");
                PyErr_SetString(PyExc_ValueError,
                    "MC integ. failed to converge; returning C = -1.");
                return -1.;
            }
            cvec[j] = exp(res_mcint);
        }

        // 2. Compute the bounds on the MI with p^r
        avg_c = 0.;
        for (j = 0; j < ninputs; j++)
            avg_c += pvec[j]*cvec[j];
        lowbound = log(avg_c);
        upbound = log(amax(ninputs, cvec));

        // 3.  Compute the error
        if (lowbound > 0.){
            rerr = (upbound - lowbound) / lowbound;
        }
        else{
            printf("Strange: lowbound = %f\n", lowbound);
            rerr = rtol + 1.;  // iterate more
        }

        // 4. If the error is too large, compute the next prob distrib.
        // but not if this is the last iteration,
        // because it's p^r, not p^{r+1} that gave the return MI
        // TODO: is it better to return p^{r+1} that we have for free?
        if (rerr > rtol){
            for (j = 0; j < ninputs; j++){
                pvec[j] = pvec[j] * cvec[j] / avg_c;
            }
        }
        /* Print some information on the current iteration.
        * Since we are using Python C-API even in this .h file, print
        * to Python's stdout with printf rather than printf,
        * which goes to terminal if running a Jupyter notebook, say. */
        if (niter >= 10000){
            printf("blahut_arimc failed to converge after 10^4 iterations\n");
            return lowbound;
        }
        printf("Completed Blahut-Arimoto iteration %d\n", niter);
        printf("Used %ld samples for the last MC integration\n", totsamp);
        printf("I_low = %f nits, I_up = %f nits, rel. error = %f\n",
                                        lowbound, upbound, rerr);
        printf("p_vector = {");
        for (j = 0; j < ninputs; j++){
            printf("%f ", pvec[j]);
        }
        printf("}\n\n");

    }
    printf("blahut_arimc converged after %d iterations\n", niter);
    printf("The optimal input p is that obtained after one fewer iteration\n");
    return lowbound;
}



/* Sampling method for a multivariate input-output gaussian: in the mcint
function, just use gen_multivariate_normal_samples with the jth (pre-sliced)
cholL and mean vector when computing c_j(p^r).
*/
