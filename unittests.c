/* Testing functions defined in the headers

* To compile (note: adapt the location of the anaconda3/ folder):

gcc "dSFMT/dSFMT.c" unittests.c blahut_arimoto.c gaussfuncs.c mcint.c \
-O3 -std=c99 -msse2 -fno-strict-aliasing -DHAVE_SSE2=1 -DDSFMT_MEXP=19937 \
-I/Users/francoisb/anaconda3/include/python3.7m -lpython -o unittests.out

* Or on the physics server:
gcc "dSFMT/dSFMT.c" unittests.c blahut_arimoto.c gaussfuncs.c mcint.c -O3 \
-std=c99 -msse2 -fno-strict-aliasing -DHAVE_SSE2=1 -DDSFMT_MEXP=19937  \
-I/homes/este/bourassa/anaconda3/include/python3.6m \
-L/homes/este/bourassa/anaconda3/lib -lpython3.6m -lm -o unittests.out

* For some reason the linker fails on the mbp_env virtual environment, where the
-I and -L would be in ...anaconda3/envs/mbp_env/... instead of just
...anaconda3/...
The reason seems to be a gcc version too old: gcc 4.9.4 onthe Physics server
doesn't make the cut.
Therefore, use also
    -B /este/users/bourassa/anaconda3/envs/mbp_env/compiler_compat
and
    -Wl,-rpath=/este/users/bourassa/anaconda3/envs/mbp_env/lib -Wl,--sysroot=/
The B flag gets a ld version compatible with python 3.7; the rpath args
seem to complete the ld in the env. (-Wl, tells gcc this is an arg for ld)

* So the full command to compile in a virtual environment on the server is:

gcc "dSFMT/dSFMT.c" unittests.c blahut_arimoto.c gaussfuncs.c mcint.c \
-O3 -std=c99 -msse2 -fno-strict-aliasing -DHAVE_SSE2=1 -DDSFMT_MEXP=19937  \
-B /este/users/bourassa/anaconda3/envs/mbp_env/compiler_compat \
-Wl,-rpath=/este/users/bourassa/anaconda3/envs/mbp_env/lib -Wl,--sysroot=/ \
-I/homes/este/bourassa/anaconda3/envs/mbp_env/include/python3.7m \
-L/homes/este/bourassa/anaconda3/envs/mbp_env/lib -lpython3.7m -lm -o unittests.out

I found out the necessary options by trial and error on the options used by
setuptools to automatically compile chancapmcmodule on the platform.
The command gets so long I feel like I should compile this unittests.c
with a Python script or a Makefile.
See for instance:
https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
*/

#include <math.h>
#include "mcint.h"
#include "gaussfuncs.h"
#include "blahut_arimoto.h"  // inclues math.h, gaussfuncs.h, mcint.h
#include <time.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// Prototypes
int compare_2d(int m, int n, double [m][n], double mat2[m][n]);
int compare_1d(int, double [], double []);
void compute_1d_moments(size_t, double*, double*, double*, double*, double*);
void savetobin(size_t, double*, char []);
void save2dtobin(size_t s0, size_t s1, double arr[s0][s1], char filepath[]);

/* Array comparison */
int
compare_2d(int m, int n, double mat1[m][n], double mat2[m][n])
{
    // Sum the absolute differences
    int i = 0, j = 0;
    double absdiff = 0.;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            absdiff += fabs(mat1[i][j] - mat2[i][j]);
        }
    }
    if (absdiff > 1e-14*m*n) return 1;
    else return 0;
    // If this is above some threshold, return error.
}

int
compare_1d(int m, double vec1[], double vec2[])
{
    int i = 0;
    double absdiff = 0.;
    for (i = 0; i < m; i++){
        absdiff += fabs(vec1[i] - vec2[i]);
    }
    if (absdiff > 1e-12) return 1;
    else return 0;
}

/* Statistics verification */
void
compute_1d_moments(size_t nsamples, double *container, double *mean,
                    double *nu2, double *skew, double *kurt)
{
    size_t i = 0;
    double term2 = 0.;
    double m1 = 0., m2 = 0., m3 = 0., m4 = 0.;
    // Mean and variance
    for (i = 0; i < nsamples; i++){
        // non-centered moments
        m1 += container[i];
        term2 = container[i] * container[i];
        m2 += term2;
        m3 += term2 * container[i];
        m4 += term2 * term2;
    }
    m1 /= nsamples;
    m2 /= nsamples;
    m3 /= nsamples;
    m4 /= nsamples;

    // Variance nu2 (centered second moment)
    *mean = m1;
    *nu2 = nsamples / (nsamples - 1) * (m2 - m1*m1);

    // Skewness expressed in terms of non-centered moments
    *skew = (m3 - 3. * m1 * (*nu2) - m1*m1*m1) / pow(*nu2, 3/2);

    // Excess kurtosis defined from non-centered moments.
    // Don't care about biases of order N-1/N
    *kurt = (m4 - 4*m3*m1 + 6*m2*m1*m1 - 3*m1*m1*m1) / ((*nu2)*(*nu2)) - 3.;
}

/* Saving results for further analysis in Python */
void
savetobin(size_t sizea, double *arr, char filepath[])
{
    FILE *ptr;
    ptr = fopen(filepath, "wb");
    fwrite(arr, sizeof(double), sizea, ptr);
    printf("\tDone saving %zu elements in %s\n", sizea, filepath);
    fclose(ptr);   // Don't forget to close the file before trying to reopen!
}

void
save2dtobin(size_t s0, size_t s1, double arr[s0][s1], char filepath[])
{
    FILE *ptr;
    ptr = fopen(filepath, "wb");
    fwrite(arr, sizeof(double[s1]), s0, ptr);
    printf("\tDone saving %zu x %zu elements in %s\n", s0, s1, filepath);
    fclose(ptr);   // Don't forget to close the file before trying to reopen!
}

/* INTEGRANDS FOR MC INTEGRATION */
// f_X ^ 2 for a multivariate normal.
double
integrand_test_normsquared(size_t d, size_t ni, size_t aj, double x[d],
    double means[ni][d], double invLs[ni][d][d], double pv[ni])
{
    // Only call with aj = 0, the first matrix.
    if (aj != 0)
        return -1.;
    double integr = 0.;
    integr = pdf_multivariate_normal_fast(d, x, means[aj], invLs[aj]);
    return integr * integr;
}
/* MAIN */
int
main()
{
    printf("\n");   // Skip a line for readability.
    int success = 0;   // 1 or -1
    int i = 0, j = 0;
    clock_t start, diff;
    int msec;


    /* LINEAR ALGEBRA TESTS */
    // matprod
    double matA [3][3] = {{1., 4., 3.}, {2., -1., 0.}, {-2., 4., 2.}};
    double matB [3][2] = {{1., 0.}, {-1., 2.}, {-2., 1.}};
    double matO [3][2];
    // Expected
    double expected00 [3][2] = {{-9., 11.}, {3., -2.}, {-10., 10.}};
    success = matprod(3, 3, 2, matA, matB, matO);
    if (success != 0){
        printf("matprod failed\n");
        return 1;
    }

    success = compare_2d(3, 2, matO, expected00);
    if (success != 0){
        printf("matprod failed a test\n");
        return 1;
    }
    else {
        printf("matprod worked on 3x3 times 3x2 matrices\n");
    }

    // Matrix on the left, vector on the right
    double vecU [3] = {-1., 1., 2.};
    double vecV1 [3];
    double expected01 [3] = {9., -3., 10.};   // A dot U
    success = ldot(3, 3, matA, vecU, vecV1);
    if (success != 0){
        printf("ldot failed\n");
        return 1;
    }
    success = compare_1d(3, vecV1, expected01);
    if (success != 0){
        printf("ldot failed a test\n");
        for (i = 0; i < 3; i++){
            printf("%f ", vecV1[i]);
        }
        printf("\n");
        return 1;
    }
    else {
        printf("ldot worked on 3x3 matrix times length-3 vector\n");
    }

    // row vector, matrix on the right
    double vecU2 [3] = {3., -1., 2.};
    double vecV2 [2];
    double expected02 [2] = {0., 0.};
    success = rdot(3, 2, vecU2, matB, vecV2);
    if (success != 0){
        printf("rdot failed\n");
        return 1;
    }
    success = compare_1d(2, vecV2, expected02);
    if (success != 0){
        printf("rdot failed a test\n");
        for (i = 0; i < 2; i++){
            printf("%f ", vecV2[i]);
        }
        printf("\n");
        return 1;
    }
    else {
        printf("rdot worked on length-3 vector times 3x2 matrix\n");
    }


    /* CHOLESKY TESTS */
    // Test Cholesky decomposition, with an acceptable matrix and a wrong one.
    // Example taken from Youtube: https://youtu.be/TprfUB3nI8Y
    double matA2 [3][3] = {{2, 4, -3}, {4, 14, -9}, {-3, -9, 12}};
    double matL [3][3];
    success = cholesky(3, matA2, matL);
    if (success != 0){
        printf("Cholesky decomposition failed on a valid input\n");
        return 1;
    }
    // Check the output is indeed lower triangular, and zero elsewhere
    for (i = 0; i < 3; i++){
        for (j = i+1; j < 3; j++){
            if (matL[i][j] != 0.){
                printf("Cholesky returned a non-lower triangular L\n");
                printf("Faulty element: L[%d][%d] = %f\n", i, j, matL[i][j]);
                return 1;
            }
        }
    }
    // Check that the matrix product of matL and its transpose factorize A2
    double matLT [3][3];
    transpose_mat(3, 3, matL, matLT);
    matprod(3, 3, 3, matL, matLT, matA);
    success = compare_2d(3, 3, matA, matA2);
    if (success != 0){
        printf("Cholesky decomposition failed to deliver LL^T = A\n");
        return 1;
    }
    else
        printf("cholesky correctly factorized A = LL^T\n");

    // Test the inversion of a cholesky
    double invL [3][3];
    success = inv_triang(3, matL, invL);
    if (success == 1){
        printf("inv_triang failed to recognize an invertible matrix\n");
    }
    // Check that we recover the identity matrix
    double identity [3][3] = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
    double product [3][3];
    matprod(3, 3, 3, matL, invL, product);
    success = compare_2d(3, 3, product, identity);
    if (success != 0){
        printf("inv_triang failed to correctly invert a triangular matrix\n");
        return 1;
    }
    matprod(3, 3, 3, invL, matL, product);
    if (success != 0){
        printf("inv_triang failed to correctly invert a triangular matrix\n");
        return 1;
    }
    else{
        printf("inv_triang successfully inverted matL. L^-1 = \n");
        for (i = 0; i < 3; i++)
            printf("\t[%f %f %f]\n", invL[i][0], invL[i][1], invL[i][2]);
    }

    // Test whether the algorithm can handle incorrect inputs
    double matA3 [3][3] = {{2,  4, -3},
                           {4,  2, -3},
                           {-3, -3, 9}}; // Negative eigenvalue
    // This should have failed!
    success = cholesky(3, matA3, matL);
    if (success != 1){
        printf("Cholesky decomposition failed to detect a nonpositive matrix\n");
        return 1;
    }
    else
        printf("Cholesky detected successfully non-positive definite matrix\n");

    // Break the symmetry of A2
    matA2[0][1] = 1;
    success = cholesky(3, matA2, matL);
    if (success != 2){
        printf("Cholesky failed to detect a non-symmetric matrix\n");
        return 1;
    }
    else
        printf("Cholesky detected successfully a non-symmetric matrix\n");


    /* RANDOM SAMPLING TESTS */
    // Generate normal(0, 1) samples and check their mean, variance, skewness, kurtosis.
    start = clock();  // timing the whole operation

    size_t nsamples = 1000000;
    double *container = malloc(nsamples*sizeof(double));
    dsfmt_t rand_state;
    int seed = 43952;
    dsfmt_init_gen_rand(&rand_state, seed);
    container = gen_univariate_normal01_samples(nsamples, container, rand_state);
    double mean=0., variance=0., skewness=0., exc_kurtosis=0.;
    compute_1d_moments(
        nsamples, container, &mean, &variance, &skewness, &exc_kurtosis
    );
    savetobin(nsamples, container, "tests/1d_normal_samples.bin");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("That all took: %d seconds %d ms\n", msec/1000, msec%1000);

    if (fabs(mean) > 1/sqrt(nsamples)){
        printf("Mean %f seems too different from 0\n", mean);
        return 1;
    }
    // variance of variance estimate in 1/N, std on it is thus 1/sqrt(N)
    else if (fabs(variance - 1) > 1/sqrt(nsamples)){
        printf("Variance %f seems too different from 1\n", variance);
        return 1;
    }
    // variance of skewness is in N^2/N^3, so std on it is 1/sqrt(N) too.
    else if (fabs(skewness) > sqrt(6)/sqrt(nsamples)){
        printf("Skewness %f seems too different from 0\n", skewness);
        return 1;
    }
    else if (fabs(exc_kurtosis) > sqrt(24.)/sqrt(nsamples)){
        printf("Exc. kurtosis %f seems too different from 0\n", exc_kurtosis);
        return 1;
    }
    else {
        printf(
            "All statistics of the gaussian seem OK: \n"
            "mean = %f, variance = %f, skewness = %f, excess kurtosis = %f\n",
            mean, variance, skewness, exc_kurtosis
        );
    }

    // Now, try generating multivariate normal samples
    start = clock();

    double matCov [2][2] = {{1., -0.5}, {-0.5, 1.}};
    double cholLCov [2][2];
    success = cholesky(2, matCov, cholLCov);
    if (success != 0){
        printf("Cholesky failed on a perfectly fine 2x2 matrix");
        return 1;
    }
    double matMean [2] = {3., -3.};
    nsamples = 10000;
    double (*sample_vecs)[2] = malloc(sizeof(double[nsamples][2]));
    gen_multivariate_normal_samples(10000, 2, matMean,
        cholLCov, sample_vecs, rand_state);

    // Write that to an array, analyze in Python
    save2dtobin(nsamples, 2, sample_vecs, "tests/2d_multinormal_samples.bin");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("That 2D sampling took: %d seconds %d ms\n", msec/1000, msec%1000);

    /* PROB. DENSITY FUNCTION TESTS */
    // Evaluate f_X(x) on a grid, save in a file, reopen in Python to
    // compare to the true distribution.
    double dx = 0.05, dy = 0.05;
    double xy [2] = {-10., 10.};
    double (*f_X)[401] = malloc(sizeof(double[401][401]));

    // grid as we are facing the matrix, y increasing to the top of the matrix
    start = clock();
    for (j = 0; j < 401; j++){
        for (i = 0; i < 401; i++){
            f_X[j][i] = pdf_multivariate_normal(2, xy, matMean, cholLCov);
            xy[0] += dx;
        }
        xy[1] -= dy;
        xy[0] = -10.;  // reinitialize this x for the next loop
    }
    // Write that to an array, analyze in Python
    save2dtobin(401, 401, f_X, "tests/multinormal_pdf.bin");
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Covering the grid with default version took: %d seconds %d ms\n", msec/1000, msec%1000);

    // Check that the same results are obtained with the fast version
    // Clock the time difference as well, which is the only motivation.
    start = clock();
    double invLCov [2][2];
    double (*f_X2)[401] = malloc(sizeof(double[401][401]));
    inv_triang(2, cholLCov, invLCov);
    xy[0] = -10.;
    xy[1] = 10.;
    for (j = 0; j < 401; j++){
        for (i = 0; i < 401; i++){
            f_X2[j][i] = pdf_multivariate_normal_fast(2, xy, matMean, invLCov);
            xy[0] += dx;
        }
        xy[1] -= dy;
        xy[0] = -10.;  // reinitialize this x for the next loop
    }
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Covering the grid with fast version took: %d seconds %d ms\n", msec/1000, msec%1000);

    // Check that the result is the same
    success = compare_2d(401, 401, f_X, f_X2);
    if (success != 0){
        printf("The fast version gave different results\n");
        return 1;
    }
    else
        printf("The fast version gave the same result\n");

    /* MONTE CARLO INTEGRATION TESTS */
    /* mcint_gaussdist(size_t dim, size_t n_ins, size_t argj,
        double integrand(size_t d, double x[d], size_t ni, size_t aj,
            double means[ni][d], double invLs[ni][d][d], double pv[ni]),
        double pvec[n_ins], double mat1[n_ins][dim], double mat2[n_ins][dim][dim],
        double matinv2[n_ins][dim][dim], dsfmt_t rand_state, size_t nsmp,
        double tol, double *abserror, double *integ_res)
    */
    double p_vector [2] = {0.25, 0.75};  // don't start at the optimum
    double matmeans[2][2] = {{6, 6}, {-6, -6}};  // Two opposite gaussians
    double matcovs[2][2][2] = {{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}};
    double matcholLs[2][2][2];
    double matinvLs[2][2][2];
    // Find L and L^-1 for each identity matrix... just another check
    for (i = 0; i < 2; i++){
        success = cholesky(2, matcovs[i], matcholLs[i]);
        success += inv_triang(2, matcholLs[i], matinvLs[i]);
        success += compare_2d(2, 2, matcovs[i], matcholLs[i]);
        success += compare_2d(2, 2, matcovs[i], matinvLs[i]);
        if (success != 0){
            printf("inv_triang failed on the identity matrix...\n");
            return 1;
        }
    }
    double integral_result = 0., integral_err = 0.;
    double rtol = 0.001;
    nsamples = 100000;
    nsamples = mcint_gaussdist(
        2, 2, 0, integrand_test_normsquared, p_vector, matmeans, matcholLs,
        matinvLs, rand_state, nsamples, rtol, &integral_result, &integral_err
    );
    if (integral_err / integral_result > rtol){
        printf("mcint_gaussdist failed to stop after desired accuracy\n");
        printf("Integration result: %f pm %f", integral_result, integral_err);
        return 1;
    }
    // This integral result should give 1/(4*pi)
    if (fabs(integral_result * (4. * M_PI) - 1.) > rtol){
        printf("mcint_gaussdist gave %f pm %f", integral_result, integral_err);
        return 1;
    }
    else {
        printf("mcint_gaussdist integrated f_x^2 within %f %%, giving:\n",
                rtol*100);
        printf("\t%f pm %f\n", integral_result, integral_err);
        printf("And used %ld samples to do so\n", nsamples);
    }
    nsamples = 100000;
    // Also test the cj integrand. Evaluate it at one of the means, say,
    // of the binary gaussian channel
    /* cj_integrand_gaussout(size_t dim, size_t n_ins, size_t argj,
        double x[dim], double means[n_ins][dim], double invLs[n_ins][dim][dim],
        double p_vec[n_ins])
    */
    integral_result = cj_integrand_gaussout(2, 2, 0, matmeans[0], matmeans,
        matinvLs, p_vector);
    if (fabs(integral_result / log(4.) * (2.*M_PI) - 1.) > 1e-3){
        printf("The c_j integrand is not behaving as expected\n");
        printf("%f \n", integral_result);
        return 1;
    }
    else{
        printf("c_j integrand gives the expected result for a simple case\n");
    }


    /* ULTIMATE TEST: BLAHUT-ARIMOTO CHANNEL CAPACITY */
    /* Test on a binary gaussian channel with low noise.
    * Expect C = 0.9999... bits, p_opt = 1/2, 1/2 */
    /* blahut_arimc_gaussout(size_t dimension, size_t ninputs,
        double pvec[ninputs], double means[ninputs][dimension],
        double covmats[ninputs][dimension][dimension], double rtol)
    */
    double capacity = 0.;
    seed = 946373;
    capacity = blahut_arimc_gaussout(2, 2, p_vector, matmeans, matcovs, rtol, seed);
    // Convert to bits (the algorithm returns nits)
    capacity = capacity / log(2.);
    if (fabs(capacity - 1.) > rtol * 1.){
        printf("Channel capacity = %f bits NOT WHAT EXPECTED\n", capacity);
        return 1;
    }
    else
        printf("Channel capacity = %f bits IS CORRECT\n", capacity);
    if (fabs(p_vector[0] - p_vector[1]) > rtol*0.5){
        printf("But optimal input distribution {%f %f} NOT WHAT EXPECTED\n",
            p_vector[0], p_vector[1]);
        return 1;
    }
    else {
        printf("Optimal input distribution {%f %f} IS CORRECT\n\n",
            p_vector[0], p_vector[1]);
    }

    // Another Blahut-Arimoto test: 4 inputs symmetrically distributed
    // So the optimal prob distrib is 1/4 for each; unsure about capacity.
    // Make this a bit more challenging by causing a little overlap
    double p_vector2 [4] = {0.1, 0.3, 0.5, 0.1};  // don't start at the optimum
    double matmeans2[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    double matcovs2[4][2][2] = {{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}},
                                {{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}};

    rtol = 0.01;
    nsamples = 100000;
    seed = 946373;
    capacity = blahut_arimc_gaussout(
        2, 4, p_vector2, matmeans2, matcovs2, rtol, seed);
    // Convert to bits (the algorithm returns nits)
    capacity = capacity / log(2.);

    printf("Capacity for 4 overlapping identical gaussians = %f bits\n", capacity);

    if (fabs(p_vector2[0] - 0.25) > rtol ||
        fabs(p_vector2[1] - 0.25) > rtol ||
        fabs(p_vector2[2] - 0.25) > rtol ||
        fabs(p_vector2[3] - 0.25) > rtol)
    {
        printf("But optimal input distribution {%f %f %f %f} NOT EXPECTED\n",
            p_vector2[0], p_vector2[1], p_vector2[2], p_vector2[3]);
        return 1;
    }
    else {
        printf("Optimal input distribution {%f %f %f %f} IS CORRECT\n\n",
            p_vector2[0], p_vector2[1], p_vector2[2], p_vector2[3]);
    }

    /* Another test, comparison with a method using stochastic grad. descent
    * and the Kraskov binless estimator (Grabowski, 2019):
    * https://github.com/pawel-czyz/channel-capacity-estimator
    See my Python test code for more test cases taken from this paper.
    */
    double matcovs3[3][2][2] = {{{1., 0.}, {0., 1.}},
                                {{1., 0.}, {0., 1.}},
                                {{1., 0.}, {0., 1.}}};
    double matmeans3[3][2] = {{0., 0.}, {0., 1.}, {3., 3.}};
    double p_vector3[3] = {0.25, 0.25, 0.5};
    rtol = 0.01;
    capacity = blahut_arimc_gaussout(
        2, 3, p_vector3, matmeans3, matcovs3, rtol, seed);
    // Convert to bits (the algorithm returns nits)
    capacity = capacity / log(2.);

    // Their relative tolerance is 0.04, because the Kraskov estimator
    // is biased usually above the true value, by a few 1/100ths of a bit,
    // for small k/N (k=nb neighbors, N=nb pts).
    // Then their optimal distrib may be different from the exact solution
    // even more, because the MI estimate is based on nearest-neighbors only.
    if (fabs(capacity - 1.0154510500713743) > 0.04 * 1.0154510500713743){
        printf("Channel capacity = %f bits NOT WHAT EXPECTED\n", capacity);
        return 1;
    }
    else
        printf("Channel capacity = %f bits IS CORRECT\n", capacity);

    if (fabs(p_vector3[0] - 0.33343804) > 0.03
        || fabs(p_vector3[1] - 0.19158363) > 0.03
        || fabs(p_vector3[2] - 0.4749783) > 0.03){
        printf("But optimal input distribution {%f %f %f} NOT WHAT EXPECTED\n",
            p_vector3[0], p_vector3[1], p_vector3[2]);
        return 1;
    }
    else {
        printf("Optimal input distribution {%f %f %f} IS CORRECT\n\n",
            p_vector3[0], p_vector3[1], p_vector3[2]);
    }

    printf("ALL TESTS PASSED SUCCESSFULLY! (but is that enough?)\n\n");
    free(container);
    free(sample_vecs);
    free(f_X);
    free(f_X2);
    return 0;
}
