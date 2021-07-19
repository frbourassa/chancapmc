#include <time.h>
#include <stdlib.h>

#define DSFMT_MEXP 19937
#include "/Users/francoisb/cpackages/dSFMT-src-2.2.3/dSFMT.h"


int main(){
    // Initialization
    int nb_rand;
    nb_rand = get_min_array_size();  //  at least (SFMT_MEXP / 128) * 2
    size_t ntest = 10000000;
    double *array = malloc(ntest * sizeof(double)); // on the heap

    dsfmt_t rand_state;
    int seed = 14928;
    dsfmt_init_gen_rand(&rand_state, seed);
    int msec;
    clock_t start, diff;
    // First call is slower
    // Check that I can pass in part of an array only
    dsfmt_fill_array_close1_open2(&rand_state, &array[1000], ntest-1000);


    // close1_open2
    start = clock();
    dsfmt_fill_array_close1_open2(&rand_state, array, ntest);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time for close1_open2: %d seconds %d ms\n", msec/1000, msec%1000);

    // open0_close1
    start = clock();
    dsfmt_fill_array_open_close(&rand_state, array, ntest);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time for open0_close1: %d seconds %d ms\n", msec/1000, msec%1000);

    free(array); 
    return 0;
}
