/* Module to launche the Blahut-Arimoto for a discrete-input,
gaussian continuous-output channel (i.e. optimize MI of the channel by
optimizing the discrete input distribution), using Monte Carlo integration
for the integrals, by just passing three numpy arrays and a couple
more parameters from Python:
Args:
    pymeans (2darray): array of output mean vectors for each conditional
        distribution. Indexed [input condition, output dimension]
    pycovs (3darray): array of output covariance matrices for each
        conditional distribution.
        Indexed [input condition, output dimension, output dimension]
    pyinputs (1darray): list of discrete input variable values x_i.
        They are not directly used (we just use an integer index i) but
        we care about the length of this array.
    captol (float): relative tolerance on the channel capacity. Default 0.01
    mctol (float): relative error tolerance for MC integration, default 0.01

NOTE: The first dimension of all three arrays should, obviously, be the same,
    and equal to the number of possible input values.

Returns (computed in C and converted to Python objects before return):
    chancap (float): channel capacity, which is just a float!
    p_opt (1darray): optimal probability mass function for the discrete inputs.


The functions are organized in the following manner:

    chancapmcmodule.c: interface between Python and C code; calls the
        Blahut-Arimoto algorithm and defines functions for the gaussian
        channel probability density with the inputted Python arrays,
        wrapping the generic gaussians in gaussfuncs.c

    gaussfuncs.h : defines gaussian distribution and sampling in pure C,
        using a Mersenne Twister extension.
        A bit slower than defining gaussians in-place, but cleaner because the
        Python interface and Mersenne Twister already add lots of overhead code

    blahut_arimoto.h : Blahut-Arimoto algorithm in pure C

    mcint.h : Monte Carlo integration in pure C. takes a sampling distribution
        as input, so no need to include the Mersenne Twister here.

WARNING: this code uses the C99 standard for variable-length arrays, because
this is more efficient than pointers-of-pointers arrays (double **)


@author: frbourassa
November 25, 2020
*/

// Python-related headers and tags
#include "blahut_arimoto.h"  // includes stddef.h
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// Prototypes
int copy_1darray(PyArrayObject * pyarr, size_t n, double carr[n]);
int copy_2darray(PyArrayObject * pyarr, size_t m, size_t n, double carr[m][n]);
int copy_3darray(PyArrayObject * pyarr, size_t n0, size_t n1, size_t n2,
    double carr[n0][n1][n2]);
PyArrayObject *create_1d_PyArray(size_t nins, double pvec[nins]);
static PyObject * chancapmcmodule_ba_discretein_gaussout(PyObject *self,
    PyObject *args);
PyMODINIT_FUNC PyInit_chancapmc(void);


/* Utility functions to copy 3d, 2d or 1d float64 arrays to C ndarrays
created outside of the call, on the stack. */
int
copy_1darray(PyArrayObject * pyarr, size_t n, double carr[n])
{
    if ((size_t) PyArray_SIZE(pyarr) != n)
        return 1;
    size_t i = 0;
    for (i = 0; i < n; i++){
        carr[i] = * (double *) PyArray_GETPTR1(pyarr, i);
    }
    return 0;
}

int
copy_2darray(PyArrayObject * pyarr, size_t m, size_t n, double carr[m][n])
{
    // Create rows
    if ((size_t) PyArray_DIMS(pyarr)[0] != m
        || (size_t) PyArray_DIMS(pyarr)[1] != n){
        return 1;
    }
    size_t i = 0, j = 0;
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            carr[i][j] = *(double *) PyArray_GETPTR2(pyarr, i, j);
        }
    }
    return 0;
}

int
copy_3darray(PyArrayObject * pyarr, size_t n0, size_t n1, size_t n2, double carr[n0][n1][n2])
{
    // Create rows
    if ((size_t) PyArray_DIMS(pyarr)[0] != n0
        || (size_t) PyArray_DIMS(pyarr)[1] != n1
        || (size_t) PyArray_DIMS(pyarr)[2] != n2){
        return 1;
    }
    size_t i = 0, j = 0, k = 0;
    for (i = 0; i < n0; i++){
        for (j = 0; j < n1; j++){
            for (k = 0; k < n2; k++)
            carr[i][j][k] = *(double *) PyArray_GETPTR3(pyarr, i, j, k);
        }
    }
    return 0;
}

// Just a souvenir, would work with C < C99
double ***
copy_3darray_alloc(PyArrayObject * pyarr)
{
    // Create rows
    double ***carr;
    carr = PyMem_Malloc(PyArray_DIMS(pyarr)[0] * sizeof(**carr));
    if (carr == NULL){
        PyErr_SetString(PyExc_ValueError, "Could not copy 3d array");
        return NULL;
    }
    int i = 0, j = 0, k = 0;
    for (i = 0; i < PyArray_DIMS(pyarr)[0]; i++){
        // Create another matrix in the stack of matrices
        carr[i] = PyMem_Malloc(PyArray_DIMS(pyarr)[1] * sizeof(*(carr[i])));
        if (carr[i] == NULL){
            PyErr_SetString(PyExc_ValueError, "Memory issue to copy 3d array");
            return NULL;
        }
        // In each row of one of the matrices, create columns
        for (j = 0; j < PyArray_DIMS(pyarr)[1]; j++){
            carr[i][j] = PyMem_Malloc(PyArray_DIMS(pyarr)[2] * sizeof(*(carr[i][j])));
            if (carr[i][j] == NULL){
                PyErr_SetString(PyExc_ValueError, "Memory issue to copy 3d array");
                return NULL;
            }
            // And fill that column with a copy of the original array.
            for (k = 0; k < PyArray_DIMS(pyarr)[2]; k++){
                carr[i][j][k] = *(double *) PyArray_GETPTR3(pyarr, i, j, k);
            }
        }
    }
    return carr;
}

PyArrayObject *
create_1d_PyArray(size_t nins, double cvec[nins])
{
    PyArrayObject *pyvec = NULL;
    npy_intp dims[1] = {nins};
    // Can't just give &cvec[0] because that C vector might be freed
    // at the end of the caller's function if it's a pure C vector.
    pyvec = (PyArrayObject *) PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    // Copy cvec into pyvec
    size_t j = 0;
    for (j = 0; j < nins; j++){
        *(double *) PyArray_GETPTR1(pyvec, j) = cvec[j];
    }
    // caller's responsibility to DECREF this new reference.
    return pyvec;
}

/* Main function called from Python and listed in the struct below */
static PyObject *
chancapmcmodule_ba_discretein_gaussout(PyObject *self, PyObject *args)
{
    /* Process args tuple. */
    printf("Processing inputs, checking dimensions. \n");
    PyObject *origmeans = NULL, *origcovs = NULL, *originputs = NULL;
    double captol = 0.001;  // optional argument.
    int seed = 946373; // optional seed argument.
    if (!PyArg_ParseTuple(args, "OOO|di", &origmeans, &origcovs, &originputs,
                            &captol, &seed)){
        return NULL;  // in case there is an error
    }

    /* First, make sure we have C-contiguous arrays of doubles;
    if not, the FromAny code makes copies. Based on the example in
        https://numpy.org/doc/stable/user/c-info.how-to-extend.html#example

    I do this because it's unclear whether GETPTR works for strided,
    misaligned, etc. array. Otherwise, iterating arbitrary arrays
    becomes insanely tedious; see examples in
    https://numpy.org/devdocs/reference/c-api/iterator.html#simple-multi-iteration-example
    */
    PyArrayObject *pymeans = NULL, *pycovs = NULL,  *pyinputs = NULL;
    // PyArray_ContiguousFromAny(PyObject* op, int typenum, int min_depth, int max_depth)
    pymeans = (PyArrayObject *) PyArray_ContiguousFromAny(origmeans, NPY_DOUBLE, 1, 3);
    pycovs = (PyArrayObject *) PyArray_ContiguousFromAny(origcovs, NPY_DOUBLE, 1, 3);
    pyinputs = (PyArrayObject *) PyArray_ContiguousFromAny(originputs, NPY_DOUBLE, 1, 3);
    if (pyinputs == NULL || pycovs == NULL || pymeans == NULL){
        Py_XDECREF(pymeans);
        Py_XDECREF(pycovs);
        Py_XDECREF(pyinputs);
        return NULL;
    }
    // Don't forget to DECREF the above if any error arises!

    /* Checking again that the input dimensions are correct */
    unsigned int meandim = PyArray_NDIM(pymeans);
    unsigned int covdim = PyArray_NDIM(pycovs);
    unsigned int inputdim = PyArray_NDIM(pyinputs);
    if (meandim != 2 || covdim != 3 || inputdim != 1) {
        PyErr_SetString(PyExc_ValueError, "means array must have dimension 2, "
                                    "covs array must have dimension 3, "
                                    "inputs arrau must have dimension 1, "
                                    "captol is an optional float, "
                                    "seed is an optional int. ");
        Py_XDECREF(pymeans); Py_XDECREF(pycovs); Py_XDECREF(pyinputs);
        return NULL;
    }

    /* Saving array shapes in size_t VLA arrays (rather than npy_intp) */
    printf("Converting shape arrays to simple ints\n");
    size_t meanshape [meandim];
    size_t covshape [covdim];
    size_t ninputs = (size_t) PyArray_SIZE(pyinputs);
    size_t i = 0;
    for (i = 0; i < meandim; i++)
        meanshape[i] = (size_t) PyArray_DIMS(pymeans)[i];

    for (i = 0; i < covdim; i++)
        covshape[i] = (size_t) PyArray_DIMS(pycovs)[i];

    if (meanshape[0] != ninputs || covshape[0] != ninputs){
        PyErr_SetString(PyExc_ValueError,
            "arrays means, covs, and inputs should have equal 1st dimension");
        Py_DECREF(pymeans); Py_DECREF(pycovs); Py_DECREF(pyinputs);
        return NULL;
    }
    else if (meanshape[1] != covshape[1] || meanshape[1] != covshape[2]){
        PyErr_SetString(PyExc_ValueError,
            "arrays means and covs should have equal 2nd and 3rd dimension");
        Py_DECREF(pymeans); Py_DECREF(pycovs); Py_DECREF(pyinputs);
        return NULL;
    }

    /* Copy C arrays, for easier slicing and avoiding Python API in headers */
    // Use VLA defined on the stack: we expect those to be small.
    size_t dimension = meanshape[1];
    double covs [ninputs][dimension][dimension];
    double means [ninputs][dimension];
    double inputs [ninputs];
    // We don't need this inputs vector, we just use input indices throughout
    // not the actual values of the input RV. TODO.

    // Initialize the starting probability mass function (uniform distrib.)
    double p_vector [ninputs];
    // For testing purposes, don't start with uniform initial distribution
    double sum_indices = ninputs * (ninputs+1) / 2;
    for (i = 0; i < ninputs; i++){
        if (captol > 1.)
            p_vector[i] = (i+1) / sum_indices;
        else
            p_vector[i] = 1. / ninputs;
    }

    printf("Copying 3D array covs\n");
    int code_ret;
    code_ret = copy_3darray(pycovs, covshape[0], covshape[1], covshape[2], covs);
    if (code_ret != 0) goto fail;
    printf("Copying 2D array means\n");
    code_ret = copy_2darray(pymeans, meanshape[0], meanshape[1], means);
    if (code_ret != 0) goto fail;
    printf("Copying 1D array inputs\n");
    code_ret = copy_1darray(pyinputs, ninputs, inputs);
    if (code_ret != 0) goto fail;

    // This means it is just a memory test
    if (captol > 10.){
        Py_DECREF(pymeans);
        Py_DECREF(pycovs);
        Py_DECREF(pyinputs);
        Py_INCREF(Py_None);
        return Py_None;
    }
    // This means it is a test case where we want captol = 0.001. We used
    // 10. > captol > 1. to initialize p_vector not uniform, because
    // that's the expected optimum.
    else if (captol > 1.){
        captol = 0.001;
    }

    // Run Blahut-Arimoto with the mean and covariance matrices
    double capacity = -1.;
    capacity = blahut_arimc_gaussout(
        dimension, ninputs, p_vector, means, covs, captol, seed);
    if (capacity < 0.) goto fail;

    // Convert to bits (the algorithm returns nits)
    capacity = capacity / log(2.);

    // Create a Python array from p_vector
    PyArrayObject *pypvec = NULL;
    pypvec = create_1d_PyArray(ninputs, p_vector);
    if (pypvec == NULL) goto fail;
    /* else we now own the reference to pypvec, got to DECREF it if fail, and
    * if everything goes OK, we return it and pass the ref. to the caller */

    // Wrap the capacity and the optimal input distribution in a Python Tuple
    PyObject *cap_and_vec = PyTuple_New(2);
    /* PyFloat_FromDouble returns a new reference, but the Tuple is
    * responsible for DECREF'ing it; if tuple is DECREF'ed, so is the float */
    PyTuple_SetItem(cap_and_vec, 0, PyFloat_FromDouble(capacity));
    /* We give ownership of pypvec to the tuple; no need to DECREF
    * pypvec ourselves anymore */
    PyTuple_SetItem(cap_and_vec, 1, (PyObject *) pypvec);

    // Free the copies of arrays we made before copying them again to C
    Py_DECREF(pymeans);
    Py_DECREF(pycovs);
    Py_DECREF(pyinputs);

    // Return the capacity and the optimal probability distribution
    // The caller is responsible for DECREF'ing the tuple if needed.
    return cap_and_vec;

    fail:
        Py_XDECREF(pymeans);
        Py_XDECREF(pycovs);
        Py_XDECREF(pyinputs);
        return NULL;
}


/* Define the module methods structure */
// This is a list of the methods callable as module.name from Python
// Each entry: {name, function, argument type, docstring}
char docs [] =  "Args:\n\tmeans (2darray): mean vectors for each input value"
                "\n\tcovs(3darray): covariance matrices for each input value"
                "\n\tinputs (1darray): discrete values of the input RV"
                "\n\tcaptol (float): rel. tol. for convergence of chan. cap."
                "\n\tmctol (float): rel. tol. for Monte Carlo integrals.";
static PyMethodDef ChanCapMcModuleMethods[] = {
    {"ba_discretein_gaussout", chancapmcmodule_ba_discretein_gaussout,
        METH_VARARGS, docs},
    {NULL, NULL, 0, NULL}  /* SENTINEL */
};


/* Define the Module specification structure.
    This is a list of a few attributes of the module, such as its name
    Note that this could be bypassed by using PyInit_Module (which takes
the methods struct and a name) rather than PyModule_Create (which just takes
the module structure defined here)
*/
static struct PyModuleDef chancapmcmodule = {
    PyModuleDef_HEAD_INIT,
    "chancapmc",  // module name
    "Module to compute the channel capacity of a discrete input, "
    "continuous vector output channel using the Blahut-Arimoto algorithm "
    "and Monte Carlo integration. ",  // Docstring, can be NULL
    -1,  /* size of per-interpreter state of the module,
            or -1 if the module keeps state in global variables. */
    ChanCapMcModuleMethods  // Methods struct; methods accessed as chancapmc.method
};


/* Initialization function; the only non-static one. */
PyMODINIT_FUNC
PyInit_chancapmc(void)
{
    import_array();
    PyObject *m;
    m = PyModule_Create(&chancapmcmodule); // arg: address of PyModuleDef struct
    // The INCREF is done in PyModule_Create; m is borrowed.
    if (m == NULL)
        return NULL;
    else
        return m;
}
