""" See
https://medium.com/@joshua.massover/python-c-extension-example-cef86ffab4ed

Execute from the directory containing this script as
    python setup_chancapmc.py build_ext --inplace
build_ext to keep things fully local and not mess with the python installation
-- inplace to copy the final built module to the local directory.

Later, from mi_param_space directory, import as follows in Python:
    import chancapmc.testmod


Using dSFMT Mersenne Twister. To include, need to #include "dSFMT-src/dSFMT.h"
in whatever code uses the generator, to add dSFMT-src/dSFMT.c to the c files
being compiled, and to compile with the option -DDSFMT_MEXP=<chosen exp>,
typically 19937. No need to define the constant DSFMT_MEXP in a file
(although I do in gaussfuncs.h).

See docs in dSFMT-src/html/ to know how to 1) test the package
    and 2) how to include it. For better performance on CPUs > Pentium IV,
    use the options -O3 -msse2 -fno-strict-aliasing -DHAVE_SSE2=1

On linux (at least the physics servers) we also need -fPIC (because of VLAs?)
But it's always good to use anyways for something compiled to an .so file.
"""
from setuptools import setup, Extension
import numpy as np

def main():
    # Add -DSFMT_MEXP=19937 and add file "dSFMT/dSFMT.c" to the compilation.
    ext = Extension("chancapmc",
            sources=["chancapmcmodule.c", "blahut_arimoto.c", "gaussfuncs.c",
                "mcint.c", "dSFMT/dSFMT.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-msse2", "-fno-strict-aliasing",
                "-DHAVE_SSE2=1", "-DDSFMT_MEXP=19937", "-fPIC"]
            )
    setup(name="chancapmc",
          version="0.1.0",
          description="Python interface to a C Blahut-Arimoto algorithm",
          author="frbourassa",
          author_email="francois.bourassa4@mail.mcgill.ca",
          ext_modules=[ext]
    )

if __name__ == "__main__":
    main()
