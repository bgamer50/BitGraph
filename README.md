Version 0.1.0 (October 18, 2019)

BitGraph is a graph backend that uses the Gremlin traversal language.  It is currently under development.  Version 0.1.0 deprecated OpenCL acceleration, in favor of a new solution currently under development.
It is licensed under the Apache 2.0 license.

Currently BitGraph supports the following platforms:
  - Linux

BitGraph is a header library, so setting the include directory will suffice.
Currently, clang is the only officially supported compiler due to test environment constraints.

High-level instructions for building BitGraph projects:
1. Ensure Gremlin++ is included.
2. Ensure the Boost libraries are included (at minimum all dependencies for boost/any.hpp).
3. If using the GPU, install clang 13 and set the following variables in the Makefile: CC (i.e. /opt/llvm/bin/clang++), CUDALIB (i.e. /usr/local/cuda/lib64), CUDAARCH (i.e. sm_53)
   If you don't know the arch version, it can be looked up using the deviceQuery program that is included as a CUDA sample program in the CUDA toolkit (i.e. /usr/local/cuda/samples/1_Utilities/deviceQuery).
