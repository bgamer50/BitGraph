Version 0.1.0 (October 18, 2019)

BitGraph is a graph backend that uses the Gremlin traversal language.  It is currently under development.  Version 0.1.0 deprecated OpenCL acceleration, in favor of a new solution currently under development.
It is licensed under the Apache 2.0 license.

Currently BitGraph supports the following platforms:
  - Windows (Cygwin64)
  - Linux

BitGraph is a header library, so setting the include directory will suffice.

High-level instructions for building BitGraph projects:
1. Ensure Gremlin++ is included.
2. Ensure the Boost libraries are included (at minimum all dependencies for boost/any.hpp).

