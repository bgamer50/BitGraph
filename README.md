Version 1.0.0 (Expected September 2023)

This will be a major refactor of BitGraph that uses _Maelstrom_ for almost all operations, including sparse matrix operations, vector operations, and hash tables.
Building on work from before, it will delegate even more work to Gremlin++ and its default set of steps, which have been significantly overhauled.
Major performance and memory improvements are expected with this release.

---------------------------------------------------------------

Version 0.6.1 (October 11, 2022)

BitGraph is a graph backend that uses the Gremlin traversal language.  It contains both the original CPU backend, and a new Hybrid backend that stores the graph structure on the GPU and properties on the CPU.
It is licensed under the Apache 2.0 license.  Several papers on BitGraph are in the works, with publication expected in 2023.

Version 0.6.1 primarily includes updates to match the semantic changes in Gremlin++ v0.6.1, as well as some other less major updates.  The main Makefile now builds with g++11 and nvcc.  Improved builds
(most likely through conda) will be included in the next update).

BitGraph is a header library, so to use BitGraph in your project, simply set it as an include directory.

---------------------------------------------------------------

Version 0.6.0 (June 11, 2022)

BitGraph is a graph backend that uses the Gremlin traversal language.  It contains both the original CPU backend, and a new Hybrid backend that stores the graph structure on the GPU and properties on the CPU.
It is licensed under the Apache 2.0 license.  Several papers on BitGraph are in the works, with publication expected in late 2022-2023.

BitGraph is a header library, so to use BitGraph in your project, simply set it as an include directory.

---------------------------------------------------------------
Version 0.1.0 (October 18, 2019)

BitGraph is a graph backend that uses the Gremlin traversal language.  It is currently under development.  Version 0.1.0 deprecated OpenCL acceleration, in favor of a new solution currently under development.
It is licensed under the Apache 2.0 license.

Currently BitGraph supports the following platforms:
  - Linux

BitGraph is a header library, so setting the include directory will suffice.
