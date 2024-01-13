cmake_minimum_required(VERSION 3.10)

project(
    BITGRAPH
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-forward-unknown-to-host-compiler --expt-relaxed-constexpr --expt-extended-lambda")

# Minimum supported arch is 75
set(CMAKE_CUDA_ARCHITECTURES "86")

set(POSITION_INDEPENDENT_CODE True)
set(INTERFACE_POSITION_INDEPENDENT_CODE True)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

add_library(
    bitgraph SHARED

    structure/BitGraph_construct.cpp
    structure/BitGraph_access.cpp
    structure/BitGraph_convert.cpp
    structure/BitGraph_query.cpp

    traversal/BitGraphTraversalSource.cpp

    step/BitGraphVStep.cpp
    
    strategy/BitGraphStrategy.cpp
)

target_compile_options(
    bitgraph
    PUBLIC
        -O3
        -funsafe-math-optimizations
        -frename-registers
        -funroll-loops
        -fsized-deallocation
)

target_include_directories(
    bitgraph
    PUBLIC
        "include/" # bitgraph path
        "../gremlin++" # gremlinxx path
        "../maelstrom/include" # maelstrom path
)

function(AddBitGraphExec EXEC_NAME)
    
    add_executable(${EXEC_NAME} ${ARGN})

    set_target_properties(
        ${EXEC_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "bin"
    )

    target_link_directories(
        ${EXEC_NAME}
        PRIVATE
          "../maelstrom/" 
          "../gremlin++/"
     )

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE
            bitgraph   
            gremlinxx
            maelstrom
    )

endfunction()

AddBitGraphExec(
    "components.exe" "examples/components.cu"
)

AddBitGraphExec(
    "edge_query.exe" "examples/edge_query.cu"
)

AddBitGraphExec(
    "shortest_path.exe" "examples/shortest_path.cu"
)

function(AddBitGraphTest EXEC_NAME)
    
    add_executable(${EXEC_NAME} ${ARGN})

    set_target_properties(
        ${EXEC_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "test/bin"
    )

    target_link_directories(
       ${EXEC_NAME}
       PRIVATE
         "../maelstrom/" 
         "../gremlin++/"
    )

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE
            bitgraph
            gremlinxx
            maelstrom
    )

endfunction()

AddBitGraphTest(
    "test_basic.exe" "test/test_basic.cpp"
)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(PyBitGraph PyBind/PyBitGraph.cpp)
target_link_libraries(
    PyBitGraph 
    PRIVATE  
        bitgraph 
        gremlinxx 
        maelstrom
)
target_link_directories(
       PyBitGraph
       PRIVATE
         "../maelstrom/" 
         "../gremlin++/"
)