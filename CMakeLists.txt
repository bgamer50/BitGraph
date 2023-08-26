cmake_minimum_required(VERSION 3.10)

project(
    BITGRAPH
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-forward-unknown-to-host-compiler --expt-relaxed-constexpr --expt-extended-lambda")

# Minimum supported arch is 75
set(CMAKE_CUDA_ARCHITECTURES "86")

set(POSITION_INDEPENDENT_CODE True)
set(INTERFACE_POSITION_INDEPENDENT_CODE True)

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

function(AddBitGraphTest EXEC_NAME)
    
    add_executable(${EXEC_NAME} ${ARGN})

    set_target_properties(
        ${EXEC_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "test/bin"
            LINK_OPTIONS "-g"
            COMPILE_OPTIONS "-g"
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