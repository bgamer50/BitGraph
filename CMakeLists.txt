cmake_minimum_required(VERSION 3.10)

project(
    BITGRAPH
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(POSITION_INDEPENDENT_CODE True)
set(INTERFACE_POSITION_INDEPENDENT_CODE True)

add_library(
    bitgraph

    structure/BitGraph_construct.cpp
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

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE
            bitgraph   
            "../gremlin++/libgremlinxx.a"
            "../maelstrom/maelstrom.a"
    )

endfunction()

# todo execs

function(AddBitGraphTest EXEC_NAME)
    
    add_executable(${EXEC_NAME} ${ARGN})

    set_target_properties(
        ${EXEC_NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "test/bin"
    )

    target_link_libraries(
        ${EXEC_NAME}
        PRIVATE
            bitgraph   
            "../gremlin++/libgremlinxx.a"
            "../maelstrom/maelstrom.a"
    )

endfunction()

# todo tests