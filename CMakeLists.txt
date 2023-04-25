cmake_minimum_required(VERSION 3.10)

project(
    BITGRAPH
    VERSION 0.7
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

# set(CMAKE_CUDA_ARCHITECTURES "61;62;70;75;86")
set(CMAKE_CUDA_ARCHITECTURES "61")

set(POSITION_INDEPENDENT_CODE True)
set(INTERFACE_POSITION_INDEPENDENT_CODE True)

add_library(
    bitgraph

    structure/matrix/GPUSparseMatrix.cu
    structure/matrix/CPUSparseMatrix.cpp
    structure/matrix/Adjacency.cu
    
    structure/memory/TypeErasure.cu
    structure/memory/primitives/hash_primitives/HashFilterValid.cu
    structure/memory/primitives/hash_primitives/HashGetScan.cu
    structure/memory/primitives/hash_primitives/HashInsert.cu
    structure/memory/primitives/hash_primitives/HashSetScatter.cu
    structure/memory/primitives/hash_primitives/HashSetSort.cu
    structure/memory/primitives/ArrayPrimitives.cu
    structure/memory/ArrayFunctions.cu
    structure/memory/GPUDynamicCastingHelper.cu
    structure/memory/GPUHashTable.cu
    structure/memory/GPUPropertyTable.cu

    structure/BitVertex.cpp
    structure/BitEdge.cpp
    structure/CPUGraph.cpp
    structure/GPUVertex.cu
    structure/GPUGraph.cu

    traversal/CPUGraphTraversalSource.cpp
    traversal/GPUGraphTraversalSource.cu

    step/cpu/BitGraphStep.cpp
    step/cpu/CPUSubgraphExtractionStep.cpp
    step/cpu/HasWithIndexStep.cpp
    step/cpu/IndexStep.cpp

    step/gpu/GPUTraversalHelper.cu
    step/gpu/GPUBindStep.cu
    step/gpu/GPUUnbindStep.cu
    step/gpu/GPUVertexStep.cu
    step/gpu/GPUGlobalReductionStep.cu

    step/hybrid/GPUAddPropertyStep.cu
    step/hybrid/GPUGraphStep.cu
    step/hybrid/GPUPropertyStep.cu

    strategy/BitGraphStrategy.cpp
    strategy/GPUGraphStrategy.cu

    algorithm/ConnectedComponentsGPUGraphAlgorithm.cu
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
        "/mnt/bitgraph" # bitgraph path
        "/mnt/gremlin++" # gremlinxx path
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
            "/mnt/gremlin++/libgremlinxx.a"
    )

endfunction()

AddBitGraphExec(components.exe examples/components.cu)
AddBitGraphExec(ast.exe examples/ast.cu)
AddBitGraphExec(deg.exe examples/deg.cu)
AddBitGraphExec(dos.exe examples/dos.cu)
AddBitGraphExec(lca.exe examples/lca.cu)

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
            "/mnt/gremlin++/libgremlinxx.a"
    )

endfunction()

AddBitGraphTest(TestGPUHashTable.exe test/TestGPUHashTable.cu)
