#pragma once

#include <inttypes.h>
#include "gremlinxx/gremlinxx.h"

// TODO write a function for properly updating this data structure
typedef struct gpu_traverser_info {
    void* traversers;
    size_t num_traversers;
    gremlinxx::comparison::C traverser_dtype;
    TraverserSet original_traversers;
    std::vector<std::pair<size_t*, size_t>> paths;
} gpu_traverser_info_t;

enum reduction_type{MIN, MAX, SUM, MUL};

/**
    Copy data from a traversal over graph elements (Vertex,Edge)
    to the GPU.
**/
template<typename T>
void* to_gpu(TraverserSet& traversers);

void* C_TO_GPU(gremlinxx::comparison::C c, TraverserSet& traversers);

/**
    Removes duplicates from A (modifies the original device array).
    Returns the original indices of the unique elements and the
    number of unique elements (size of index array and new size of A).

    This method is NOT stable (may choose unique indices that are not
    the first occurrence of that value).

    V_ptr: The array pointing to each deduplicated value's origin
    V_size: The length of V and V_ptr (# of deduplicated elements).
**/
std::tuple<size_t*, size_t> pick_unique(size_t* A, size_t N);

/**
   Collapses a path down into a single output origin array.
   traverser_info: the traverser info (contains path & other info)
**/
std::vector<size_t> collapse_path(gpu_traverser_info_t& traverser_info, bool free_memory);

template<typename T>
void retrieve_new_traversers(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);

void C_RETRIEVE_NEW_TRAVERSERS(GraphTraversal* parent_traversal, TraverserSet& output_traversers, gpu_traverser_info_t& traverser_info);
