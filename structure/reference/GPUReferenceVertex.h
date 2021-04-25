#ifndef GPU_REFERENCE_VERTEX_H
#define GPU_REFERENCE_VERTEX_H

#include "structure/reference/ReferenceVertex.h"
#include <string>

class GPUReferenceVertex : public ReferenceVertex {
    public:
        size_t gpu_vertex_id;

        GPUReferenceVertex(size_t gpu_vertex_id, size_t cpu_vertex_id, std::string vertex_label)
        : ReferenceVertex(cpu_vertex_id, vertex_label) {
            this->gpu_vertex_id = gpu_vertex_id;
        }

        /**
            Construct a new GPUReferenceVertex from a CPU Vertex.
            The new reference Vertex won't be linked to the original CPU Vertex.
        **/
        GPUReferenceVertex(BitVertex* v, size_t gpu_vertex_id) {
            this(gpu_vertex_id, v->id(), v->label());
        }

};

#endif