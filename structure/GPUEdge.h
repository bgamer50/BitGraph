#ifndef GPU_EDGE_H
#define GPU_EDGE_H

#include "structure/reference/ReferenceEdge.h"
#include "structure/Vertex.h"
#include "structure/GPUVertex.h"

#include <string>

typedef std::pair<int32_t, int32_t> edge_pair_t;

inline std::string get_gpu_edge_id(int32_t out, int32_t in) { return "E{" + std::to_string(out) + "->" + std::to_string(in) + "}"; }

class GPUEdge : public ReferenceEdge {
    public:
        std::string gpu_edge_id; // should not be used for comparisons; technically the (out, in) pair is the id.

        GPUEdge(uint64_t cpu_edge_id, std::string label, GPUVertex* out, GPUVertex* in)
        : ReferenceEdge(cpu_edge_id, label, out, in) {
            this->gpu_edge_id = get_gpu_edge_id(out->gpu_vertex_id, in->gpu_vertex_id);
        }
};

#endif