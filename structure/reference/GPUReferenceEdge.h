#ifndef GPU_REFERENCE_EDGE_H
#define GPU_REFERECNE_EDGE_H

#include "structure/reference/ReferenceEdge.h"
#include "structure/reference/ReferenceVertex.h"
#include "structure/reference/GPUReferenceVertex.h"

#include <string>

struct edge_pair_hash {
    size_t operator() (const std::pair<int32_t, int32_t>& p) const {
        size_t h = 0;
        boost::hash_combine(h, boost::hash_value(p.first));
        boost::hash_combine(h, boost::hash_value(p.second));
        return h;
    }
};

struct edge_pair_equals {
    size_t operator() (const std::pair<int32_t, int32_t>& p, const std::pair<int32_t, int32_t>& q) const {
        return p.first == q.first && p.second == q.second;
    }
};

inline std::string get_gpu_edge_id(int32_t out, int32_t in) { return "E{" + std::to_string(out) + "->" + std::to_string(in) + "}"; }

class GPUReferenceEdge : public ReferenceEdge {
    public:
        std::string gpu_edge_id; // should not be used for comparisons; technically the (out, in) pair is the id.

        GPUReferenceEdge(uint64_t cpu_edge_id, std::string label, GPUReferenceVertex* out, GPUReferenceVertex* in)
        : public ReferenceEdge(cpu_edge_id, label, out, in) {
            this->gpu_edge_id = get_gpu_edge_id(out->gpu_vertex_id, in->gpu_edge_id);
        }
};

#endif