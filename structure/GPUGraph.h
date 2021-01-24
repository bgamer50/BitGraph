#ifndef GPU_GRAPH_H
#define GPU_GRAPH_H

#include "structure/Graph.h"
#include "structure/CPUGraph.h"
#include "structure/Vertex.h"

#include <boost/any.hpp>

typedef std::unordered_map<std::string, std::unordered_map<uint64_t, boost::any>> property_table_t;

typedef struct sparse_adj_matrix {
    size_t nnz; // number of nonzero elements
    std::vector<float> values; // element values
    std::vector<size_t> row_ptr; // row pointer
    std::vector<size_t> col_ptr; // column pointer
} sparse_adj_matrix_t;

sparse_adj_matrix_t sparse_make(size_t r, size_t c) {
    sparse_adj_matrix_t M;
    M.nnz = 0;
    M.row_ptr.resize(r+1, 0.0);
    return M;
}

float sparse_get(sparse_adj_matrix_t& M, size_t i, size_t j) {
    size_t rs = M.row_ptr[i];
    size_t re = M.row_ptr[i+1];
    for(size_t k = rs; k < re; ++k) {
        if(j == M.col_ptr[k]) return M.values[k];
    }

    return 0.0;
}

void sparse_set(sparse_adj_matrix_t& M, size_t i, size_t j, float val) {
    size_t rs = M.row_ptr[i];
    size_t re = M.row_ptr[i+1];
    for(size_t k = rs; k < re; ++k) {
        if(j == M.col_ptr[k]) { 
            if(val != 0.0) M.values[k] = val;
            else {
                M.nnz -= 1;
                M.col_ptr.erase(M.col_ptr.begin() + k);
                M.values.erase(M.values.begin() + k);
                for(int r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] -= 1;
            }

            return;
        }
    }

    if(rs == re) {
        M.col_ptr.insert(M.col_ptr.begin() + rs, j);
        M.values.insert(M.values.begin() + rs, val);
    }
    else {
        int c = rs;
        while(M.col_ptr[c] < j) ++c;
        M.col_ptr.insert(M.col_ptr.begin() + c, j);
        M.values.insert(M.values.begin() + c, val);
    }

    M.nnz += 1;
    for(int r = i + 1; r < M.row_ptr.size(); ++r) M.row_ptr[r] += 1;
}

class GPUGraph : public Graph {
    private:
        // sparse adjacency matrix
        
        // property tables
        property_table_t property_table;
        
        // convert CPU ids to sequential ids
        std::vector<uint64_t> vertex_ids;
        std::unordered_map<uint64_t, size_t> vertex_id_map;

    public:
        GPUGraph(CPUGraph& cpu_graph) 
        : Graph() {
            this->vertex_ids.resize(cpu_graph.numVertices());

            std::vector<Vertex*>& vertices = cpu_graph.access_vertices();
            for(int k = 0; k < vertices.size(); ++k) {
                Vertex* v = vertices[k];
                uint64_t vid = boost::any_cast<uint64_t>(v->id());

                this->vertex_ids[k] = vid;
                this->vertex_id_map[vid] = k;
            }

            sparse_adj_matrix_t M = sparse_make(this->vertex_ids.size(), this->vertex_ids.size());
            for(Edge* e : cpu_graph.edges()) {
                size_t out = this->vertex_id_map[boost::any_cast<uint64_t>(e->outV()->id())];
                size_t in = this->vertex_id_map[boost::any_cast<uint64_t>(e->inV()->id())];
                sparse_set(out, in);
            }

            // copy the sparse matrix to the gpu
            
        }

};

#endif