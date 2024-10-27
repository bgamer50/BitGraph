#include "bitgraph/structure/BitGraph.h"
#include "bitgraph/structure/Index.h"

#include "maelstrom/containers/vector.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/topk.h"

#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/gpu/GpuIndexIVFPQ.h"
#include "faiss/gpu/GpuIndexFlat.h"

namespace bitgraph {

    void BitGraph::make_vertex_embedding_index(std::string emb_name) {
        auto p = this->vertex_embeddings.find(emb_name);
        if(p == this->vertex_embeddings.end()) {
            throw std::runtime_error(
                "A vertex embedding with the name " + emb_name + " does not exist."
            );
        }
        auto& stored_emb = this->vertex_embeddings[emb_name];

        if(stored_emb.get_dtype() != maelstrom::float32) {
            throw std::runtime_error("Only float32 dtype is supported for indexing.");
        }
        if(stored_emb.get_mem_type() != maelstrom::HOST && stored_emb.get_mem_type() != maelstrom::PINNED) {
            throw std::runtime_error("Only host and pinned memtypes are supported for indexing.");
        }
        if(this->get_vertex_dtype().prim_type != maelstrom::INT64) {
            throw std::runtime_error("Only the int64 vertex type is supported for indexing.");
        }

        size_t td = stored_emb.size() / this->num_vertices();

        auto* res = new faiss::gpu::StandardGpuResources();
        int ncentroids = static_cast<int>(4 * sqrt(stored_emb.size()));
        int m = static_cast<int>(td/256);
        if(m <= 1) m = 2;
        int mod = td % m;
        if(mod != 0) {
            m += mod;
        }
        
        auto* vecs = new faiss::gpu::GpuIndexFlatL2(res, td);

        this->embedding_indices[emb_name] = std::make_shared<faiss::gpu::GpuIndexIVFPQ>(res, vecs, td, 8, m, 8);

        std::cout << "training..." << std::endl;
        std::cout << "td: " << td << std::endl;

        std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(this->embedding_indices[emb_name])->train(
            this->num_vertices() / 256,
            static_cast<float*>(stored_emb.data())
        );

        std::cout << "adding..." << std::endl;
        std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(this->embedding_indices[emb_name])->add(
            this->num_vertices(),
            static_cast<float*>(stored_emb.data())
        );

    }

    maelstrom::vector search_embedding_index_knn(maelstrom::vector& emb, std::any& emb_index, maelstrom::vector& query, size_t emb_stride, size_t k) {
        auto index = std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(emb_index);
        size_t num_queries = query.size() / emb_stride;

        maelstrom::vector D(maelstrom::PINNED, maelstrom::int64, num_queries * k);
        maelstrom::vector I(maelstrom::PINNED, maelstrom::float32, num_queries * k);

        maelstrom::vector query_pinned_view = maelstrom::vector(
            maelstrom::PINNED,
            query.get_dtype(),
            query.data(),
            query.size(),
            false
        );

        index->search(
            num_queries,
            static_cast<float*>(query_pinned_view.data()),
            k,
            static_cast<float*>(D.data()),
            static_cast<int64_t*>(I.data())
        );

        if(num_queries > 1) {
            auto nearest_ix = maelstrom::topk(D, k, true);
            return maelstrom::select(
                I,
                nearest_ix
            );
        }

        return I;
    }

}