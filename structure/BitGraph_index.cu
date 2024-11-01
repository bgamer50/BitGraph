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

        // TODO fix memory leak
        auto* res = new faiss::gpu::StandardGpuResources();
        int ncentroids = static_cast<int>(4 * sqrt(this->num_vertices()));
        int m = static_cast<int>(td/16);
        if(m <= 1) m = 2;
        int mod = td % m;
        if(mod != 0) {
            m += mod;
        }
        
        faiss::gpu::GpuIndexIVFPQConfig config;
        config.useFloat16LookupTables = true;

        auto* vecs = new faiss::gpu::GpuIndexFlatL2(res, td);

        this->embedding_indices[emb_name] = std::make_shared<faiss::gpu::GpuIndexIVFPQ>(res, vecs, td, ncentroids, m, 8, faiss::METRIC_L2, config);

        std::cout << "training embedding index (this may take a while)..." << std::endl;

        std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(this->embedding_indices[emb_name])->train(
            this->num_vertices(),
            static_cast<float*>(stored_emb.data())
        );

        std::cout << "adding vectors to embedding index (this may take a while)..." << std::endl;
        std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(this->embedding_indices[emb_name])->add(
            this->num_vertices(),
            static_cast<float*>(stored_emb.data())
        );

    }

    maelstrom::vector search_embedding_index_knn(maelstrom::vector& emb, std::any& emb_index, maelstrom::vector& query, size_t emb_stride, size_t k) {
        if(query.get_dtype() != maelstrom::float32) {
            throw std::runtime_error("Searching embeddings only supported for float32 dtype");
        }

        auto index = std::any_cast<std::shared_ptr<faiss::gpu::GpuIndexIVFPQ>>(emb_index);
        size_t num_queries = query.size() / emb_stride;

        // FAISS wants all vectors on host (NOT PINNED or host-advised MANAGED)
        float* D = new float[num_queries * k];
        int64_t* I = new int64_t[num_queries * k];
        float* query_h = new float[query.size()];
        cudaMemcpy(query_h, query.data(), query.size() * sizeof(float), cudaMemcpyDefault);

        index->nprobe = 4;
        index->search(
            num_queries,
            query_h,
            k,
            D,
            I
        );
        index->nprobe = 1;

        delete query_h;

        auto found_ix = maelstrom::vector(
            maelstrom::PINNED,
            maelstrom::int64,
            I,
            num_queries * k,
            false
        );
        delete I;

        if(num_queries > 1) {
            maelstrom::vector D_pinned(
                maelstrom::PINNED,
                maelstrom::float32,
                D,
                num_queries * k,
                false
            );

            auto nearest_ix = maelstrom::topk(D_pinned, k, true);

            found_ix = maelstrom::select(
                found_ix,
                nearest_ix
            );
        }

        delete D;

        return found_ix;
    }

}