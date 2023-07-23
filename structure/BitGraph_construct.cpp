#include "bitgraph/structure/BitGraph.h"

#include "maelstrom/util/any_utils.cuh"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/set.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/sort.h"

namespace bitgraph {

    gremlinxx::Vertex BitGraph::vertex_from_id(std::any v_id) {
        size_t vertex_id = std::any_cast<size_t>(
            maelstrom::safe_any_cast(v_id, maelstrom::uint64)
            );

        std::string label = std::any_cast<std::string>(this->vertex_labels.get(vertex_id));
        return gremlinxx::Vertex{
            label,
            vertex_id
        };
    }

    maelstrom::dtype_t BitGraph::make_vertex_dtype(maelstrom::dtype_t raw_vertex_dtype) {
        return maelstrom::dtype_t{
            "bitgraph_vertex",
            raw_vertex_dtype.prim_type,
            [this, raw_vertex_dtype](void* data) {
                std::any d_data = raw_vertex_dtype.deserialize(data);
                return this->vertex_from_id(d_data);
            },
            [this, raw_vertex_dtype](std::any data) {
                gremlinxx::Vertex v = std::any_cast<gremlinxx::Vertex>(data);
                return maelstrom::safe_any_cast(v.id, raw_vertex_dtype);
            }
        };
    }

    maelstrom::dtype_t BitGraph::make_edge_dtype(maelstrom::dtype_t raw_edge_dtype) {
        return maelstrom::dtype_t{
            "bitgraph_edge",
            raw_edge_dtype.prim_type,
            [this, raw_edge_dtype](void* data) {
                std::vector<std::any> d_data_vec = {raw_edge_dtype.deserialize(data)};
                auto m_data_vec = maelstrom::make_vector_from_anys(this->structure_storage, d_data_vec);
                
                maelstrom::vector rows, cols, vals, rels;
                std::tie(rows, cols, vals, rels) = this->matrix->get_entries_1d(m_data_vec);

                return gremlinxx::Edge{
                    std::any_cast<std::string>(rels.get(0)),
                    std::any_cast<size_t>(vals.get(0)),
                    std::any_cast<Vertex>(rows.get(0)),
                    std::any_cast<Vertex>(cols.get(0))
                };
            },
            [this, raw_edge_dtype](std::any data) {
                gremlinxx::Edge e = std::any_cast<Edge>(data);
                std::vector<std::any> anys = {e.id};
                auto id_vec = maelstrom::make_vector_from_anys(this->structure_storage, this->edge_id_dtype, anys);
                return maelstrom::safe_any_cast(this->matrix->get_1d_index_from_value(id_vec).get(0), raw_edge_dtype);
            }
        };
    }

    BitGraph::BitGraph(maelstrom::dtype_t vertex_dtype, maelstrom::dtype_t edge_dtype, maelstrom::storage structure_storage, maelstrom::storage default_property_storage, maelstrom::storage traverser_storage)
    : gremlinxx::Graph() {
        this->structure_storage = structure_storage;
        this->default_property_storage = default_property_storage;
        this->traverser_storage = traverser_storage;

        this->vertex_dtype = make_vertex_dtype(vertex_dtype);
        this->edge_dtype = make_edge_dtype(edge_dtype);
        this->edge_id_dtype = edge_dtype;
        
        auto string_dtype = this->string_index.get_dtype();

        // TODO should this be property storage?
        this->vertex_labels = maelstrom::vector(
            this->structure_storage,
            string_dtype
        );

        
        this->matrix = std::make_unique<maelstrom::basic_sparse_matrix>(
            maelstrom::vector(structure_storage, vertex_dtype),
            maelstrom::vector(structure_storage, vertex_dtype),
            maelstrom::vector(structure_storage, edge_id_dtype),
            maelstrom::vector(structure_storage, this->edge_label_index.get_dtype()),
            maelstrom::COO,
            0,
            0,
            false
        );

    }

    gremlinxx::Vertex BitGraph::add_vertex(std::string label) {
        this->add_vertices(1, label);
        return gremlinxx::Vertex{label, this->next_vertex_id - 1};
    }

    maelstrom::vector BitGraph::add_vertices(size_t n_new_vertices, std::string label) {
        // matrix has to be COO to update
        this->matrix->to_coo();

        // update matrix
        size_t total_num_vertices = this->matrix->num_rows() + n_new_vertices;
        this->matrix->set(
            maelstrom::vector(),
            total_num_vertices,
            maelstrom::vector(),
            total_num_vertices
        );

        // labels
        auto string_dtype = this->string_index.get_dtype();
        maelstrom::vector new_labels(
            this->structure_storage,
            string_dtype,
            n_new_vertices
        );
        maelstrom::set(new_labels, label);
        this->vertex_labels.insert(this->vertex_labels.size(), new_labels);
        new_labels.clear();


        auto new_vertices = maelstrom::arange(
            this->traverser_storage,
            maelstrom::safe_any_cast(this->next_vertex_id, this->vertex_dtype),
            maelstrom::safe_any_cast(this->next_vertex_id + n_new_vertices, this->vertex_dtype)
        );
        
        this->next_vertex_id += n_new_vertices;
        return new_vertices;
    }

    gremlinxx::Edge BitGraph::add_edge(Vertex from_vertex, Vertex to_vertex, std::string label) {
        std::vector from_vertex_anys = {std::any(from_vertex)};
        auto from_vertex_vec = maelstrom::make_vector_from_anys(this->structure_storage, this->vertex_dtype, from_vertex_anys);
        
        std::vector to_vertex_anys = {std::any(to_vertex)};
        auto to_vertex_vec = maelstrom::make_vector_from_anys(this->structure_storage, this->vertex_dtype, to_vertex_anys);

        auto added_edge_vec = this->add_edges(
            from_vertex_vec,
            to_vertex_vec,
            label
        );

        if(added_edge_vec.size() != 1) {
            std::stringstream sx;
            sx << "Invalid edge size (expected 1 edge) but got " << added_edge_vec.size() << " edges";
            throw std::runtime_error(sx.str());
        }
        
        return std::any_cast<Edge>(added_edge_vec.get(0));
    }

    maelstrom::vector BitGraph::add_edges(maelstrom::vector& from_vertices, maelstrom::vector& to_vertices, std::string label) {
        if(from_vertices.size() != to_vertices.size()) throw std::runtime_error("from vertices size must match to vertices size");

        // matrix has to be COO to update
        this->matrix->to_coo();

        maelstrom::vector from_vertices_view(from_vertices, true);
        maelstrom::vector to_vertices_view(to_vertices, true);

        maelstrom::vector new_edge_ids = maelstrom::arange(
            this->structure_storage,
            maelstrom::safe_any_cast(this->next_edge_id, this->edge_id_dtype),
            maelstrom::safe_any_cast(this->next_edge_id + from_vertices.size(), this->edge_id_dtype)
        );

        maelstrom::vector new_labels = maelstrom::vector(
            this->structure_storage,
            this->edge_label_index.get_dtype(),
            from_vertices.size()
        );
        maelstrom::set(new_labels, label);

        // behavior is undefined if passing nonexisting vertices
        size_t old_size = this->matrix->num_nonzero();
        this->matrix->set(
            from_vertices_view,
            this->matrix->num_rows(),
            to_vertices_view,
            this->matrix->num_cols(),
            new_edge_ids,
            new_labels
        );
        new_edge_ids.clear();
        new_labels.clear();

        // this is the confusing part - have to return "raw" edge ids, not the ones used for lookup
        // First step is doing a sort
        auto perm = this->matrix->sort(true);
        auto filter_ix = maelstrom::filter(perm, maelstrom::GREATER_THAN_OR_EQUAL, old_size);
        perm = maelstrom::select(perm, filter_ix);

        auto new_ix = maelstrom::arange(this->traverser_storage, this->matrix->num_nonzero());
        new_ix = maelstrom::select(new_ix, filter_ix);
        filter_ix.clear();

        auto sorted_ix = maelstrom::sort(perm);
        perm.clear();
        return maelstrom::select(new_ix, sorted_ix).astype(this->edge_dtype);
    }

}