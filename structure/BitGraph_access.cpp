#include "bitgraph/structure/BitGraph.h"

#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/arange.h"
#include "maelstrom/algorithms/assign.h"
#include "maelstrom/algorithms/set.h"

#include "maelstrom/util/any_utils.h"

namespace bitgraph {

    std::pair<maelstrom::vector, maelstrom::vector> BitGraph::get_vertex_properties(std::string property_name, maelstrom::vector& vertices, bool return_values) {
        auto p = this->vertex_properties.find(property_name);
        if(p == this->vertex_properties.end()) {
            std::stringstream sx;
            sx << "Vertex Property " << property_name << " does not exist in this graph";
            throw std::invalid_argument(sx.str());
        }

        auto& table = p->second;        

        if(vertices.empty()) {
            return std::make_pair(
                maelstrom::vector(vertices.get_mem_type(), table->get_val_dtype()),
                maelstrom::vector(vertices.get_mem_type(), maelstrom::uint64)
            );
        }

        auto found_values = table->get(vertices);
        auto found_values_prim_view = maelstrom::as_primitive_vector(found_values, true);
        auto filter_ix = maelstrom::filter(found_values_prim_view, maelstrom::NOT_EQUALS, table->val_not_found());

        if(return_values) {
            return std::make_pair(
                std::move(maelstrom::select(found_values, filter_ix)),
                std::move(filter_ix)
            );
        }

        return std::make_pair(
            maelstrom::vector(),
            std::move(filter_ix)
        );
    }

    std::pair<maelstrom::vector, maelstrom::vector> BitGraph::get_edge_properties(std::string property_name, maelstrom::vector& edges, bool return_values) {
        auto p = this->edge_properties.find(property_name);
        if(p == this->edge_properties.end()) {
            std::stringstream sx;
            sx << "Edge Property " << property_name << " does not exist in this graph";
            throw std::invalid_argument(sx.str());
        }

        auto& table = p->second;        

        if(edges.empty()) {
            return std::make_pair(
                maelstrom::vector(edges.get_mem_type(), table->get_val_dtype()),
                maelstrom::vector(edges.get_mem_type(), maelstrom::uint64)
            );
        }

        auto found_values = table->get(edges);
        auto found_values_prim_view = maelstrom::as_primitive_vector(found_values, true);
        auto filter_ix = maelstrom::filter(found_values_prim_view, maelstrom::NOT_EQUALS, table->val_not_found());

        if(return_values) {
            return std::make_pair(
                std::move(maelstrom::select(found_values, filter_ix)),
                std::move(filter_ix)
            );
        }

        return std::make_pair(
            maelstrom::vector(),
            std::move(filter_ix)
        );
    }

    maelstrom::vector BitGraph::get_vertex_embeddings(std::string emb_name, maelstrom::vector& vertices) {
        auto p = this->vertex_embeddings.find(emb_name);
        if(p == this->vertex_embeddings.end()) {
            throw std::invalid_argument("Attempted to get a nonexistent embedding");
        }
        auto& emb = p->second;

        size_t emb_stride = p->second.size() / this->num_vertices();
        auto vertices_host_view = maelstrom::as_host_vector(vertices).astype(maelstrom::uint64);

        maelstrom::vector dest(
            vertices.get_mem_type(),
            emb.get_dtype()
        );
        dest.reserve(emb_stride * vertices.size());

        for(size_t k = 0; k < vertices_host_view.size(); ++k) {
            size_t v = std::any_cast<size_t>(vertices_host_view.get(k));
            auto ix_k = maelstrom::arange(maelstrom::DEVICE, v * emb_stride, (v+1) * emb_stride);
            
            auto z = maelstrom::select(emb, ix_k);
            dest.insert(z);
        }

        return dest;
    }

    void BitGraph::set_vertex_properties(std::string property_name, maelstrom::vector& vertices, maelstrom::vector& property_values) {
        if(vertices.get_dtype() != this->vertex_dtype) {
            std::stringstream sx;
            sx << "Vertex array did not have proper datatype! (Expected " << this->vertex_dtype.name << " but got " << vertices.get_dtype().name << ")";
            throw std::invalid_argument(sx.str());
        }

        auto p = this->vertex_properties.find(property_name);
        if(p == this->vertex_properties.end()) {
            this->vertex_properties[property_name] = std::unique_ptr<maelstrom::hash_table>(
                new maelstrom::hash_table(
                    this->default_property_storage,
                    this->vertex_dtype,
                    property_values.get_dtype(),
                    property_values.size()
                )
            );
        }

        auto mem_type = this->vertex_properties[property_name]->get_mem_type();
        if(mem_type == maelstrom::MANAGED || mem_type == maelstrom::DEVICE) {
            auto vertices_d = maelstrom::as_device_vector(vertices);
            auto property_values_d = maelstrom::as_device_vector(property_values);
            this->vertex_properties[property_name]->set(
                vertices_d,
                property_values_d
            );
        } else {
            this->vertex_properties[property_name]->set(
                vertices,
                property_values
            );
        }
    }

    void BitGraph::set_edge_properties(std::string property_name, maelstrom::vector& edges, maelstrom::vector& property_values) {
        if(edges.get_dtype() != this->edge_dtype) {
            throw std::runtime_error("Edge array did not have proper datatype!");
        }

        auto p = this->edge_properties.find(property_name);
        if(p == this->edge_properties.end()) {
            this->edge_properties[property_name] = std::unique_ptr<maelstrom::hash_table>(
                new maelstrom::hash_table(
                    this->default_property_storage,
                    this->edge_dtype,
                    property_values.get_dtype(),
                    property_values.size()
                )
            );
        }

        auto mem_type = this->edge_properties[property_name]->get_mem_type();
        if(mem_type == maelstrom::MANAGED || mem_type == maelstrom::DEVICE) {
            auto edges_d = maelstrom::as_device_vector(edges);
            auto property_values_d = maelstrom::as_device_vector(property_values);
            this->edge_properties[property_name]->set(
                edges,
                property_values
            );
        } else {
            this->edge_properties[property_name]->set(
                edges,
                property_values
            );
        }
    }

    void BitGraph::set_vertex_embeddings(std::string emb_name, maelstrom::vector& vertices, maelstrom::vector& embeddings, std::any default_val) {
        const size_t n_vertices = this->num_vertices();

        if(vertices.empty()) {
            this->vertex_embeddings[emb_name] = maelstrom::vector(
                this->default_property_storage,
                embeddings.get_dtype()
            );
            this->vertex_embeddings[emb_name].reserve(embeddings.size());
            this->vertex_embeddings[emb_name].insert(embeddings);
        } else {
            const size_t emb_stride = embeddings.size() / vertices.size();

            auto p = this->vertex_embeddings.find(emb_name);
            if(p == this->vertex_embeddings.end()) {
                this->vertex_embeddings[emb_name] = maelstrom::vector(
                    this->default_property_storage,
                    embeddings.get_dtype(),
                    n_vertices * emb_stride
                );
                maelstrom::set(this->vertex_embeddings[emb_name], default_val);
            }

            auto emb_dtype = embeddings.get_dtype();
            auto& stored_emb = this->vertex_embeddings[emb_name];
            auto vertices_host_view = maelstrom::as_host_vector(vertices).astype(maelstrom::uint64);
            auto range_ix = maelstrom::arange(maelstrom::DEVICE, emb_stride);
            for(size_t k = 0; k < vertices_host_view.size(); ++k) {
                size_t current_emb = std::any_cast<size_t>(vertices_host_view.get(k));
                maelstrom::vector current_stored_emb(
                    stored_emb.get_mem_type(),
                    emb_dtype,
                    static_cast<unsigned char*>(stored_emb.data()) + (current_emb * emb_stride * maelstrom::size_of(emb_dtype)),
                    emb_stride,
                    true
                );

                maelstrom::vector current_new_emb(
                    embeddings.get_mem_type(),
                    emb_dtype,
                    static_cast<unsigned char*>(embeddings.data()) + (k * emb_stride * maelstrom::size_of(emb_dtype)),
                    emb_stride,
                    true
                );

                maelstrom::assign(
                    current_stored_emb,
                    range_ix,
                    current_new_emb
                );
            }
        }
    }

    void BitGraph::declare_vertex_property(std::string property_name, maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t initial_size) {
        if(initial_size == 0) {
            auto n_vertices = this->num_vertices();
            initial_size = n_vertices > 0 ? n_vertices : 62;
        }

        auto p = this->vertex_properties.find(property_name);
        if(p == this->vertex_properties.end()) {
            this->vertex_properties[property_name] = std::unique_ptr<maelstrom::hash_table>(
                new maelstrom::hash_table(
                    this->default_property_storage,
                    this->vertex_dtype,
                    dtype,
                    initial_size
                )
            );
        } else {
            std::stringstream sx;
            sx << "Cannot declare vertex property " << property_name;
            sx << " because it already exists!";
            throw std::runtime_error(sx.str());
        }
    }

    void BitGraph::declare_edge_property(std::string property_name, maelstrom::storage mem_type, maelstrom::dtype_t dtype, size_t initial_size) {
        if(initial_size == 0) {
            auto n_edges = this->num_edges();
            initial_size = n_edges > 0 ? n_edges : 62;
        }

        auto p = this->edge_properties.find(property_name);
        if(p == this->edge_properties.end()) {
            this->edge_properties[property_name] = std::unique_ptr<maelstrom::hash_table>(
                new maelstrom::hash_table(
                    this->default_property_storage,
                    this->edge_dtype,
                    dtype,
                    initial_size
                )
            );
        } else {
            std::stringstream sx;
            sx << "Cannot declare edge property " << property_name;
            sx << " because it already exists!";
            throw std::runtime_error(sx.str());
        }
    }

    size_t BitGraph::get_vertex_property_num_entries(std::string prop_name) {
        return this->vertex_properties[prop_name]->size();
    }

    size_t BitGraph::get_edge_property_num_entries(std::string prop_name) {
        return this->vertex_properties[prop_name]->size();
    }

    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> BitGraph::view_vertex_property(std::string prop_name, bool view_keys, bool view_values) {
        std::optional<maelstrom::vector> keys;
        std::optional<maelstrom::vector> values;
        if(view_keys) {
            if(view_values) {
                maelstrom::vector k, v;
                std::tie(k, v) = this->vertex_properties[prop_name]->get_items();
                keys.emplace(std::move(k));
                values.emplace(std::move(v));
            } else {
                keys = std::make_optional(
                    std::move(this->vertex_properties[prop_name]->get_keys())
                );
            }
        } else {
            values = std::make_optional(
                std::move(this->vertex_properties[prop_name]->get_values())
            );
        }

        return std::make_pair(
            std::move(keys),
            std::move(values)
        );
    }

    std::pair<std::optional<maelstrom::vector>, std::optional<maelstrom::vector>> BitGraph::view_edge_property(std::string prop_name, bool view_keys, bool view_values) {
        std::optional<maelstrom::vector> keys;
        std::optional<maelstrom::vector> values;
        if(view_keys) {
            if(view_values) {
                maelstrom::vector k, v;
                std::tie(k, v) = this->edge_properties[prop_name]->get_items();
                keys.emplace(std::move(k));
                values.emplace(std::move(v));
            } else {
                keys = std::make_optional(
                    std::move(this->edge_properties[prop_name]->get_keys())
                );
            }
        } else {
            values = std::make_optional(
                std::move(this->edge_properties[prop_name]->get_values())
            );
        }

        return std::make_pair(
            std::move(keys),
            std::move(values)
        );
    }

    std::vector<std::string> BitGraph::get_vertex_property_names() {
        std::vector<std::string> names;
        names.reserve(this->vertex_properties.size());
        for(auto& p : this->vertex_properties) names.push_back(p.first);
        return names;
    }

    std::vector<std::string> BitGraph::get_edge_property_names() {
        std::vector<std::string> names;
        names.reserve(this->edge_properties.size());
        for(auto& p : this->edge_properties) names.push_back(p.first);
        return names;
    }

    maelstrom::vector BitGraph::get_vertex_labels(maelstrom::vector& vertices) {
        return maelstrom::select(
            this->vertex_labels,
            vertices
        );
    }

    maelstrom::vector BitGraph::get_edge_labels(maelstrom::vector& edges) {
        this->to_canonical_coo();
        return this->matrix->get_relations_1d(edges);
    }

    /**
     * Returns a copy of this Graph's vertices.
     **/
    std::vector<gremlinxx::Vertex> BitGraph::vertices() {
        size_t num_vertices = this->matrix->num_rows();

        std::vector<gremlinxx::Vertex> detached_vertices;
        detached_vertices.reserve(num_vertices);
        for(size_t k = 0; k < num_vertices; ++k) {
            detached_vertices.push_back(
                std::any_cast<gremlinxx::Vertex>(maelstrom::safe_any_cast(k, this->vertex_dtype))
            );
        }

        return std::move(detached_vertices);
    }

    /**
     * Returns a copy of this Graph's edges.
    **/
    std::vector<gremlinxx::Edge> BitGraph::edges() {
        size_t num_edges = this->matrix->num_nonzero();

        std::vector<gremlinxx::Edge> detached_edges;
        detached_edges.reserve(num_edges);
        for(size_t k = 0; k < num_edges; ++k) {
            detached_edges.push_back(
                std::any_cast<gremlinxx::Edge>(maelstrom::safe_any_cast(k, this->edge_dtype))
            );
        }

        return std::move(detached_edges);
    }

}