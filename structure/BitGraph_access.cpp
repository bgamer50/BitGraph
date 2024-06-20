#include "bitgraph/structure/BitGraph.h"

#include "maelstrom/algorithms/filter.h"
#include "maelstrom/algorithms/select.h"
#include "maelstrom/algorithms/arange.h"

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