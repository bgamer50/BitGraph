#pragma once

#include "gremlinxx/gremlinxx.h"

#include "maelstrom/storage/datatype.h"
#include "maelstrom/storage/strings.h"
#include "maelstrom/containers/vector.h"
#include "maelstrom/containers/hash_table.h"
#include "maelstrom/containers/sparse_matrix.h"

#include "bitgraph/traversal/BitGraphTraversalSource.h"
#include "bitgraph/step/BitGraphVStep.h"

#include <limits>
#include <memory>

#define BITGRAPH_INVALID_STRING ("_INVALID_")

namespace bitgraph {

    // TODO factory

    class BitGraph: public gremlinxx::Graph {
        using GraphTraversal = gremlinxx::GraphTraversal;
        using GraphTraversalSource = gremlinxx::GraphTraversalSource;
        using Vertex = gremlinxx::Vertex;
        using Edge = gremlinxx::Edge;

        friend bitgraph::BitGraphVStep;

        private:
            maelstrom::storage structure_storage;
            maelstrom::storage default_property_storage;
            maelstrom::storage traverser_storage;

            maelstrom::dtype_t vertex_dtype;
            maelstrom::dtype_t edge_dtype;

            size_t next_vertex_id = 0; // will get auto-converted to the vertex_dtype
            size_t next_edge_id = 0; // will get auto-converted to the edge_dtype

            // Graph Structure
            std::unique_ptr<maelstrom::sparse_matrix> matrix;

            // Element Properties
            std::unordered_map<std::string, std::unique_ptr<maelstrom::hash_table>> vertex_properties;
            // TODO edge properties

            // String index
            maelstrom::string_index<uint64_t> string_index{BITGRAPH_INVALID_STRING, std::numeric_limits<uint64_t>::max()};
            maelstrom::string_index<uint8_t> edge_label_index{BITGRAPH_INVALID_STRING, std::numeric_limits<uint8_t>::max()};

            // Vertex Labels
            maelstrom::vector vertex_labels;

            maelstrom::dtype_t make_vertex_dtype(maelstrom::dtype_t raw_dtype);
            maelstrom::dtype_t make_edge_dtype(maelstrom::dtype_t raw_dtype);
            Vertex vertex_from_id(std::any v_id);

            void to_canonical_csc();
            void to_canonical_csr();
            void to_canonical_coo();

        public:
            BitGraph(
                maelstrom::dtype_t vertex_dtype=maelstrom::int32,
                maelstrom::dtype_t edge_dtype=maelstrom::int32,
                maelstrom::storage structure_storage=maelstrom::DEVICE,
                maelstrom::storage default_property_storage=maelstrom::MANAGED,
                maelstrom::storage traverser_storage=maelstrom::DEVICE);

            using gremlinxx::Graph::traversal;
            inline virtual GraphTraversalSource* traversal() {
                return new BitGraphTraversalSource(this);
            }

            using gremlinxx::Graph::vertices;
            virtual std::vector<Vertex> vertices();

            using gremlinxx::Graph::edges;
            virtual std::vector<Edge> edges();

            using gremlinxx::Graph::add_vertex;
            virtual Vertex add_vertex(std::string label="");

            using gremlinxx::Graph::add_vertices;
			virtual maelstrom::vector add_vertices(size_t n_new_vertices, std::string label="");

            using gremlinxx::Graph::add_edge;
            virtual Edge add_edge(Vertex from_vertex, Vertex to_vertex, std::string label);

            using gremlinxx::Graph::add_edges;
			virtual maelstrom::vector add_edges(maelstrom::vector& from_vertices, maelstrom::vector& to_vertices, std::string label);

            using gremlinxx::Graph::set_vertex_properties;
            virtual void set_vertex_properties(std::string property_name, maelstrom::vector& vertices, maelstrom::vector& property_values);

            using gremlinxx::Graph::get_vertex_properties;
            virtual std::pair<maelstrom::vector, maelstrom::vector> get_vertex_properties(std::string property_name, maelstrom::vector& vertices, bool return_values=true);

            using gremlinxx::Graph::get_vertex_property_names;
            virtual std::vector<std::string> get_vertex_property_names();

            using gremlinxx::Graph::get_vertex_labels;
            virtual maelstrom::vector get_vertex_labels(maelstrom::vector& vertices);

            using gremlinxx::Graph::get_vertex_dtype;
            inline virtual maelstrom::dtype_t get_vertex_dtype() { return this->vertex_dtype; }

            using gremlinxx::Graph::get_edge_dtype;
			inline virtual maelstrom::dtype_t get_edge_dtype() { return this->edge_dtype; }

            inline virtual maelstrom::dtype_t get_string_dtype() { return this->string_index.get_dtype(); }

            inline virtual maelstrom::dtype_t get_edge_label_dtype() { return this->edge_label_index.get_dtype(); }

			using gremlinxx::Graph::V;
			virtual std::pair<maelstrom::vector, maelstrom::vector> V(maelstrom::vector& current_vertices, std::vector<std::string>& labels, gremlinxx::Direction direction);

			using gremlinxx::Graph::E;
			virtual std::pair<maelstrom::vector, maelstrom::vector> E(maelstrom::vector& current_vertices, std::vector<std::string>& labels, gremlinxx::Direction direction);

			using gremlinxx::Graph::toV;
			virtual std::pair<maelstrom::vector, maelstrom::vector> toV(maelstrom::vector& current_edges, gremlinxx::Direction direction);

            using gremlinxx::Graph::num_vertices;
            inline virtual size_t num_vertices() { return this->matrix->num_rows(); }

            using gremlinxx::Graph::num_edges;
            inline virtual size_t num_edges() { return this->matrix->num_nonzero(); }
    };

}