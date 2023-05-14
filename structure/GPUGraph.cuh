#pragma once

#include "gremlinxx/gremlinxx.h"

#include "structure/matrix/CPUSparseMatrix.h"
#include "structure/matrix/GPUSparseMatrix.cuh"
#include "structure/matrix/Adjacency.cuh"
#include "structure/memory/GPUPropertyTable.cuh"
#include "structure/memory/TypeErasure.cuh"
class GPUGraphAlgorithm;

#include "structure/CPUGraph.h"

#include <cuda_runtime.h>

#include <boost/any.hpp>
#include <boost/functional/hash.hpp>
#include <utility>

#include <unordered_map>
#include <vector>

#define IN_ADJACENCY_MATRIX "IN_ADJACENCY_MATRIX"
#define BOTH_ADJACENCY_MATRIX "BOTH_ADJACENCY_MATRIX"

class GPUVertex;
class GPUEdge;

// The value type for this graph's sparse matrices
// Holds the multiplicity of each edge
// Can be substituted for some other values by
// shallow-copying the matrix and replacing values
// (and setting the right dtype, of course).
using matrix_value_t = uint32_t;

using namespace bitgraph::memory;

class GPUGraph : public Graph {
    private:
        // sparse adjacency matrix
        bitgraph::matrix::sparse_matrix_device adjacency_matrix;
        
        // property tables
        GPUPropertyTable property_table;

        // string index
        StringIndex string_index;
        
        // convert CPU ids to sequential ids
        std::vector<Vertex*> vertex_list; // GPU -> CPU
        std::unordered_map<uint64_t, Vertex*> vertex_id_map; // CPU -> GPU

        std::unordered_map<std::pair<size_t, size_t>, Edge*, boost::hash<std::pair<size_t, size_t>>> edge_list; // GPU -> CPU
        std::unordered_map<uint64_t, Edge*> edge_id_map; // CPU -> GPU

        // index edge labels for subgraph extraction
        std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> edge_label_index;

        // whether to cache various matrices (i.e. both-adjacency matrix, in-adjacency matrix)
        bool matrix_caching_enabled = true;
        size_t matrix_cache_max_size = 3;

        // matrix cache
        std::unordered_map<std::string, bitgraph::matrix::sparse_matrix_device> matrix_cache;
        std::unordered_map<std::string, size_t> matrix_cache_usage_tracker;

        std::optional<bitgraph::matrix::sparse_matrix_device> matrix_cache_find(std::string key) { 
            for(auto it = matrix_cache_usage_tracker.begin(); it != matrix_cache_usage_tracker.end(); ++it) ++(it->second);
            
            auto f = matrix_cache.find(key);
            if(f == matrix_cache.end()) return {};
            
            matrix_cache_usage_tracker[key] = 0;
            return f->second;            
        }

    public:
        GPUGraph(CPUGraph& cpu_graph);

        using Graph::traversal;
        virtual GraphTraversalSource* traversal();

        using Graph::vertices;
        inline virtual std::vector<Vertex*> vertices() { 
            throw std::runtime_error("Out-of-traversal vertex access unsupported by GPU Graph!"); 
        }

        using Graph::edges;
        inline virtual std::vector<Edge*> edges() {
            throw std::runtime_error("Out-of-traversal edge acess unsupported by GPU Graph!");
        }

        using Graph::add_vertex;
        inline virtual Vertex* add_vertex(std::string label) {
            throw std::runtime_error("Vertex addition unsupported by GPU Graph!");
        }
        inline virtual Vertex* add_vertex() {
            throw std::runtime_error("Vertex addition unsupported by GPU Graph!");
        }

        using Graph::add_edge;
        inline virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label) {
            throw std::runtime_error("Edge addition unsupported by GPU Graph!");
        };

        // GPUGraph-specific accessors

        /*
            Provides direct access to this GPUGraph's vertex list.
        */
        inline std::vector<Vertex*>& access_vertices() {
            return this->vertex_list;
        }

        /*
            Returns the vertex with the given cpu id.
        */
        inline Vertex* get_vertex_with_cpu_id(uint64_t vid) {
            return this->vertex_id_map[vid];
        }

        /*
            Provides direct access to this GPUGraph's edgelist.
            Currently unsupported.
        */
        inline std::vector<Edge*>& access_edges() {
            throw std::runtime_error("Cannot currently access edges!");
        }

        /*
            Returns the edge with the given cpu id.
        */
        inline Edge* get_edge_with_cpu_id(uint64_t eid) {
            return this->edge_id_map[eid];
        }

        /*
            Inserts a matrix into this GPUGraph's matrix cache.  Associates the
            matrix with given key.

            Arguments
            ---------
                key: std::string
                    Name of the matrix being inserted.
                matrix: bitgraph::matrix::sparse_matrix_device
                    The matrix to insert.
        */
        void matrix_cache_insert(std::string key, bitgraph::matrix::sparse_matrix_device matrix);

        inline bitgraph::matrix::sparse_matrix_device get_out_adjacency_matrix() { return this->adjacency_matrix; }

        inline bitgraph::matrix::sparse_matrix_device get_in_adjacency_matrix() {
            auto in_adjacency_matrix_opt = this->matrix_cache_find(IN_ADJACENCY_MATRIX);
            if(in_adjacency_matrix_opt) return in_adjacency_matrix_opt.value();

            bitgraph::matrix::sparse_matrix_device in_adjacency_matrix = transpose_csr_matrix(this->adjacency_matrix);
            if(this->matrix_caching_enabled) this->matrix_cache_insert(IN_ADJACENCY_MATRIX, in_adjacency_matrix);
            return in_adjacency_matrix;
        }

        inline bitgraph::matrix::sparse_matrix_device get_both_adjacency_matrix() {
            throw std::runtime_error("Getting the both() matrix is currently unsupported!");
        }

        inline bitgraph::matrix::sparse_matrix_device get_adjacency_matrix(Direction dir) {
            switch(dir) {
                case OUT:
                    return this->get_out_adjacency_matrix();
                case IN: 
                    return this->get_in_adjacency_matrix();
                case BOTH: 
                    return this->get_both_adjacency_matrix();
            }

            throw std::runtime_error("Illegal direction provided to get_adjacency_matrix");
        }

        inline GPUPropertyTable& access_property_table() {
            return this->property_table;
        }

        inline bitgraph::matrix::sparse_matrix_device& access_adjacency_matrix() {
            return this->adjacency_matrix;
        }

        inline bool is_matrix_caching_enabled() {
            return this->matrix_caching_enabled;
        }

        inline void set_matrix_caching(bool should_cache) {
            if(!should_cache) {
                this->matrix_cache.clear();
                this->matrix_cache_usage_tracker.clear();
            }
            this->matrix_caching_enabled = should_cache;
        }

        inline size_t num_vertices() {
            return this->vertex_list.size();
        }

        /*
            Returns a TypeErasedVector of the correct dtype containing the values of the given property
            for the given vertices.  If strict is true, an error will be thrown if one or more of the
            given vertices do not have that value.  Otherwise, will return only values for vertices that
            have them (but always in order of the original vertices).

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to get the values of.
            gpu_vertex_ids: TypeErasedVector&
                Reference to a type erased vector of vertex ids to get property values for.
            strict: bool
                If false, will skip vertices that do not have a value for the given property.
                If true, will throw an exception if one or more vertices do not have a value for the given property.
            
            Returns
            -------
            TypeErasedVector
                The values of the given property for the given vertices, if they exist.
        */
        TypeErasedVector get_properties(std::string property_key, TypeErasedVector& gpu_vertex_ids, bool strict=true);

        /*
            Returns a TypeErasedVector of the correct dtype containing the values of the given property
            for the given vertices.  If strict is true, an error will be thrown if one or more of the
            given vertices do not have that value.  Otherwise, will return only values for vertices that
            have them (but always in order of the original vertices).

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to get the values of.
            gpu_vertex_ids: std::vector<size_t>&
                Reference to a vector of vertex ids to get property values for.
            strict: bool
                If false, will skip vertices that do not have a value for the given property.
                If true, will throw an exception if one or more vertices do not have a value for the given property.
            
            Returns
            -------
            TypeErasedVector
                The values of the given property for the given vertices, if they exist.
        */
        TypeErasedVector get_properties(std::string property_key, std::vector<size_t>& gpu_vertex_ids, bool strict=true);

        /*
            Returns the value of the given property for the given vertex.  If the given vertex
            does not have the given property, and strict is false, an empty any will be
            returned.  Otherwise, if strict is true, an exception will be thrown.

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to get the value of.
            gpu_vertex_id: size_t
                The id of the vertex to get the property value for.
            strict: bool
                If false, will return an empty any if the property does not exist on this vertex.
                If true, will throw an exception if the property does not exist on this vertex.
            
            Returns
            -------
            TypeErasedVector
                The values of the given property for the given vertices, if they exist.
        */
        boost::any get_property(std::string property_key, size_t gpu_vertex_id, bool strict=true);

        /*
            Sets the value of the given property for the given vertices.

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to set the values of.
            gpu_vertex_ids: TypeErasedVector&
                Reference to a type erased vector containing the vertices to set the values of.
            values: TypeErasedVector&
                Reference to a type erased vector containing the values to be set.
        */
        void set_properties(std::string property_key, TypeErasedVector& gpu_vertex_ids, TypeErasedVector& values);

        /*
            Sets the value of the given property for the given vertices.

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to set the values of.
            gpu_vertex_ids: std::vector<size_t>&
                Reference to a std::vector containing the vertices to set the values of.
            values: std::vector<boost::any>&
                Reference to a std::vector containing the values to set.
        */
        void set_properties(std::string property_key, std::vector<size_t>& gpu_vertex_ids, std::vector<boost::any>& values);

        /*
            Sets the value of the given property for the given vertex.

            Arguments
            ---------
            property_key: std::string
                The key corresponding to the property to set the value of.
            gpu_vertex_id: size_t
                Vertex to set the value of.
            value: boost::any
                The value to be set.
        */
        void set_property(std::string property_key, size_t gpu_vertex_id, boost::any value);

        /*
            Declares a property with the given name, data type, and initial allocated size.

            Arguments
            ---------
            property_name: std::string
                Name of the new property.
            property_dtype: gremlinxx::comparison::C
                Data type of the new property.
            initial_max_size: size_t
                Initial number of property values to allocate space for.
        */
        inline void declare_property(std::string property_name, gremlinxx::comparison::C property_dtype, size_t initial_max_size) {
            this->property_table.declare_property(property_name, property_dtype, initial_max_size);
        }

        std::tuple<TypeErasedVector, TypeErasedVector> query_adjacency(bitgraph::matrix::ADJ adj, Direction dir, TypeErasedVector& input_objects);

        std::unordered_map<std::string, boost::any> algorithm(GPUGraphAlgorithm* algo);
};

