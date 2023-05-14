	#pragma once

#define CPUGRAPH_INITIAL_LIST_SIZE 100000

#include "maelstrom/storage/datatype.h"
#include "gremlinxx/gremlinxx.h"
#include "index/Index.h"
#include <string>
#include <vector>
#include <list>
#include <unordered_map>
class BitVertex;

enum IndexType {VERTEX_INDEX, EDGE_INDEX};

class CPUGraph : public Graph {
	private:
		std::vector<Vertex*> vertex_list = std::vector<Vertex*>(CPUGRAPH_INITIAL_LIST_SIZE);
		std::vector<Edge*> edge_list;
		std::unordered_map<std::string, Index*> vertex_index;
		std::unordered_map<uint64_t, Vertex*> vertex_id_map;
		uint64_t next_edge_id = 0;
		uint64_t next_vertex_id = 0;
		uint64_t num_vertices = 0;
		maelstrom::dtype_t maelstrom_vertex_dtype;
	public:

		/*
			Main constructor for the CPUGraph.
		*/
		CPUGraph(): Graph() {
			maelstrom::dtype_t bitgraph_vertex_t{
				"bitgraph_cpu_vertex",
				maelstrom::primitive_t::UINT64,
				[this](void* v){ return boost::any(this->vertex_list[*static_cast<uint64_t*>(v)]); },
				[this](boost::any b) { return boost::any(this->vertex_list[boost::any_cast<uint64_t>(b)]); }
			};

			this->maelstrom_vertex_dtype = bitgraph_vertex_t;
		}

		/*
			A copy of the vector containing the CPUGraph's vertices.
		*/
		std::vector<Vertex*> vertices() { 
			std::vector<Vertex*> view(this->vertex_list.begin(), this->vertex_list.begin() + this->num_vertices);
			return view; 
		}

		/*
			Access the raw vertex vector.  Should not be called by users.
			Size of this list DOES NOT reflect the actual # of Vertices,
			which should be strictly checked using numVertices().
		*/
		std::vector<Vertex*>& access_vertices() {
			return this->vertex_list;
		}

		uint64_t numVertices() { return this->num_vertices; }
		uint64_t numEdges() { return this->edge_list.size(); }

		/*
			The list containing the CPUGraph's edges.
		*/
		std::vector<Edge*> edges() { 
			std::vector<Edge*> view(this->edge_list);
			return view;
		}

		/*
			Adds a new Vertex (w/label) to this CPUGraph.
		*/
		virtual Vertex* add_vertex(std::string label);

		/*
			Adds a new Vertex (w/o label) to this CPUGraph.
		*/
		virtual Vertex* add_vertex();

		Vertex* get_vertex(boost::any& id);

		/*
			Adds a new Edge to this CPUGraph.
		*/
		virtual Edge* add_edge(Vertex* from_vertex, Vertex* to_vertex, std::string label);

		virtual maelstrom::dtype_t get_vertex_dtype();

		/*
			Get a traversal source for this CPUGraph.
		*/
		virtual GraphTraversalSource* traversal();

		bool is_indexed(std::string key) {
			return vertex_index.count(key) > 0;
		}

		void clear_index(BitVertex* v, std::string property_key, boost::any value);

		void update_index(BitVertex* v, std::string property_key, boost::any value);
		
		void create_index(IndexType type, std::string property_key, std::function<int64_t(boost::any&)> hash_func, std::function<bool(boost::any&, boost::any&)> equals_func);

		Index* get_index(std::string key) {
			return vertex_index.find(key)->second;
		}
};

#define NEXT_VERTEX_ID_CPU() ((uint64_t)( next_vertex_id++ ))
#define NEXT_EDGE_ID_CPU() ((uint64_t)( next_edge_id++ ))

