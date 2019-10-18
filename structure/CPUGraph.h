#ifndef CPU_GRAPH_H
#define CPU_GRAPH_H

#define CPUGRAPH_INITIAL_LIST_SIZE 100000

#include "index/Index.h"
#include "structure/Graph.h"
#include "structure/Vertex.h"
#include "structure/Element.h"
#include "structure/Direction.h"
#include <string>
#include <vector>
#include <list>
#include <unordered_map>
class BitVertex;

enum IndexType {VERTEX_INDEX, EDGE_INDEX};

class CPUGraph : public Graph {
	private:
		std::vector<Vertex*> vertex_list = std::vector<Vertex*>(CPUGRAPH_INITIAL_LIST_SIZE);
		std::list<Edge*> edge_list;
		std::unordered_map<std::string, Index*> vertex_index;
		std::unordered_map<uint64_t, Vertex*> vertex_id_map;
		uint64_t next_edge_id = 0;
		uint64_t next_vertex_id = 0;
		uint64_t num_vertices = 0;
	public:

		/*
			Main constructor for the CPUGraph.
		*/
		CPUGraph(): Graph() {}

		/*
			The list containing the CPUGraph's vertices.
			TODO may want to optimize this or switch to vector.
		*/
		std::list<Vertex*> vertices() { 
			std::list<Vertex*> view;
			for(int k = 0; k < num_vertices; ++k) view.push_back(vertex_list[k]);
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

		/*
			The list containing the CPUGraph's edges.
		*/
		std::list<Edge*>& edges() { return edge_list; }

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

#define NEXT_VERTEX_ID_CPU() ( (uint64_t)( next_vertex_id++ ))
#define NEXT_EDGE_ID_CPU() ((uint64_t)( next_edge_id++ ))

#include "CPUGraphTraversalSource.h"
#include "structure/BitVertex.h"

/*
	Get a traversal source for this CPUGraph.
*/
GraphTraversalSource* CPUGraph::traversal() {
	CPUGraph* ref = this;
	return new CPUGraphTraversalSource(ref);
}

/*
	Adds a new Vertex (w/label) to this CPUGraph.
	Currently not part of the higher-level api.
*/
Vertex* CPUGraph::add_vertex(std::string label) {
	if(this->vertex_list.size() == this->num_vertices) this->vertex_list.resize(2*num_vertices);

	Vertex* v = new BitVertex(this, NEXT_VERTEX_ID_CPU(), label);
	uint64_t id_val = boost::any_cast<uint64_t>(v->id());
	vertex_list[this->num_vertices++] = v;
	vertex_id_map.insert(std::pair<uint64_t, Vertex*>{id_val, v});
	return v;
}

/*
	Adds a new Vertex (w/o label) to this CPUGraph.
*/
Vertex* CPUGraph::add_vertex() {
	if(this->vertex_list.size() == this->num_vertices) this->vertex_list.resize(2*num_vertices);

	Vertex* v = new BitVertex(this, NEXT_VERTEX_ID_CPU());
	uint64_t id_val = boost::any_cast<uint64_t>(v->id());
	vertex_list[this->num_vertices++] = v;
	vertex_id_map.insert(std::pair<uint64_t, Vertex*>{id_val, v});
	return v;
}

Vertex* CPUGraph::get_vertex(boost::any& id) {
	uint64_t id_val = boost::any_cast<uint64_t>(id);
	return vertex_id_map.find(id_val)->second;
}

/*
	Adds a new Edge to this CPUGraph.
*/
Edge* CPUGraph::add_edge(Vertex* out, Vertex* in, std::string label) {
	BitEdge* new_edge = new BitEdge(this, NEXT_EDGE_ID_CPU(), out, in, label);
	static_cast<BitVertex*>(out)->addEdge(new_edge, OUT);
	static_cast<BitVertex*>(in)->addEdge(new_edge, IN);
	edge_list.push_back(new_edge);
	return new_edge;
}

void CPUGraph::clear_index(BitVertex* v, std::string property_key, boost::any value) {
	auto f = vertex_index.find(property_key);
	if(f == vertex_index.end()) throw std::runtime_error("Property not indexed!\n");
	
	f->second->remove(v, value);
}

void CPUGraph::update_index(BitVertex* v, std::string property_key, boost::any value) {
	auto f = vertex_index.find(property_key);
	if(f == vertex_index.end()) throw std::runtime_error("Property not indexed!\n");
	f->second->insert(v, value);
}

void CPUGraph::create_index(IndexType type, std::string property_key, std::function<int64_t(boost::any&)> hash_func, std::function<bool(boost::any&, boost::any&)> equals_func) {
	Index* idx = new Index(hash_func, equals_func);
	vertex_index.insert(std::pair<std::string, Index*>(property_key, idx));
}

#endif