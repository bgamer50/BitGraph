#include "gremlinxx/gremlinxx.h"
#include "structure/CPUGraph.h"
#include "traversal/CPUGraphTraversalSource.h"
#include "structure/BitVertex.h"
#include "structure/BitEdge.h"

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
	vertex_list[this->num_vertices++] = v; // TODO this makes deletion impossible
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
	vertex_list[this->num_vertices++] = v; // TODO this makes deletion impossible
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

