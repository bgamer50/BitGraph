#include "BitEdge.h"

BitEdge::BitEdge(uint64_t id, Vertex* out, Vertex* in, std::string label) {
	this->edge_id = id;
	this->out_vertex = out;
	this->in_vertex = in;
	this->edge_label = label;
}

boost::any BitEdge::id() { return boost::any(this->edge_id); }
Vertex* BitEdge::outV() { return this->out_vertex; }
Vertex* BitEdge::inV() { return this->in_vertex; }
std::string BitEdge::label() { return std::string(this->edge_label); }
