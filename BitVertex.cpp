#include "BitVertex.h"

BitVertex::BitVertex(uint64_t vid) {
	this->vertex_id = vid;
	this->has_label = false;
}

BitVertex::BitVertex(uint64_t vid, std::string v_label) {
	this->vertex_id = vid;
	this->has_label = true;
	this->vertex_label = v_label;
}

/*
	Get the unique id of the Vertex.
	In CPUGraph this is indirectly
	derived from its initial position 
	in the list of Vertices.
*/
void const* BitVertex::id() {
	return &vertex_id;
}

/*
	Make sure to return NULL if there is
	no label for the Vertex
*/
std::string const* BitVertex::label() {
	return has_label ? &vertex_label : NULL;
}

/*
	Nifty helper method that returns
	whether or not this Vertex has a label.
*/
bool BitVertex::hasLabel() {
	return has_label;
}