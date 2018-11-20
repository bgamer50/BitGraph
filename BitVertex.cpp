#include "Vertex.h"
#include "BitVertex.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>

BitVertex::BitVertex(uint64_t vid) {
	this->vertex_id = vid;
	this->has_label = false;
	this->magic = VERTEX_MAGIC_NUMBER;
}

BitVertex::BitVertex(uint64_t vid, std::string v_label) {
	this->vertex_id = vid;
	this->has_label = true;
	this->vertex_label = v_label;
	this->magic = VERTEX_MAGIC_NUMBER;
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

/*
	Add an edge to this Vertex in the
	given direction.  Checks to make
	sure the Edge makes sense.
*/
void BitVertex::addEdge(BitEdge* new_edge, Direction dir) {
	add_edge_mutex.lock();

	if(dir == OUT && new_edge->outV() == this) edges_out.push_back(new_edge);
	else if(dir == IN && new_edge->inV() == this) edges_in.push_back(new_edge);

	add_edge_mutex.unlock();
}

/*
	Get edges in a particular direction.
*/
std::vector<BitEdge*> BitVertex::edges(Direction dir) {
	switch(dir) {
		case OUT: {
			return this->edges_out;
		}
		case IN: {
			return this->edges_in;
		}
		case BOTH: {
			std::vector<BitEdge*> both_edges;

			for_each(edges_in.begin(), edges_in.end(), [&, this](BitEdge* edg) { both_edges.push_back(edg); });
			for_each(edges_out.begin(), edges_out.end(), [&, this](BitEdge* edg) { both_edges.push_back(edg); });

			return both_edges;
		}
		default: {
			// should never occur
		}
	}
}

/*
	Get the property with the given key.
*/
VertexProperty<void*>* BitVertex::property(std::string key) {
    std::cout << "32324324\n";
	return this->my_properties.find(key)->second;
}

/*
	Get the property with the given key.
*/
VertexProperty<void*>* BitVertex::property(Cardinality card, std::string key, void* value) {
	auto old_prop = this->my_properties.find(key);
	if(card == SINGLE) {
		VertexProperty<void*>* prop = new VertexProperty<void*>(SINGLE, key, {value});
		if(old_prop != this->my_properties.end()) delete old_prop->second;
		this->my_properties[key] = prop;
	}
	else if(card == SET || card == LIST) {
		if(old_prop != this->my_properties.end()) {
			std::vector<void*> vals = old_prop->second->values();
			vals.push_back(value);
			delete old_prop->second;
			this->my_properties[key] = new VertexProperty<void*>(card, key, vals);
		}
		else {
			this->my_properties[key] = new VertexProperty<void*>(card, key, {value});
		}
	}
	else {
		throw std::runtime_error("Illegal cardinality!");
	}

	return this->my_properties[key];
}


VertexProperty<void*>* BitVertex::property(std::string key, void* value) {
	return this->property(SINGLE, key, value);
}
