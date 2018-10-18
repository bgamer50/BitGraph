#ifndef BIT_VERTEX_H
#define BIT_VERTEX_H

#include "Vertex.h"
#include "Direction.h"
#include "BitEdge.h"
#include <inttypes.h>

/*
Vertex type that uses a uint64_t for identifiers,
allowing for up to 2^64 = 10^19 vertices.
*/
class BitVertex : public Vertex {
	private:
		// The id of this BitVertex
		uint64_t vertex_id;
		
		// The outgoing edges
		std::vector<BitEdge*> edges_out;

		// The incoming edges
		std::vector<BitEdge*> edges_in;

		// The label, if it has one
		std::string vertex_label;

		// Boolean whether or not it has a label
		bool has_label;
	public:
		BitVertex(uint64_t vid);
		BitVertex(uint64_t vid, std::string v_label);
		virtual void const* id();
		virtual std::string const* label();
		virtual bool hasLabel();
		void addEdge(BitEdge* new_edge, Direction dir);
};

#endif