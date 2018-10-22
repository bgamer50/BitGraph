#ifndef BIT_EDGE_H
#define BIT_EDGE_H

#include "Edge.h"
#include <inttypes.h>

class BitEdge : public Edge {
	private:
		// The unique id of this BitEdge
		uint64_t edge_id;

		// The out-vertex of this BitEdge
		Vertex* out_vertex;

		// The in-vertex of this BitEdge
		Vertex* in_vertex;

		// The label of this BitEdge
		std::string edge_label;

	public:
		/*
			Construct a new BitEdge with the given in and
			out vertices and the given label.  Per TP3
			standard, all Edges must have a label.
		*/
		BitEdge(uint64_t id, Vertex* out, Vertex* in, std::string label);

		/*
			Get a pointer to the id of the given Edge.
		*/
		virtual void const* id();
		virtual std::string const* label();
		virtual Vertex* outV();
		virtual Vertex* inV();
};

#endif