#ifndef BIT_EDGE_H
#define BIT_EDGE_H

#include "Edge.h"
#include <inttypes.h>

class BitEdge : public Edge {
	private:

	public:
		/*
			Construct a new BitEdge with the given in and
			out vertices and the given label.  Per TP3
			standard, all Edges must have a label.
		*/
		BitEdge(Vertex* out, Vertex* in, std::string label);

		/*
			Get a pointer to the id of the given Edge.
		*/
		virtual void const* id();
		virtual std::string const* label();
		virtual Vertex* outV();
		virtual Vertex* inV();
};

#endif