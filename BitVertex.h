#ifndef BIT_VERTEX_H
#define BIT_VERTEX_H

#define VERTEX_MAGIC_NUMBER 15134994420398101408ull

#include "Vertex.h"
#include "Direction.h"
#include "BitEdge.h"
#include <inttypes.h>
#include <mutex>
#include <vector>

/*
Vertex type that uses a uint64_t for identifiers,
allowing for up to 2^64 = 10^19 vertices.
*/
class BitVertex : public Vertex {
	public:
		// Magic number used for casting checks
		uint64_t magic;

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

		// Mutex that prevents concurrent edge addition
		std::mutex add_edge_mutex;

	public:
		BitVertex(uint64_t vid);
		BitVertex(uint64_t vid, std::string v_label);
		virtual void const* id();
		virtual std::string const* label();
		virtual bool hasLabel();
		void addEdge(BitEdge* new_edge, Direction dir);
		std::vector<BitEdge*> edges(Direction dir);
};

#endif