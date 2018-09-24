#ifndef BIT_VERTEX_H
#define BIT_VERTEX_H

#include "Vertex.h"
#include <inttypes.h>

/*
Vertex type that uses a uint64_t for identifiers,
allowing for up to 2^64 = 10^19 vertices.
*/
class BitVertex : public Vertex {
	private:
		uint64_t vertex_id;
		std::string vertex_label;
		bool has_label;
	public:
		BitVertex(uint64_t vid);
		BitVertex(uint64_t vid, std::string v_label);
		virtual void const* id();
		virtual std::string const* label();
		virtual bool hasLabel();
};

#endif