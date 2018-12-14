#ifndef BIT_VERTEX_H
#define BIT_VERTEX_H

#define VERTEX_MAGIC_NUMBER 15134994420398101408ull

#include <inttypes.h>
#include <mutex>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <boost/any.hpp>

#include "Vertex.h"
#include "Direction.h"
#include "BitEdge.h"

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

		// The properties
		std::map<std::string, VertexProperty<boost::any>*> my_properties;

		// Mutex that prevents concurrent edge addition
		std::mutex add_edge_mutex;

		// Mutex that prevents concurrent property addition
		std::mutex add_prop_mutex;

	public:
		BitVertex(uint64_t vid);
		BitVertex(uint64_t vid, std::string v_label);
		virtual void const* id();
		virtual std::string const* label();
		virtual bool hasLabel();
		void addEdge(BitEdge* new_edge, Direction dir);
		std::vector<BitEdge*> edges(Direction dir);
		
		virtual VertexProperty<boost::any>* property(std::string key);
		virtual VertexProperty<boost::any>* property(Cardinality cardinality, std::string key, boost::any value);
		virtual VertexProperty<boost::any>* property(std::string key, boost::any value);
};

#endif