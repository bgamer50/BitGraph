#ifndef BIT_VERTEX_H
#define BIT_VERTEX_H

#define VERTEX_MAGIC_NUMBER 15134994420398101408ull

#include <inttypes.h>
#include <mutex>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <boost/any.hpp>

#include "structure/Vertex.h"
#include "structure/Direction.h"
#include "structure/BitEdge.h"
#include "structure/CPUGraph.h"

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

		// The Graph this BitVertex belongs to
		CPUGraph* graph;
		
		// The outgoing edges
		std::vector<BitEdge*> edges_out;

		// The incoming edges
		std::vector<BitEdge*> edges_in;

		// The label, if it has one
		std::string vertex_label;

		// Boolean whether or not it has a label
		bool has_label;

		// The properties - currently can't support list/set per breaking API changes
		std::unordered_map<std::string, VertexProperty*> my_properties;

		// Mutex that prevents concurrent edge addition
		//std::mutex add_edge_mutex;

		// Mutex that prevents concurrent property addition
		//std::mutex add_prop_mutex;

		// Who needs thread safety anyways?

	public:
		BitVertex(CPUGraph* graph, uint64_t vid) {
			this->graph = graph;
			this->vertex_id = vid;
			this->has_label = false;
			this->magic = VERTEX_MAGIC_NUMBER;
		}

		BitVertex(CPUGraph* graph, uint64_t vid, std::string v_label) {
			this->graph = graph;
			this->vertex_id = vid;
			this->has_label = true;
			this->vertex_label = v_label;
			this->magic = VERTEX_MAGIC_NUMBER;
		}

		/*
			Return the Graph this Vertex belongs to.
		*/
		virtual Graph* getGraph() {
			return static_cast<Graph*>(this->graph);
		}
		
		/*
			Get the unique id of the Vertex.
			In CPUGraph this is indirectly
			derived from its initial position
			in the list of Vertices.
		*/
		virtual boost::any id() {
			return boost::any(vertex_id);
		}

		/*
			Make sure to return NULL if there is
			no label for the Vertex
		*/
		virtual std::string label() {
			return has_label ? std::string(vertex_label) : NULL;
		}

		/*
			Nifty helper method that returns
			whether or not this Vertex has a label.
		*/
		virtual bool hasLabel() {
			return has_label;
		}

		/*
			Add an edge to this Vertex in the
			given direction.  Checks to make
			sure the Edge makes sense.
		*/
		void addEdge(BitEdge* new_edge, Direction dir) {
			//add_edge_mutex.lock();

			if(dir == OUT && new_edge->outV() == this) edges_out.push_back(new_edge);
			else if(dir == IN && new_edge->inV() == this) edges_in.push_back(new_edge);

			//add_edge_mutex.unlock();
		}

		/*
			Get edges in a particular direction.
		*/
		virtual std::vector<Edge*> edges(Direction dir) {
			switch(dir) {
				case OUT: {
					return std::vector<Edge*>{this->edges_out.begin(), this->edges_out.end()};
				}
				case IN: {
					return std::vector<Edge*>{this->edges_in.begin(), this->edges_in.end()};
				}
				case BOTH: 
				default: {
					std::vector<Edge*> both_edges;
					both_edges.insert(both_edges.end(), this->edges_in.begin(), this->edges_in.end());
					both_edges.insert(both_edges.end(), this->edges_out.begin(), this->edges_out.end());

					return both_edges;
				}
			}
		}

		/*
			Get the property with the given key.
		*/		
		virtual Property* property(std::string key) {
			auto v = this->my_properties.find(key);
			if(v == my_properties.end()) return nullptr;
			return v->second;
		}

		/*
			Get all the properties with the given keys.
			Should support multiproperties if available.
		*/
		virtual std::vector<Property*> properties(std::vector<std::string> keys) {
			std::vector<Property*> props;
			for(std::string& key : keys) props.push_back(this->property(key));
			return props;
		}

		/*
			Set the property with the given key to the given value.
		*/
		virtual Property* property(Cardinality card, std::string key, boost::any& value) {
			auto old_prop = this->my_properties.find(key);
			
			// Update the indexes if necessary.
			if(this->graph->is_indexed(key)) {
				/*
				If the cardinality is single and there is already an entry for it,
				then clear the index.
				*/
				if(card == SINGLE && old_prop != this->my_properties.end()) {
					this->graph->clear_index(this, key, old_prop->second->value());
				}

				// Perform the index update.  This should be fine for set cardinality.
				graph->update_index(this, key, value);

			}

			if(card == SINGLE) {
				this->my_properties[key] = new VertexProperty(key, value);
			}
			else if(card == SET || card == LIST) { //TODO support list/set
				throw std::runtime_error("Multiproperties currently unsupported!");
			}
			else {
				throw std::runtime_error("Illegal cardinality!");
			}

			return this->my_properties[key];
		}

		/*
			Set the property with the given key to the given value.
		*/
		virtual Property* property(std::string key, boost::any& value) {
			return this->property(SINGLE, key, value);
		}
};

#endif