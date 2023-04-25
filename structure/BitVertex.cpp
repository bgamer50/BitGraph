#include "structure/BitVertex.h"
#include "structure/CPUGraph.h"
#include "structure/BitEdge.h"

Graph* BitVertex::getGraph() {
	return static_cast<Graph*>(this->graph);
}

Property* BitVertex::property(Cardinality card, std::string key, boost::any& value) {
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

void BitVertex::addEdge(BitEdge* new_edge, Direction dir) {
	//add_edge_mutex.lock();

	if(dir == OUT && new_edge->outV() == this) edges_out.push_back(new_edge);
	else if(dir == IN && new_edge->inV() == this) edges_in.push_back(new_edge);

	//add_edge_mutex.unlock();
}

/*
    Get edges in a particular direction.
*/
std::vector<Edge*> BitVertex::edges(Direction dir) {
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
Property* BitVertex::property(std::string key) {
    auto v = this->my_properties.find(key);
    if(v == my_properties.end()) return nullptr;
    return v->second;
}

/*
    Get all the properties with the given keys.
    Should support multiproperties if available.
*/
std::vector<Property*> BitVertex::properties(std::vector<std::string> keys) {
    std::vector<Property*> props;
    if(keys.empty()) {
        props.resize(this->my_properties.size());
        size_t i = 0;
        for(std::pair<std::string, VertexProperty*> p : this->my_properties) props[i++] = p.second;
    } 
    else {
        props.resize(keys.size());
        size_t i = 0;
        for(std::string& key : keys) props[i++] = this->property(key);
    }

    return props;
}

/*
    Set the property with the given key to the given value.
*/
Property* BitVertex::property(std::string key, boost::any& value) {
    return this->property(SINGLE, key, value);
}

std::vector<Property*> BitVertex::properties() {
    return this->properties({});
}