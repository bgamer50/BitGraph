#include <algorithm>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <boost/any.hpp>

#include "NoOpStep.h"
#include "GraphTraversal.h"
#include "VertexStep.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddVertexStep.h"
#include "AddEdgeStartStep.h"
#include "AddEdgeStep.h"
#include "AddPropertyStep.h"
#include "HasStep.h"
#include "Traverser.h"
#include "BitEdge.h"
#include "BitVertex.h"
#include "Graph.h"
class CPUGraph;

#ifndef CPU_GRAPH_TRAVERSAL_H
#define CPU_GRAPH_TRAVERSAL_H

class CPUGraphTraversal : public GraphTraversal {
	public:
		CPUGraphTraversal(GraphTraversalSource* src)
		: GraphTraversal(src) {
			// do nothing
		}

		CPUGraphTraversal(GraphTraversalSource* src, GraphTraversal* anonymous_traversal)
		: GraphTraversal(src) {
			std::vector<TraversalStep*> new_steps = anonymous_traversal->getSteps();
			std::for_each(new_steps.begin(), new_steps.end(), [this](TraversalStep* s){this->steps.push_back(s);});
		}

		/*
			Perform BitGraph-specific optimizations
		*/
		virtual void getInitialTraversal() {
			// Have GraphTraversal do its optimizations first.
			GraphTraversal::getInitialTraversal();

			// No optimizations if this is empty.
			if(this->steps.empty()) return;

			// Convert g.E() into g.V().outE()
			if(this->steps[0]->uid == GRAPH_STEP) {
				GraphStep* graph_step = (GraphStep*)this->steps[0];
				if(graph_step->getType() == EDGE) {
					this->steps[0] = new GraphStep(VERTEX, {});
					this->steps.emplace(this->steps.begin() + 1, new VertexStep(OUT, EDGE));
				}
			}
		}

		/**
			Processes the traversal if not already processed and returns the next traversal result.
			Official API call.
		**/
		virtual boost::any next() {
			std::vector<Traverser*>* traversers = this->get_initial_traversers();
			traversers = this->execute(traversers);

			if(traversers->empty()) throw std::runtime_error("No Traversers were available!");

			Traverser* trv = traversers->at(0);
			boost::any data = trv->get();
			free(traversers);

			return data;
		}

		/**
			Processes the traversal if not already processed and executes the given function on
			each of the traversal results.
		**/
		virtual void forEachRemaining(std::function<void(boost::any&)> func) {
			std::vector<Traverser*>* traversers = this->get_initial_traversers();
			traversers = this->execute(traversers);

			if(traversers->empty()) throw std::runtime_error("No Traversers were available!");

			std::for_each(traversers->begin(), traversers->end(), [&func](Traverser* trv) {
				boost::any t = trv->get();
				func(t);
			});

			delete traversers;
		}

		/**
			Given some initial traversers, execute this CPUGraphTraversal.  This method does most
			of the real work once there is data to operate on.
		**/
		virtual std::vector<Traverser*>* execute(std::vector<Traverser*>* traversers) {
			for(auto index = 0; index < this->steps.size(); index++) {
				switch(this->steps[index]->uid) {
					case GRAPH_STEP: // we implicitly assume this is a Vertex Graph step.
						{
							cout << "graph step\n";
							// For each traverser, a traverser should be created for each Vertex and passed to the next step
							std::vector<Vertex*> vertices = this->getGraph()->vertices();

							// Make map of element ids to counts
							std::vector<boost::any> element_ids = ((GraphStep*)this->steps[index])->get_element_ids();
							std::map<uint64_t, unsigned long> element_id_counts;
							std::for_each(element_ids.begin(), element_ids.end(), [&](boost::any id_ctr){
								uint64_t id = boost::any_cast<uint64_t>(id_ctr);
								if(element_id_counts.find(id) != element_id_counts.end()) element_id_counts[id]++;
								else element_id_counts[id] = 1;
							});

							std::vector<Traverser*>* new_traversers = new std::vector<Traverser*>;
							// For each traverser...
							std::for_each(traversers->begin(), traversers->end(), [&](Traverser* trv) {
								std::for_each(vertices.begin(), vertices.end(), [&](Vertex* v) {
									// TODO this is still inefficient; there is probably a better way to get vertices.
									uint64_t id = boost::any_cast<uint64_t>(v->id());
									if(element_id_counts.find(id) != element_id_counts.end()) {
										for(auto k = 0; k < element_id_counts[id]; k++) new_traversers->push_back(new Traverser(v));
										// TODO retain side effect information
									}
								});
							});

							delete traversers;
							traversers = new_traversers;
							break;
						}
					case ADD_VERTEX_STEP:
						{
							cout << "add vertex step\n";
							// For each traverser, a new Vertex should be created and replace the original traverser
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
								Vertex* v = this->getGraph()->add_vertex();
								trv->replace_data(v);
							});
							break;
						}
					case ADD_EDGE_STEP:
						{
							std::cout << "add edge step\n";
							AddEdgeStep* add_edge_step = ((AddEdgeStep*)this->steps[index]);
							// For each traverser
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
								// Need to check if there is enough info to add the Edge, then add it
								// if we can.
								// For now we will be strict and require that the user explicitly specify from() and to() traversals.
								GraphTraversalSource* my_traversal_source = this->getTraversalSource();
								if(my_traversal_source == NULL) throw std::runtime_error("Cannot call this step from an anonymous traversal!");

								GraphTraversal* out_traversal = add_edge_step->get_out_traversal();
								GraphTraversal* in_traversal = add_edge_step->get_in_traversal();
								
								CPUGraphTraversal* from_traversal = out_traversal == nullptr ? nullptr : new CPUGraphTraversal(my_traversal_source, out_traversal);
								CPUGraphTraversal* to_traversal = in_traversal == nullptr ? nullptr : new CPUGraphTraversal(my_traversal_source, in_traversal);

								std::string label = ((AddEdgeStep*)this->steps[index])->get_label();

								Vertex* from_vertex;
								Vertex* to_vertex;
								bool used_current_traverser = false;
								if(from_traversal == nullptr) {
									from_vertex = boost::any_cast<Vertex*>(trv->get());
									used_current_traverser = true;
								} else { from_vertex = boost::any_cast<Vertex*>(this->get_next(traversers, from_traversal)); }

								if(to_traversal == nullptr) {
									if(used_current_traverser) {
										throw std::runtime_error("No from/to step was provided.");
									}
									to_vertex = boost::any_cast<Vertex*>(trv->get());
								} else { to_vertex = boost::any_cast<Vertex*>(this->get_next(traversers, to_traversal)); }

								std::cout << "from vertex: " << boost::any_cast<uint64_t>(from_vertex->id()) << "\n";
								std::cout << "to vertex: " << boost::any_cast<uint64_t>(to_vertex->id()) << "\n";

								Graph* my_graph = this->getGraph();
								Edge* new_edge = my_graph->add_edge(static_cast<BitVertex*>(from_vertex), static_cast<BitVertex*>(to_vertex), label);
								trv->replace_data(new_edge);

								// Clean up completed traversals
								delete from_traversal;
								delete to_traversal;
							});

							break;
						}
					case NO_OP_STEP: {
						break;
					}

					case ADD_PROPERTY_STEP: {
						std::cout << "add property step\n";
						AddPropertyStep* add_property_step = dynamic_cast<AddPropertyStep*>(this->steps[index]);

						if(add_property_step->get_value().type() == typeid(GraphTraversal*)) {
							GraphTraversal* traversal = boost::any_cast<GraphTraversal*>(add_property_step->get_value());
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
								BitVertex* v = static_cast<BitVertex*>(boost::any_cast<Vertex*>(trv->get()));
								CPUGraphTraversal* new_trv = new CPUGraphTraversal(this->getTraversalSource(), traversal);
								
								// Execute traversal
								std::vector<Traverser*>* temp_traversers = new std::vector<Traverser*>;
								temp_traversers->push_back(new Traverser(trv->get()));
								boost::any prop_value = this->get_next(temp_traversers, new_trv);
								delete temp_traversers->at(0);
								delete temp_traversers;

								// Store the property
								v->property(add_property_step->get_key(), prop_value);
								std::cout << "property stored!\n";
							});
						}
						else {
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
								Vertex* v = boost::any_cast<Vertex*>(trv->get());
								add_property_step->apply(v);
							});
						}

						// Regardless of what happens (except an error), the traversers should continue through unmodified.
						break;
					}

					case HAS_STEP: {
					    cout << "has step\n";
                        HasStep* has_step = dynamic_cast<HasStep*>(this->steps[index]);
                        
                        std::vector<Traverser*>* new_traversers = new std::vector<Traverser*>;
                        std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
							Vertex* v = boost::any_cast<Vertex*>(trv->get());
                            //cout << "vertex id " << boost::any_cast<uint64_t>(v->id()) << "\n";
							//cout << "applying has step...\n";
							//std::cout << "Vertex " << boost::any_cast<uint64_t>(v->id()) << " has a=" << boost::any_cast<std::string>(v->property("a")->value()) << "\n";
                            bool advance = has_step->apply(v);
                            //cout << "application of has step was successful\n";
                            if(advance) new_traversers->push_back(trv); else delete trv;
                        });
                        delete traversers;
                        traversers = new_traversers;
                        break;
					}

					case ID_STEP: {
						std::cout << "id step\n";
						IdStep* id_step = dynamic_cast<IdStep*>(this->steps[index]);

						try {
							std::for_each(traversers->begin(), traversers->end(), [](Traverser* trv) {
								trv->replace_data(boost::any_cast<Vertex*>(trv->get())->id());
							});
						} catch(const std::exception& err) {
							std::for_each(traversers->begin(), traversers->end(), [](Traverser* trv) {
								trv->replace_data(boost::any_cast<Edge*>(trv->get())->id());
							});
						} catch(...) {
							throw std::runtime_error("Error: Traverser does not appear to contain an element.");
						}

						break;
					}

					case VERTEX_STEP: {
						cout << "vertex step\n";
						VertexStep* vertex_step = dynamic_cast<VertexStep*>(this->steps[index]);

						std::set<std::string> labels = vertex_step->get_labels();
						bool label_required = !labels.empty();
						Direction direction = vertex_step->get_direction();

						std::vector<Traverser*>* new_traversers = new std::vector<Traverser*>;
						std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
							BitVertex* v = static_cast<BitVertex*>(boost::any_cast<Vertex*>(trv->get()));
							for(BitEdge* e : v->edges(direction)) {
								if(label_required && labels.count(e->label()) == 0) continue;

								switch(direction) {
									case IN: {
										Vertex* w = e->outV();
										new_traversers->push_back(new Traverser(w));
										break;
									}
									case OUT: {
										Vertex* w = e->inV();
										new_traversers->push_back(new Traverser(w));
										break;
									}

									case BOTH: {
										Vertex* u = e->outV();
										Vertex* w = u == v ? e->inV() : u;
										new_traversers->push_back(new Traverser(w));
										break;
									}
								}

							}
						});

						delete traversers;
						traversers = new_traversers;
						break;
					}

					default: {

						// TODO throw exception
						break;
					}
				}

			}

			return traversers;
		}

		/**
			Performs each step of the traversal.
		**/
		virtual void iterate() {
            auto traversers = this->get_initial_traversers();
			this->execute(traversers);
		}

		/*
			Create a copy of this traversal's traversers.  Used when some anonymous traversal needs to be passed the
			existing traversers; obviously we to maintain the current traverser status.
		*/
		inline std::vector<Traverser*>* copy_traversers(std::vector<Traverser*>* old_traversers) {
			std::vector<Traverser*>* new_traversers = new std::vector<Traverser*>;
			std::for_each(old_traversers->begin(), old_traversers->end(), [&new_traversers](Traverser* trv){
				new_traversers->push_back(new Traverser(trv->get()));
			});

			return new_traversers;
		}

		/*
			Get the next object from the given traversal.  Makes a copy of the given traversers.
		*/
		inline boost::any get_next(std::vector<Traverser*>* traversers, CPUGraphTraversal* traversal) {
			// Call execute in the given traversal with a copy of the current traversers.
			std::vector<Traverser*>* temporary_traversers = this->copy_traversers(traversers);
			temporary_traversers = traversal->execute(temporary_traversers);

			// Make sure the traversal evaluated to something.
			if(temporary_traversers->empty()) throw std::runtime_error("Traversal evaluated to empty set!");

			// Correct behavior is grabbing the first traverser and ignoring any others.
			Traverser* out_traverser = temporary_traversers->at(0);
			boost::any next_object = out_traverser->get();

			// Ensure memory for the temporary traversers is cleared.
			std::for_each(temporary_traversers->begin(), temporary_traversers->end(), [](Traverser* trv) { delete trv; });
			delete temporary_traversers;

			return next_object;
		}

    private:
        std::vector<Traverser*>* get_initial_traversers() {
            // Get the initial optimized traversal.
			this->getInitialTraversal();

			// Return and do nothing if there is nothing to execute.
			if(this->steps.empty()) {
				return nullptr;
			}

			std::vector<Traverser*>* traversers = new std::vector<Traverser*>; // Will use these to traverse

			// Handle the start step (GraphStep(Vertex), GraphStep(Edge), or AddEdgeStartStep)
			switch(this->steps[0]->uid) {
				case GRAPH_STEP: {
					GraphStep* graph_step = (GraphStep*)this->steps[0];
					if(graph_step->getType() == VERTEX) {
						std::set<uint64_t> ids; for(boost::any id_ctr : graph_step->get_element_ids()) ids.insert(boost::any_cast<uint64_t>(id_ctr));

						std::vector<Vertex*> vertices = this->getGraph()->vertices();
						
						if(!ids.empty()) {
							// Create one traverser for each Vertex requested.
							// TODO this is sort of backwards; ideally the Vertices are in a BST or similar structure.
							for(Vertex* v : vertices) {
								uint64_t id = boost::any_cast<uint64_t>(v->id());
								if(0 < ids.count(id)) traversers->push_back(new Traverser(v));
							}
						}
						else {
							// Create one traverser for each Vertex in the Graph.
							std::for_each(vertices.begin(), vertices.end(), [&traversers](Vertex* v){ traversers->push_back(new Traverser(v)); });
						}
					}
					else if(graph_step->getType() == EDGE) {
						// This should never be true since getInitialTraversal() is called.
						throw std::runtime_error("Illegal state for traverser; likely initialization failure.");
					}
					break;
				}
				case ADD_VERTEX_START_STEP: {
					Vertex* v = this->getGraph()->add_vertex();
					traversers->push_back(new Traverser(v));
					break;
				}
				case ADD_EDGE_START_STEP: {
					// Need to check if there is enough info to add the Edge, then add it
					// if we can.
					// from() and to() are both always required here.
					GraphTraversalSource* my_traversal_source = this->getTraversalSource();
					if(my_traversal_source == NULL) throw std::runtime_error("Cannot call this step from an anonymous traversal!\n");

					CPUGraphTraversal* from_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStartStep*)this->steps[0])->get_out_traversal());
					CPUGraphTraversal* to_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStartStep*)this->steps[0])->get_in_traversal());
					std::string label = static_cast<AddEdgeStartStep*>(this->steps[0])->get_label();

					Vertex* from_vertex;
					Vertex* to_vertex;

					// Add a dummy traverser so that execute() will actually do something.
					traversers->push_back(new Traverser(NULL));

					std::cout << "about to execute\n";

					if(from_traversal == NULL) {
						throw std::runtime_error("No from step was provided.");
					} else { from_vertex = boost::any_cast<Vertex*>(this->get_next(traversers, from_traversal)); }

					if(to_traversal == NULL) {
						throw std::runtime_error("No to step was provided.");
					} else { to_vertex = boost::any_cast<Vertex*>(this->get_next(traversers, to_traversal)); }

					// Remove the dummy traverser
					traversers->erase(traversers->begin());

					Graph* graph = this->getGraph();
					BitEdge* new_edge = (BitEdge*)graph->add_edge(static_cast<BitVertex*>(from_vertex), static_cast<BitVertex*>(to_vertex), label);
					traversers->push_back(new Traverser(new_edge));
					break;
				}
				default: {
					// TODO throw exception
				}
			}

			delete this->steps[0];
			this->steps[0] = new NoOpStep();

			return traversers;
        }
};

#endif
