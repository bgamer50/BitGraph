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
#include "ApplicableStep.h"
#include "HasStep.h"
#include "Traverser.h"
#include "BitEdge.h"
#include "BitVertex.h"
#include "CPUGraph.h"

#ifndef CPU_GRAPH_TRAVERSAL_H
#define CPU_GRAPH_TRAVERSAL_H

template<typename U, typename W>
class CPUGraphTraversal : public GraphTraversal<U, W> {
	public:
		CPUGraphTraversal(GraphTraversalSource* src)
		: GraphTraversal<U, W>(src) {
			// do nothing
		}

		CPUGraphTraversal(GraphTraversalSource* src, GraphTraversal<void*, W>* anonymous_traversal)
		: GraphTraversal<U, W>(src) {
			std::vector<TraversalStep*> new_steps = anonymous_traversal->getSteps();
			std::for_each(new_steps.begin(), new_steps.end(), [this](TraversalStep* s){this->steps.push_back(s);});
		}

		/*
			Perform BitGraph-specific optimizations
		*/
		virtual void getInitialTraversal() {
			// Have GraphTraversal do its optimizations first.
			GraphTraversal<U, W>::getInitialTraversal();

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
		virtual W* next() {
			std::vector<Traverser<void*>*>* traversers = this->get_initial_traversers();
			traversers = this->execute(traversers);

			if(traversers->empty()) throw std::runtime_error("No Traversers were available!");

			Traverser<void*>* trv = traversers->at(0);
			void* data = *trv->get();
			free(traversers);

			return (W*)data;
		}

		/**
			Processes the traversal if not already processed and executes the given function on
			each of the traversal results.
		**/
		virtual void forEachRemaining(std::function<void(void*)> func) {
			std::vector<Traverser<void*>*>* traversers = this->get_initial_traversers();
			traversers = this->execute(traversers);

			if(traversers->empty()) if(traversers->empty()) throw std::runtime_error("No Traversers were available!");
			std::cout << "size: " << traversers->size() << "\n";

			std::for_each(traversers->begin(), traversers->end(), [&func](Traverser<void*>* trv) {
				std::cout << "asdf\n";
				func(*trv->get());
			});

			delete traversers;
		}

		/**
			Given some initial traversers, execute this CPUGraphTraversal.  This method does most
			of the real work once there is data to operate on.
		**/
		virtual std::vector<Traverser<void*>*>* execute(std::vector<Traverser<void*>*>* traversers) {
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

							std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
							// For each traverser...
							std::for_each(traversers->begin(), traversers->end(), [&](Traverser<void*>* trv) {
								std::for_each(vertices.begin(), vertices.end(), [&](Vertex* v) {
									// TODO this is still inefficient; there is probably a better way to get vertices.
									uint64_t id = boost::any_cast<uint64_t>(v->id());
									if(element_id_counts.find(id) != element_id_counts.end()) {
										for(auto k = 0; k < element_id_counts[id]; k++) new_traversers->push_back(new Traverser<void*>((void*)v));
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
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								Vertex* v = ((CPUGraph*)this->getGraph())->add_vertex();
								trv->replace_data((void*)v);
							});
							break;
						}
					case ADD_EDGE_STEP:
						{
							std::cout << "add edge step\n";
							AddEdgeStep* add_edge_step = ((AddEdgeStep*)this->steps[index]);
							// For each traverser
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								// Need to check if there is enough info to add the Edge, then add it
								// if we can.
								// For now we will be strict and require that the user explicitly specify from() and to() traversals.
								GraphTraversalSource* my_traversal_source = this->getTraversalSource();
								if(my_traversal_source == NULL) throw std::runtime_error("Cannot call this step from an anonymous traversal!");

								GraphTraversal<void*, Vertex>* out_traversal = add_edge_step->get_out_traversal();
								GraphTraversal<void*, Vertex>* in_traversal = add_edge_step->get_in_traversal();
								
								CPUGraphTraversal<void*, Vertex>* from_traversal = out_traversal == nullptr ? nullptr : new CPUGraphTraversal<void*, Vertex>(my_traversal_source, out_traversal);
								CPUGraphTraversal<void*, Vertex>* to_traversal = in_traversal == nullptr ? nullptr : new CPUGraphTraversal<void*, Vertex>(my_traversal_source, in_traversal);

								std::string label = ((AddEdgeStep*)this->steps[index])->get_label();

								BitVertex* from_vertex;
								BitVertex* to_vertex;
								bool used_current_traverser = false;
								if(from_traversal == nullptr) {
									from_vertex = (BitVertex*)(*trv->get());
									used_current_traverser = true;
									if(from_vertex->magic != VERTEX_MAGIC_NUMBER) {
										throw std::runtime_error("Attempted to use implicit from() on object that is not a Vertex.");
									}
								} else { from_vertex = (BitVertex*)this->get_next(traversers, (CPUGraphTraversal<void*, void*>*)from_traversal); }

								if(to_traversal == nullptr) {
									if(used_current_traverser) {
										throw std::runtime_error("No from/to step was provided.");
									}
									to_vertex = (BitVertex*)(*trv->get());
									if(to_vertex->magic != VERTEX_MAGIC_NUMBER) {
										throw std::runtime_error("Attempted to use implicit to() on object that is not a Vertex.");
									}
								} else { to_vertex = (BitVertex*)this->get_next(traversers, (CPUGraphTraversal<void*, void*>*)to_traversal); }

								std::cout << "from vertex: " << boost::any_cast<uint64_t>(from_vertex->id()) << "\n";
								std::cout << "to vertex: " << boost::any_cast<uint64_t>(to_vertex->id()) << "\n";

								BitEdge* new_edge = (BitEdge*)((CPUGraph*)this->getGraph())->add_edge(from_vertex, to_vertex, label);
								trv->replace_data((void*)new_edge);

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

						AddPropertyStep* add_property_step = dynamic_cast<AddPropertyStep*>(this->steps[index]);

						try {
							GraphTraversal* trv = boost::any_cast<GraphTraversal*>(add_property_step->get_value())
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								Vertex* v = static_cast<Vertex*>(*trv->get());
								CPUGraphTraversal* new_trv = new CPUGraphTraversal(this->getTraversalSource(), trv);
								boost::any prop_value = boost::any(this->get_next(traversers, new_trv));
								//TODO this breaks void* entirely.  I doubt it will work.
							});
						} catch(std::exception& err) {
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								Vertex* v = static_cast<Vertex*>(*trv->get());
								add_property_step->apply(v);
							});
						}
						break;
					}

					case HAS_STEP: {
					    cout << "has step\n";
                        HasStep* has_step = dynamic_cast<HasStep*>(this->steps[index]);
                        
                        std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
                        std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
                            Vertex* v = static_cast<Vertex*>(*trv->get());
                            //cout << "vertex id " << boost::any_cast<uint64_t>(v->id()) << "\n";
							//cout << "applying has step...\n";
                            bool advance = has_step->apply(v);
                            //cout << "application of has step was successful\n";
                            if(advance) new_traversers->push_back(trv); else delete trv;
                        });
                        delete traversers;
                        traversers = new_traversers;
                        break;
					}

					case VERTEX_STEP: {
						cout << "vertex step\n";
						VertexStep* vertex_step = dynamic_cast<VertexStep*>(this->steps[index]);

						std::set<std::string> labels = vertex_step->get_labels();
						bool label_required = !labels.empty();
						Direction direction = vertex_step->get_direction();

						std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
						std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
							BitVertex* v = static_cast<BitVertex*>(*trv->get());
							for(BitEdge* e : v->edges(direction)) {
								if(label_required && labels.count(e->label()) == 0) continue;

								Vertex* w = direction == OUT ? e->inV() : e->outV();
								new_traversers->push_back(new Traverser<void*>(static_cast<void*>(w)));
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
		inline std::vector<Traverser<void*>*>* copy_traversers(std::vector<Traverser<void*>*>* old_traversers) {
			std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
			std::for_each(old_traversers->begin(), old_traversers->end(), [&new_traversers](Traverser<void*>* trv){
				new_traversers->push_back(new Traverser<void*>(*trv->get()));
			});

			return new_traversers;
		}

		/*
			Get the next object from the given traversal.  Makes a copy of the given traversers.
		*/
		inline void* get_next(std::vector<Traverser<void*>*>* traversers, CPUGraphTraversal<void*, void*>* traversal) {
			// Call execute in the given traversal with a copy of the current traversers.
			std::vector<Traverser<void*>*>* temporary_traversers = this->copy_traversers(traversers);
			temporary_traversers = traversal->execute(temporary_traversers);

			// Make sure the traversal evaluated to something.
			if(temporary_traversers->empty()) throw std::runtime_error("Traversal evaluated to empty set!");

			// Correct behavior is grabbing the first traverser and ignoring any others.
			Traverser<void*>* out_traverser = temporary_traversers->at(0);
			void* next_object = *out_traverser->get();

			// Ensure memory for the temporary traversers is cleared.
			std::for_each(temporary_traversers->begin(), temporary_traversers->end(), [](Traverser<void*>* trv) { delete trv; });
			delete temporary_traversers;

			return next_object;
		}

    private:
        std::vector<Traverser<void*>*>* get_initial_traversers() {
            // Get the initial optimized traversal.
			this->getInitialTraversal();

			// Return and do nothing if there is nothing to execute.
			if(this->steps.empty()) {
				return nullptr;
			}

			std::vector<Traverser<void*>*>* traversers = new std::vector<Traverser<void*>*>; // Will use these to traverse

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
								if(0 < ids.count(id)) traversers->push_back((Traverser<void*>*)new Traverser<Vertex*>(v));
							}
						}
						else {
							// Create one traverser for each Vertex in the Graph.
							std::for_each(vertices.begin(), vertices.end(), [&traversers](Vertex* v){ traversers->push_back((Traverser<void*>*)new Traverser<Vertex*>(v)); });
						}
					}
					else if(graph_step->getType() == EDGE) {
						// This should never be true since getInitialTraversal() is called.
						throw std::runtime_error("Illegal state for traverser; likely initialization failure.");
					}
					break;
				}
				case ADD_VERTEX_START_STEP: {
					Vertex* v = ((CPUGraph*)this->getGraph())->add_vertex();
					traversers->push_back(new Traverser<void*>((void*)v));
					break;
				}
				case ADD_EDGE_START_STEP: {
					// Need to check if there is enough info to add the Edge, then add it
					// if we can.
					// from() and to() are both always required here.
					GraphTraversalSource* my_traversal_source = this->getTraversalSource();
					if(my_traversal_source == NULL) throw std::runtime_error("Cannot call this step from an anonymous traversal!\n");

					CPUGraphTraversal<void*, Vertex>* from_traversal = new CPUGraphTraversal<void*, Vertex>(my_traversal_source, ((AddEdgeStep*)this->steps[0])->get_out_traversal());
					CPUGraphTraversal<void*, Vertex>* to_traversal = new CPUGraphTraversal<void*, Vertex>(my_traversal_source, ((AddEdgeStep*)this->steps[0])->get_in_traversal());
					std::string label = ((AddEdgeStep*)this->steps[0])->get_label();

					BitVertex* from_vertex;
					BitVertex* to_vertex;

					// Add a dummy traverser so that execute() will actually do something.
					traversers->push_back(new Traverser<void*>(NULL));

					if(from_traversal == NULL) {
						throw std::runtime_error("No from step was provided.");
					} else { from_vertex = (BitVertex*)this->get_next(traversers, (CPUGraphTraversal<void*, void*>*)from_traversal); }

					if(to_traversal == NULL) {
						throw std::runtime_error("No to step was provided.");
					} else { to_vertex = (BitVertex*)this->get_next(traversers, (CPUGraphTraversal<void*, void*>*)to_traversal); }

					// Remove the dummy traverser
					traversers->erase(traversers->begin());

					BitEdge* new_edge = (BitEdge*)((CPUGraph*)this->getGraph())->add_edge(from_vertex, to_vertex, label);
					traversers->push_back(new Traverser<void*>((void*)new_edge));
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
