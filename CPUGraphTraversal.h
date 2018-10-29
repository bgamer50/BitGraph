#include "NoOpStep.h"
#include "GraphTraversal.h"
#include "VertexStep.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddVertexStep.h"
#include "AddEdgeStartStep.h"
#include "AddEdgeStep.h"
#include "HasStep.h"
#include "Traverser.h"
#include <algorithm>
#include <stdlib.h>
#include <exception>

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
			for_each(new_steps.begin(), new_steps.end(), [this](TraversalStep* s){this->steps.push_back(s);});
		}

		/*
			Perform BitGraph-specific optimizations
		*/
		virtual void getInitialTraversal() {
			// Have GraphTraversal do its optimizations first.
			GraphTraversal::getInitialTraversal();

			// No optimizations if this is empty.
			if(steps.empty()) return;

			// Convert g.E() into g.V().outE()
			if(steps[0]->uid == GRAPH_STEP) {
				GraphStep* graph_step = (GraphStep*)steps[0];
				if(graph_step->getType() == EDGE) {
					steps[0] = new GraphStep(VERTEX, {});
					steps.emplace(steps.begin() + 1, new VertexStep(OUT, EDGE));
				}
			}
		}

		/**
			Processes the traversal if not already processed and returns the next traversal result.
		**/
		virtual void* next() {
			return NULL;
		}

		/**
			Processes the traversal if not already processed and executes the given function on
			each of the traversal results.
		**/
		virtual void forEachRemaining(std::function<void (void*)> func) {
			return;
		}

		/**
			Given some initial traversers, execute this CPUGraphTraversal.
		**/
		virtual std::vector<Traverser<void*>*>* execute(std::vector<Traverser<void*>*>* traversers) {
			for(unsigned int index = 0; index < steps.size(); index++) {
				switch(steps[index]->uid) {
					case GRAPH_STEP: // we implicitly assume this is a Vertex Graph step.
						{
							// For each traverser, a traverser should be created for each Vertex and passed to the next step
							std::vector<Vertex*> vertices = getGraph()->vertices();

							// Make map of element ids to counts
							std::vector<void*> element_ids = ((GraphStep*)steps[index])->get_element_ids();
							std::map<uint64_t, unsigned long> element_id_counts;
							for_each(element_ids.begin(), element_ids.end(), [&](void* id_ptr){
								uint64_t id = *((uint64_t*)id_ptr);
								if(element_id_counts.find(id) != element_id_counts.end()) element_id_counts[id]++;
								else element_id_counts[id] = 1;
							});

							std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
							// For each traverser...
							for_each(traversers->begin(), traversers->end(), [&](Traverser<void*>* trv) {
								for_each(vertices.begin(), vertices.end(), [&](Vertex* v) { 
									// TODO this is still inefficient; there is probably a better way to get vertices.
									uint64_t id = *((uint64_t*)v->id());
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
							// For each traverser, a new Vertex should be created and replace the original traverser
							for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								Vertex* v = ((CPUGraph*)this->getGraph())->add_vertex();
								trv->replace_data((void*)v);
							});
							break;
						}
					case ADD_EDGE_STEP:
						{
							// For each traverser
							for_each(traversers->begin(), traversers->end(), [&, this](Traverser<void*>* trv) {
								// Need to check if there is enough info to add the Edge, then add it
								// if we can.
								// For now we will be strict and require that the user explicity specify from() and to() traversals.
								GraphTraversalSource* my_traversal_source = this->getTraversalSource();
								if(my_traversal_source == NULL) throw std::runtime_error("Cannot call this step from an anonymous traversal!");

								CPUGraphTraversal* from_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStep*)steps[index])->get_out_traversal());
								CPUGraphTraversal* to_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStep*)steps[index])->get_in_traversal());
								std::string label = ((AddEdgeStep*)steps[index])->get_label();

								BitVertex* from_vertex;
								BitVertex* to_vertex;
								bool used_current_traverser = false;
								if(from_traversal == NULL) {
									from_vertex = (BitVertex*)(*trv->get());
									used_current_traverser = true;
									if(from_vertex->magic != VERTEX_MAGIC_NUMBER) {
										throw std::runtime_error("Attempted to use implicit from() on object that is not a Vertex."); 
									}
								} else { from_vertex = (BitVertex*)this->get_next(traversers, from_traversal); }

								if(to_traversal == NULL) {
									if(used_current_traverser) {
										throw std::runtime_error("No from/to step was provided.");
									}
									to_vertex = (BitVertex*)(*trv->get());
									if(to_vertex->magic != VERTEX_MAGIC_NUMBER) {
										throw std::runtime_error("Attempted to use implicit to() on object that is not a Vertex.");
									}
								} else { to_vertex = (BitVertex*)this->get_next(traversers, to_traversal); }
								
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
			// Get the initial optimized traversal.
			this->getInitialTraversal();

			// Return and do nothing if there is nothing to execute.
			if(steps.empty()) {
				return;
			}

			std::vector<Traverser<void*>*>* traversers = new std::vector<Traverser<void*>*>; // Will use these to traverse

			// Handle the start step (GraphStep(Vertex), GraphStep(Edge), or AddEdgeStartStep)
			switch(steps[0]->uid) {
				case GRAPH_STEP: {
					GraphStep* graph_step = (GraphStep*)steps[0];
					if(graph_step->getType() == VERTEX) {
						// Create one traverser for each Vertex in the Graph.
						std::vector<Vertex*> vertices = getGraph()->vertices();
						for_each(vertices.begin(), vertices.end(), [&traversers](Vertex* v){ traversers->push_back(new Traverser<void*>((void*)v)); });
					}
					else if(graph_step->getType() == EDGE) {
						// TODO this should never be true since getInitialTraversal() is called.
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

					CPUGraphTraversal* from_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStep*)steps[0])->get_out_traversal());
					CPUGraphTraversal* to_traversal = new CPUGraphTraversal(my_traversal_source, ((AddEdgeStep*)steps[0])->get_in_traversal());
					std::string label = ((AddEdgeStep*)steps[0])->get_label();

					BitVertex* from_vertex;
					BitVertex* to_vertex;

					// Add a dummy traverser so that execute() will actually do something.
					traversers->push_back(new Traverser<void*>(NULL));

					if(from_traversal == NULL) {
						throw std::runtime_error("No from step was provided.");
					} else { from_vertex = (BitVertex*)this->get_next(traversers, from_traversal); }

					if(to_traversal == NULL) {
						throw std::runtime_error("No to step was provided.");
					} else { to_vertex = (BitVertex*)this->get_next(traversers, to_traversal); }

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

			this->execute(traversers);
		}

		/*
			Create a copy of this traversal's traversers.  Used when some anonymous traversal needs to be passed the
			existing traversers; obviously we to maintain the current traverser status.
		*/
		inline std::vector<Traverser<void*>*>* copy_traversers(std::vector<Traverser<void*>*>* old_traversers) {
			std::vector<Traverser<void*>*>* new_traversers = new std::vector<Traverser<void*>*>;
			for_each(old_traversers->begin(), old_traversers->end(), [&new_traversers](Traverser<void*>* trv){
				new_traversers->push_back(new Traverser<void*>(*trv->get()));
			});

			return new_traversers;
		}

		/*
			Get the next object from the given traversal.  Makes a copy of the given traversers.
		*/
		inline void* get_next(std::vector<Traverser<void*>*>* traversers, CPUGraphTraversal* traversal) {
			// Call execute in the given traversal with a copy of the current traversers.
			std::vector<Traverser<void*>*>* temporary_traversers = this->copy_traversers(traversers);
			temporary_traversers = traversal->execute(temporary_traversers);

			// Make sure the traversal evaluated to something.
			if(temporary_traversers->empty()) throw std::runtime_error("Traversal evaluated to empty set!");
			
			// Correct behavior is grabbing the first traverser and ignoring any others.
			Traverser<void*>* out_traverser = temporary_traversers->at(0);
			void* next_object = *out_traverser->get();

			// Ensure memory for the temporary traversers is cleared.
			for_each(temporary_traversers->begin(), temporary_traversers->end(), [](Traverser<void*>* trv) { delete trv; });
			delete temporary_traversers;

			return next_object;
		}
};

#endif