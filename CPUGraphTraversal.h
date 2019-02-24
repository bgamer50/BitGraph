#include <algorithm>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <list>
#include <unordered_set>
#include <boost/any.hpp>

#include "Direction.h"
#include "NoOpStep.h"
#include "GraphTraversal.h"
#include "VertexStep.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddVertexStep.h"
#include "AddEdgeStartStep.h"
#include "AddEdgeStep.h"
#include "AddPropertyStep.h"
#include "PropertyStep.h"
#include "IdentityStep.h"
#include "IndexStep.h"
#include "HasStep.h"
#include "Traverser.h"
#include "BitEdge.h"
#include "BitVertex.h"
#include "Graph.h"
#include "GPUFilterStep.h"
class CPUGraph;

#ifndef CPU_GRAPH_TRAVERSAL_H
#define CPU_GRAPH_TRAVERSAL_H
//#define VERBOSE

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
		virtual void getInitialTraversal();

		/**
			Processes the traversal if not already processed and returns the next traversal result.
			Official API call.
		**/
		virtual boost::any next() {
			std::list<Traverser*>* traversers = this->get_initial_traversers();
			traversers = this->execute(traversers);

			if(traversers->empty()) throw std::runtime_error("No Traversers were available!");

			Traverser* trv = traversers->front();
			boost::any data = trv->get();
			free(traversers);

			return data;
		}

		/**
			Processes the traversal if not already processed and executes the given function on
			each of the traversal results.
		**/
		virtual void forEachRemaining(std::function<void(boost::any&)> func) {
			std::list<Traverser*>* traversers = this->get_initial_traversers();
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
		virtual std::list<Traverser*>* execute(std::list<Traverser*>* traversers) {
			for(auto index = 0; index < this->steps.size(); index++) {
				switch(this->steps[index]->uid) {
					case GRAPH_STEP: // we implicitly assume this is a Vertex Graph step.
						{
							#ifdef VERBOSE
							cout << "graph step\n";
							#endif
							GraphStep* graph_step = dynamic_cast<GraphStep*>(this->steps[index]);
							std::list<Traverser*>* new_traversers = execute_graph_step(graph_step, traversers);

							delete traversers;
							traversers = new_traversers;
							break;
						}
					case ADD_VERTEX_STEP:
						{
							#ifdef VERBOSE
							std::cout << "add vertex step\n";
							#endif
							// For each traverser, a new Vertex should be created and replace the original traverser
							std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
								Vertex* v = this->getGraph()->add_vertex();
								trv->replace_data(v);
							});
							break;
						}
					case ADD_EDGE_STEP:
						{
							#ifdef VERBOSE
							std::cout << "add edge step\n";
							#endif
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

								//std::cout << "from vertex: " << boost::any_cast<uint64_t>(from_vertex->id()) << "\n";
								//std::cout << "to vertex: " << boost::any_cast<uint64_t>(to_vertex->id()) << "\n";

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
						#ifdef VERBOSE
						std::cout << "add property step\n";
						#endif
						AddPropertyStep* add_property_step = dynamic_cast<AddPropertyStep*>(this->steps[index]);

						this->execute_add_property_step(add_property_step, traversers);
						// Regardless of what happens (except an error), the traversers should continue through unmodified.
						break;
					}

					case HAS_STEP: {
						#ifdef VERBOSE
						cout << "has step\n";
						#endif
                        HasStep* has_step = dynamic_cast<HasStep*>(this->steps[index]);
						std::list<Traverser*>* new_traversers = this->execute_has_step(has_step, traversers);

						delete traversers;
                        traversers = new_traversers;
                        break;                        
					}

					case GPU_FILTER_STEP: {
						#ifdef VERBOSE
						cout << "gpu filter step\n";
						#endif
						GPUFilterStep* gpu_filter_step = dynamic_cast<GPUFilterStep*>(this->steps[index]);
						gpu_filter_step->apply(traversers);
						break;
					}

					case MIN_STEP: {
						#ifdef VERBOSE
						std::cout << "min step\n";
						#endif
						this->execute_min_step(traversers, dynamic_cast<MinStep*>(this->steps[index]));
						break;
					}

					case ID_STEP: {
						#ifdef VERBOSE
						std::cout << "id step\n";
						#endif
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
						#ifdef VERBOSE
						cout << "vertex step\n";
						#endif
						VertexStep* vertex_step = dynamic_cast<VertexStep*>(this->steps[index]);

						std::set<std::string> labels = vertex_step->get_labels();
						bool label_required = !labels.empty();
						Direction direction = vertex_step->get_direction();

						std::list<Traverser*> new_traversers;
						std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
							BitVertex* v = static_cast<BitVertex*>(boost::any_cast<Vertex*>(trv->get()));
							for(BitEdge* e : v->edges(direction)) {
								if(label_required && labels.count(e->label()) == 0) continue;

								switch(direction) {
									case IN: {
										Vertex* w = e->outV();
										new_traversers.push_back(new Traverser(w));
										break;
									}
									case OUT: {
										Vertex* w = e->inV();
										new_traversers.push_back(new Traverser(w));
										break;
									}
									case BOTH: {
										Vertex* u = e->outV();
										Vertex* w = u == v ? e->inV() : u;
										new_traversers.push_back(new Traverser(w));
										break;
									}
								}

							}
						});

						traversers->clear();
						for(auto it = new_traversers.begin(); it != new_traversers.end(); ++it) traversers->push_back(*it);
						break;
					}

					case PROPERTY_STEP: {
						#ifdef VERBOSE
						std::cout << "property step\n";
						#endif

						PropertyStep* property_step = dynamic_cast<PropertyStep*>(this->steps[index]);
						std::vector<std::string>& keys = property_step->get_keys();

						std::list<Traverser*>* new_traversers = new std::list<Traverser*>;
						if(property_step->get_type() == PROPERTY) {
							for(auto it = traversers->begin(); it != traversers->end(); ++it) {
								for(std::string key : keys) {
									try {
										Vertex* v = boost::any_cast<Vertex*>((*it)->get());
										VertexProperty<boost::any>* p = v->property(key);
										if(p != nullptr) new_traversers->push_back(new Traverser(p));
									} catch(const std::exception& err) {
										throw std::runtime_error("Error: Traverser does not appear to contain a Vertex.");
									}
								}
							}
						} else {
							for(auto it = traversers->begin(); it != traversers->end(); ++it) {
								for(std::string key : keys) {
									try {
										Vertex* v = boost::any_cast<Vertex*>((*it)->get());
										VertexProperty<boost::any>* p = v->property(key);
										if(p != nullptr) new_traversers->push_back(new Traverser(p->value()));
									} catch(const std::exception& err) {
										throw std::runtime_error("Error: Traverser does not appear to contain a Vertex.");
									}
								}
							}
						}

						delete traversers;
						traversers = new_traversers;

						break;
					}

					case COALESCE_STEP: {
						#ifdef VERBOSE
						std::cout << "coalesce step\n";
						#endif

						std::vector<GraphTraversal*> traversals = dynamic_cast<CoalesceStep*>(this->steps[index])->get_traversals();
						std::list<Traverser*>* new_traversers = new std::list<Traverser*>;
						for(auto it = traversals.begin(); it != traversals.end(); ++it) {
							GraphTraversal* trv = *it;
							CPUGraphTraversal current_traversal = CPUGraphTraversal(this->getTraversalSource(), trv);
							std::list<Traverser*>* temp_traversers = get_all(traversers, &current_traversal);
							for(auto ju = temp_traversers->begin(); ju != temp_traversers->end(); ++ju) new_traversers->push_back(*ju);
							delete temp_traversers;
						}

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

		/*
			Executes the min step.  May be overridden by subclasses (i.e. GPUGraphTraversal)
		*/
		virtual void execute_min_step(std::list<Traverser*>* traversers, MinStep* min_step) {
			Traverser* min = traversers->front();
			for(auto it = traversers->begin(); it != traversers->end(); ++it) {
				min = min_step->min(*it, min);
			}

			traversers->clear();
			traversers->push_back(min);
		}

		/*
			Executes the add property step.
		*/
		void execute_add_property_step(AddPropertyStep* add_property_step, std::list<Traverser*>* traversers);

		/*
			Executes the has step.  May be overridden by subclasses (i.e. GPUGraphTraversal)
		*/
		virtual std::list<Traverser*>* execute_has_step(HasStep* has_step, std::list<Traverser*>* traversers);

		/*
			Executes the index step.
		*/
		std::list<Traverser*>* execute_index_step(IndexStep* index_step, std::list<Traverser*>* traversers);

		/*
			Executes the graph step.
		*/
		std::list<Traverser*>* execute_graph_step(GraphStep* graph_step, std::list<Traverser*>* traversers);

		/*
			Executes the graph step (start).
		*/
		void execute_graph_step_start(GraphStep* graph_step, std::list<Traverser*>* traversers);

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
		inline std::list<Traverser*>* copy_traversers(std::list<Traverser*>* old_traversers) {
			std::list<Traverser*>* new_traversers = new std::list<Traverser*>;
			std::for_each(old_traversers->begin(), old_traversers->end(), [&new_traversers](Traverser* trv){
				new_traversers->push_back(new Traverser(trv->get()));
			});

			return new_traversers;
		}

		inline std::list<Traverser*>* get_all(std::list<Traverser*>* traversers, CPUGraphTraversal* traversal) {
			// Call execute in the given traversal with a copy of the current traversers.
			std::list<Traverser*>* temporary_traversers = this->copy_traversers(traversers);
			temporary_traversers = traversal->execute(temporary_traversers);
			return temporary_traversers;
		}

		/*
			Get the next object from the given traversal.  Makes a copy of the given traversers.
		*/
		inline boost::any get_next(std::list<Traverser*>* traversers, CPUGraphTraversal* traversal) {
			// Call execute in the given traversal with a copy of the current traversers.
			std::list<Traverser*>* temporary_traversers = this->copy_traversers(traversers);
			temporary_traversers = traversal->execute(temporary_traversers);

			// Make sure the traversal evaluated to something.
			if(temporary_traversers->empty()) throw std::runtime_error("Traversal evaluated to empty set!");

			// Correct behavior is grabbing the first traverser and ignoring any others.
			Traverser* out_traverser = temporary_traversers->front();
			boost::any next_object = out_traverser->get();

			// Ensure memory for the temporary traversers is cleared.
			std::for_each(temporary_traversers->begin(), temporary_traversers->end(), [](Traverser* trv) { delete trv; });
			delete temporary_traversers;

			return next_object;
		}

    private:
        std::list<Traverser*>* get_initial_traversers() {
            // Get the initial optimized traversal.
			this->getInitialTraversal();

			// Return and do nothing if there is nothing to execute.
			if(this->steps.empty()) {
				return nullptr;
			}

			std::list<Traverser*>* traversers = new std::list<Traverser*>; // Will use these to traverse

			// Handle the start step (GraphStep(Vertex), GraphStep(Edge), or AddEdgeStartStep)
			switch(this->steps[0]->uid) {
				case IDENTITY_STEP: {
					// do nothing
					break;
				}
				case GRAPH_STEP: {
					#ifdef VERBOSE
					std::cout << "graph step (start)\n";
					#endif

					GraphStep* graph_step = (GraphStep*)this->steps[0];
					execute_graph_step_start(graph_step, traversers);
					break;
				}
				case ADD_VERTEX_START_STEP: {
					#ifdef VERBOSE
					std::cout << "add vertex start step\n";
					#endif

					Vertex* v = this->getGraph()->add_vertex();
					traversers->push_back(new Traverser(v));
					break;
				}
				case ADD_EDGE_START_STEP: {
					#ifdef VERBOSE
					std::cout << "add edge start step\n";
					#endif
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

				case INDEX_STEP: {
					#ifdef VERBOSE
					cout << "index step\n";
					#endif
					IndexStep* index_step = dynamic_cast<IndexStep*>(this->steps[0]);
					std::list<Traverser*>* new_traversers = this->execute_index_step(index_step, traversers);

					delete traversers;
					traversers = new_traversers;
					break;               
				}

				default: {
					throw std::runtime_error("Invalid step!");
					break;
				}
			}

			delete this->steps[0];
			this->steps[0] = new NoOpStep();

			return traversers;
        }

	virtual CPUGraphTraversal* from_anonymous_traversal(GraphTraversal* anonymous) {
		return new CPUGraphTraversal(this->getTraversalSource(), anonymous);
	}
};

#include "CPUGraph.h"

void CPUGraphTraversal::getInitialTraversal() {
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
		else if(graph_step->getType() == VERTEX) {
			if(graph_step->get_element_ids().empty()) {
				if(this->steps[1]->uid == HAS_STEP) {
					HasStep* has_step = (HasStep*)this->steps[1];
					CPUGraph* bg = dynamic_cast<CPUGraph*>(this->getGraph());
					if(bg->is_indexed(has_step->get_key())) {
						IndexStep* index_list_step = new IndexStep(has_step->get_key(), has_step->get_value());
						delete this->steps[0];
						this->steps[0] = index_list_step;
						delete this->steps[1];
						this->steps.erase(this->steps.begin() + 1);
					}
				}
			}
		}
	}
}

void CPUGraphTraversal::execute_add_property_step(AddPropertyStep* add_property_step, std::list<Traverser*>* traversers) {
	if(add_property_step->get_value().type() == typeid(GraphTraversal*)) {
		GraphTraversal* traversal = boost::any_cast<GraphTraversal*>(add_property_step->get_value());
		std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
			BitVertex* v = static_cast<BitVertex*>(boost::any_cast<Vertex*>(trv->get()));
			CPUGraphTraversal* new_trv = from_anonymous_traversal(traversal);
			
			// Execute traversal
			std::list<Traverser*>* temp_traversers = new std::list<Traverser*>;
			temp_traversers->push_back(new Traverser(trv->get()));
			boost::any prop_value = this->get_next(temp_traversers, new_trv);
			delete temp_traversers->front();
			delete temp_traversers;

			// Store the property
			std::string key = add_property_step->get_key();
			
			CPUGraph* graph = dynamic_cast<CPUGraph*>(this->getGraph());
			if(graph->is_indexed(key)) {
				if(v->property(key) != nullptr) graph->clear_index(v, key, v->property(key)->value());
				graph->update_index(v, key, prop_value);
			}

			v->property(key, prop_value);
			//std::cout << "property stored!\n";
		});
	}
	else {
		std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
			Vertex* v = boost::any_cast<Vertex*>(trv->get());

			CPUGraph* graph = dynamic_cast<CPUGraph*>(this->getGraph());
			std::string key = add_property_step->get_key();
			if(graph->is_indexed(key)) {
				if(v->property(key) != nullptr) graph->clear_index(dynamic_cast<BitVertex*>(v), key, v->property(key)->value());
				graph->update_index(dynamic_cast<BitVertex*>(v), key, add_property_step->get_value());
			}

			add_property_step->apply(v);
		});
	}
}

std::list<Traverser*>* CPUGraphTraversal::execute_has_step(HasStep* has_step, std::list<Traverser*>* traversers) {
	std::list<Traverser*>* new_traversers = new std::list<Traverser*>;

	if(dynamic_cast<CPUGraph*>(this->getGraph())->is_indexed(has_step->get_key())) {
		Index* idx = dynamic_cast<CPUGraph*>(this->getGraph())->get_index(has_step->get_key());
		std::unordered_set<Element*> elements = idx->get_elements(has_step->get_value());
		std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
			Element* e = static_cast<Element*>(boost::any_cast<Vertex*>(trv->get()));
			bool advance = elements.count(e) > 0;
			if(advance) new_traversers->push_back(trv); else delete trv;
		});
	}

	else {
		std::for_each(traversers->begin(), traversers->end(), [&, this](Traverser* trv) {
			Vertex* v = boost::any_cast<Vertex*>(trv->get());
			bool advance = has_step->apply(v);
			if(advance) new_traversers->push_back(trv); else delete trv;
		});
	}

	return new_traversers;
}

std::list<Traverser*>* CPUGraphTraversal::execute_index_step(IndexStep* index_step, std::list<Traverser*>* traversers) {
	std::list<Traverser*>* new_traversers = new std::list<Traverser*>;

	Index* idx = dynamic_cast<CPUGraph*>(this->getGraph())->get_index(index_step->get_key());
	std::unordered_set<Element*> elements = idx->get_elements(index_step->get_value());
	std::for_each(elements.begin(), elements.end(), [&, this](Element* e) {
		Traverser* trv = new Traverser((Vertex*)e); // TODO this won't handle edges
		new_traversers->push_back(trv);
	});

	return new_traversers;
}

std::list<Traverser*>* CPUGraphTraversal::execute_graph_step(GraphStep* graph_step, std::list<Traverser*>* traversers) {
	// For each traverser, a traverser should be created for each Vertex and passed to the next step
	CPUGraph* bg = dynamic_cast<CPUGraph*>(this->getGraph());

	// Make map of element ids to counts
	std::vector<boost::any> element_ids = graph_step->get_element_ids();
	std::map<uint64_t, unsigned long> element_id_counts;
	std::for_each(element_ids.begin(), element_ids.end(), [&](boost::any id_ctr){
		uint64_t id = boost::any_cast<uint64_t>(id_ctr);
		if(element_id_counts.find(id) != element_id_counts.end()) element_id_counts[id]++;
		else element_id_counts[id] = 1;
	});

	std::list<Traverser*>* new_traversers = new std::list<Traverser*>;
	// For each traverser...
	std::for_each(traversers->begin(), traversers->end(), [&](Traverser* trv) {
		std::for_each(element_ids.begin(), element_ids.end(), [&](boost::any id_ctr) {
			Vertex* v = bg->get_vertex(id_ctr);
			uint64_t id = boost::any_cast<uint64_t>(id_ctr);
			for(auto k = 0; k < element_id_counts[id]; k++) new_traversers->push_back(new Traverser(v));
			//TODO retain side effects, path
		});
	});

	return new_traversers;
}

void CPUGraphTraversal::execute_graph_step_start(GraphStep* graph_step, std::list<Traverser*>* traversers) {
	if(graph_step->getType() == VERTEX) {
		std::list<Vertex*>& vertices = this->getGraph()->vertices();
		
		if(!graph_step->get_element_ids().empty()) {
			// Create one traverser for each Vertex requested.
			// TODO handle multiplicity
			CPUGraph* bg = dynamic_cast<CPUGraph*>(this->getGraph());
			for(boost::any& id_ctr : graph_step->get_element_ids()) {
				Vertex* v = bg->get_vertex(id_ctr);
				traversers->push_back(new Traverser(v));
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
}

//#undef VERBOSE
#endif
