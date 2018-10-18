#include "GraphTraversal.h"
#include "VertexStep.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddVertexStep.h"
#include "HasStep.h"
#include "Traverser.h"
#include "NoOpStep.h"
#include <algorithm>

class CPUGraphTraversal : public GraphTraversal {
	public:
		CPUGraphTraversal(GraphTraversalSource* src)
		: GraphTraversal(src) {
			// do nothing
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
					steps[0] = new GraphStep(VERTEX);
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
			Performs each step of the traversal.
		**/
		virtual void iterate() {
			// Get the initial optimized traversal using the parent's method.
			this->getInitialTraversal();

			// Return and do nothing if there is nothing to execute.
			if(steps.empty()) {
				return;
			}

			std::vector<Traverser<void*>*> traversers; //TODO actually use these to traverse

			// Handle the start step (GraphStep(Vertex), GraphStep(Edge), or AddEdgeStartStep)
			switch(steps[0]->type) {
				case GRAPH_STEP: {
					GraphStep* graph_step = (GraphStep*)steps[0];
					if(graph_step->getType() == VERTEX) {
						// Create one traverser for each Vertex in the Graph.
						std::vector<Vertex*> vertices = getGraph()->vertices();
						for_each(vertices.begin(), vertices.end(), [](Vertex* v){ traversers.push_back((void*)v); });
					}
					else if(graph_step->getType() == EDGE) {
						// Create one traverser for each Edge in the Graph.

					}
					break;
				}
				case ADD_EDGE_START_STEP: {
					break;
				}
				default: {
					// TODO throw exception
				}
			}

			unsigned int index = 1;
			while(index < steps.size()) {
				switch(steps[index]->uid) {
					case GRAPH_STEP:
						{
							std::vector<Vertex*> vertices = getGraph()->vertices();
							for_each(vertices.begin(), vertices.end(), [](Vertex* v){printf("%llu\n", *((uint64_t*)v->id()));});
							// For each traverser, a traverser should be created for each Vertex and passed to the next step
							break;
						}
					case ADD_VERTEX_STEP:
						{
							// For each traverser ...
							((CPUGraph*)getGraph())->add_vertex();
							// For each traverser, a new Vertex should be created and replace the original traverser
							break;
						}
					case ADD_VERTEX_START_STEP:
						{
							((CPUGraph*)getGraph())->add_vertex();
						}
					case ADD_EDGE_STEP:
						{
														
							// Need to check if there is enough info to add the Edge, then add it
							// if we can.


							// We must throw an exception if there is an issue with the addEdge step.
							
						}
				}

				index++;
			}
		};
};