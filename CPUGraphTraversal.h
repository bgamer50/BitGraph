#include "GraphTraversal.h"
#include "VertexStep.h"
#include "GraphStep.h"
#include "HasStep.h"

class CPUGraphTraversal : public GraphTraversal {
	public:
		CPUGraphTraversal(GraphTraversalSource* src)
		: GraphTraversal(src) {
			// do nothing
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
		void process() {
			unsigned int index = 0;
			while(index < steps.size()) {
				switch(steps[index]->uid) {
					case GRAPH_STEP:
						getGraph()->vertices();
						break;
					default:
						break;
				}

				index++;
			}
		};
};