#ifndef CPU_GRAPH_TRAVERSAL_SOURCE_H
#define CPU_GRAPH_TRAVERSAL_SOURCE_H

#include "GraphTraversalSource.h"
#include "CPUGraphTraversal.h"
#include "step/graph/GraphStep.h"
#include "step/vertex/AddVertexStartStep.h"
#include "step/edge/AddEdgeStartStep.h"

class CPUGraph;
class GPUGraphTraversalSource;

class CPUGraphTraversalSource : public GraphTraversalSource {
	public:
		CPUGraphTraversalSource(CPUGraph* gr);

		GPUGraphTraversalSource* withGPU();

		GraphTraversal* V() {
			GraphTraversal* trv = get_appropriate_traversal();
			trv->appendStep(new GraphStep(VERTEX, {}));
			return trv;
		}

		GraphTraversal* V(Vertex* v) {
			GraphTraversal* trv = get_appropriate_traversal();
			trv->appendStep(new GraphStep(VERTEX, {v->id()}));
			return trv;
		}

		/*
			Although the API allows for making this a special call,
			CPUGraph treats this as shorthand for g.V().outE()
		*/
		GraphTraversal* E() {
			return (GraphTraversal*)this->V()->outE();
		}
		
		GraphTraversal* addV() {
			GraphTraversal* trv = get_appropriate_traversal();
			trv->appendStep(new AddVertexStartStep());
			return trv;
		}
		
		GraphTraversal* addV(std::string label) {
			GraphTraversal* trv = get_appropriate_traversal();
			trv->appendStep(new AddVertexStartStep(label));
			return trv;
		}

		GraphTraversal* addE(std::string label) {
			GraphTraversal* trv = get_appropriate_traversal();
			trv->appendStep(new AddEdgeStartStep(label));
			return trv;
		}

		virtual GraphTraversal* get_appropriate_traversal();
};

#include "CPUGraph.h"
#include "strategy/TraversalStrategy.h"
#include "strategy/BitGraphStrategy.h"
//#include "GPUGraphTraversalSource.h"

CPUGraphTraversalSource::CPUGraphTraversalSource(CPUGraph* gr)
	: GraphTraversalSource(gr) {
		this->strategies.push_back([this](std::vector<TraversalStep*>& t) {
			bitgraph_strategy(static_cast<CPUGraph*>(this->getGraph()), t);
		});
}

GraphTraversal* CPUGraphTraversalSource::get_appropriate_traversal() {
	return new CPUGraphTraversal(this);
}

/*
GPUGraphTraversalSource* CPUGraphTraversalSource::withGPU() {
	return new GPUGraphTraversalSource(static_cast<CPUGraph*>(this->getGraph()));
}
*/

#endif