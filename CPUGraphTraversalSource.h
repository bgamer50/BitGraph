#ifndef CPU_GRAPH_TRAVERSAL_SOURCE_H
#define CPU_GRAPH_TRAVERSAL_SOURCE_H

#include "GraphTraversalSource.h"
#include "CPUGraphTraversal.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddEdgeStartStep.h"
class CPUGraph;

class CPUGraphTraversalSource : public GraphTraversalSource {
	public:
		CPUGraphTraversalSource(CPUGraph* gr)
		: GraphTraversalSource(gr) {
			// do nothing
		}

		GraphTraversal* V() {
			GraphTraversal* trv = new CPUGraphTraversal(this);
			trv->appendStep(new GraphStep(VERTEX, {}));
			return trv;
		}

		GraphTraversal* V(Vertex* v) {
			GraphTraversal* trv = new CPUGraphTraversal(this);
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
			GraphTraversal* trv = new CPUGraphTraversal(this);
			trv->appendStep(new AddVertexStartStep());
			return trv;
		}
		
		GraphTraversal* addV(std::string label) {
			GraphTraversal* trv = new CPUGraphTraversal(this);
			trv->appendStep(new AddVertexStartStep(label));
			return trv;
		}

		GraphTraversal* addE(std::string label) {
			GraphTraversal* trv = new CPUGraphTraversal(this);
			trv->appendStep(new AddEdgeStartStep(label));
			return trv;
		}
};

#endif