#include "CPUGraphTraversalSource.h"
#include "CPUGraphTraversal.h"
#include "GraphTraversal.h"
#include "CPUGraph.h"
#include "GraphStep.h"

CPUGraphTraversalSource::CPUGraphTraversalSource(CPUGraph* gr)
: GraphTraversalSource(gr) {
	// do nothing
}

GraphTraversal* CPUGraphTraversalSource::V() {
	GraphTraversal* trv = new CPUGraphTraversal(this);
	trv->appendStep(new GraphStep(VERTEX, {}));
	return trv;
}

/*
	Although the API allows for making this a special call,
	CPUGraph treats this as shorthand for g.V().outE()
*/
GraphTraversal* CPUGraphTraversalSource::E() {
	return this->V()->outE();
}

GraphTraversal* CPUGraphTraversalSource::addV(){

}