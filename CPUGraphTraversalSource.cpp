#include "CPUGraphTraversalSource.h"
#include "CPUGraphTraversal.h"
#include "GraphTraversal.h"
#include "CPUGraph.h"
#include "GraphStep.h"
#include "AddVertexStartStep.h"
#include "AddEdgeStartStep.h"

CPUGraphTraversalSource::CPUGraphTraversalSource(CPUGraph* gr)
: GraphTraversalSource(gr) {
	// do nothing
}

GraphTraversal<Vertex, Vertex>* CPUGraphTraversalSource::V() {
	GraphTraversal<Vertex, Vertex>* trv = new CPUGraphTraversal<Vertex, Vertex>(this);
	trv->appendStep(new GraphStep(VERTEX, {}));
	return trv;
}

/*
	Although the API allows for making this a special call,
	CPUGraph treats this as shorthand for g.V().outE()
*/
GraphTraversal<Edge, Edge>* CPUGraphTraversalSource::E() {
	return (GraphTraversal<Edge, Edge>*)this->V()->outE();
}

GraphTraversal<Vertex, Vertex>* CPUGraphTraversalSource::addV() {
	GraphTraversal<Vertex, Vertex>* trv = new CPUGraphTraversal<Vertex, Vertex>(this);
	trv->appendStep(new AddVertexStartStep());
	return trv;
}

GraphTraversal<Vertex, Vertex>* CPUGraphTraversalSource::addV(std::string label) {
	GraphTraversal<Vertex, Vertex>* trv = new CPUGraphTraversal<Vertex, Vertex>(this);
	trv->appendStep(new AddVertexStartStep(label));
	return trv;
}

GraphTraversal<Edge, Edge>* CPUGraphTraversalSource::addE(std::string label) {
	GraphTraversal<Edge, Edge>* trv = new CPUGraphTraversal<Edge, Edge>(this);
	trv->appendStep(new AddEdgeStartStep(label));
	return trv;
}