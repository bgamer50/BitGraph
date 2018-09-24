#include <stdio.h>
#include "CPUGraph.h"
#include "P.h"
#include <string>

int main(int argc, char* argv[]) {
	CPUGraph graph;
	std::string explanation = graph.traversal()->addV()->has("a", "b")->explain();
	printf("%s\n", explanation.c_str());

	graph.traversal()->addV()->addV()->addV()->V()->iterate();
}