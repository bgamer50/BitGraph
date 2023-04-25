#pragma once

#include "gremlinxx/gremlinxx.h"

class CPUGraph;

void bitgraph_strategy(CPUGraph* bg, std::vector<TraversalStep*>& steps);