#pragma once

#include "gremlinxx/gremlinxx.h"

#include "step/hybrid/GPUGraphStep.cuh"
#include "step/gpu/GPUVertexStep.cuh"
#include "step/gpu/GPUBindStep.cuh"
#include "step/gpu/GPUUnbindStep.cuh"
#include "step/hybrid/GPUPropertyStep.cuh"
#include "step/hybrid/GPUAddPropertyStep.cuh"

void gpugraph_strategy(std::vector<TraversalStep*>& steps);

