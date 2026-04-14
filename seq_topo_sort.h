#pragma once
#include <cstdint>
#include <vector>

#include "graph.h"

// BFS-based topological sort with value propagation.
// Source nodes start with value 1.0; each node accumulates the sum of its
// predecessors' values.  Returns elapsed time in milliseconds.
double sequential_topo_sort(const CSRGraph &g, std::vector<float> &values);