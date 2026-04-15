#pragma once
#include <cstdint>
#include <vector>

#include "graph.h"

void generate_graph(const GraphConfig &cfg, std::vector<Edge> &edges);
void permute_graph(uint32_t N, std::vector<Edge> &edges, uint64_t seed);
CSRGraph build_csr(uint32_t N, const std::vector<Edge> &edges);
CSRGraph load_graph(const char *path);
void compute_in_degrees(const CSRGraph &g, std::vector<uint32_t> &in_deg);