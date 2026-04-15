#pragma once
#include <cstdint>
#include <vector>

struct Edge {
    uint32_t src, dst;
};

struct GraphConfig {
    uint32_t num_vertices   = 1000;
    uint32_t num_layers     = 0;      // 0 = auto
    uint32_t avg_width      = 0;      // 0 = auto
    double   width_var      = 10.0;   // Dirichlet shape: large = uniform, small = uneven
    double   edge_density   = 5.0;    // avg out-degree per vertex
    uint32_t max_fan_out    = 10;
    uint32_t max_fan_in     = 10;
    double   skip_prob      = 0.2;    // per-edge probability of targeting a non-adjacent layer
    uint32_t max_skip       = 0;      // max layers to skip (0 = unlimited)
    uint64_t seed           = 42;
};

struct CSRGraph {
    uint32_t              num_vertices;
    uint32_t              num_edges;
    std::vector<uint32_t> row_offsets; // size N+1
    std::vector<uint32_t> col_indices; // size E
};
