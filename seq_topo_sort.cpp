#include <chrono>
#include <cstdio>
#include <queue>
#include <vector>

#include "graph.h"
#include "graph_gen.h"
#include "seq_topo_sort.h"

double sequential_topo_sort(const CSRGraph &g, std::vector<float> &values) {
    uint32_t N = g.num_vertices;
    std::vector<uint32_t> in_deg;
    compute_in_degrees(g, in_deg);

    values.assign(N, 0.0f);
    std::queue<uint32_t> q;
    for (uint32_t i = 0; i < N; i++) {
        if (in_deg[i] == 0) {
            values[i] = 1.0f;
            q.push(i);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    uint32_t processed = 0;
    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        processed++;
        for (uint32_t e = g.row_offsets[u]; e < g.row_offsets[u + 1]; e++) {
            uint32_t v = g.col_indices[e];
            values[v] += values[u];
            if (--in_deg[v] == 0)
                q.push(v);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (processed != N)
        fprintf(stderr, "WARNING: sequential processed %u / %u nodes (cycle?)\n",
                processed, N);
    return ms;
}