#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "graph.h"
#include "graph_gen.h"

/*
 * Layered DAG generation:
 *   - Vertices are partitioned into layers with randomized widths.
 *   - Edges only go from earlier to later layers, guaranteeing acyclicity.
 *   - Each edge has skip_prob chance of targeting a further layer instead of
 *     the immediately next one, up to max_skip layers ahead.
 *   - Fan-in is capped so no single vertex has too many predecessors.
 */
void generate_graph(const GraphConfig &cfg, std::vector<Edge> &edges) {
    const uint32_t N = cfg.num_vertices;

    uint32_t num_layers = cfg.num_layers;
    uint32_t avg_width  = cfg.avg_width;
    if (num_layers == 0 && avg_width == 0) {
        avg_width = static_cast<uint32_t>(std::sqrt(static_cast<double>(N)));
        if (avg_width < 1) avg_width = 1;
        num_layers = (N + avg_width - 1) / avg_width;
    } else if (num_layers == 0) {
        num_layers = (N + avg_width - 1) / avg_width;
    } else if (avg_width == 0) {
        avg_width = (N + num_layers - 1) / num_layers;
    }

    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Dirichlet-distributed layer sizes via gamma sampling.
    // High shape (~100) = nearly equal layers, low shape (~0.5) = very uneven.
    double shape = std::max(0.01, cfg.width_var);
    std::gamma_distribution<double> gamma_dist(shape, 1.0);
    std::vector<double> weights(num_layers);
    double total_weight = 0.0;
    for (uint32_t l = 0; l < num_layers; l++) {
        weights[l] = gamma_dist(rng);
        total_weight += weights[l];
    }
    std::vector<uint32_t> layer_size(num_layers);
    uint32_t assigned = 0;
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t sz = std::max(1u, static_cast<uint32_t>(weights[l] / total_weight * N));
        layer_size[l] = sz;
        assigned += sz;
    }
    // Redistribute rounding error one vertex at a time
    while (assigned < N) { layer_size[rng() % num_layers]++; assigned++; }
    while (assigned > N) {
        uint32_t l = rng() % num_layers;
        if (layer_size[l] > 1) { layer_size[l]--; assigned--; }
    }

    std::vector<uint32_t> layer_start(num_layers + 1);
    layer_start[0] = 0;
    for (uint32_t l = 0; l < num_layers; l++)
        layer_start[l + 1] = layer_start[l] + layer_size[l];

    std::vector<uint32_t> in_deg(N, 0);
    edges.reserve(static_cast<size_t>(N * cfg.edge_density * 1.2));

    uint32_t max_skip = cfg.max_skip;

    for (uint32_t l = 0; l + 1 < num_layers; l++) {
        uint32_t src_begin = layer_start[l];
        uint32_t src_end   = layer_start[l + 1];

        for (uint32_t u = src_begin; u < src_end; u++) {
            std::poisson_distribution<uint32_t> pois(cfg.edge_density);
            uint32_t out_degree = std::max(1u, pois(rng));
            if (out_degree > cfg.max_fan_out) out_degree = cfg.max_fan_out;

            for (uint32_t e = 0; e < out_degree; e++) {
                uint32_t target_layer = l + 1;
                if (unif(rng) < cfg.skip_prob && l + 2 < num_layers) {
                    uint32_t max_target = (max_skip > 0)
                        ? std::min(l + 1 + max_skip, num_layers - 1)
                        : num_layers - 1;
                    if (max_target >= l + 2)
                        target_layer = l + 2 + (rng() % (max_target - l - 1));
                }

                uint32_t dst_begin = layer_start[target_layer];
                uint32_t dst_end   = layer_start[target_layer + 1];
                uint32_t dst_count = dst_end - dst_begin;
                if (dst_count == 0) continue;

                uint32_t v = dst_begin + (rng() % dst_count);
                if (in_deg[v] < cfg.max_fan_in) {
                    edges.push_back({u, v});
                    in_deg[v]++;
                }
            }
        }
    }

    // Ensure every non-source node has at least one predecessor
    for (uint32_t l = 1; l < num_layers; l++) {
        uint32_t dst_begin  = layer_start[l];
        uint32_t dst_end    = layer_start[l + 1];
        uint32_t prev_begin = layer_start[l - 1];
        uint32_t prev_end   = layer_start[l];
        uint32_t prev_count = prev_end - prev_begin;
        if (prev_count == 0) continue;

        for (uint32_t v = dst_begin; v < dst_end; v++) {
            if (in_deg[v] == 0) {
                uint32_t u = prev_begin + (rng() % prev_count);
                edges.push_back({u, v});
                in_deg[v]++;
            }
        }
    }
}

void permute_graph(uint32_t N, std::vector<Edge> &edges, uint64_t seed) {
    std::vector<uint32_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::mt19937_64 rng(seed ^ 0xDEADBEEFull);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (auto &e : edges) {
        e.src = perm[e.src];
        e.dst = perm[e.dst];
    }
}

CSRGraph build_csr(uint32_t N, const std::vector<Edge> &edges) {
    uint32_t E = static_cast<uint32_t>(edges.size());

    std::vector<uint32_t> degree(N, 0);
    for (uint32_t i = 0; i < E; i++) degree[edges[i].src]++;

    CSRGraph g;
    g.num_vertices = N;
    g.num_edges    = E;
    g.row_offsets.resize(N + 1);
    g.col_indices.resize(E);

    g.row_offsets[0] = 0;
    for (uint32_t i = 0; i < N; i++)
        g.row_offsets[i + 1] = g.row_offsets[i] + degree[i];

    std::vector<uint32_t> offset(N, 0);
    for (uint32_t i = 0; i < E; i++) {
        uint32_t u = edges[i].src;
        g.col_indices[g.row_offsets[u] + offset[u]] = edges[i].dst;
        offset[u]++;
    }

    return g;
}

/*
 * File format:
 *   <num_vertices> <num_edges>
 *   <src_0> <dst_0>
 *   ...
 */
CSRGraph load_graph(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); exit(1); }

    uint32_t N, E;
    if (fscanf(fp, "%u %u", &N, &E) != 2) {
        fprintf(stderr, "Bad header in %s\n", path);
        exit(1);
    }

    std::vector<uint32_t> src(E), dst(E);
    for (uint32_t i = 0; i < E; i++) {
        if (fscanf(fp, "%u %u", &src[i], &dst[i]) != 2) {
            fprintf(stderr, "Bad edge %u in %s\n", i, path);
            exit(1);
        }
    }
    fclose(fp);

    std::vector<uint32_t> degree(N, 0);
    for (uint32_t i = 0; i < E; i++) degree[src[i]]++;

    CSRGraph g;
    g.num_vertices = N;
    g.num_edges    = E;
    g.row_offsets.resize(N + 1);
    g.col_indices.resize(E);

    g.row_offsets[0] = 0;
    for (uint32_t i = 0; i < N; i++)
        g.row_offsets[i + 1] = g.row_offsets[i] + degree[i];

    std::vector<uint32_t> offset(N, 0);
    for (uint32_t i = 0; i < E; i++) {
        uint32_t u = src[i];
        g.col_indices[g.row_offsets[u] + offset[u]] = dst[i];
        offset[u]++;
    }

    return g;
}

void compute_in_degrees(const CSRGraph &g, std::vector<uint32_t> &in_deg) {
    in_deg.assign(g.num_vertices, 0);
    for (uint32_t i = 0; i < g.num_edges; i++)
        in_deg[g.col_indices[i]]++;
}