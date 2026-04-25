#define CHUNK_SIZE 64

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <omp.h>

#include "graph.h"
#include "graph_gen.h"
#include "seq_topo_sort.h"

using namespace std;

// ── OpenMP Sort Variants ──────────────────────────────────────────────────────

/*
 * Basic parallel variant:
 *   - Each frontier level is processed with a parallel-for over frontier nodes.
 *   - Value accumulation uses OpenMP atomic updates.
 *   - Next-frontier slots are claimed with an atomic counter, identical in
 *     structure to the CUDA baseline (one thread per frontier node).
 */
static double omp_topo_sort_basic(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;
    const auto &row_offsets = g.row_offsets;
    const auto &col_indices = g.col_indices;

    vector<uint32_t> in_deg_u;
    compute_in_degrees(g, in_deg_u);
    vector<int32_t> in_deg(in_deg_u.begin(), in_deg_u.end());

    values.assign(N, 0.0f);
    vector<uint32_t> frontier_a(N), frontier_b(N);
    uint32_t cur_size = 0;

    // Build initial frontier (sources)
    for (uint32_t i = 0; i < N; i++) {
        if (in_deg[i] == 0) {
            values[i] = 1.0f;
            frontier_a[cur_size++] = i;
        }
    }

    uint32_t *cur  = frontier_a.data();
    uint32_t *next = frontier_b.data();
    uint32_t total  = cur_size;
    uint32_t levels = 0;

    auto t0 = chrono::high_resolution_clock::now();

    while (cur_size > 0) {
        uint32_t next_size = 0;

        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) default(none) \
            shared(cur, cur_size, next, next_size, row_offsets, col_indices, \
                   in_deg, values)
        for (uint32_t i = 0; i < cur_size; i++) {
            uint32_t u     = cur[i];
            float    val_u = values[u];
            uint32_t begin = row_offsets[u];
            uint32_t end   = row_offsets[u + 1];

            for (uint32_t e = begin; e < end; e++) {
                uint32_t v = col_indices[e];

                #pragma omp atomic
                values[v] += val_u;

                int32_t new_deg;
                #pragma omp atomic capture
                new_deg = --in_deg[v];

                if (new_deg == 0) {
                    uint32_t pos;
                    #pragma omp atomic capture
                    pos = next_size++;
                    next[pos] = v;
                }
            }
        }

        swap(cur, next);
        cur_size = next_size;
        total   += cur_size;
        levels++;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    if (total != N)
        fprintf(stderr, "WARNING: OMP basic processed %u / %u nodes\n", total, N);
    fprintf(stderr, "OMP basic: %u levels traversed\n", levels);
    return ms;
}

/*
 * Thread-local frontier variant:
 *   - Each thread accumulates newly-ready nodes in a private vector.
 *   - After each frontier level the local buffers are merged into the shared
 *     next-frontier with a single pass, avoiding per-node atomic contention on
 *     the frontier counter (analogous to the CUDA block-local kernel).
 */
static double omp_topo_sort_local(const CSRGraph &g, vector<float> &values, const int nthreads) {
    uint32_t N = g.num_vertices;
    const auto &row_offsets = g.row_offsets;
    const auto &col_indices = g.col_indices;

    vector<uint32_t> in_deg_u;
    compute_in_degrees(g, in_deg_u);
    vector<int32_t> in_deg(in_deg_u.begin(), in_deg_u.end());

    values.assign(N, 0.0f);
    vector<uint32_t> frontier_a(N), frontier_b(N);
    uint32_t cur_size = 0;

    for (uint32_t i = 0; i < N; i++) {
        if (in_deg[i] == 0) {
            values[i] = 1.0f;
            frontier_a[cur_size++] = i;
        }
    }

    uint32_t *cur  = frontier_a.data();
    uint32_t *next = frontier_b.data();
    uint32_t total  = cur_size;
    uint32_t levels = 0;

    vector<vector<uint32_t>> local_q(nthreads);

    auto t0 = chrono::high_resolution_clock::now();

    while (cur_size > 0) {
        #pragma omp parallel default(none) \
            shared(cur, cur_size, row_offsets, col_indices, in_deg, values, local_q)
        {
            int tid = omp_get_thread_num();
            local_q[tid].clear();

            #pragma omp for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < cur_size; i++) {
                uint32_t u     = cur[i];
                float    val_u = values[u];

                for (uint32_t e = row_offsets[u]; e < row_offsets[u + 1]; e++) {
                    uint32_t v = col_indices[e];

                    #pragma omp atomic
                    values[v] += val_u;

                    int32_t new_deg;
                    #pragma omp atomic capture
                    new_deg = --in_deg[v];

                    if (new_deg == 0)
                        local_q[tid].push_back(v);
                }
            }
        } // implicit barrier: all threads done before merge

        // Merge thread-local queues into next frontier
        uint32_t next_size = 0;
        for (int t = 0; t < nthreads; t++) {
            for (uint32_t v : local_q[t])
                next[next_size++] = v;
        }

        swap(cur, next);
        cur_size = next_size;
        total   += cur_size;
        levels++;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    if (total != N)
        fprintf(stderr, "WARNING: OMP local processed %u / %u nodes\n", total, N);
    fprintf(stderr, "OMP thread-local: %u levels traversed\n", levels);
    return ms;
}

// ── Verification ──────────────────────────────────────────────────────────────

static bool verify(const vector<float> &seq,
                   const vector<float> &omp_vals,
                   const char *label,
                   float tol = 1e-3f) {
    uint32_t N = seq.size();
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < N && mismatches < 10; i++) {
        float diff  = fabsf(seq[i] - omp_vals[i]);
        float denom = fmaxf(fabsf(seq[i]), 1e-8f);
        if (diff / denom > tol) {
            fprintf(stderr, "  [%s] Mismatch node %u: seq=%.6f omp=%.6f (rel=%.6f)\n",
                    label, i, seq[i], omp_vals[i], diff / denom);
            mismatches++;
        }
    }
    return mismatches == 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Graph generation (default, in-memory):\n"
        "  -n  <num_vertices>    Number of vertices          (default 1000)\n"
        "  -l  <num_layers>      Number of layers            (0 = auto)\n"
        "  -w  <avg_width>       Avg width of each layer     (0 = auto)\n"
        "  -wv <width_var>       Layer width shape (big=even) (default 10.0)\n"
        "  -d  <edge_density>    Avg out-degree per vertex   (default 5.0)\n"
        "  -fo <max_fan_out>     Max fan-out per vertex      (default 10)\n"
        "  -fi <max_fan_in>      Max fan-in per vertex       (default 10)\n"
        "  -s  <skip_prob>       Per-edge skip probability   (default 0.2)\n"
        "  -sk <max_skip>        Max layers to skip (0=any)  (default 0)\n"
        "  -r  <seed>            RNG seed                    (default 42)\n"
        "Or load from file:\n"
        "  -f  <path>            Load graph from file instead of generating\n"
        "  -t  <num_threads>     OpenMP thread count         (default: all cores)\n"
        "  -h                    Show this help\n",
        prog);
}

int main(int argc, char **argv) {
    GraphConfig cfg;
    const char *file_path  = nullptr;
    int         num_threads = 0; // 0 = let OpenMP decide

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n")  && i+1 < argc) cfg.num_vertices  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-l")  && i+1 < argc) cfg.num_layers    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w")  && i+1 < argc) cfg.avg_width     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-wv") && i+1 < argc) cfg.width_var     = atof(argv[++i]);
        else if (!strcmp(argv[i], "-d")  && i+1 < argc) cfg.edge_density  = atof(argv[++i]);
        else if (!strcmp(argv[i], "-fo") && i+1 < argc) cfg.max_fan_out   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-fi") && i+1 < argc) cfg.max_fan_in    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")  && i+1 < argc) cfg.skip_prob     = atof(argv[++i]);
        else if (!strcmp(argv[i], "-sk") && i+1 < argc) cfg.max_skip      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")  && i+1 < argc) cfg.seed          = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-f")  && i+1 < argc) file_path         = argv[++i];
        else if (!strcmp(argv[i], "-t")  && i+1 < argc) num_threads       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (num_threads > 0)
        omp_set_num_threads(num_threads);
    fprintf(stderr, "OpenMP threads: %d\n", omp_get_max_threads());

    CSRGraph g;
    if (file_path) {
        fprintf(stderr, "Loading graph from %s ...\n", file_path);
        g = load_graph(file_path);
    } else {
        fprintf(stderr, "Generating DAG: %u vertices, seed=%lu ...\n",
                cfg.num_vertices, static_cast<unsigned long>(cfg.seed));
        vector<Edge> edges;
        generate_graph(cfg, edges);
        permute_graph(cfg.num_vertices, edges, cfg.seed);
        fprintf(stderr, "Generated %zu edges, building CSR ...\n", edges.size());
        g = build_csr(cfg.num_vertices, edges);
    }
    fprintf(stderr, "Graph: %u vertices, %u edges (avg degree %.2f)\n",
            g.num_vertices, g.num_edges,
            static_cast<double>(g.num_edges) / g.num_vertices);

    // Sequential baseline
    vector<float> seq_values;
    double seq_ms = sequential_topo_sort(g, seq_values);

    // OMP basic (atomic counter for frontier)
    vector<float> basic_values;
    double basic_ms = omp_topo_sort_basic(g, basic_values);

    // OMP thread-local frontier buffers
    vector<float> local_values;
    double local_ms = omp_topo_sort_local(g, local_values, num_threads);

    // Report
    printf("\n%-25s %10.3f ms\n", "Sequential", seq_ms);
    printf("%-25s %10.3f ms  (%.2fx vs seq)\n",
           "OMP basic", basic_ms, seq_ms / basic_ms);
    printf("%-25s %10.3f ms  (%.2fx vs seq, %.2fx vs basic)\n",
           "OMP thread-local", local_ms, seq_ms / local_ms, basic_ms / local_ms);

    bool p1 = verify(seq_values, basic_values, "basic");
    bool p2 = verify(seq_values, local_values, "local");
    printf("\nVerification OMP basic:           %s\n", p1 ? "PASSED" : "FAILED");
    printf("Verification OMP thread-local:    %s\n",  p2 ? "PASSED" : "FAILED");

    return 0;
}