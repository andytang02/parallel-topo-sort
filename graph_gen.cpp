#include <bits/stdc++.h>
#include <filesystem>
using namespace std;

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
    double   skip_prob      = 0.2;   // per-edge probability of targeting a non-adjacent layer
    uint32_t max_skip       = 0;      // max layers to skip (0 = unlimited)
    uint64_t seed           = 42;
    const char *out_path    = "graphs/graph.txt";
};

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
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
        "  -o  <output_file>     Output file path            (default graphs/graph.txt)\n"
        "  -h                    Show this help\n",
        prog);
}

static GraphConfig parse_args(int argc, char **argv) {
    GraphConfig cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n")  && i+1 < argc) cfg.num_vertices = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-l")  && i+1 < argc) cfg.num_layers  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w")  && i+1 < argc) cfg.avg_width   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-wv") && i+1 < argc) cfg.width_var   = atof(argv[++i]);
        else if (!strcmp(argv[i], "-d")  && i+1 < argc) cfg.edge_density = atof(argv[++i]);
        else if (!strcmp(argv[i], "-fo") && i+1 < argc) cfg.max_fan_out  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-fi") && i+1 < argc) cfg.max_fan_in   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")  && i+1 < argc) cfg.skip_prob    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-sk") && i+1 < argc) cfg.max_skip     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")  && i+1 < argc) cfg.seed         = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-o")  && i+1 < argc) cfg.out_path     = argv[++i];
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); exit(1); }
    }
    return cfg;
}

/*
 * Layered DAG generation:
 *   - Vertices are partitioned into layers with randomized widths.
 *   - Edges only go from earlier to later layers, guaranteeing acyclicity.
 *   - Each edge has skip_prob chance of targeting a further layer instead of
 *     the immediately next one, up to max_skip layers ahead.
 *   - Fan-in is capped so no single vertex has too many predecessors.
 */
static void generate_graph(const GraphConfig &cfg,
                           vector<Edge> &edges) {
    const uint32_t N = cfg.num_vertices;

    uint32_t num_layers = cfg.num_layers;
    uint32_t avg_width = cfg.avg_width;
    if (num_layers == 0 && avg_width == 0) {
        avg_width = static_cast<uint32_t>(sqrt(static_cast<double>(N)));
        if (avg_width < 1) avg_width = 1;
        num_layers = (N + avg_width - 1) / avg_width;
    } else if (num_layers == 0) {
        num_layers = (N + avg_width - 1) / avg_width;
    } else if (avg_width == 0) {
        avg_width = (N + num_layers - 1) / num_layers;
    }

    mt19937_64 rng(cfg.seed);
    uniform_real_distribution<double> unif(0.0, 1.0);

    // Dirichlet-distributed layer sizes: shape controls uniformity
    // High shape (~100) = nearly equal layers, low shape (~0.5) = very uneven
    double shape = max(0.01, cfg.width_var);
    gamma_distribution<double> gamma(shape, 1.0);
    vector<double> weights(num_layers);
    double total_weight = 0.0;
    for (uint32_t l = 0; l < num_layers; l++) {
        weights[l] = gamma(rng);
        total_weight += weights[l];
    }
    vector<uint32_t> layer_size(num_layers);
    uint32_t assigned = 0;
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t sz = max(1u, static_cast<uint32_t>(weights[l] / total_weight * N));
        layer_size[l] = sz;
        assigned += sz;
    }
    // Redistribute rounding error one vertex at a time
    while (assigned < N) { layer_size[rng() % num_layers]++; assigned++; }
    while (assigned > N) {
        uint32_t l = rng() % num_layers;
        if (layer_size[l] > 1) { layer_size[l]--; assigned--; }
    }

    // Build layer boundaries from sizes
    vector<uint32_t> layer_start(num_layers + 1);
    layer_start[0] = 0;
    for (uint32_t l = 0; l < num_layers; l++)
        layer_start[l + 1] = layer_start[l] + layer_size[l];

    vector<uint32_t> in_deg(N, 0);
    edges.reserve(static_cast<size_t>(N * cfg.edge_density * 1.2));

    uint32_t max_skip = cfg.max_skip;

    for (uint32_t l = 0; l + 1 < num_layers; l++) {
        uint32_t src_begin = layer_start[l];
        uint32_t src_end   = layer_start[l + 1];

        for (uint32_t u = src_begin; u < src_end; u++) {
            poisson_distribution<uint32_t> pois(cfg.edge_density);
            uint32_t out_degree = max(1u, pois(rng));
            if (out_degree > cfg.max_fan_out) out_degree = cfg.max_fan_out;

            for (uint32_t e = 0; e < out_degree; e++) {
                // Decide target layer: next layer, or skip ahead
                uint32_t target_layer = l + 1;
                if (unif(rng) < cfg.skip_prob && l + 2 < num_layers) {
                    uint32_t max_target = (max_skip > 0)
                        ? min(l + 1 + max_skip, num_layers - 1)
                        : num_layers - 1;
                    // uniform over [l+2, max_target]
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
        uint32_t dst_begin = layer_start[l];
        uint32_t dst_end   = layer_start[l + 1];
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

/*
 * Output format:
 *   <num_vertices> <num_edges>
 *   <src_0> <dst_0>
 *   <src_1> <dst_1>
 *   ...
 */
static void write_graph(const char *path, uint32_t N,
                         const vector<Edge> &edges) {
    filesystem::path p(path);
    if (p.has_parent_path())
        filesystem::create_directories(p.parent_path());

    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); exit(1); }

    fprintf(fp, "%u %zu\n", N, edges.size());
    for (const auto &e : edges)
        fprintf(fp, "%u %u\n", e.src, e.dst);

    fclose(fp);
}

int main(int argc, char **argv) {
    GraphConfig cfg = parse_args(argc, argv);

    fprintf(stderr, "Generating DAG: %u vertices, seed=%lu\n",
                     cfg.num_vertices, cfg.seed);

    vector<Edge> edges;
    generate_graph(cfg, edges);

    fprintf(stderr, "Generated %zu edges (avg degree %.2f)\n",
                     edges.size(),
                     static_cast<double>(edges.size()) / cfg.num_vertices);

    write_graph(cfg.out_path, cfg.num_vertices, edges);

    fprintf(stderr, "Written to %s\n", cfg.out_path);
    return 0;
}
