#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "graph.h"
#include "graph_gen.h"

/*
 * Output format:
 *   <num_vertices> <num_edges>
 *   <src_0> <dst_0>
 *   ...
 */
static void write_graph(const char *path, uint32_t N,
                         const std::vector<Edge> &edges) {
    std::filesystem::path p(path);
    if (p.has_parent_path())
        std::filesystem::create_directories(p.parent_path());

    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); exit(1); }

    fprintf(fp, "%u %zu\n", N, edges.size());
    for (const auto &e : edges)
        fprintf(fp, "%u %u\n", e.src, e.dst);

    fclose(fp);
}

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

static GraphConfig parse_args(int argc, char **argv, const char **out_path) {
    GraphConfig cfg;
    *out_path = "graphs/graph.txt";
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
        else if (!strcmp(argv[i], "-o")  && i+1 < argc) *out_path         = argv[++i];
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); exit(1); }
    }
    return cfg;
}

int main(int argc, char **argv) {
    const char *out_path;
    GraphConfig cfg = parse_args(argc, argv, &out_path);

    fprintf(stderr, "Generating DAG: %u vertices, seed=%lu\n",
                     cfg.num_vertices, cfg.seed);

    std::vector<Edge> edges;
    generate_graph(cfg, edges);

    fprintf(stderr, "Generated %zu edges (avg degree %.2f)\n",
                     edges.size(),
                     static_cast<double>(edges.size()) / cfg.num_vertices);

    write_graph(out_path, cfg.num_vertices, edges);

    fprintf(stderr, "Written to %s\n", out_path);
    return 0;
}