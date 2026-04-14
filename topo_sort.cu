#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "graph.h"
#include "graph_gen.h"
#include "seq_topo_sort.h"

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static constexpr int BLOCK_SIZE      = 256;
static constexpr int LOCAL_QUEUE_CAP = 1024;
static constexpr int WARP_SIZE       = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// ── CUDA Kernels ─────────────────────────────────────────────────────────────

// Collect source nodes (in-degree == 0) into the initial frontier and assign
// them their base value.
__global__ void init_frontier_kernel(const int32_t *in_deg,
                                     uint32_t      *frontier,
                                     uint32_t      *frontier_size,
                                     float         *values,
                                     uint32_t       N) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (in_deg[tid] == 0) {
        values[tid] = 1.0f;
        uint32_t pos = atomicAdd(frontier_size, 1u);
        frontier[pos] = tid;
    }
}

// Process every node in the current frontier: propagate its value to
// successors and atomically decrement their in-degrees.  When a successor's
// in-degree reaches zero it is appended to next_frontier.
__global__ void process_frontier_kernel(const uint32_t *frontier,
                                        uint32_t        frontier_size,
                                        const uint32_t *row_offsets,
                                        const uint32_t *col_indices,
                                        int32_t        *in_deg,
                                        float          *values,
                                        uint32_t       *next_frontier,
                                        uint32_t       *next_frontier_size) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    uint32_t u     = frontier[tid];
    float    val_u = values[u];
    uint32_t begin = row_offsets[u];
    uint32_t end   = row_offsets[u + 1];

    for (uint32_t e = begin; e < end; e++) {
        uint32_t v = col_indices[e];
        atomicAdd(&values[v], val_u);
        if (atomicSub(&in_deg[v], 1) == 1) {
            uint32_t pos = atomicAdd(next_frontier_size, 1u);
            next_frontier[pos] = v;
        }
    }
}

// Block-local frontier buffering: each block collects newly-ready nodes in
// shared memory, then flushes with a single bulk atomicAdd on the global
// counter.  Reduces contention on next_frontier_size from O(new_nodes) to
// O(num_blocks).
__global__ void process_frontier_blocal_kernel(
        const uint32_t *frontier,
        uint32_t        frontier_size,
        const uint32_t *row_offsets,
        const uint32_t *col_indices,
        int32_t        *in_deg,
        float          *values,
        uint32_t       *next_frontier,
        uint32_t       *next_frontier_size)
{
    __shared__ uint32_t s_queue[LOCAL_QUEUE_CAP];
    __shared__ uint32_t s_qcount;
    __shared__ uint32_t s_goffset;

    if (threadIdx.x == 0) s_qcount = 0;
    __syncthreads();

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        uint32_t u     = frontier[tid];
        float    val_u = values[u];
        for (uint32_t e = row_offsets[u]; e < row_offsets[u + 1]; e++) {
            uint32_t v = col_indices[e];
            atomicAdd(&values[v], val_u);
            if (atomicSub(&in_deg[v], 1) == 1) {
                uint32_t pos = atomicAdd(&s_qcount, 1u);
                if (pos < LOCAL_QUEUE_CAP)
                    s_queue[pos] = v;
                else {
                    uint32_t gpos = atomicAdd(next_frontier_size, 1u);
                    next_frontier[gpos] = v;
                }
            }
        }
    }

    __syncthreads();
    uint32_t lcount = s_qcount < LOCAL_QUEUE_CAP ? s_qcount : LOCAL_QUEUE_CAP;
    if (threadIdx.x == 0 && lcount > 0)
        s_goffset = atomicAdd(next_frontier_size, lcount);
    __syncthreads();

    for (uint32_t j = threadIdx.x; j < lcount; j += blockDim.x)
        next_frontier[s_goffset + j] = s_queue[j];
}

// Warp-per-node edge-parallel processing: each warp of 32 threads
// cooperatively processes one frontier node's edges.
// Edges within each node are accessed at consecutive addresses across lanes,
// giving coalesced col_indices reads.
__global__ void process_frontier_warp_kernel(
        const uint32_t *frontier,
        uint32_t        frontier_size,
        const uint32_t *row_offsets,
        const uint32_t *col_indices,
        int32_t        *in_deg,
        float          *values,
        uint32_t       *next_frontier,
        uint32_t       *next_frontier_size)
{
    __shared__ uint32_t s_queue[LOCAL_QUEUE_CAP];
    __shared__ uint32_t s_qcount;
    __shared__ uint32_t s_goffset;

    if (threadIdx.x == 0) s_qcount = 0;
    __syncthreads();

    uint32_t warp_id_in_block = threadIdx.x / WARP_SIZE;
    uint32_t lane              = threadIdx.x & (WARP_SIZE - 1);
    uint32_t global_warp       = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    uint32_t total_warps       = gridDim.x * WARPS_PER_BLOCK;

    for (uint32_t idx = global_warp; idx < frontier_size; idx += total_warps) {
        uint32_t u     = frontier[idx];
        float    val_u = values[u];
        uint32_t begin = row_offsets[u];
        uint32_t end   = row_offsets[u + 1];

        for (uint32_t e = begin + lane; e < end; e += WARP_SIZE) {
            uint32_t v = col_indices[e];
            atomicAdd(&values[v], val_u);
            if (atomicSub(&in_deg[v], 1) == 1) {
                uint32_t pos = atomicAdd(&s_qcount, 1u);
                if (pos < LOCAL_QUEUE_CAP)
                    s_queue[pos] = v;
                else {
                    uint32_t gpos = atomicAdd(next_frontier_size, 1u);
                    next_frontier[gpos] = v;
                }
            }
        }
    }

    __syncthreads();
    uint32_t lcount = s_qcount < LOCAL_QUEUE_CAP ? s_qcount : LOCAL_QUEUE_CAP;
    if (threadIdx.x == 0 && lcount > 0)
        s_goffset = atomicAdd(next_frontier_size, lcount);
    __syncthreads();

    for (uint32_t j = threadIdx.x; j < lcount; j += blockDim.x)
        next_frontier[s_goffset + j] = s_queue[j];
}

// ── CUDA Driver Helpers ───────────────────────────────────────────────────────

struct CudaGraphBuffers {
    uint32_t *d_row_offsets, *d_col_indices;
    int32_t  *d_in_deg;
    float    *d_values;
    uint32_t *d_frontier_a, *d_frontier_b;
    uint32_t *d_frontier_size, *d_next_frontier_size;
};

static CudaGraphBuffers alloc_and_upload(const CSRGraph &g,
                                          const vector<int32_t> &h_in_deg) {
    uint32_t N = g.num_vertices;
    uint32_t E = g.num_edges;
    CudaGraphBuffers b;

    CUDA_CHECK(cudaMalloc(&b.d_row_offsets,        (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_col_indices,         E       * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_in_deg,              N       * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_values,              N       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_frontier_a,          N       * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_frontier_b,          N       * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_frontier_size,       sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.d_next_frontier_size,  sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(b.d_row_offsets, g.row_offsets.data(),
                          (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_col_indices, g.col_indices.data(),
                          E * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.d_in_deg, h_in_deg.data(),
                          N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b.d_values, 0, N * sizeof(float)));

    return b;
}

static void free_cuda_buffers(CudaGraphBuffers &b) {
    cudaFree(b.d_row_offsets);  cudaFree(b.d_col_indices);
    cudaFree(b.d_in_deg);       cudaFree(b.d_values);
    cudaFree(b.d_frontier_a);   cudaFree(b.d_frontier_b);
    cudaFree(b.d_frontier_size);cudaFree(b.d_next_frontier_size);
}

// Initialize the frontier on device; returns initial frontier size.
static uint32_t init_frontier(CudaGraphBuffers &b, uint32_t N) {
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(b.d_frontier_size, &zero, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
        b.d_in_deg, b.d_frontier_a, b.d_frontier_size, b.d_values, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_size;
    CUDA_CHECK(cudaMemcpy(&h_size, b.d_frontier_size, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    return h_size;
}

// ── CUDA Sort Drivers ─────────────────────────────────────────────────────────

static double cuda_topo_sort_baseline(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    CudaGraphBuffers b = alloc_and_upload(g, h_in_deg);
    uint32_t cur_size  = init_frontier(b, N);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur  = b.d_frontier_a;
    uint32_t *next = b.d_frontier_b;
    uint32_t  total  = cur_size;
    uint32_t  levels = 0;
    uint32_t  zero   = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(b.d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        int nblocks = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        process_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur, cur_size,
            b.d_row_offsets, b.d_col_indices,
            b.d_in_deg, b.d_values,
            next, b.d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&cur_size, b.d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        swap(cur, next);
        total += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA baseline processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), b.d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    free_cuda_buffers(b);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA baseline: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

static double cuda_topo_sort_blocal(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    CudaGraphBuffers b = alloc_and_upload(g, h_in_deg);
    uint32_t cur_size  = init_frontier(b, N);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur  = b.d_frontier_a;
    uint32_t *next = b.d_frontier_b;
    uint32_t  total  = cur_size;
    uint32_t  levels = 0;
    uint32_t  zero   = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(b.d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        int nblocks = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        process_frontier_blocal_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur, cur_size,
            b.d_row_offsets, b.d_col_indices,
            b.d_in_deg, b.d_values,
            next, b.d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&cur_size, b.d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        swap(cur, next);
        total += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA blocal processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), b.d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    free_cuda_buffers(b);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA block-local: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

static double cuda_topo_sort_warp(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    CudaGraphBuffers b = alloc_and_upload(g, h_in_deg);
    uint32_t cur_size  = init_frontier(b, N);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur  = b.d_frontier_a;
    uint32_t *next = b.d_frontier_b;
    uint32_t  total  = cur_size;
    uint32_t  levels = 0;
    uint32_t  zero   = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(b.d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        int nblocks = (cur_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        process_frontier_warp_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur, cur_size,
            b.d_row_offsets, b.d_col_indices,
            b.d_in_deg, b.d_values,
            next, b.d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&cur_size, b.d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        swap(cur, next);
        total += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA warp processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), b.d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    free_cuda_buffers(b);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA warp-per-node: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

// ── Verification ──────────────────────────────────────────────────────────────

static bool verify(const vector<float> &seq,
                   const vector<float> &cuda_vals,
                   float tol = 1e-3f) {
    uint32_t N = seq.size();
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < N && mismatches < 10; i++) {
        float diff  = fabsf(seq[i] - cuda_vals[i]);
        float denom = fmaxf(fabsf(seq[i]), 1e-8f);
        if (diff / denom > tol) {
            fprintf(stderr, "  Mismatch node %u: seq=%.6f cuda=%.6f (rel=%.6f)\n",
                    i, seq[i], cuda_vals[i], diff / denom);
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
        "  -h                    Show this help\n",
        prog);
}

int main(int argc, char **argv) {
    GraphConfig cfg;
    const char *file_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n")  && i+1 < argc) cfg.num_vertices  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-l")  && i+1 < argc) cfg.num_layers   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w")  && i+1 < argc) cfg.avg_width    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-wv") && i+1 < argc) cfg.width_var    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-d")  && i+1 < argc) cfg.edge_density = atof(argv[++i]);
        else if (!strcmp(argv[i], "-fo") && i+1 < argc) cfg.max_fan_out  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-fi") && i+1 < argc) cfg.max_fan_in   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s")  && i+1 < argc) cfg.skip_prob    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-sk") && i+1 < argc) cfg.max_skip     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")  && i+1 < argc) cfg.seed         = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-f")  && i+1 < argc) file_path        = argv[++i];
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

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

    // Sequential
    vector<float> seq_values;
    double seq_ms = sequential_topo_sort(g, seq_values);

    // CUDA baseline
    vector<float> base_values;
    double base_ms = cuda_topo_sort_baseline(g, base_values);

    // CUDA block-local frontier
    vector<float> bl_values;
    double bl_ms = cuda_topo_sort_blocal(g, bl_values);

    // CUDA warp-per-node
    vector<float> warp_values;
    double warp_ms = cuda_topo_sort_warp(g, warp_values);

    // Report
    printf("\n%-22s %10.3f ms\n", "Sequential", seq_ms);
    printf("%-22s %10.3f ms  (%.2fx vs seq)\n",
           "CUDA baseline", base_ms, seq_ms / base_ms);
    printf("%-22s %10.3f ms  (%.2fx vs seq, %.2fx vs baseline)\n",
           "CUDA block-local", bl_ms, seq_ms / bl_ms, base_ms / bl_ms);
    printf("%-22s %10.3f ms  (%.2fx vs seq, %.2fx vs baseline)\n",
           "CUDA warp-per-node", warp_ms, seq_ms / warp_ms, base_ms / warp_ms);

    bool p1 = verify(seq_values, base_values);
    bool p2 = verify(seq_values, bl_values);
    bool p3 = verify(seq_values, warp_values);
    printf("\nVerification baseline:     %s\n", p1 ? "PASSED" : "FAILED");
    printf("Verification block-local:  %s\n",  p2 ? "PASSED" : "FAILED");
    printf("Verification warp-per-node:%s\n",  p3 ? " PASSED" : " FAILED");

    return 0;
}