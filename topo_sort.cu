#include <bits/stdc++.h>
#include <cuda_runtime.h>

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

static constexpr int BLOCK_SIZE = 256;
static constexpr int LOCAL_QUEUE_CAP = 1024;

struct CSRGraph {
    uint32_t num_vertices;
    uint32_t num_edges;
    vector<uint32_t> row_offsets;
    vector<uint32_t> col_indices;
};

static CSRGraph load_graph(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); exit(1); }

    uint32_t N, E;
    if (fscanf(fp, "%u %u", &N, &E) != 2) {
        fprintf(stderr, "Bad header in %s\n", path);
        exit(1);
    }

    vector<uint32_t> src(E), dst(E);
    for (uint32_t i = 0; i < E; i++) {
        if (fscanf(fp, "%u %u", &src[i], &dst[i]) != 2) {
            fprintf(stderr, "Bad edge %u in %s\n", i, path);
            exit(1);
        }
    }
    fclose(fp);

    vector<uint32_t> degree(N, 0);
    for (uint32_t i = 0; i < E; i++) degree[src[i]]++;

    CSRGraph g;
    g.num_vertices = N;
    g.num_edges    = E;
    g.row_offsets.resize(N + 1);
    g.col_indices.resize(E);

    g.row_offsets[0] = 0;
    for (uint32_t i = 0; i < N; i++)
        g.row_offsets[i + 1] = g.row_offsets[i] + degree[i];

    vector<uint32_t> offset(N, 0);
    for (uint32_t i = 0; i < E; i++) {
        uint32_t u = src[i];
        g.col_indices[g.row_offsets[u] + offset[u]] = dst[i];
        offset[u]++;
    }

    return g;
}

static void compute_in_degrees(const CSRGraph &g, vector<uint32_t> &in_deg) {
    in_deg.assign(g.num_vertices, 0);
    for (uint32_t i = 0; i < g.num_edges; i++)
        in_deg[g.col_indices[i]]++;
}

// Graph Generation
struct Edge { uint32_t src, dst; };

struct GraphConfig {
    uint32_t num_vertices   = 1000;
    uint32_t num_layers     = 0;
    uint32_t avg_width      = 0;
    double   width_var      = 10.0;
    double   edge_density   = 5.0;
    uint32_t max_fan_out    = 10;
    uint32_t max_fan_in     = 10;
    double   skip_prob      = 0.2;
    uint32_t max_skip       = 0;
    uint64_t seed           = 42;
};

static void generate_graph(const GraphConfig &cfg, vector<Edge> &edges) {
    const uint32_t N = cfg.num_vertices;

    uint32_t num_layers = cfg.num_layers;
    uint32_t avg_width  = cfg.avg_width;
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

    double shape = max(0.01, cfg.width_var);
    gamma_distribution<double> gamma_dist(shape, 1.0);
    vector<double> weights(num_layers);
    double total_weight = 0.0;
    for (uint32_t l = 0; l < num_layers; l++) {
        weights[l] = gamma_dist(rng);
        total_weight += weights[l];
    }
    vector<uint32_t> layer_size(num_layers);
    uint32_t assigned = 0;
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t sz = max(1u, static_cast<uint32_t>(weights[l] / total_weight * N));
        layer_size[l] = sz;
        assigned += sz;
    }
    while (assigned < N) { layer_size[rng() % num_layers]++; assigned++; }
    while (assigned > N) {
        uint32_t l = rng() % num_layers;
        if (layer_size[l] > 1) { layer_size[l]--; assigned--; }
    }

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
                uint32_t target_layer = l + 1;
                if (unif(rng) < cfg.skip_prob && l + 2 < num_layers) {
                    uint32_t max_target = (max_skip > 0)
                        ? min(l + 1 + max_skip, num_layers - 1)
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

static void permute_graph(uint32_t N, vector<Edge> &edges, uint64_t seed) {
    vector<uint32_t> perm(N);
    iota(perm.begin(), perm.end(), 0u);
    mt19937_64 rng(seed ^ 0xDEADBEEFull);
    shuffle(perm.begin(), perm.end(), rng);
    for (auto &e : edges) {
        e.src = perm[e.src];
        e.dst = perm[e.dst];
    }
}

static CSRGraph build_csr(uint32_t N, const vector<Edge> &edges) {
    uint32_t E = static_cast<uint32_t>(edges.size());

    vector<uint32_t> degree(N, 0);
    for (uint32_t i = 0; i < E; i++) degree[edges[i].src]++;

    CSRGraph g;
    g.num_vertices = N;
    g.num_edges    = E;
    g.row_offsets.resize(N + 1);
    g.col_indices.resize(E);

    g.row_offsets[0] = 0;
    for (uint32_t i = 0; i < N; i++)
        g.row_offsets[i + 1] = g.row_offsets[i] + degree[i];

    vector<uint32_t> offset(N, 0);
    for (uint32_t i = 0; i < E; i++) {
        uint32_t u = edges[i].src;
        g.col_indices[g.row_offsets[u] + offset[u]] = edges[i].dst;
        offset[u]++;
    }

    return g;
}

// Sequential Baseline
static double sequential_topo_sort(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;
    vector<uint32_t> in_deg;
    compute_in_degrees(g, in_deg);

    values.assign(N, 0.0f);
    queue<uint32_t> q;
    for (uint32_t i = 0; i < N; i++) {
        if (in_deg[i] == 0) {
            values[i] = 1.0f;
            q.push(i);
        }
    }

    auto t0 = chrono::high_resolution_clock::now();

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

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();

    if (processed != N)
        fprintf(stderr, "WARNING: sequential processed %u / %u nodes (cycle?)\n",
                processed, N);
    return ms;
}

// CUDA Kernels

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

static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

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
    uint32_t lane = threadIdx.x & (WARP_SIZE - 1);
    uint32_t global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    uint32_t total_warps = gridDim.x * WARPS_PER_BLOCK;

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

// baseline driver
static double cuda_topo_sort_baseline(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;
    uint32_t E = g.num_edges;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    // Device allocations
    uint32_t *d_row_offsets, *d_col_indices;
    int32_t  *d_in_deg;
    float    *d_values;
    uint32_t *d_frontier_a, *d_frontier_b;
    uint32_t *d_frontier_size, *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_in_deg, N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frontier_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

    // Copy graph to device
    CUDA_CHECK(cudaMemcpy(d_row_offsets, g.row_offsets.data(),
                          (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, g.col_indices.data(),
                          E * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_deg, h_in_deg.data(),
                          N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_values, 0, N * sizeof(float)));

    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Build initial frontier
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
        d_in_deg, d_frontier_a, d_frontier_size, d_values, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_frontier_size;
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Timed BFS loop
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur_frontier  = d_frontier_a;
    uint32_t *next_frontier = d_frontier_b;
    uint32_t  cur_size      = h_frontier_size;
    uint32_t  total         = cur_size;
    uint32_t  levels        = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        nblocks = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        process_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur_frontier, cur_size,
            d_row_offsets, d_col_indices,
            d_in_deg, d_values,
            next_frontier, d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));

        swap(cur_frontier, next_frontier);
        cur_size = h_frontier_size;
        total   += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_in_deg);
    cudaFree(d_values);
    cudaFree(d_frontier_a);
    cudaFree(d_frontier_b);
    cudaFree(d_frontier_size);
    cudaFree(d_next_frontier_size);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA baseline: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

// Block-local driver
static double cuda_topo_sort_blocal(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;
    uint32_t E = g.num_edges;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    uint32_t *d_row_offsets, *d_col_indices;
    int32_t  *d_in_deg;
    float    *d_values;
    uint32_t *d_frontier_a, *d_frontier_b;
    uint32_t *d_frontier_size, *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_in_deg, N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frontier_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_row_offsets, g.row_offsets.data(),
                          (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, g.col_indices.data(),
                          E * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_deg, h_in_deg.data(),
                          N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_values, 0, N * sizeof(float)));

    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
        d_in_deg, d_frontier_a, d_frontier_size, d_values, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_frontier_size;
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur_frontier  = d_frontier_a;
    uint32_t *next_frontier = d_frontier_b;
    uint32_t  cur_size      = h_frontier_size;
    uint32_t  total         = cur_size;
    uint32_t  levels        = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        nblocks = (cur_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        process_frontier_blocal_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur_frontier, cur_size,
            d_row_offsets, d_col_indices,
            d_in_deg, d_values,
            next_frontier, d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));

        swap(cur_frontier, next_frontier);
        cur_size = h_frontier_size;
        total   += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA blocal processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_row_offsets);  cudaFree(d_col_indices);
    cudaFree(d_in_deg);       cudaFree(d_values);
    cudaFree(d_frontier_a);   cudaFree(d_frontier_b);
    cudaFree(d_frontier_size);cudaFree(d_next_frontier_size);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA block-local: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

// Warp-per-node driver
static double cuda_topo_sort_warp(const CSRGraph &g, vector<float> &values) {
    uint32_t N = g.num_vertices;
    uint32_t E = g.num_edges;

    vector<uint32_t> h_in_deg_u;
    compute_in_degrees(g, h_in_deg_u);
    vector<int32_t> h_in_deg(h_in_deg_u.begin(), h_in_deg_u.end());

    uint32_t *d_row_offsets, *d_col_indices;
    int32_t  *d_in_deg;
    float    *d_values;
    uint32_t *d_frontier_a, *d_frontier_b;
    uint32_t *d_frontier_size, *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_in_deg, N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_values, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frontier_a, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_b, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_row_offsets, g.row_offsets.data(),
                          (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, g.col_indices.data(),
                          E * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_deg, h_in_deg.data(),
                          N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_values, 0, N * sizeof(float)));

    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &zero, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_frontier_kernel<<<nblocks, BLOCK_SIZE>>>(
        d_in_deg, d_frontier_a, d_frontier_size, d_values, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_frontier_size;
    CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_frontier_size, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    uint32_t *cur_frontier  = d_frontier_a;
    uint32_t *next_frontier = d_frontier_b;
    uint32_t  cur_size      = h_frontier_size;
    uint32_t  total         = cur_size;
    uint32_t  levels        = 0;

    while (cur_size > 0) {
        CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &zero, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        nblocks = (cur_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        process_frontier_warp_kernel<<<nblocks, BLOCK_SIZE>>>(
            cur_frontier, cur_size,
            d_row_offsets, d_col_indices,
            d_in_deg, d_values,
            next_frontier, d_next_frontier_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_frontier_size, d_next_frontier_size,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));

        swap(cur_frontier, next_frontier);
        cur_size = h_frontier_size;
        total   += cur_size;
        levels++;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    if (total != N)
        fprintf(stderr, "WARNING: CUDA warp processed %u / %u nodes\n", total, N);

    values.resize(N);
    CUDA_CHECK(cudaMemcpy(values.data(), d_values, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_row_offsets);  cudaFree(d_col_indices);
    cudaFree(d_in_deg);       cudaFree(d_values);
    cudaFree(d_frontier_a);   cudaFree(d_frontier_b);
    cudaFree(d_frontier_size);cudaFree(d_next_frontier_size);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    fprintf(stderr, "CUDA warp-per-node: %u levels traversed\n", levels);
    return static_cast<double>(elapsed_ms);
}

static bool verify(const vector<float> &seq,
                   const vector<float> &cuda_vals,
                   float tol = 1e-3f) {
    uint32_t N = seq.size();
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < N && mismatches < 10; i++) {
        float diff = fabsf(seq[i] - cuda_vals[i]);
        float denom = fmaxf(fabsf(seq[i]), 1e-8f);
        if (diff / denom > tol) {
            fprintf(stderr, "  Mismatch node %u: seq=%.6f cuda=%.6f (rel=%.6f)\n",
                    i, seq[i], cuda_vals[i], diff / denom);
            mismatches++;
        }
    }
    return mismatches == 0;
}

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
        if (!strcmp(argv[i], "-n")  && i+1 < argc) cfg.num_vertices = atoi(argv[++i]);
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
           "CUDA baseline",  base_ms, seq_ms / base_ms);
    printf("%-22s %10.3f ms  (%.2fx vs seq, %.2fx vs baseline)\n",
           "CUDA block-local",  bl_ms, seq_ms / bl_ms, base_ms / bl_ms);
    printf("%-22s %10.3f ms  (%.2fx vs seq, %.2fx vs baseline)\n",
           "CUDA warp-per-node", warp_ms, seq_ms / warp_ms, base_ms / warp_ms);

    bool p1 = verify(seq_values, base_values);
    bool p2 = verify(seq_values, bl_values);
    bool p3 = verify(seq_values, warp_values);
    printf("\nVerification baseline:     %s\n", p1 ? "PASSED" : "FAILED");
    printf("Verification block-local:  %s\n", p2 ? "PASSED" : "FAILED");
    printf("Verification warp-per-node:%s\n", p3 ? " PASSED" : " FAILED");

    return 0;
}
