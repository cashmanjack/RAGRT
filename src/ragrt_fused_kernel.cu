#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_TASKS 1048576
#define NUM_3D_SUBSPACES 32
#define MIN_BASE_SCORE 0.0f

__device__ __forceinline__ void atomicMaxHalf(half* addr, half val) {
#if __CUDA_ARCH__ >= 700
    unsigned short* address_as_us = (unsigned short*)addr;
    unsigned short old = *address_as_us, assumed;
    while (__half_as_ushort(val) > old) {
        assumed = old;
        old = atomicCAS(address_as_us, assumed, __half_as_ushort(val));
        if (assumed == old) break;
    }
#endif
}

__global__ void collect_posting_tasks_csr_warp_kernel(
    const int* __restrict__ out_c, const float* __restrict__ out_v, const int* __restrict__ out_hit_count,
    const int* __restrict__ topc, const float* __restrict__ scores,
    const int* __restrict__ csr_row_ptrs, const uint8_t* __restrict__ csr_col_eids,
    const int64_t* __restrict__ csr_offsets, const uint16_t* __restrict__ csr_lengths,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    int* __restrict__ d_num_tasks, int* __restrict__ d_num_dropped,
    int* __restrict__ d_task_qt, int64_t* __restrict__ d_task_offset,
    int* __restrict__ d_task_length, float* __restrict__ d_task_base_score
) {
    __shared__ int   s_top_eid[8][64];
    __shared__ float s_top_val[8][64];

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    if (warp_id >= total_rays) return;

    int qt = warp_id / NUM_3D_SUBSPACES;
    int s  = warp_id % NUM_3D_SUBSPACES;

    if (qt >= ntok || s >= NUM_3D_SUBSPACES) return;

    int n_hits = out_hit_count[warp_id];
    if (n_hits > max_hits) n_hits = max_hits;
    int want = (k_eids < n_hits) ? k_eids : n_hits;
    int nsel = 0;

    if (lane_id == 0 && want > 0) {
        int base_hit = warp_id * max_hits;
        for (int r = 0; r < want; ++r) {
            float best_v = -1e30f;
            int   best_i = -1;
            for (int i = 0; i < n_hits; ++i) {
                int eid = out_c[base_hit + i];
                if (eid < 0 || eid >= 256) continue;
                bool dup = false;
                for (int k = 0; k < nsel; ++k) {
                    if (s_top_eid[warp_in_block][k] == eid) { dup = true; break; }
                }
                if (dup) continue;
                float v = out_v[base_hit + i];
                if (v > best_v) { best_v = v; best_i = i; }
            }
            if (best_i < 0) break;
            s_top_eid[warp_in_block][nsel] = out_c[base_hit + best_i];
            s_top_val[warp_in_block][nsel] = best_v;
            ++nsel;
        }
    }
    __syncwarp();
    nsel = __shfl_sync(0xFFFFFFFF, nsel, 0);

    int total_combos = nsel * k_cent;
    int64_t s_row_base = (int64_t)s * (num_centroids + 1);

    for (int combo = lane_id; combo < total_combos; combo += 32) {
        int h = combo / k_cent;
        int c = combo % k_cent;

        int eid = s_top_eid[warp_in_block][h];
        int cid = topc[qt * k_cent + c];
        if (cid < 0 || cid >= num_centroids) continue;

        int row_start = csr_row_ptrs[s_row_base + cid];
        int row_end   = csr_row_ptrs[s_row_base + cid + 1];
        if (row_end <= row_start) continue;

        // Binary search on uint8_t EIDs (0 .. 255)
        int found_idx = -1;
        int lo = row_start, hi = row_end;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            int e = (int)csr_col_eids[mid];
            if (e == eid) { found_idx = mid; break; }
            else if (e < eid) lo = mid + 1;
            else hi = mid;
        }

        if (found_idx >= 0) {
            int length = (int)csr_lengths[found_idx];
            if (length <= 0) continue;

            float base_score = scores[qt * num_centroids + cid] + s_top_val[warp_in_block][h];
            if (base_score <= MIN_BASE_SCORE) continue;

            int task_idx = atomicAdd(d_num_tasks, 1);
            if (task_idx < MAX_TASKS) {
                d_task_qt[task_idx]         = qt;
                d_task_offset[task_idx]     = csr_offsets[found_idx];
                d_task_length[task_idx]     = length;
                d_task_base_score[task_idx] = base_score;
            } else {
                atomicAdd(d_num_dropped, 1);
            }
        }
    }
}

__global__ void cooperative_score_passages_csr_fp16_filtered_kernel(
    const int* __restrict__ d_num_tasks, int* __restrict__ d_queue_counter,
    const int* __restrict__ d_task_qt, const int64_t* __restrict__ d_task_offset,
    const int* __restrict__ d_task_length, const float* __restrict__ d_task_base_score,
    const uint8_t* __restrict__ map_packed_24, int num_passages, half* __restrict__ qt_pid_max_fp16,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask
) {
    int total_tasks = *d_num_tasks;
    if (total_tasks > MAX_TASKS) total_tasks = MAX_TASKS;
    int lane_id = threadIdx.x % 32;

    while (true) {
        int task_id = 0;
        if (lane_id == 0) task_id = atomicAdd(d_queue_counter, 1);
        task_id = __shfl_sync(0xFFFFFFFF, task_id, 0);

        if (task_id >= total_tasks) break;

        int qt           = d_task_qt[task_id];
        int64_t offset   = d_task_offset[task_id];
        int length       = d_task_length[task_id];
        half base_score_h = __float2half(d_task_base_score[task_id]);

        int prev_pid = -1;
        for (int chunk = 0; chunk < length; chunk += 32) {
            int idx = chunk + lane_id;
            int pid = -1;
            if (idx < length) {
                const uint8_t* p24 = map_packed_24 + (offset + idx) * 3;
                pid = p24[0] | (p24[1] << 8) | (p24[2] << 16);
            }
            int left_pid = __shfl_up_sync(0xFFFFFFFF, pid, 1);
            bool dup = (pid >= 0) &&
                       ((lane_id > 0 && pid == left_pid) || (lane_id == 0 && pid == prev_pid));
            if (idx < length && !dup && pid >= 0 && pid < num_passages) {
                if (query_mask == 0 || (doc_predicates[pid] & query_mask) != 0) {
                    atomicMaxHalf(&qt_pid_max_fp16[qt * num_passages + pid], base_score_h);
                }
            }
            prev_pid = __shfl_sync(0xFFFFFFFF, pid, 31);
        }
    }
}

__global__ void sum_passages_fp16_to_fp32_kernel(
    const half* __restrict__ qt_pid_max_fp16, int ntok, int num_passages, float* __restrict__ passage_scores
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_passages) return;

    float total = 0.0f;
    #pragma unroll 8
    for (int q = 0; q < ntok; ++q) {
        float val = __half2float(qt_pid_max_fp16[q * num_passages + pid]);
        if (val > 0.0f) total += val;
    }
    passage_scores[pid] = total;
}

__global__ void rt_maxsim_rescore_from_approx_kernel(
    const half* __restrict__ approx,
    const int*  __restrict__ pids,
    int num_cands, int ntok, int num_passages,
    float* __restrict__ out_scores
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cands) return;
    int pid = pids[i];
    float total = 0.0f;
    if (pid >= 0 && pid < num_passages) {
        #pragma unroll 8
        for (int q = 0; q < ntok; ++q) {
            float val = __half2float(approx[q * num_passages + pid]);
            if (val > 0.0f) total += val;
        }
    }
    out_scores[i] = total;
}

static bool g_buffers_allocated = false;
static int *g_d_num_tasks[2]={nullptr, nullptr}, *g_d_task_queue_counter[2]={nullptr, nullptr}, *g_d_task_qt[2]={nullptr, nullptr}, *g_d_task_length[2]={nullptr, nullptr};
static int *g_d_num_dropped_tasks[2]={nullptr, nullptr};
static int64_t *g_d_task_offset[2]={nullptr, nullptr};
static float *g_d_task_base_score[2]={nullptr, nullptr};
static half *g_d_qt_pid_max_fp16[2]={nullptr, nullptr};
static int g_allocated_passages = 0;

static void ensure_alloc(int num_passages) {
    if (!g_buffers_allocated || g_allocated_passages != num_passages) {
        if (g_buffers_allocated) {
            for (int b = 0; b < 2; ++b) {
                cudaFree(g_d_num_tasks[b]);
                cudaFree(g_d_task_queue_counter[b]);
                cudaFree(g_d_task_qt[b]);
                cudaFree(g_d_task_offset[b]);
                cudaFree(g_d_task_length[b]);
                cudaFree(g_d_task_base_score[b]);
                cudaFree(g_d_qt_pid_max_fp16[b]);
                cudaFree(g_d_num_dropped_tasks[b]);
            }
        }
        for (int b = 0; b < 2; ++b) {
            cudaMalloc(&g_d_num_tasks[b], sizeof(int));
            cudaMalloc(&g_d_task_queue_counter[b], sizeof(int));
            cudaMalloc(&g_d_num_dropped_tasks[b], sizeof(int));
            cudaMalloc(&g_d_task_qt[b], MAX_TASKS * sizeof(int));
            cudaMalloc(&g_d_task_offset[b], MAX_TASKS * sizeof(int64_t));
            cudaMalloc(&g_d_task_length[b], MAX_TASKS * sizeof(int));
            cudaMalloc(&g_d_task_base_score[b], MAX_TASKS * sizeof(float));
            cudaMalloc(&g_d_qt_pid_max_fp16[b], 32 * num_passages * sizeof(half));
        }
        g_allocated_passages = num_passages;
        g_buffers_allocated = true;
    }
}

extern "C" void launch_ragrt_fused_stage23_csr256(
    const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
    const int* d_topc, const float* d_scores,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    const int* d_csr_row_ptrs, const uint8_t* d_csr_col_eids,
    const int64_t* d_csr_offsets, const uint16_t* d_csr_lengths, const uint8_t* d_map_packed_24,
    float* d_passage_scores, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask,
    int buf_idx, cudaStream_t stream
) {
    ensure_alloc(num_passages);
    int active_ntok = (ntok <= 32) ? ntok : 32;

    cudaMemsetAsync(g_d_num_tasks[buf_idx], 0, sizeof(int), stream);
    cudaMemsetAsync(g_d_task_queue_counter[buf_idx], 0, sizeof(int), stream);
    cudaMemsetAsync(g_d_num_dropped_tasks[buf_idx], 0, sizeof(int), stream);
    cudaMemsetAsync(g_d_qt_pid_max_fp16[buf_idx], 0, active_ntok * num_passages * sizeof(half), stream);

    int threads = 256;
    int blocks_collect = (total_rays * 32 + threads - 1) / threads;
    collect_posting_tasks_csr_warp_kernel<<<blocks_collect, threads, 0, stream>>>(
        d_out_c, d_out_v, d_out_hit_count, d_topc, d_scores,
        d_csr_row_ptrs, d_csr_col_eids, d_csr_offsets, d_csr_lengths,
        total_rays, max_hits, k_eids, k_cent, active_ntok, num_centroids,
        g_d_num_tasks[buf_idx], g_d_num_dropped_tasks[buf_idx],
        g_d_task_qt[buf_idx], g_d_task_offset[buf_idx], g_d_task_length[buf_idx], g_d_task_base_score[buf_idx]
    );

    int coop_blocks = 108;
    cooperative_score_passages_csr_fp16_filtered_kernel<<<coop_blocks, threads, 0, stream>>>(
        g_d_num_tasks[buf_idx], g_d_task_queue_counter[buf_idx], g_d_task_qt[buf_idx], g_d_task_offset[buf_idx],
        g_d_task_length[buf_idx], g_d_task_base_score[buf_idx], d_map_packed_24, num_passages, g_d_qt_pid_max_fp16[buf_idx],
        d_doc_predicates, query_mask
    );

    int pass_blocks = (num_passages + threads - 1) / threads;
    sum_passages_fp16_to_fp32_kernel<<<pass_blocks, threads, 0, stream>>>(
        g_d_qt_pid_max_fp16[buf_idx], active_ntok, num_passages, d_passage_scores
    );
}

extern "C" half* ragrt_approx_score_buffer(int buf_idx) {
    if (!g_buffers_allocated || buf_idx < 0 || buf_idx > 1) return nullptr;
    return g_d_qt_pid_max_fp16[buf_idx];
}

extern "C" int ragrt_dropped_task_count(int buf_idx) {
    int h = 0;
    if (!g_buffers_allocated || buf_idx < 0 || buf_idx > 1) return 0;
    cudaMemcpy(&h, g_d_num_dropped_tasks[buf_idx], sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}

extern "C" void launch_rt_maxsim_rescore_from_approx(
    const half* d_approx, const int* d_pids,
    int num_cands, int ntok, int num_passages,
    float* d_out_scores, cudaStream_t stream
) {
    if (num_cands <= 0) return;
    int active_ntok = (ntok <= 32) ? ntok : 32;
    int threads = 256;
    int blocks = (num_cands + threads - 1) / threads;
    rt_maxsim_rescore_from_approx_kernel<<<blocks, threads, 0, stream>>>(
        d_approx, d_pids, num_cands, active_ntok, num_passages, d_out_scores
    );
}




// Append to ragrt_fused_kernel.cu
extern "C" void launch_ragrt_stage2_only(
    const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
    const int* d_topc, const float* d_scores,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    const int* d_csr_row_ptrs, const uint8_t* d_csr_col_eids,
    const int64_t* d_csr_offsets, const uint16_t* d_csr_lengths,
    int buf_idx, cudaStream_t stream
) {
    ensure_alloc(g_allocated_passages > 0 ? g_allocated_passages : 2430000);
    int active_ntok = (ntok <= 32) ? ntok : 32;

    cudaMemsetAsync(g_d_num_tasks[buf_idx], 0, sizeof(int), stream);
    cudaMemsetAsync(g_d_task_queue_counter[buf_idx], 0, sizeof(int), stream);
    cudaMemsetAsync(g_d_num_dropped_tasks[buf_idx], 0, sizeof(int), stream);

    int threads = 256;
    int blocks_collect = (total_rays * 32 + threads - 1) / threads;
    collect_posting_tasks_csr_warp_kernel<<<blocks_collect, threads, 0, stream>>>(
        d_out_c, d_out_v, d_out_hit_count, d_topc, d_scores,
        d_csr_row_ptrs, d_csr_col_eids, d_csr_offsets, d_csr_lengths,
        total_rays, max_hits, k_eids, k_cent, active_ntok, num_centroids,
        g_d_num_tasks[buf_idx], g_d_num_dropped_tasks[buf_idx],
        g_d_task_qt[buf_idx], g_d_task_offset[buf_idx], g_d_task_length[buf_idx], g_d_task_base_score[buf_idx]
    );
}

extern "C" void launch_ragrt_stage3_only(
    const uint8_t* d_map_packed_24, float* d_passage_scores, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask, int ntok,
    int buf_idx, cudaStream_t stream
) {
    int active_ntok = (ntok <= 32) ? ntok : 32;
    cudaMemsetAsync(g_d_qt_pid_max_fp16[buf_idx], 0, active_ntok * num_passages * sizeof(half), stream);

    int threads = 256;
    int coop_blocks = 108;
    cooperative_score_passages_csr_fp16_filtered_kernel<<<coop_blocks, threads, 0, stream>>>(
        g_d_num_tasks[buf_idx], g_d_task_queue_counter[buf_idx], g_d_task_qt[buf_idx], g_d_task_offset[buf_idx],
        g_d_task_length[buf_idx], g_d_task_base_score[buf_idx], d_map_packed_24, num_passages, g_d_qt_pid_max_fp16[buf_idx],
        d_doc_predicates, query_mask
    );

    int pass_blocks = (num_passages + threads - 1) / threads;
    sum_passages_fp16_to_fp32_kernel<<<pass_blocks, threads, 0, stream>>>(
        g_d_qt_pid_max_fp16[buf_idx], active_ntok, num_passages, d_passage_scores
    );
}
